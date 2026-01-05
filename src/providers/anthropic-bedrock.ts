/**
 * Anthropic Bedrock provider for pi-ai.
 * 
 * This is a modified version of the standard Anthropic provider that uses
 * the @anthropic-ai/bedrock-sdk instead of the regular @anthropic-ai/sdk.
 * 
 * Authentication is handled via AWS credentials (environment variables or
 * ~/.aws/credentials file) instead of an Anthropic API key.
 */

import { AnthropicBedrock } from "@anthropic-ai/bedrock-sdk";
import type {
  Api,
  AssistantMessage,
  Context,
  Message,
  Model,
  StreamOptions,
  TextContent,
  ThinkingContent,
  ToolCall,
  ToolResultMessage,
  UserMessage,
  ImageContent,
} from "@mariozechner/pi-ai";
import { calculateCost } from "@mariozechner/pi-ai/dist/models.js";
import { AssistantMessageEventStream } from "@mariozechner/pi-ai/dist/utils/event-stream.js";
import { parseStreamingJson } from "@mariozechner/pi-ai/dist/utils/json-parse.js";
import { sanitizeSurrogates } from "@mariozechner/pi-ai/dist/utils/sanitize-unicode.js";
import { transformMessages } from "@mariozechner/pi-ai/dist/providers/transorm-messages.js";

// Define the Bedrock API type - this extends the standard Api type
export type BedrockApi = "anthropic-bedrock";

export interface AnthropicBedrockOptions extends StreamOptions {
  thinkingEnabled?: boolean;
  thinkingBudgetTokens?: number;
  interleavedThinking?: boolean;
  toolChoice?: "auto" | "any" | "none" | { type: "tool"; name: string };
  // AWS-specific options
  awsRegion?: string;
  awsAccessKey?: string;
  awsSecretKey?: string;
  awsSessionToken?: string;
}

interface ContentBlockWithIndex {
  index?: number;
}

type TextContentWithIndex = TextContent & ContentBlockWithIndex;
type ThinkingContentWithIndex = ThinkingContent & ContentBlockWithIndex;
type ToolCallWithIndex = ToolCall & ContentBlockWithIndex & { partialJson?: string };
type ContentWithIndex = TextContentWithIndex | ThinkingContentWithIndex | ToolCallWithIndex;

/**
 * Convert content blocks to Anthropic API format
 */
function convertContentBlocks(content: Array<TextContent | ImageContent>) {
  // If only text blocks, return as concatenated string for simplicity
  const hasImages = content.some((c) => c.type === "image");
  if (!hasImages) {
    return sanitizeSurrogates(content.map((c) => (c as TextContent).text || "").join("\n"));
  }
  // If we have images, convert to content block array
  const blocks = content.map((block) => {
    if (block.type === "text") {
      return {
        type: "text" as const,
        text: sanitizeSurrogates(block.text),
      };
    }
    const imgBlock = block as ImageContent;
    return {
      type: "image" as const,
      source: {
        type: "base64" as const,
        media_type: imgBlock.mimeType as "image/jpeg" | "image/png" | "image/gif" | "image/webp",
        data: imgBlock.data,
      },
    };
  });
  // If only images (no text), add placeholder text block
  const hasText = blocks.some((b) => b.type === "text");
  if (!hasText) {
    blocks.unshift({
      type: "text" as const,
      text: "(see attached image)",
    });
  }
  return blocks;
}

export const streamAnthropicBedrock = (
  model: Model<Api>,
  context: Context,
  options?: AnthropicBedrockOptions
): AssistantMessageEventStream => {
  const stream = new AssistantMessageEventStream();
  
  (async () => {
    const output: AssistantMessage = {
      role: "assistant",
      content: [],
      api: "anthropic-messages" as Api, // Bedrock uses the same message format
      provider: model.provider,
      model: model.id,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "stop",
      timestamp: Date.now(),
    };
    
    try {
      const client = createBedrockClient(model, options);
      const params = buildParams(model, context, options);
      
      const anthropicStream = client.messages.stream(
        { ...params, stream: true },
        { signal: options?.signal }
      );
      
      stream.push({ type: "start", partial: output });
      const blocks = output.content as ContentWithIndex[];
      
      for await (const event of anthropicStream) {
        if (event.type === "message_start") {
          output.usage.input = (event.message.usage as any).input_tokens || 0;
          output.usage.output = (event.message.usage as any).output_tokens || 0;
          output.usage.cacheRead = (event.message.usage as any).cache_read_input_tokens || 0;
          output.usage.cacheWrite = (event.message.usage as any).cache_creation_input_tokens || 0;
          output.usage.totalTokens =
            output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
          calculateCost(model, output.usage);
        } else if (event.type === "content_block_start") {
          if (event.content_block.type === "text") {
            const block: TextContentWithIndex = {
              type: "text",
              text: "",
              index: event.index,
            };
            output.content.push(block);
            stream.push({ type: "text_start", contentIndex: output.content.length - 1, partial: output });
          } else if (event.content_block.type === "thinking") {
            const block: ThinkingContentWithIndex = {
              type: "thinking",
              thinking: "",
              thinkingSignature: "",
              index: event.index,
            };
            output.content.push(block);
            stream.push({ type: "thinking_start", contentIndex: output.content.length - 1, partial: output });
          } else if (event.content_block.type === "tool_use") {
            const block: ToolCallWithIndex = {
              type: "toolCall",
              id: event.content_block.id,
              name: event.content_block.name,
              arguments: event.content_block.input as Record<string, any>,
              partialJson: "",
              index: event.index,
            };
            output.content.push(block);
            stream.push({ type: "toolcall_start", contentIndex: output.content.length - 1, partial: output });
          }
        } else if (event.type === "content_block_delta") {
          if (event.delta.type === "text_delta") {
            const index = blocks.findIndex((b) => b.index === event.index);
            const block = blocks[index];
            if (block && block.type === "text") {
              block.text += event.delta.text;
              stream.push({
                type: "text_delta",
                contentIndex: index,
                delta: event.delta.text,
                partial: output,
              });
            }
          } else if (event.delta.type === "thinking_delta") {
            const index = blocks.findIndex((b) => b.index === event.index);
            const block = blocks[index];
            if (block && block.type === "thinking") {
              block.thinking += (event.delta as any).thinking;
              stream.push({
                type: "thinking_delta",
                contentIndex: index,
                delta: (event.delta as any).thinking,
                partial: output,
              });
            }
          } else if (event.delta.type === "input_json_delta") {
            const index = blocks.findIndex((b) => b.index === event.index);
            const block = blocks[index] as ToolCallWithIndex;
            if (block && block.type === "toolCall") {
              block.partialJson += event.delta.partial_json;
              block.arguments = parseStreamingJson(block.partialJson || "");
              stream.push({
                type: "toolcall_delta",
                contentIndex: index,
                delta: event.delta.partial_json,
                partial: output,
              });
            }
          } else if ((event.delta as any).type === "signature_delta") {
            const index = blocks.findIndex((b) => b.index === event.index);
            const block = blocks[index];
            if (block && block.type === "thinking") {
              block.thinkingSignature = block.thinkingSignature || "";
              block.thinkingSignature += (event.delta as any).signature;
            }
          }
        } else if (event.type === "content_block_stop") {
          const index = blocks.findIndex((b) => b.index === event.index);
          const block = blocks[index];
          if (block) {
            delete block.index;
            if (block.type === "text") {
              stream.push({
                type: "text_end",
                contentIndex: index,
                content: block.text,
                partial: output,
              });
            } else if (block.type === "thinking") {
              stream.push({
                type: "thinking_end",
                contentIndex: index,
                content: block.thinking,
                partial: output,
              });
            } else if (block.type === "toolCall") {
              const toolBlock = block as ToolCallWithIndex;
              toolBlock.arguments = parseStreamingJson(toolBlock.partialJson || "");
              delete toolBlock.partialJson;
              stream.push({
                type: "toolcall_end",
                contentIndex: index,
                toolCall: block as ToolCall,
                partial: output,
              });
            }
          }
        } else if (event.type === "message_delta") {
          if ((event.delta as any).stop_reason) {
            output.stopReason = mapStopReason((event.delta as any).stop_reason);
          }
          output.usage.input = (event.usage as any).input_tokens || 0;
          output.usage.output = (event.usage as any).output_tokens || 0;
          output.usage.cacheRead = (event.usage as any).cache_read_input_tokens || 0;
          output.usage.cacheWrite = (event.usage as any).cache_creation_input_tokens || 0;
          output.usage.totalTokens =
            output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
          calculateCost(model, output.usage);
        }
      }
      
      if (options?.signal?.aborted) {
        throw new Error("Request was aborted");
      }
      if (output.stopReason === "aborted" || output.stopReason === "error") {
        throw new Error("An unknown error occurred");
      }
      
      stream.push({ type: "done", reason: output.stopReason as "stop" | "length" | "toolUse", message: output });
      stream.end();
    } catch (error) {
      for (const block of output.content as ContentWithIndex[]) delete block.index;
      output.stopReason = options?.signal?.aborted ? "aborted" : "error";
      output.errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
      stream.push({ type: "error", reason: output.stopReason as "aborted" | "error", error: output });
      stream.end();
    }
  })();
  
  return stream;
};

function createBedrockClient(
  model: Model<Api>,
  options?: AnthropicBedrockOptions
): AnthropicBedrock {
  const betaFeatures = ["fine-grained-tool-streaming-2025-05-14"];
  if (options?.interleavedThinking !== false) {
    betaFeatures.push("interleaved-thinking-2025-05-14");
  }
  
  const defaultHeaders = {
    accept: "application/json",
    "anthropic-beta": betaFeatures.join(","),
    ...(model.headers || {}),
  };
  
  // AWS credentials can come from:
  // 1. Explicit options
  // 2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
  // 3. AWS credential chain (~/.aws/credentials)
  const clientOptions: ConstructorParameters<typeof AnthropicBedrock>[0] = {
    awsRegion: options?.awsRegion || process.env.AWS_REGION || "us-east-1",
    defaultHeaders,
  };
  
  if (options?.awsAccessKey && options?.awsSecretKey) {
    clientOptions.awsAccessKey = options.awsAccessKey;
    clientOptions.awsSecretKey = options.awsSecretKey;
    if (options?.awsSessionToken) {
      clientOptions.awsSessionToken = options.awsSessionToken;
    }
  }
  
  // If model has a custom baseUrl, use it (for VPC endpoints, etc.)
  if (model.baseUrl && !model.baseUrl.includes("api.anthropic.com")) {
    clientOptions.baseURL = model.baseUrl;
  }
  
  return new AnthropicBedrock(clientOptions);
}

function buildParams(
  model: Model<Api>,
  context: Context,
  options?: AnthropicBedrockOptions
): any {
  const params: any = {
    model: model.id,
    messages: convertMessages(context.messages, model),
    max_tokens: options?.maxTokens || (model.maxTokens / 3) | 0,
    stream: true,
  };
  
  if (context.systemPrompt) {
    params.system = [
      {
        type: "text",
        text: sanitizeSurrogates(context.systemPrompt),
        cache_control: {
          type: "ephemeral",
        },
      },
    ];
  }
  
  if (options?.temperature !== undefined) {
    params.temperature = options.temperature;
  }
  
  if (context.tools) {
    params.tools = convertTools(context.tools);
  }
  
  if (options?.thinkingEnabled && model.reasoning) {
    params.thinking = {
      type: "enabled",
      budget_tokens: options.thinkingBudgetTokens || 1024,
    };
  }
  
  if (options?.toolChoice) {
    if (typeof options.toolChoice === "string") {
      params.tool_choice = { type: options.toolChoice };
    } else {
      params.tool_choice = options.toolChoice;
    }
  }
  
  return params;
}

function sanitizeToolCallId(id: string): string {
  return id.replace(/[^a-zA-Z0-9_-]/g, "_");
}

function convertMessages(messages: Message[], model: Model<Api>): any[] {
  const params: any[] = [];
  const transformedMessages = transformMessages(messages, model) as Message[];
  
  for (let i = 0; i < transformedMessages.length; i++) {
    const msg = transformedMessages[i];
    
    if (msg.role === "user") {
      const userMsg = msg as UserMessage;
      if (typeof userMsg.content === "string") {
        if (userMsg.content.trim().length > 0) {
          params.push({
            role: "user",
            content: sanitizeSurrogates(userMsg.content),
          });
        }
      } else {
        const blocks = userMsg.content.map((item) => {
          if (item.type === "text") {
            return {
              type: "text" as const,
              text: sanitizeSurrogates(item.text),
            };
          } else {
            const imgItem = item as ImageContent;
            return {
              type: "image" as const,
              source: {
                type: "base64" as const,
                media_type: imgItem.mimeType,
                data: imgItem.data,
              },
            };
          }
        });
        
        let filteredBlocks = !model?.input.includes("image") 
          ? blocks.filter((b: any) => b.type !== "image") 
          : blocks;
        filteredBlocks = filteredBlocks.filter((b: any) => {
          if (b.type === "text") {
            return b.text.trim().length > 0;
          }
          return true;
        });
        
        if (filteredBlocks.length === 0) continue;
        params.push({
          role: "user",
          content: filteredBlocks,
        });
      }
    } else if (msg.role === "assistant") {
      const assistantMsg = msg as AssistantMessage;
      const blocks: any[] = [];
      for (const block of assistantMsg.content) {
        if (block.type === "text") {
          if (block.text.trim().length === 0) continue;
          blocks.push({
            type: "text",
            text: sanitizeSurrogates(block.text),
          });
        } else if (block.type === "thinking") {
          if (block.thinking.trim().length === 0) continue;
          if (!block.thinkingSignature || block.thinkingSignature.trim().length === 0) {
            blocks.push({
              type: "text",
              text: sanitizeSurrogates(block.thinking),
            });
          } else {
            blocks.push({
              type: "thinking",
              thinking: sanitizeSurrogates(block.thinking),
              signature: block.thinkingSignature,
            });
          }
        } else if (block.type === "toolCall") {
          blocks.push({
            type: "tool_use",
            id: sanitizeToolCallId(block.id),
            name: block.name,
            input: block.arguments,
          });
        }
      }
      if (blocks.length === 0) continue;
      params.push({
        role: "assistant",
        content: blocks,
      });
    } else if (msg.role === "toolResult") {
      const toolMsg = msg as ToolResultMessage;
      const toolResults: any[] = [];
      toolResults.push({
        type: "tool_result",
        tool_use_id: sanitizeToolCallId(toolMsg.toolCallId),
        content: convertContentBlocks(toolMsg.content as Array<TextContent | ImageContent>),
        is_error: toolMsg.isError,
      });
      
      let j = i + 1;
      while (j < transformedMessages.length && transformedMessages[j].role === "toolResult") {
        const nextMsg = transformedMessages[j] as ToolResultMessage;
        toolResults.push({
          type: "tool_result",
          tool_use_id: sanitizeToolCallId(nextMsg.toolCallId),
          content: convertContentBlocks(nextMsg.content as Array<TextContent | ImageContent>),
          is_error: nextMsg.isError,
        });
        j++;
      }
      i = j - 1;
      
      params.push({
        role: "user",
        content: toolResults,
      });
    }
  }
  
  // Add cache_control to the last user message
  if (params.length > 0) {
    const lastMessage = params[params.length - 1];
    if (lastMessage.role === "user") {
      if (Array.isArray(lastMessage.content)) {
        const lastBlock = lastMessage.content[lastMessage.content.length - 1];
        if (lastBlock && (lastBlock.type === "text" || lastBlock.type === "image" || lastBlock.type === "tool_result")) {
          lastBlock.cache_control = { type: "ephemeral" };
        }
      }
    }
  }
  
  return params;
}

function convertTools(tools: any[]): any[] {
  if (!tools) return [];
  return tools.map((tool) => {
    const jsonSchema = tool.parameters;
    return {
      name: tool.name,
      description: tool.description,
      input_schema: {
        type: "object",
        properties: jsonSchema.properties || {},
        required: jsonSchema.required || [],
      },
    };
  });
}

function mapStopReason(reason: string): "stop" | "length" | "toolUse" | "error" | "aborted" {
  switch (reason) {
    case "end_turn":
      return "stop";
    case "max_tokens":
      return "length";
    case "tool_use":
      return "toolUse";
    case "refusal":
      return "error";
    case "pause_turn":
      return "stop";
    case "stop_sequence":
      return "stop";
    default:
      return "stop";
  }
}
