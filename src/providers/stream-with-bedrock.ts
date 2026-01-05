/**
 * Custom stream function that routes to the appropriate provider,
 * including support for Anthropic Bedrock.
 * 
 * This is a drop-in replacement for pi-ai's streamSimple that adds
 * support for the "anthropic-bedrock" provider.
 */

import {
  type Api,
  type Model,
  type Context,
  type SimpleStreamOptions,
  type AssistantMessageEventStream,
  streamSimple,
} from "@mariozechner/pi-ai";
import {
  streamAnthropicBedrock,
  type AnthropicBedrockOptions,
} from "./anthropic-bedrock.js";

/**
 * Extended stream function that supports Anthropic Bedrock in addition
 * to all standard pi-ai providers.
 * 
 * For Bedrock models:
 * - Set provider to "anthropic-bedrock" in models.json
 * - AWS credentials are read from environment or ~/.aws/credentials
 * - Model ID should be the Bedrock model identifier (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
 */
export function streamWithBedrock(
  model: Model<Api>,
  context: Context,
  options?: SimpleStreamOptions
): AssistantMessageEventStream {
  // Check if this is a Bedrock model by provider name
  if (model.provider === "anthropic-bedrock") {
    // Map SimpleStreamOptions to AnthropicBedrockOptions
    const bedrockOptions: AnthropicBedrockOptions = {
      temperature: options?.temperature,
      maxTokens: options?.maxTokens,
      signal: options?.signal,
      apiKey: options?.apiKey,
    };
    
    // Handle reasoning/thinking settings
    if (options?.reasoning && model.reasoning) {
      bedrockOptions.thinkingEnabled = true;
      // Map reasoning effort to thinking budget
      const budgets = {
        minimal: 1024,
        low: 2048,
        medium: 8192,
        high: 16384,
        xhigh: 16384, // Clamp xhigh to high for now
      };
      bedrockOptions.thinkingBudgetTokens = budgets[options.reasoning];
      // Increase maxTokens to account for thinking tokens
      const minOutputTokens = 1024;
      const maxTokens = Math.min(
        (bedrockOptions.maxTokens || 32000) + bedrockOptions.thinkingBudgetTokens,
        model.maxTokens
      );
      if (maxTokens <= bedrockOptions.thinkingBudgetTokens) {
        bedrockOptions.thinkingBudgetTokens = Math.max(0, maxTokens - minOutputTokens);
      }
      bedrockOptions.maxTokens = maxTokens;
    }
    
    return streamAnthropicBedrock(model, context, bedrockOptions);
  }
  
  // For all other providers, use the standard pi-ai streamSimple
  return streamSimple(model, context, options);
}

/**
 * Check if a model is a Bedrock model.
 */
export function isBedrockModel(model: Model<Api>): boolean {
  return model.provider === "anthropic-bedrock";
}
