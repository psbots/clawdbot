import type { AgentTool, AgentToolResult } from "@mariozechner/pi-agent-core";
import {
  codingTools,
  createEditTool,
  createReadTool,
  createWriteTool,
  readTool,
} from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";

import { detectMime } from "../media/mime.js";
import { startWebLoginWithQr, waitForWebLogin } from "../web/login-qr.js";
import {
  type BashToolDefaults,
  createBashTool,
  createProcessTool,
  type ProcessToolDefaults,
} from "./bash-tools.js";
import { createClawdisTools } from "./clawdis-tools.js";
import type { SandboxContext, SandboxToolPolicy } from "./sandbox.js";
import { assertSandboxPath } from "./sandbox-paths.js";
import { sanitizeToolResultImages } from "./tool-images.js";

// NOTE(steipete): Upstream read now does file-magic MIME detection; we keep the wrapper
// to normalize payloads and sanitize oversized images before they hit providers.
type ToolContentBlock = AgentToolResult<unknown>["content"][number];
type ImageContentBlock = Extract<ToolContentBlock, { type: "image" }>;
type TextContentBlock = Extract<ToolContentBlock, { type: "text" }>;

async function sniffMimeFromBase64(
  base64: string,
): Promise<string | undefined> {
  const trimmed = base64.trim();
  if (!trimmed) return undefined;

  const take = Math.min(256, trimmed.length);
  const sliceLen = take - (take % 4);
  if (sliceLen < 8) return undefined;

  try {
    const head = Buffer.from(trimmed.slice(0, sliceLen), "base64");
    return await detectMime({ buffer: head });
  } catch {
    return undefined;
  }
}

function rewriteReadImageHeader(text: string, mimeType: string): string {
  // pi-coding-agent uses: "Read image file [image/png]"
  if (text.startsWith("Read image file [") && text.endsWith("]")) {
    return `Read image file [${mimeType}]`;
  }
  return text;
}

async function normalizeReadImageResult(
  result: AgentToolResult<unknown>,
  filePath: string,
): Promise<AgentToolResult<unknown>> {
  const content = Array.isArray(result.content) ? result.content : [];

  const image = content.find(
    (b): b is ImageContentBlock =>
      !!b &&
      typeof b === "object" &&
      (b as { type?: unknown }).type === "image" &&
      typeof (b as { data?: unknown }).data === "string" &&
      typeof (b as { mimeType?: unknown }).mimeType === "string",
  );
  if (!image) return result;

  if (!image.data.trim()) {
    throw new Error(`read: image payload is empty (${filePath})`);
  }

  const sniffed = await sniffMimeFromBase64(image.data);
  if (!sniffed) return result;

  if (!sniffed.startsWith("image/")) {
    throw new Error(
      `read: file looks like ${sniffed} but was treated as ${image.mimeType} (${filePath})`,
    );
  }

  if (sniffed === image.mimeType) return result;

  const nextContent = content.map((block) => {
    if (
      block &&
      typeof block === "object" &&
      (block as { type?: unknown }).type === "image"
    ) {
      const b = block as ImageContentBlock & { mimeType: string };
      return { ...b, mimeType: sniffed } satisfies ImageContentBlock;
    }
    if (
      block &&
      typeof block === "object" &&
      (block as { type?: unknown }).type === "text" &&
      typeof (block as { text?: unknown }).text === "string"
    ) {
      const b = block as TextContentBlock & { text: string };
      return {
        ...b,
        text: rewriteReadImageHeader(b.text, sniffed),
      } satisfies TextContentBlock;
    }
    return block;
  });

  return { ...result, content: nextContent };
}

// biome-ignore lint/suspicious/noExplicitAny: TypeBox schema type from pi-agent-core uses a different module instance.
type AnyAgentTool = AgentTool<any, unknown>;

function extractEnumValues(schema: unknown): unknown[] | undefined {
  if (!schema || typeof schema !== "object") return undefined;
  const record = schema as Record<string, unknown>;
  if (Array.isArray(record.enum)) return record.enum;
  if ("const" in record) return [record.const];
  return undefined;
}

function mergePropertySchemas(existing: unknown, incoming: unknown): unknown {
  if (!existing) return incoming;
  if (!incoming) return existing;

  const existingEnum = extractEnumValues(existing);
  const incomingEnum = extractEnumValues(incoming);
  if (existingEnum || incomingEnum) {
    const values = Array.from(
      new Set([...(existingEnum ?? []), ...(incomingEnum ?? [])]),
    );
    const merged: Record<string, unknown> = {};
    for (const source of [existing, incoming]) {
      if (!source || typeof source !== "object") continue;
      const record = source as Record<string, unknown>;
      for (const key of ["title", "description", "default", "type"]) {
        if (!(key in merged) && key in record) merged[key] = record[key];
      }
    }
    const types = new Set(
      values.map((value) => (value === null ? "null" : typeof value)),
    );
    if (types.size === 1) merged.type = Array.from(types)[0];
    merged.enum = values;
    return merged;
  }

  if (
    typeof existing === "object" &&
    typeof incoming === "object" &&
    !Array.isArray(existing) &&
    !Array.isArray(incoming)
  ) {
    const e = existing as Record<string, unknown>;
    const i = incoming as Record<string, unknown>;

    // Merge properties
    const mergedProperties: Record<string, unknown> = {
      ...(e.properties as Record<string, unknown>),
    };
    const iProps = (i.properties as Record<string, unknown>) ?? {};
    for (const [key, value] of Object.entries(iProps)) {
      mergedProperties[key] = mergePropertySchemas(mergedProperties[key], value);
    }

    // Merge required (only those present in BOTH if we want to be safe, but for Gemini flattening
    // we often want all possible properties to be optional unless strictly required in all).
    const eReq = Array.isArray(e.required) ? e.required : [];
    const iReq = Array.isArray(i.required) ? i.required : [];
    const mergedRequired = eReq.filter((r) => iReq.includes(r));

    return {
      ...e,
      ...i,
      type: (e.type as string) ?? (i.type as string) ?? "object",
      properties:
        Object.keys(mergedProperties).length > 0 ? mergedProperties : undefined,
      required: mergedRequired.length > 0 ? mergedRequired : undefined,
    };
  }

  return existing;
}

function mergeSchemas(schemas: unknown[]): Record<string, unknown> {
  if (schemas.length === 0) return {};
  let merged = schemas[0];
  for (let i = 1; i < schemas.length; i++) {
    merged = mergePropertySchemas(merged, schemas[i]);
  }
  return (merged as Record<string, unknown>) ?? {};
}

function cleanSchemaForGemini(schema: unknown): unknown {
  if (!schema || typeof schema !== "object") return schema;
  if (Array.isArray(schema)) return schema.map(cleanSchemaForGemini);

  const obj = schema as Record<string, unknown>;

  // If this level has any unions, flatten them first
  if (
    Array.isArray(obj.anyOf) ||
    Array.isArray(obj.oneOf) ||
    Array.isArray(obj.allOf)
  ) {
    const variants = [
      ...(Array.isArray(obj.anyOf) ? obj.anyOf : []),
      ...(Array.isArray(obj.oneOf) ? obj.oneOf : []),
      ...(Array.isArray(obj.allOf) ? obj.allOf : []),
    ];
    if (variants.length > 0) {
      return cleanSchemaForGemini(mergeSchemas(variants));
    }
  }

  const cleaned: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(obj)) {
    // Skip unsupported schema features for Gemini:
    // - patternProperties: not in OpenAPI 3.0 subset
    // - const: convert to enum with single value instead
    if (key === "patternProperties") {
      // Gemini doesn't support patternProperties - skip it
      continue;
    }

    // Convert const to enum (Gemini doesn't support const)
    if (key === "const") {
      cleaned.enum = [value];
      continue;
    }

    // Gemini always requires 'type' at the top level and in properties.
    // If we have anyOf/oneOf/allOf, we should still keep the type if it was provided or merged.

    if (key === "properties" && value && typeof value === "object") {
      // Recursively clean nested properties
      const props = value as Record<string, unknown>;
      cleaned[key] = Object.fromEntries(
        Object.entries(props).map(([k, v]) => [k, cleanSchemaForGemini(v)]),
      );
    } else if (key === "items" && value && typeof value === "object") {
      // Recursively clean array items schema
      cleaned[key] = cleanSchemaForGemini(value);
    } else if (
      key === "additionalProperties" &&
      value &&
      typeof value === "object"
    ) {
      // Recursively clean additionalProperties schema
      cleaned[key] = cleanSchemaForGemini(value);
    } else {
      cleaned[key] = value;
    }
  }

  return cleaned;
}

function normalizeToolParameters(tool: AnyAgentTool): AnyAgentTool {
  const schema =
    tool.parameters && typeof tool.parameters === "object"
      ? (tool.parameters as Record<string, unknown>)
      : undefined;
  if (!schema) return tool;

  return {
    ...tool,
    parameters: cleanSchemaForGemini(schema),
  };
}

function normalizeToolNames(list?: string[]) {
  if (!list) return [];
  return list.map((entry) => entry.trim().toLowerCase()).filter(Boolean);
}

function filterToolsByPolicy(
  tools: AnyAgentTool[],
  policy?: SandboxToolPolicy,
) {
  if (!policy) return tools;
  const deny = new Set(normalizeToolNames(policy.deny));
  const allowRaw = normalizeToolNames(policy.allow);
  const allow = allowRaw.length > 0 ? new Set(allowRaw) : null;
  return tools.filter((tool) => {
    const name = tool.name.toLowerCase();
    if (deny.has(name)) return false;
    if (allow) return allow.has(name);
    return true;
  });
}

function wrapSandboxPathGuard(tool: AnyAgentTool, root: string): AnyAgentTool {
  return {
    ...tool,
    execute: async (toolCallId, args, signal, onUpdate) => {
      const record =
        args && typeof args === "object"
          ? (args as Record<string, unknown>)
          : undefined;
      const filePath = record?.path;
      if (typeof filePath === "string" && filePath.trim()) {
        await assertSandboxPath({ filePath, cwd: root, root });
      }
      return tool.execute(toolCallId, args, signal, onUpdate);
    },
  };
}

function createSandboxedReadTool(root: string) {
  const base = createReadTool(root);
  return wrapSandboxPathGuard(createClawdisReadTool(base), root);
}

function createSandboxedWriteTool(root: string) {
  const base = createWriteTool(root);
  return wrapSandboxPathGuard(base as unknown as AnyAgentTool, root);
}

function createSandboxedEditTool(root: string) {
  const base = createEditTool(root);
  return wrapSandboxPathGuard(base as unknown as AnyAgentTool, root);
}

function createWhatsAppLoginTool(): AnyAgentTool {
  return {
    label: "WhatsApp Login",
    name: "whatsapp_login",
    description:
      "Generate a WhatsApp QR code for linking, or wait for the scan to complete.",
    parameters: Type.Object({
      action: Type.Union([Type.Literal("start"), Type.Literal("wait")]),
      timeoutMs: Type.Optional(Type.Number()),
      force: Type.Optional(Type.Boolean()),
    }),
    execute: async (_toolCallId, args) => {
      const action = (args as { action?: string })?.action ?? "start";
      if (action === "wait") {
        const result = await waitForWebLogin({
          timeoutMs:
            typeof (args as { timeoutMs?: unknown }).timeoutMs === "number"
              ? (args as { timeoutMs?: number }).timeoutMs
              : undefined,
        });
        return {
          content: [{ type: "text", text: result.message }],
          details: { connected: result.connected },
        };
      }

      const result = await startWebLoginWithQr({
        timeoutMs:
          typeof (args as { timeoutMs?: unknown }).timeoutMs === "number"
            ? (args as { timeoutMs?: number }).timeoutMs
            : undefined,
        force:
          typeof (args as { force?: unknown }).force === "boolean"
            ? (args as { force?: boolean }).force
            : false,
      });

      if (!result.qrDataUrl) {
        return {
          content: [
            {
              type: "text",
              text: result.message,
            },
          ],
          details: { qr: false },
        };
      }

      const text = [
        result.message,
        "",
        "Open WhatsApp → Linked Devices and scan:",
        "",
        `![whatsapp-qr](${result.qrDataUrl})`,
      ].join("\n");
      return {
        content: [{ type: "text", text }],
        details: { qr: true },
      };
    },
  };
}

function createClawdisReadTool(base: AnyAgentTool): AnyAgentTool {
  return {
    ...base,
    execute: async (toolCallId, params, signal) => {
      const result = (await base.execute(
        toolCallId,
        params,
        signal,
      )) as AgentToolResult<unknown>;
      const record =
        params && typeof params === "object"
          ? (params as Record<string, unknown>)
          : undefined;
      const filePath =
        typeof record?.path === "string" ? String(record.path) : "<unknown>";
      const normalized = await normalizeReadImageResult(result, filePath);
      return sanitizeToolResultImages(normalized, `read:${filePath}`);
    },
  };
}

function normalizeSurface(surface?: string): string | undefined {
  const trimmed = surface?.trim().toLowerCase();
  return trimmed ? trimmed : undefined;
}

function shouldIncludeDiscordTool(surface?: string): boolean {
  const normalized = normalizeSurface(surface);
  if (!normalized) return false;
  return normalized === "discord" || normalized.startsWith("discord:");
}

function shouldIncludeSlackTool(surface?: string): boolean {
  const normalized = normalizeSurface(surface);
  if (!normalized) return false;
  return normalized === "slack" || normalized.startsWith("slack:");
}

export function createClawdisCodingTools(options?: {
  bash?: BashToolDefaults & ProcessToolDefaults;
  surface?: string;
  sandbox?: SandboxContext | null;
  sessionKey?: string;
}): AnyAgentTool[] {
  const bashToolName = "bash";
  const sandbox = options?.sandbox?.enabled ? options.sandbox : undefined;
  const sandboxRoot = sandbox?.workspaceDir;
  const base = (codingTools as unknown as AnyAgentTool[]).flatMap((tool) => {
    if (tool.name === readTool.name) {
      return sandboxRoot
        ? [createSandboxedReadTool(sandboxRoot)]
        : [createClawdisReadTool(tool)];
    }
    if (tool.name === bashToolName) return [];
    if (sandboxRoot && (tool.name === "write" || tool.name === "edit")) {
      return [];
    }
    return [tool as AnyAgentTool];
  });
  const bashTool = createBashTool({
    ...options?.bash,
    sandbox: sandbox
      ? {
        containerName: sandbox.containerName,
        workspaceDir: sandbox.workspaceDir,
        containerWorkdir: sandbox.containerWorkdir,
        env: sandbox.docker.env,
      }
      : undefined,
  });
  const processTool = createProcessTool({
    cleanupMs: options?.bash?.cleanupMs,
  });
  const tools: AnyAgentTool[] = [
    ...base,
    ...(sandboxRoot
      ? [
        createSandboxedEditTool(sandboxRoot),
        createSandboxedWriteTool(sandboxRoot),
      ]
      : []),
    bashTool as unknown as AnyAgentTool,
    processTool as unknown as AnyAgentTool,
    createWhatsAppLoginTool(),
    ...createClawdisTools({
      browserControlUrl: sandbox?.browser?.controlUrl,
      agentSessionKey: options?.sessionKey,
      agentSurface: options?.surface,
    }),
  ];
  const allowDiscord = shouldIncludeDiscordTool(options?.surface);
  const allowSlack = shouldIncludeSlackTool(options?.surface);
  const filtered = tools.filter((tool) => {
    if (tool.name === "discord") return allowDiscord;
    if (tool.name === "slack") return allowSlack;
    return true;
  });
  const sandboxed = sandbox
    ? filterToolsByPolicy(filtered, sandbox.tools)
    : filtered;
  return sandboxed.map(normalizeToolParameters);
}
