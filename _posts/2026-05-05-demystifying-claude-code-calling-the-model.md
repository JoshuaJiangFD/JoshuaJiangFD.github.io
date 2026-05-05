---
title: "Demystifying Claude Code: Calling the Model"
date: 2026-05-04 11:00:00 -0700
categories: [Claude Code]
tags: [Claude Code, LLM, API]
mermaid: true
---

This post details how `claude.ts` transforms messages, builds the API request, and streams the response back to the query loop. This is what happens inside Step 8 (API CALL) of the message pipeline (Managing Message Context), and is referenced from Session Orchestration section 4.

---

## 1. Entry Point

`query.ts` calls `deps.callModel()` at line 659, which delegates to `queryModelWithStreaming()` in `claude.ts`:

```typescript
// query.ts:659
for await (const message of deps.callModel({
  messages: prependUserContext(messagesForQuery, userContext),
  systemPrompt: fullSystemPrompt,
  thinkingConfig: toolUseContext.options.thinkingConfig,
  tools: toolUseContext.options.tools,
  signal: toolUseContext.abortController.signal,
  options: { model, querySource, fallbackModel, ... },
}))
```

Note that `prependUserContext` runs in `query.ts` before passing messages in. Everything else happens inside `claude.ts`.

---

## 2. The Full Message Transformation Pipeline

```
messagesForQuery (from query.ts, after Steps 1-6)
    |
    v
prependUserContext(messages, userContext)         -- query.ts:660
  Prepends a synthetic UserMessage containing
  <system-reminder> with key-value context
  (claudeMd, gitStatus, currentDate, etc.)
    |
    v
queryModelWithStreaming()                        -- claude.ts:752
  |
  v
queryModel()                                     -- claude.ts:1017
  |
  +-- normalizeMessagesForAPI(messages, tools)    -- claude.ts:1266
  |     Strips system/progress/attachment msgs
  |     Merges consecutive user messages
  |     Converts attachments to user content
  |     Strips stale tool_reference blocks
  |     Strips errored media blocks
  |     -> (UserMessage | AssistantMessage)[]
  |
  +-- ensureToolResultPairing(messagesForAPI)     -- claude.ts:1301
  |     Inserts synthetic error tool_results
  |     for orphaned tool_uses
  |     Strips orphaned tool_results
  |     (fixes resume/teleport gaps)
  |
  +-- stripAdvisorBlocks(messagesForAPI)          -- claude.ts:1305
  |     (only if advisor beta header not present)
  |
  +-- stripExcessMediaItems(messagesForAPI, 100)  -- claude.ts:1312
  |     Drops oldest images/docs if >100 media
  |     items (API rejects with confusing error)
  |
  +-- [optional] prepend deferred tool list       -- claude.ts:1337
  |     Adds <available-deferred-tools> message
  |     (only if tool search on + delta disabled)
  |
  +-- addCacheBreakpoints(messagesForAPI, ...)    -- claude.ts:1701
  |     Converts to MessageParam[] (API format)
  |     Adds cache_control markers
  |     Inserts cache_edits blocks (cached MC)
  |     -> MessageParam[] (ready for HTTP)
  |
  v
anthropic.beta.messages.stream(params)           -- the actual API call
```

---

## 3. prependUserContext (api.ts:449)

```typescript
function prependUserContext(
  messages: Message[],
  context: { [k: string]: string },
): Message[]
```

This function prepends a single synthetic `UserMessage` at the start of the array containing all session-level context:

```xml
<system-reminder>
As you answer the user's questions, you can use the following context:
# claudeMd
[contents of CLAUDE.md files]
# gitStatus
[current git status]
# currentDate
Today's date is 2026-05-05.

      IMPORTANT: this context may or may not be relevant to your tasks.
</system-reminder>
```

This runs in `query.ts` before messages enter `claude.ts`. The model sees it as the first user message in the conversation.

---

## 4. normalizeMessagesForAPI (messages.ts:1989)

```typescript
function normalizeMessagesForAPI(
  messages: Message[],
  tools: Tools = [],
): (UserMessage | AssistantMessage)[]
```

This is the biggest transformation. It converts the internal message format (6+ types) into what the Anthropic API accepts (strictly `user` and `assistant` roles only). The function performs the following operations in order:

1. **Reorders attachments** — bubbles attachment messages up until they hit a tool result or assistant message.
2. **Strips virtual messages** — display-only messages (e.g., REPL inner tool calls) that should never reach the API.
3. **Filters non-API types** — removes `progress`, `system` (except `local_command`), and synthetic API error messages.
4. **Converts `local_command` system messages** — turns them into `UserMessage`s so the model can reference previous command output.
5. **Strips tool_reference blocks** — removed entirely if tool search is off; removed for unavailable tools (e.g., disconnected MCP server) if tool search is on.
6. **Strips errored media** — removes document/image blocks that previously caused PDF-too-large or image-too-large errors.
7. **Merges consecutive user messages** — Bedrock requires strict user/assistant alternation; even the first-party API merges them server-side, so this ensures consistency.
8. **Folds attachment content into user messages** — attachment messages become part of the adjacent user message's content.

The result is a clean `(UserMessage | AssistantMessage)[]` alternating sequence.

### Where it's called

| Call site | Purpose |
|---|---|
| `claude.ts:1266` | Main API call — normalize the full conversation. |
| `query.ts:855` | Normalize streaming tool results mid-stream. |
| `query.ts:1396` | Normalize tool results after execution. |
| `compact.ts:1293` | Normalize messages for the compaction API call. |

---

## 5. System Prompt Assembly

`claude.ts` builds the system prompt from multiple pieces (line 1358-1368):

```
[1] Attribution header (fingerprint)
[2] CLI sysprompt prefix (interactive vs non-interactive markers)
[3] Main system prompt (from fetchSystemPromptParts)
[4] Advisor tool instructions (if advisor model enabled)
[5] Chrome tool search instructions (if applicable)
```

Then `buildSystemPromptBlocks()` (line 3213) converts this into `TextBlockParam[]` with `cache_control` markers for prompt caching. The system prompt is split into segments so that the stable prefix (attribution + CLI markers + main prompt) can be cached independently from the dynamic suffix (advisor, chrome instructions) that may change between requests.

---

## 6. Tool Schema Building

Before the API call, `claude.ts` builds the tool schemas (line 1235-1246):

```
All registered tools (built-in + MCP)
    |
    +-- Filter: if tool search enabled, only include
    |   non-deferred + already-discovered deferred tools
    |   (ToolSearchTool always included)
    |
    +-- Filter: if tool search disabled, remove ToolSearchTool
    |
    +-- toolToAPISchema() for each tool
    |   Converts Tool -> BetaToolUnion (API format)
    |   Sets defer_loading: true for deferred tools
    |
    +-- Append advisor server tool (if enabled)
    |
    v
allTools: BetaToolUnion[]  --> sent as `tools` param
```

The dynamic tool loading system means not all tools are sent on every request. Deferred tools only appear in the schema after the model has discovered them via `ToolSearchTool` — their `tool_use_id`s are tracked in the message history, and `extractDiscoveredToolNames()` scans for them to decide which deferred tools to include.

---

## 7. Cache Breakpoints and Message Conversion

`addCacheBreakpoints()` (line 3063) is the final step — it converts the normalized `(UserMessage | AssistantMessage)[]` into the SDK's `MessageParam[]`:

```typescript
function addCacheBreakpoints(
  messages: (UserMessage | AssistantMessage)[],
  enablePromptCaching: boolean,
  querySource?: QuerySource,
  useCachedMC?: boolean,
  newCacheEdits?: CachedMCEditsBlock | null,
  pinnedEdits?: CachedMCPinnedEdits[],
  skipCacheWrite?: boolean,
): MessageParam[]
```

For each message, the function calls `userMessageToMessageParam()` or `assistantMessageToMessageParam()` to produce the final `{ role: 'user' | 'assistant', content: [...] }` format the SDK expects.

It adds a `cache_control` marker to exactly **one** message — the last one in normal requests, or the second-to-last for fire-and-forget forks (like compaction, which shouldn't leave its own tail in the server cache).

For cached microcompact, it also inserts `cache_edits` blocks into user messages. These instruct the server to delete stale tool result content from the cached prefix without invalidating the rest of the cache.

---

## 8. The Final API Parameters

`paramsFromContext()` (line 1538) assembles the complete request object sent to the Anthropic SDK:

```typescript
{
  model: "claude-sonnet-4-6-20250514",
  messages: MessageParam[],              // from addCacheBreakpoints
  system: TextBlockParam[],              // from buildSystemPromptBlocks
  tools: BetaToolUnion[],                // tool schemas
  tool_choice: undefined,                // usually auto
  betas: ["..."],                        // feature beta headers
  metadata: { user_id: "..." },
  max_tokens: 8192,                      // or escalated 64K
  thinking: { type: "adaptive" },        // or { type: "enabled", budget_tokens: N }
  temperature: undefined,                // only set when thinking disabled
  context_management: { ... },           // if enabled
  output_config: {                       // effort, task_budget, structured output
    effort: "high",
    task_budget: { total: 500000, remaining: 350000 },
  },
  speed: "fast",                         // if fast mode active
}
```

Several of these parameters use **sticky latching** for beta headers: once a header is first sent (e.g., fast mode, cache editing, AFK mode), it keeps being sent for the rest of the session even if the feature is toggled off. This prevents mid-session cache key changes that would bust the prompt cache.

---

## 9. Retry and Streaming

The actual API call is wrapped in `withRetry()` (line 1778):

```
withRetry(getAnthropicClient, async (anthropic, attempt, context) => {
    |
    +-- anthropic.beta.messages.stream(params).withResponse()
    |
    +-- Iterate stream events:
    |     message_start    -> capture usage, headers, request ID
    |     content_block_start -> start content block
    |     content_block_delta -> accumulate text/thinking/tool_input
    |     content_block_stop  -> finalize block, yield AssistantMessage
    |     message_delta    -> capture final usage, stop_reason
    |     message_stop     -> cleanup
    |
    +-- On stream failure:
    |     +-- If retryable (429, 529, 5xx) -> retry with backoff
    |     +-- If streaming broke mid-response -> fall back to non-streaming
    |     +-- If prompt too long -> yield synthetic error AssistantMessage
    |     +-- If auth error -> re-create client and retry
    |
    v
yields: StreamEvent | AssistantMessage | SystemAPIErrorMessage
  back to query.ts Step 8
```

### Non-streaming fallback

When streaming fails mid-response (connection drop, timeout), `claude.ts` falls back to `executeNonStreamingRequest()` which makes a regular (non-streaming) API call with `anthropic.beta.messages.create()` and a per-attempt timeout (120s for remote sessions, 300s for local). This ensures the model's response is not lost even when the streaming connection is unreliable.

---

## 10. What query.ts Sees

From `query.ts`'s perspective, all of the above is invisible. It simply iterates the async generator stream:

```typescript
for await (const message of deps.callModel({ messages, ... })) {
  if (message.type === 'assistant') {
    assistantMessages.push(message)
    // check for tool_use blocks
  }
  if (message.type === 'stream_event') {
    // usage tracking, stream progress
  }
  // yield to QueryEngine for persistence
}
```

The `AssistantMessage` yielded back contains:
- `message.content` — text blocks, tool_use blocks, thinking blocks.
- `message.message.usage` — token counts (input, output, cache read/creation).
- `message.message.stop_reason` — `end_turn`, `tool_use`, or `max_tokens`.
- `message.apiError` — set if this is a synthetic error message (prompt_too_long, max_output_tokens).

---

## 11. Summary: What claude.ts Owns

| Responsibility | Details |
|---|---|
| **Message normalization** | `normalizeMessagesForAPI`, `ensureToolResultPairing`, `stripAdvisorBlocks`, `stripExcessMediaItems` |
| **Message conversion** | `userMessageToMessageParam`, `assistantMessageToMessageParam`, `addCacheBreakpoints` |
| **System prompt** | Assembly from parts, `buildSystemPromptBlocks` with cache markers |
| **Tool schemas** | `toolToAPISchema`, deferred/dynamic tool filtering |
| **Thinking config** | Adaptive vs budget, model capability checks |
| **Beta headers** | Tool search, prompt caching, cache editing, structured outputs, advisor, fast mode, AFK mode — with sticky latching |
| **Prompt caching** | `cache_control` breakpoints, global vs per-request scope, cache edits |
| **Model resolution** | Bedrock inference profiles, fallback models, fast mode |
| **Effort/budget** | `output_config.effort`, `output_config.task_budget` |
| **Retry** | `withRetry` with backoff, non-streaming fallback, auth retry |
| **Streaming** | SDK stream iteration, content block assembly, `AssistantMessage` construction |
| **Telemetry** | API timing, request IDs, cache hit rates, fingerprinting, tracing spans |
