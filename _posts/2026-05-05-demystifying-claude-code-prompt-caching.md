---
title: "Demystifying Claude Code: Prompt Caching"
date: 2026-05-04 12:00:00 -0700
categories: [Claude Code]
tags: [Claude Code, LLM, Prompt Caching]
mermaid: true
---

This post explores how Claude Code leverages prompt caching to minimize redundant computation across API requests. We cover the transformer-level mechanics of KV caching, how Claude Code places cache markers, the multi-scope caching strategy for the system prompt, all client-side mechanisms for cache stability, the cache editing system for surgical content removal, what breaks the cache and why, and how the system detects and reports cache breaks.

---

## 1. How Prompt Caching Works (Transformer Perspective)

### KV Cache at the Attention Layer

In a standard transformer forward pass, each token attends to all previous tokens. For a prompt of length N, each attention layer computes:

```
For each token i in [1..N]:
  Q_i = W_q * x_i          (query vector)
  K_i = W_k * x_i          (key vector)
  V_i = W_v * x_i          (value vector)
  
  Attention_i = softmax(Q_i * [K_1, K_2, ..., K_i]^T / sqrt(d)) * [V_1, V_2, ..., V_i]
```

Processing the full prompt requires computing K and V vectors for every token at every layer. For a 200K token prompt across ~80 layers with a hidden dimension of ~8192, this is enormous compute — the "prefill" phase.

During autoregressive generation (token by token), the standard optimization is to cache K and V vectors from previous tokens. When generating token N+1, only Q_{N+1} is computed fresh, attending against the cached K_{1..N} and V_{1..N}. This avoids recomputing them on every generated token.

### Why It Must Be Prefix-Based

Prompt caching extends the KV cache concept across API requests — instead of discarding the KV cache when the HTTP connection closes, the server persists it to fast storage (GPU memory, NVMe, or a dedicated KV cache service).

Attention is causal in decoder-only transformers — token i can only attend to tokens 1..i. If you change any token before position i, the representation at position i changes (because of residual connections from previous layers), which means K_i and V_i change. So you can only reuse KV cache for a prefix — the moment a byte differs, everything from that point forward is invalid.

```
Cached:   [A B C D E F G H]
New req:  [A B C X E F G H I]
                  ^ changed

KV cache valid for: [A B C]  (positions 1-3 only)
Must recompute:     [X E F G H I]  (positions 4+ are all invalid)
```

This is why Claude Code is so careful about byte-identical prefixes — even a single character change in an early message invalidates the KV cache for everything after it.

### What a Cache Hit Saves

```
Turn 1 (processing prompt):
  Compute K_1..K_N, V_1..V_N for all layers    <- expensive (prefill)
  Cache these KV pairs
  Generate tokens using cached KVs              <- cheap (decode)

Turn 2 (new user message appended):
  WITHOUT prompt cache: recompute K_1..K_{N+M} from scratch  <- expensive again
  WITH prompt cache: reuse cached K_1..K_N, only compute K_{N+1}..K_{N+M}  <- cheap
```

The savings are proportional to the prefix length. For a 180K token cached prefix with a 5K new message, you skip ~97% of the prefill computation.

### Memory Cost and Eviction

The KV cache for a single request is substantial:

```
KV cache size = 2 * num_layers * num_heads * head_dim * sequence_length * bytes_per_element

For a ~100B parameter model (roughly Sonnet-class):
  ~80 layers, ~64 heads, 128 head_dim, fp16 (2 bytes)
  At 200K tokens: 2 * 80 * 64 * 128 * 200000 * 2 bytes ≈ 52 GB
```

This is why the server can't cache everything forever — there's a TTL (5min default, 1hr for eligible users). The server-side KV cache management system ("Mycro", referenced in the codebase) manages a shared store with eviction policies, page-level reference counting, and tiered storage.

---

## 2. The Cache Model in Claude Code

### The `cache_control` Marker

The `cache_control` marker is a JSON field attached to one content block in the request that tells the Anthropic API server: "cache everything up to and including this point." It looks like:

```json
{
  "type": "ephemeral",
  "ttl": "1h",
  "scope": "global"
}
```

Claude Code places exactly **one** message-level `cache_control` marker per request, on the last content block of the last message (or second-to-last message for fire-and-forget forks like compaction):

```typescript
// claude.ts:3078-3089
// Exactly one message-level cache_control marker per request. Mycro's
// turn-to-turn eviction frees local-attention KV pages at any cached
// prefix position NOT in cache_store_int_token_boundaries. With two
// markers the second-to-last position is protected and its locals
// survive an extra turn even though nothing will ever resume from
// there — with one marker they're freed immediately.
const markerIndex = skipCacheWrite ? messages.length - 2 : messages.length - 1
```

Multiple markers would waste server-side GPU memory — the server keeps KV pages alive at every marker position, but only the last one is ever actually resumed from. One marker at the tail minimizes memory waste while maximizing cache hit surface.

### TTL Tiers

The marker's `ttl` field controls how long the server keeps the cached KV state:

- **5 minutes** (default) — used when the `ttl` field is omitted. Sufficient for typical interactive sessions where turns happen within seconds.
- **1 hour** — explicitly set via `ttl: '1h'` for eligible users. Available to Anthropic internal users (`USER_TYPE === 'ant'`) and Claude AI subscribers who are not using overage.

The TTL eligibility is **latched** at session start — if a user starts within rate limits and later exceeds them, the 1h TTL continues for the rest of the session. This prevents mid-session TTL flips that would change the `cache_control` bytes and bust the cache:

```typescript
// claude.ts:403-405
// Latch eligibility in bootstrap state for session stability — prevents
// mid-session overage flips from changing the cache_control TTL, which
// would bust the server-side prompt cache (~20K tokens per flip).
```

### Cache Scopes

The `scope` field controls who can share a cached prefix:

- **`null`** (no scope field) — no `cache_control` marker is placed on that block at all. The block has no cache entry of its own, but may still be part of a larger cached prefix if a later block has a marker.
- **`'org'`** — the cached KV state is shared across all users within the same organization. If two users in the same org send requests with byte-identical content up to this marker, they hit the same server-side cache entry.
- **`'global'`** — the cached KV state is shared across all users of the API, regardless of organization. Since Claude Code's base system prompt is identical for every installation, one cache entry serves the entire fleet.

The `scope` requires the `prompt_caching_scope` beta header to be sent with the request. This header is enabled automatically for first-party API users (`shouldUseGlobalCacheScope()` returns true for `getAPIProvider() === 'firstParty'`).

### The Growing-Prefix Pattern

Across a multi-turn conversation, the marker moves forward each turn:

```
Turn 1: [sys | tools | user_1 (marker)]                    → creates cache for prefix
Turn 2: [sys | tools | user_1 | asst_1 | user_2 (marker)]  → hits cache for [sys|tools|user_1]
Turn 3: [sys | tools | user_1 | asst_1 | user_2 | asst_2 | user_3 (marker)] → hits longer prefix
```

Each turn, the cached prefix grows by the previous turn's content. The marker always sits at the new tail — previous content is cache-hit, new content is cache-creation.

---

## 3. System Prompt Caching Strategy

### The Boundary Marker

The system prompt is divided into static and dynamic sections using `SYSTEM_PROMPT_DYNAMIC_BOUNDARY` (a literal string `'__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__'`). Content before the boundary is identical for all Claude Code users and can be globally cached. Content after it varies per user/session and cannot be shared.

### Static vs Dynamic Split

The system prompt is assembled in `prompts.ts` with this structure:

```
STATIC (before boundary):
  [1] getSimpleIntroSection()           — identity and role framing
  [2] getSimpleSystemSection()          — environment description
  [3] getSimpleDoingTasksSection()      — task execution guidelines
  [4] getActionsSection()               — action safety rules
  [5] getUsingYourToolsSection()        — tool usage patterns
  [6] getSimpleToneAndStyleSection()    — communication style
  [7] getOutputEfficiencySection()      — output length/efficiency rules
  --- SYSTEM_PROMPT_DYNAMIC_BOUNDARY ---
DYNAMIC (after boundary):
  [8] Session-specific guidance         — depends on enabled tools, mode, etc.
  [9] Memory/CLAUDE.md                  — per-project rules
  [10] Environment info                 — cwd, OS, model name
  [11] MCP instructions                 — per-user server configurations
  [12] Scratchpad instructions          — if enabled
  ... other session-variant sections
```

Any conditional content that would fragment the globally-cacheable prefix is placed after the boundary. The comment in the code explains the motivation:

```typescript
// Session-variant guidance that would fragment the cacheScope:'global'
// prefix if placed before SYSTEM_PROMPT_DYNAMIC_BOUNDARY. Each conditional
// here is a runtime bit that would otherwise multiply the Blake2b prefix
// hash variants (2^N).
```

### The Ordering

Before reaching the API, `claude.ts` prepends two headers, making the final order:

```
[0] Attribution header              → cacheScope: null (per-request fingerprint)
[1] CLI sysprompt prefix            → cacheScope: null (session type identifier)
[2] Static base instructions        → cacheScope: 'global' (fleet-wide shared)
[3] Dynamic content                 → cacheScope: null (per-user, per-session)
[4] Advisor/Chrome instructions     → appended at tail if active
```

Then `splitSysPromptPrefix()` assigns cache scopes and `buildSystemPromptBlocks()` converts them to `TextBlockParam[]` with the appropriate `cache_control` fields:

```typescript
// claude.ts:3228-3229
...(enablePromptCaching &&
  block.cacheScope !== null && {
    cache_control: getCacheControl({
      scope: block.cacheScope,
      querySource: options?.querySource,
    }),
  }),
```

When `cacheScope` is `null`, no `cache_control` is added to that block — it doesn't get its own cache entry but still participates in any encompassing cache (the per-user message-level cache that extends to the last message's marker).

### Why Attribution Header Is First but Uncached

The attribution header contains a per-request fingerprint (version + content hash). It changes every request, so caching it separately would create a unique entry per request (useless). It's placed first because the server checks it for billing/routing before processing the rest. By giving it `cacheScope: null`, it doesn't fragment the global cache — the server's prefix matching starts after it.

### The MCP Tool Interaction

MCP tools are per-user (user-configured servers with unique tool names and descriptions). When MCP tools are present and not deferred, global scope caching is disabled for the system prompt:

```typescript
// claude.ts:1210-1211
// MCP tools are per-user → dynamic tool section → can't globally cache.
const needsToolBasedCacheMarker =
  useGlobalCacheFeature && filteredTools.some(t => t.isMcp === true && !willDefer(t))
```

In this case, `splitSysPromptPrefix()` falls back to `'org'` scope instead of `'global'` — members of the same org with the same MCP setup may still share a cache entry, but fleet-wide sharing is impossible.

This is also why deferred tool loading matters for cache — a deferred MCP tool (`defer_loading: true`) doesn't appear in the tool schemas, so it doesn't pollute the globally-cacheable prefix.

---

## 4. Client-Side Cache Stability Mechanisms

The server cache is prefix-based and byte-sensitive. Claude Code goes to extraordinary lengths to ensure the bytes it sends are stable across requests.

### Tool Schema Cache (toolSchemaCache.ts)

Tool schemas are rendered once per session and memoized in a `Map<string, CachedSchema>`:

```typescript
// Session-scoped cache of rendered tool schemas. Tool schemas render at server
// position 2 (before system prompt), so any byte-level change busts the entire
// ~11K-token tool block AND everything downstream. GrowthBook gate flips
// (tengu_tool_pear, tengu_fgts), MCP reconnects, or dynamic content in
// tool.prompt() all cause this churn. Memoizing per-session locks the schema
// bytes at first render — mid-session GB refreshes no longer bust the cache.
```

Without this, GrowthBook feature flag refreshes (which happen every few minutes from disk cache) would silently change tool description text mid-session, producing different bytes on the next request and busting the ~11K token tool section plus everything after it.

### Sticky Beta Header Latching

Beta headers are part of the server's cache key. Toggling a header mid-session changes the key and busts the cache. The solution is latching — once a header is first sent, it keeps being sent for the rest of the session even if the feature is toggled off:

```typescript
let fastModeHeaderLatched = getFastModeHeaderLatched() === true
if (!fastModeHeaderLatched && isFastMode) {
  fastModeHeaderLatched = true
  setFastModeHeaderLatched(true)  // persists across requests
}
```

This applies to: fast mode (`FAST_MODE_BETA_HEADER`), AFK/auto mode (`AFK_MODE_BETA_HEADER`), cache editing (`cacheEditingBetaHeader`), and thinking-clear (`thinkingClearLatched`). The actual behavior can still be toggled (e.g., `speed: 'fast'` is only sent when fast mode is active), but the header stays constant so the cache key doesn't flip.

### Frozen Tool Result Budget Decisions

Once the tool result budget system (Step 2 in the message pipeline) decides whether a tool result should be replaced or kept, that decision is frozen by `tool_use_id`. On every subsequent iteration, the same replacement string is re-applied byte-identically from a cached `Map`:

```typescript
// toolResultStorage.ts:802
// Re-apply: pure Map lookups. No file I/O, byte-identical, cannot fail.
mustReapply.forEach(c => replacementMap.set(c.toolUseId, c.replacement))
```

A result that was replaced stays replaced forever. A result that was kept is never replaced later. This ensures the bytes of previously-sent messages never change between requests — preserving the cached prefix.

### Cache-Sharing Forks

When Claude Code needs to run a side computation (compaction summarization, session memory, prompt suggestions), it forks an agent that reuses the main conversation's cached prefix:

```typescript
// compact.ts:1179
// Use forked agent to reuse the main conversation's cached prefix
const result = await runForkedAgent({
  promptMessages: [summaryRequest],
  cacheSafeParams,  // ← same system prompt, tools, messages prefix as main thread
  ...
})
```

The `cacheSafeParams` carry the exact same system prompt, tools, and message prefix that the main thread uses. Because the forked agent sends byte-identical prefix content, the server reuses the main thread's cached KV state — the compaction agent gets a near-100% cache hit on the conversation history it's summarizing.

For this to work, the forked agent must produce byte-identical decisions for everything in the shared prefix. This is why tool result budget state is cloned (not fresh) for forks — they need to make the same replacement decisions as the main thread for shared messages.

The forked agent also uses `skipCacheWrite: true`, which shifts the `cache_control` marker to the second-to-last message. This means the fork's unique tail content doesn't create its own cache entry — it only reads from the shared prefix, never writes a fork-specific entry that would waste server memory.

### Advisor Tool Positioning

The advisor server tool is appended after the tool schemas (which carry the `cache_control` marker):

```typescript
// claude.ts:1388-1389
// toolSchemas (which carries the cache_control marker) so toggling /advisor
// only churns the small suffix, not the cached prefix.
extraToolSchemas.push({ type: 'advisor_20250301', name: 'advisor', model: advisorModel })
```

If it were inserted in the middle of the tool array, enabling/disabling advisor would change the bytes of the cached prefix. By appending after the cache marker, it only affects the uncached suffix.

### Deferred MCP Tools

MCP tools that use `defer_loading: true` don't appear in the tool schemas sent to the API at all — their definition is only loaded when the model discovers them via `ToolSearchTool`. This means connecting a new MCP server mid-session doesn't change the tool schema bytes and doesn't bust the cache. The tool becomes available to the model through the discovery mechanism without polluting the cached prefix.

---

## 5. Cache Editing (Cached Microcompact)

### The Problem

As the agentic loop runs, old tool results accumulate in the message history. These consume context space but are often no longer relevant (e.g., an old `Read` result for a file that was subsequently edited). The naive solution — clearing the content locally — would change the bytes of the cached prefix, invalidating the entire cache.

### `cache_reference` on Tool Results

To enable server-side editing, Claude Code adds a `cache_reference` identifier to every `tool_result` block that falls within the cached prefix:

```typescript
// claude.ts:3180-3201
// Add cache_reference to tool_result blocks that are strictly before
// the last cache_control marker.
if (block && isToolResultBlock(block)) {
  msg.content[j] = Object.assign({}, block, {
    cache_reference: block.tool_use_id,
  })
}
```

These references create addressable handles that the server can use to identify specific blocks within the cached KV state.

### `cache_edits` Blocks

When cached microcompact determines that old tool results should be removed, it doesn't modify the local messages. Instead, it queues a `cache_edits` block that's inserted into the request by `addCacheBreakpoints()`:

```json
{
  "type": "cache_edits",
  "edits": [
    { "cache_reference": "tool_use_id_abc", "action": "delete" },
    { "cache_reference": "tool_use_id_def", "action": "delete" }
  ]
}
```

The server receives this, identifies which token positions correspond to those `tool_result` blocks via the `cache_reference` mapping, and marks those KV entries as deleted. Subsequent attention computations skip those positions (they're masked out).

### How This Works at the KV Level

Removing tokens from the middle of the attention window is valid because it's equivalent to masking those positions in the attention pattern. The KV vectors for tokens after the deleted block are still correct — they were originally computed with the deleted block present, and removing it from attention doesn't change their stored K/V values (only changes what they attend to, which is recomputed at query time).

There is a subtle approximation: tokens after the deleted block were originally computed with that content influencing their representations through the residual stream. Deleting the block from attention means those tokens now produce slightly different attention outputs than if the block had never existed. In practice, for tool result deletions (content the model already processed and acted on), this approximation is acceptable.

### Pinned Cache Edits

Once a `cache_edits` block is sent and the server applies it, that edit must be re-sent at the same position on every subsequent request — otherwise the server would see different bytes at that position and bust the cache. Claude Code tracks this via "pinned" cache edits:

```typescript
// After first insertion, pin the edits to their message position
export function pinCacheEdits(
  userMessageIndex: number,
  block: CacheEditsBlock,
): void {
  if (cachedMCState) {
    cachedMCState.pinnedEdits.push({ userMessageIndex, block })
  }
}
```

On every subsequent request, `addCacheBreakpoints()` re-inserts all pinned edits at their original positions before adding any new ones.

### Comparison with AWS Bedrock Cache Checkpoints

AWS Bedrock offers explicit `cache_checkpoint` markers that let you place multiple cache points in the message array, enabling cache hits from intermediate positions. Claude Code's approach differs:

| Feature | Bedrock Checkpoints | Claude Code |
|---|---|---|
| Multiple cache points | Yes (multiple `cache_checkpoint` markers) | No (single `cache_control` at the tail) |
| Growing prefix cache | Yes | Yes (each turn appends, extending the hit) |
| Selective invalidation | Remove checkpoint | `cache_edits` to delete specific blocks |
| Cross-user sharing | Not documented | `scope: 'global'` for system prompt |
| TTL control | Configuration-based | `ttl: '5m'` (default) or `ttl: '1h'` (eligible users) |

Claude Code trades multi-point checkpointing for a simpler single-marker model augmented with fine-grained cache editing. The editing mechanism gives it the ability to surgically remove stale content without full prefix invalidation — arguably more powerful than multiple checkpoints for the agentic loop use case, where you want to remove old tool results from the middle without losing cache hits on everything else.

---

## 6. What Breaks the Cache

### Message Pipeline Steps: Cache-Safe vs Cache-Breaking

The 11-step message pipeline (see [Managing Message Context]({% post_url 2026-05-05-demystifying-claude-code-managing-message-context %})) processes messages before each API call. Each step has different cache implications:

**Cache-safe steps (preserve prefix bytes):**

- **Step 2 (Tool Result Budget)** — frozen decisions mean previously-seen messages never change bytes. Re-application is byte-identical from a cached Map.
- **Step 4 (Cached Microcompact)** — uses `cache_edits` for server-side deletion without modifying local bytes.
- **Step 11 (State Merge)** — appends at the tail, extending the prefix rather than modifying it.

**Steps that accept cache loss when justified:**

- **Step 3 (Snip Compact)** — removes messages from the middle of history. Changes prefix bytes and invalidates the cache. Fires when context pressure demands it.
- **Step 4 (Time-Based Microcompact)** — replaces tool result content with `"[Old tool result content cleared]"`. This modifies prefix bytes, but only fires when the time gap since the last assistant message exceeds a threshold — meaning the server cache has already expired due to TTL anyway. Cache-safe in practice because there's no cache left to break.
- **Step 5 (Context Collapse)** — replaces spans of messages with summary placeholders. Changes prefix bytes. A deliberate tradeoff of cache hit for granular context preservation (better than full compaction).
- **Step 6 (Autocompact)** — the nuclear option. Replaces the entire message array with `[boundary, summary, attachments, hookResults]`. The cache is completely invalidated. The code explicitly calls `notifyCompaction()` to reset the cache break detector's baseline.

### Non-Message Sources of Cache Breaks

Beyond the message pipeline, several other factors can change the cache key:

- **Model changes** — switching models (e.g., Sonnet → Opus fallback) changes the server-side cache partition.
- **System prompt changes** — editing CLAUDE.md files, MCP server instructions changing, or feature flag toggles that modify the system prompt text.
- **Tool schema changes** — MCP server connecting/disconnecting mid-session, new tools discovered. Mitigated by the tool schema cache (freezes at first render) and deferred loading (new tools don't appear in schemas until discovered).
- **Beta header changes** — toggling fast mode, advisor, or AFK mode. Mitigated by sticky latching.
- **Effort value changes** — changing `/effort` level modifies `output_config` in the request.
- **Fast mode toggle** — the `speed: 'fast'` field is dynamic, but the beta *header* is latched. The speed field alone doesn't change the cache key (only the header does).

### The Fundamental Tension

The message pipeline exists to keep context within the model's window, but every modification to previously-sent messages risks breaking the cache. The system resolves this tension through a hierarchy of preference:

1. **Prefer appending over modifying** — normal turns just append at the tail, extending the prefix (always cache-safe).
2. **Prefer server-side editing over local mutation** — cached microcompact uses `cache_edits` to remove stale content without changing local bytes.
3. **Accept cache loss when cache is already cold** — time-based microcompact only fires when TTL has expired, so there's no cache left to break.
4. **Accept cache loss for context survival** — when context pressure is critical (approaching the model's limit), saving the session is worth more than the cache hit. Compaction destroys the cache but keeps the conversation alive.

---

## 7. Cache Break Detection and Metrics

### Per-Request Usage Fields

Every API response includes a `usage` object with cache-specific fields:

```typescript
{
  input_tokens: number,                  // tokens NOT from cache (new computation)
  cache_read_input_tokens: number,       // tokens read from cache (cheap/free)
  cache_creation_input_tokens: number,   // tokens written to cache (slightly more expensive)
  cache_deleted_input_tokens: number,    // tokens deleted via cache_edits
  output_tokens: number,
}
```

The cache hit rate for a single request can be derived as:

```
hit_rate = cache_read / (input + cache_read + cache_creation)
```

These values come from the API's `message_delta` event at the end of streaming. Claude Code captures them via `updateUsage()` and accumulates them into session totals.

### Session-Level Accumulation (cost-tracker.ts)

The cost tracker maintains per-model cumulative totals throughout the session:

```typescript
type ModelUsage = {
  inputTokens: number,
  outputTokens: number,
  cacheReadInputTokens: number,
  cacheCreationInputTokens: number,
  costUSD: number,
}
```

When a session ends (or the user runs `/cost`), it formats this as:

```
Usage by model:
       Sonnet 4.6:  45,230 input, 12,800 output, 892,000 cache read, 23,400 cache write ($0.42)
```

A high `cacheReadInputTokens` relative to `cacheCreationInputTokens` indicates strong cache hit rates throughout the session.

### The Two-Phase Break Detector (promptCacheBreakDetection.ts)

This is the most sophisticated mechanism — a real-time detector that identifies when and why cache hits drop.

**Phase 1 (pre-request):** Before each API call, `recordPromptState()` snapshots everything that could affect the cache key:

- System prompt content (hashed)
- Per-tool schema hashes
- Model string
- Beta headers
- Fast mode, auto mode, effort value
- Global cache strategy
- Extra body params

**Phase 2 (post-response):** After the response, `checkResponseForCacheBreak()` compares the current `cache_read_input_tokens` against the previous request's value:

```typescript
// A break is detected when:
const tokenDrop = prevCacheRead - cacheReadTokens
if (cacheReadTokens >= prevCacheRead * 0.95 || tokenDrop < MIN_CACHE_MISS_TOKENS) {
  // Not a break — within 5% or below 2K token threshold
  return
}
// Break detected! Explain why using the pending changes from Phase 1...
```

If a break is detected (>5% drop AND >2K absolute token loss), it diffs the Phase 1 snapshots to identify the root cause.

### Root Cause Attribution

When a break is detected, the system explains it by checking which fields changed between the pre-request snapshot and the current state:

- **Model changed** — e.g., "model changed (claude-sonnet-4-6 → claude-opus-4-6)"
- **System prompt changed** — with character delta, e.g., "system prompt changed (+342 chars)"
- **Tools changed** — e.g., "tools changed (+2/-0 tools)" or "tool prompt/schema changed, same tool set"
- **Beta headers changed** — e.g., "betas changed (+fast-mode-2025-01-01)"
- **Fast mode toggled**
- **Global cache strategy changed** — e.g., "global cache strategy changed (system_prompt → none)"
- **Cache control changed** — scope or TTL flip
- **Auto mode toggled**
- **Overage state changed**
- **Cached microcompact toggled**
- **Effort changed** — e.g., "effort changed (default → high)"
- **Extra body params changed**

If no client-side change explains the drop, the system checks the time gap:
- Over 1 hour: "possible 1h TTL expiry (prompt unchanged)"
- Over 5 minutes: "possible 5min TTL expiry (prompt unchanged)"
- Under 5 minutes: "likely server-side (prompt unchanged, <5min gap)"

### False Positive Suppression

Several events are expected to cause cache drops and must not trigger false alarms:

- **`notifyCompaction(querySource, agentId)`** — called after autocompact, reactive compact, and session memory compaction. Resets the detector's baseline so the expected drop isn't flagged.
- **`notifyCacheDeletion(querySource)`** — called after cached microcompact applies `cache_edits`. The cache read drops because blocks were deliberately deleted, not because the prefix was invalidated.
- **TTL expiration** — detected by comparing time since last assistant message against known TTL thresholds (5min, 1hr). Attributed to TTL rather than a client-side bug.
- **Post-compaction flag** — `isPostCompaction` is logged with `tengu_api_success` events so analytics can filter expected post-compact cache misses.

### The `tengu_api_success` Event

Every successful API call logs a comprehensive analytics event containing all cache metrics:

```typescript
logEvent('tengu_api_success', {
  inputTokens: usage.input_tokens,
  cachedInputTokens: usage.cache_read_input_tokens ?? 0,
  uncachedInputTokens: usage.cache_creation_input_tokens ?? 0,
  cacheDeletedInputTokens: ...,       // from cache editing
  isPostCompaction: ...,              // first call after compact?
  globalCacheStrategy: ...,           // 'system_prompt' or 'none'
  timeSinceLastApiCallMs: ...,        // for TTL correlation
  model: ...,
  querySource: ...,
  ...
})
```

This is the primary data source for fleet-wide cache hit rate analysis. BigQuery queries (referenced in code comments as `bq-queries/prompt-caching/cache_break_pr19823_analysis.sql`) analyze these events to understand aggregate cache performance, identify regressions, and correlate breaks with specific code changes.

### The `tengu_prompt_cache_break` Event

When the detector confirms a break, it logs a diagnostic event:

```typescript
logEvent('tengu_prompt_cache_break', {
  systemPromptChanged: boolean,
  toolSchemasChanged: boolean,
  modelChanged: boolean,
  fastModeChanged: boolean,
  cacheControlChanged: boolean,
  globalCacheStrategyChanged: boolean,
  betasChanged: boolean,
  autoModeChanged: boolean,
  overageChanged: boolean,
  cachedMCChanged: boolean,
  effortChanged: boolean,
  extraBodyChanged: boolean,
  tokenDrop: number,
  prevCacheRead: number,
  currentCacheRead: number,
  timeSinceLastAssistantMsg: number,
  reason: string,              // human-readable explanation
  ...
})
```

This enables both real-time debugging (the developer sees a notification) and fleet-wide regression detection (aggregate analysis of which change types cause the most cache breaks).

### Tracking Key Isolation

The detector tracks state per `querySource` (with agentId suffix for subagents):

```typescript
function getTrackingKey(querySource, agentId): string | null {
  if (querySource === 'compact') return 'repl_main_thread'  // shares cache with main
  for (const prefix of TRACKED_SOURCE_PREFIXES) {
    if (querySource.startsWith(prefix)) return agentId || querySource
  }
  return null  // untracked (speculation, session_memory, etc.)
}
```

This prevents false positives when multiple agents run concurrently — each agent's cache read baseline is tracked independently. Compact shares the main thread's key because it uses the same `cacheSafeParams` (same server-side cache entry).

Short-lived forked agents (speculation, session_memory, prompt_suggestion) are untracked — they run 1-3 turns with a fresh agentId each time, so there's nothing meaningful to compare against.

---

## Summary: The Cache Architecture

```
Server-side prompt cache (Mycro)
    |
    +-- Global scope: system prompt prefix (shared across all users)
    +-- Org scope: system prompt + tools (shared within org)
    +-- Ephemeral scope: full request up to cache_control marker (per-user, per-session)
    |
    |  TTL: 5min (default) or 1hr (eligible users)
    |
    v
Client-side cache stability mechanisms:
    +-- Tool schema cache (freeze bytes at first render)
    +-- Sticky beta latching (never flip headers mid-session)
    +-- Frozen tool result budget decisions (byte-identical re-apply)
    +-- Cache-sharing forks (reuse main thread's cached prefix)
    +-- Advisor tool at suffix (not in cached prefix)
    +-- Deferred MCP tools (don't pollute global prefix)
    +-- System prompt boundary (static global / dynamic per-user split)
    +-- cache_reference on tool_results (enables server-side editing)
    +-- cache_edits (delete blocks without full invalidation)
    +-- Break detection (monitor and explain cache drops)
```

The overall design philosophy: the server cache is prefix-based and byte-sensitive, so the client goes to extraordinary lengths to ensure the bytes it sends are stable across requests. Every feature that could change the prefix — feature flags, MCP tools, mode toggles, tool result budgeting — has a specific mitigation to avoid accidental cache busts. When cache loss is unavoidable (compaction, context collapse), the system accepts it deliberately and notifies the detector to prevent false alarms.
