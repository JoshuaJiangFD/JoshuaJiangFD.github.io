---
title: "Demystifying Claude Code: Managing Message Context"
date: 2026-05-04 10:00:00 -0700
categories: [Claude Code]
tags: [Claude Code, LLM, Context Management]
mermaid: true
---

This post details how Claude Code manages the messages that make up the LLM's context window. Each iteration of the agentic loop runs a pipeline of 11 steps that progressively filter, compress, and transform the message array before sending it to the API. We trace every action that reads, transforms, or replaces the messages — from lightweight slicing to full conversation compaction.

For the broader orchestration architecture (how QueryEngine and query.ts interact, how the loop terminates, cross-turn state), see [Session Orchestration]({% post_url claude-code/2026-04-30-demystifying-claude-code-session-orchestration %}). For how the API call itself is constructed, see [Calling the Model]({% post_url claude-code/2026-05-05-demystifying-claude-code-calling-the-model %}).

---

## 1. The Big Picture

```
QueryEngine.submitMessage(prompt)
  |
  +-- processUserInput() -> messagesFromUserInput
  +-- mutableMessages.push(...messagesFromUserInput)     <-- USER INPUT ENTERS
  |
  +-- query(messages, systemPrompt, ...)
        |
        +-- queryLoop()   <-- the while(true) state machine
              |
              +=====================================================+
              |  EACH ITERATION (one "turn" of the agentic loop):   |
              |                                                      |
              |  1. SLICE         getMessagesAfterCompactBoundary()  |
              |  2. BUDGET        applyToolResultBudget()            |
              |  3. SNIP          snipCompactIfNeeded()              |
              |  4. MICROCOMPACT  deps.microcompact()                |
              |  5. COLLAPSE      contextCollapse.applyCollapses()   |
              |  6. AUTOCOMPACT   deps.autocompact()                 |
              |  7. BLOCKING CHK  calculateTokenWarningState()  EXIT |
              |  8. API CALL      deps.callModel()              EXIT |
              |  9. TOOL EXEC     runTools() / StreamingToolExecutor |
              |                                                 EXIT |
              | 10. ATTACHMENTS   getAttachmentMessages()            |
              | 11. STATE UPDATE  state = { messages: [...] }   EXIT |
              +=====================================================+
              
              EXIT points (reasons the loop terminates):
              
              Step 7:  blocking_limit (context too big, auto-compact off)
              Step 8:  completed (model responded with no tool_use -- NORMAL)
                       model_error, image_error, aborted_streaming
                       prompt_too_long (all recovery attempts failed)
                       stop_hook_prevented
              Step 9:  aborted_tools (user Ctrl+C during execution)
                       hook_stopped (tool hook prevented continuation)
              Step 11: max_turns (hit maxTurns limit before continuing)
```

Steps 1-6 form the **message pipeline** — a layered compression system that progressively reduces context before the API call. Each step operates on the output of the previous one, from lightweight filtering (slice, budget) to heavyweight summarization (autocompact). Steps 7-11 handle the API call, tool execution, context injection, and state merge for the next iteration.

---

## 2. Message Types

The conversation is a discriminated union type `Message` defined in `src/types/message.ts`. Four types flow through the pipeline:

- **`UserMessage`** contains user prompts and `tool_result` blocks. The Anthropic API requires tool results to be sent as `role: 'user'` messages, so from the pipeline's perspective, tool outputs are user-role responses to the assistant's requests.
- **`AssistantMessage`** carries the model's response — text blocks, `tool_use` blocks, and thinking blocks.
- **`SystemMessage`** provides internal signals: compact boundary markers, API error retry notifications, and informational telemetry. These are filtered out by `normalizeMessagesForAPI()` before reaching the API.
- **`AttachmentMessage`** injects non-conversational context (file contents, memories, plan mode instructions, notifications) into the stream. At API time, these are converted to `UserMessage`s and merged into adjacent user messages. For the full taxonomy, see [Attachment Messages]({% post_url claude-code/2026-05-14-demystifying-claude-code-attachment-messages %}).

---

## 3. The 11-Step Pipeline

### Step 1: SLICE (query.ts:365)

```
messagesForQuery = getMessagesAfterCompactBoundary()
```

This step does two things. First, it performs a **boundary slice**: it scans backward for the last compact boundary marker and drops everything before it. If no boundary exists, all messages are kept. Second, it applies a **snip projection** via `projectSnippedView()`, which filters out messages that were marked as snipped on a previous iteration (see Step 3). The REPL keeps snipped messages in memory for UI scrollback, but the model should never see them again.

The boundary slice is necessary even though `QueryEngine` already splices pre-boundary messages when compaction occurs. In the REPL path, `REPL.tsx` keeps the full message history for scrollback and passes the entire array to `query()`, so Step 1 is what actually ensures the model only sees post-compact messages. On `--resume`, the loaded history may contain the full pre-boundary chain before any compaction has fired in the current process. Subagents and forks (`runAgent.ts`, `forkedAgent.ts`) pass message arrays that may contain boundaries from inherited conversation history. And if reactive compaction fires mid-loop producing a new boundary, the next iteration needs Step 1 to slice from that new boundary.

In short, Step 1 is a defensive guarantee: regardless of how messages arrived at `query()`, the model only ever sees content from the most recent compact boundary onward.

### Step 2: TOOL RESULT BUDGET (query.ts:379-394)

```
messagesForQuery = applyToolResultBudget(...)
```

When the aggregate tool results in a single user message exceed 200K chars (`MAX_TOOL_RESULTS_PER_MESSAGE_CHARS`), the largest results are persisted to disk and replaced with a preview plus a file path the model can `Read` if it needs the full content:

```
<persisted-output>
Output too large (350 KB). Full output saved to: /path/to/.tool-results/abc123.txt

Preview (first 2 KB):
[first 2000 bytes of the original output]
...
</persisted-output>
```

This is not a silent loss — the model sees a preview to understand what the output was, and can retrieve the full content on demand. The system targets the largest fresh results first and leaves smaller ones untouched. Tools with `maxResultSizeChars: Infinity` (like `Read`) are exempt entirely since they have their own separate size limits.

Once a result is seen, its fate is frozen by `tool_use_id`: a result that was replaced stays replaced, and a result that was kept stays kept on all subsequent iterations. This ensures byte-identical messages across turns, which preserves the prompt cache. Replacement decisions are persisted for session resume via a side channel (see [Appendix A](#appendix-a-tool-result-budget-persistence)).

### Step 3: SNIP COMPACT (query.ts:401-409)

```
snipResult = snipCompactIfNeeded(messagesForQuery)
messagesForQuery = snipResult.messages
snipTokensFreed = snipResult.tokensFreed
```

This step decides whether to snip new messages on the current iteration. It looks at the current token count and removes old turn groups from the middle of history (not head or tail). The returned array has those messages already removed, so the effect is immediate for Steps 4-8 on this iteration. It also marks the removed messages as snipped in the source array and yields a `boundaryMessage` if a snip occurred. The freed token count is returned so the autocompact threshold in Step 6 can adjust accordingly.

Step 3 has a dual relationship with Step 1. Step 3 removes messages from the returned array (immediate effect) and tags them in the source. On subsequent iterations, Step 1's `projectSnippedView` filters those tagged messages out of the view — this is necessary because the REPL's in-memory array still contains them for UI scrollback. Both can fire on the same iteration: Step 1 filters old snips from previous iterations while Step 3 creates new ones for the current iteration.

### Step 4: MICROCOMPACT (query.ts:414-425)

```
microcompactResult = deps.microcompact(...)
messagesForQuery = microcompactResult.messages
```

This step clears old tool result content to free context space. It does not detect semantic staleness (it doesn't know whether a file was re-read later, for example). Instead it uses two proxy heuristics based on age and time.

The first path is **Cached Microcompact**, which is count-based. It registers every compactable tool result by `tool_use_id` in encounter order. When the total count exceeds a configured threshold, it deletes the oldest results while keeping the most recent N. This path does not modify local messages — instead it sends `cache_edits` to the API to delete the content server-side from the cached prefix, avoiding prompt cache invalidation.

The second path is **Time-Based Microcompact**, which is gap-based. It fires when the time since the last assistant message exceeds a threshold, indicating that the server's prompt cache has already expired. Since the full prefix will be rewritten regardless, it replaces old tool result content with `"[Old tool result content cleared]"`, again keeping only the most recent N results.

Only results from specific tools can be cleared: `Read`, `Bash`/shell, `Grep`, `Glob`, `WebSearch`, `WebFetch`, `Edit`, and `Write`. The most recent N results are always preserved regardless of which trigger fires.

### Step 5: CONTEXT COLLAPSE (query.ts:440-447)

```
collapseResult = contextCollapse.applyCollapses(...)
messagesForQuery = collapseResult.messages
```

This step replaces spans of messages with summaries using a read-time projection. It only runs when the `CONTEXT_COLLAPSE` feature flag is enabled and `isContextCollapseEnabled()` returns true. When active, context collapse **replaces autocompact entirely** — `shouldAutoCompact()` returns false because the two would race each other (autocompact's threshold at ~93% sits between collapse's staging trigger at ~90% and its blocking trigger at ~95%).

Context collapse operates on a two-phase staged-then-committed lifecycle (see [Appendix B](#appendix-b-context-collapse-lifecycle) for the full flow). A separate summarization agent generates summaries for spans of old messages ahead of time, holding them in a staged queue. When applied, these become committed collapses that `projectView()` replays each iteration to produce the model's view. The original messages are never deleted — the collapsed view is purely a read-time projection.

This step runs before autocompact so that if collapse alone brings the token count under the threshold, autocompact is skipped entirely — preserving granular context that a full compaction would have destroyed.

### Step 6: AUTOCOMPACT (query.ts:454-543)

```
{ compactionResult } = deps.autocompact(...)
```

This is the most impactful step. For the full decision flow and compact process, see [Message Compaction]({% post_url claude-code/2026-05-05-demystifying-claude-code-message-compaction %}).

When `deps.autocompact()` returns a successful `compactionResult`, query.ts does three things before building the final message array. First, it logs a telemetry event with metrics about the compaction — original vs compacted message counts, pre/post token counts, and API usage breakdown. Second, if a `taskBudget` is configured, it captures how many tokens the pre-compact context consumed and decrements `taskBudgetRemaining` accordingly (the server can no longer see the full pre-compact history after compaction, so this tracks cumulative spend). Third, it resets the `autoCompactTracking` state to mark a fresh compaction epoch — new `turnId`, `turnCounter` back to zero, and `consecutiveFailures` reset to zero.

Then `buildPostCompactMessages()` constructs the replacement array, each message is yielded back to QueryEngine for persistence, and `messagesForQuery` is replaced:

```
postCompactMessages = buildPostCompactMessages()
messagesForQuery = postCompactMessages
```

The result replaces the entire history with: `[boundary, summary, attachments, hookResults]`.

If compaction **fails**, the `consecutiveFailures` count is propagated into the tracking state so the circuit breaker can trip on the next iteration (stops retrying after 3 consecutive failures).

#### Concrete Example: Under Threshold vs Over Threshold

Starting point — `messagesForQuery` after Steps 1-5:

```
[0]  boundary (from earlier compact)
[1]  summary: "User asked to refactor auth..."
[2]  attachment: file_content of auth.ts
[3]  user: "now fix the login bug"
[4]  assistant: "I'll read the file..." + tool_use
[5]  user: tool_result for Read (12K chars)
[6]  assistant: "Found the bug..." + tool_use Edit
[7]  user: tool_result for Edit
[8]  attachment: edited_text_file diff
[9]  attachment: relevant_memory
```

**Case A: Under Threshold (~150K tokens)**

Autocompact checks the token count: `150K < 167K threshold`. No compaction fires, and `messagesForQuery` passes through unchanged.

**Case B: Over Threshold (~170K tokens)**

Autocompact checks: `170K >= 167K threshold`. Compaction fires and `messagesForQuery` is replaced entirely:

```
[0] NEW boundary (compact_boundary, preTokens: 170K)
[1] NEW summary ("The user debugged a login bug in auth.ts...")
[2] attachment: re-read of auth.ts (fresh content, <=5K)
[3] attachment: deferred_tools_delta (tool schemas)
[4] attachment: agent_listing_delta
[5] attachment: mcp_instructions_delta
[6] hook_result: SessionStart hook output
```

The result is ~10-15K tokens total. The old messages are gone from the model's view.

### Step 7: BLOCKING CHECK (query.ts:628-648)

```
calculateTokenWarningState(tokenCount, model)
```

If the token count has reached the hard blocking limit and auto-compact is disabled, this step yields an error message and exits the loop. This reserves space so the user can still run `/compact` manually. The check is skipped if compaction just ran on this iteration or if reactive compact is enabled (since reactive compact will handle the overflow on the API's 413 response).

### Step 8: API CALL (query.ts:659-863)

```
for await (message of deps.callModel(...))
```

This step streams the model's response and collects results into three arrays: `assistantMessages[]` holds the model's responses, `toolUseBlocks[]` accumulates any tool calls the model wants to make, and `toolResults[]` captures results from tools that execute concurrently during streaming (when streaming tool execution is enabled).

Certain error responses are withheld from the caller rather than yielded immediately: prompt-too-long, max-output-tokens, and media-size errors are held back so the loop can attempt recovery (reactive compaction, output escalation, or retry) before surfacing the error.

After streaming completes, the loop checks `needsFollowUp` (query.ts:1062). If the model's response contained **no `tool_use` blocks**, `needsFollowUp` remains `false` and the loop exits with `return { reason: 'completed' }` — **Steps 9, 10, and 11 are skipped entirely**. The model decided the turn is done by simply not calling any tools.

If `needsFollowUp` is `true` (at least one `tool_use` block was present), execution continues to Step 9.

For the full message transformation pipeline inside `claude.ts` (normalization, tool schema building, caching, retry), see [Calling the Model]({% post_url claude-code/2026-05-05-demystifying-claude-code-calling-the-model %}).

### Step 9: TOOL EXECUTION (query.ts:1380-1408)

```
for await (update of toolUpdates)
```

This step executes the tool calls that the model requested. Each tool's output becomes a `UserMessage` containing a `tool_result` block (the API requires tool results to be sent as user messages). These are collected into `toolResults[]`. Tools may also produce `AttachmentMessage`s — for example, hook results that signal whether the loop should continue. For the full tool execution lifecycle — dispatchers, the single-tool pipeline, concurrency, interrupt behavior, and hooks — see [Tool Execution]({% post_url claude-code/2026-05-12-demystifying-claude-code-tool-execution %}).

### Step 10: ATTACHMENTS (query.ts:1580-1628)

```
for await (attachment of getAttachmentMessages(...))
```

After tool execution completes, this step gathers contextual information that the model should see before its next response. These are appended to `toolResults[]` as `AttachmentMessage`s. The injections include queued commands from the message queue (follow-up prompts the user submitted while the model was working), memory prefetch results (relevant memories surfaced by the auto-memory system), skill discovery prefetch results, and file change notifications (diffs for files modified externally since the last turn).

### Step 11: STATE MERGE (query.ts:1715-1727)

At the end of each iteration, the full `State` object is rebuilt:

```typescript
const next: State = {
  messages: [...messagesForQuery, ...assistantMessages, ...toolResults],
  toolUseContext: toolUseContextWithQueryTracking,
  autoCompactTracking: tracking,
  turnCount: nextTurnCount,
  maxOutputTokensRecoveryCount: 0,
  hasAttemptedReactiveCompact: false,
  pendingToolUseSummary: nextPendingToolUseSummary,
  maxOutputTokensOverride: undefined,
  stopHookActive,
  transition: { reason: 'next_turn' },
}
state = next
```

The three arrays being merged into `messages`:

| Array | Contents | Source |
|-------|----------|--------|
| `messagesForQuery` | The conversation history sent to the API this iteration, already processed by Steps 1-6. | Steps 1-6 |
| `assistantMessages` | The model's response(s) — text blocks, tool_use blocks, thinking blocks. | Step 8 (API streaming) |
| `toolResults` | Tool result `UserMessage`s and `AttachmentMessage`s from context injection. | Steps 9-10 |

This merged array becomes `messages` for the next iteration. The `while(true)` continues and Step 1 runs again on this merged result.

---

## 4. What Happens When the User References Compacted History

After compaction, if the user says *"revert that edit you made"*, the model only has the summary (which knows *what* it did but not the exact diff), the re-read of the file (current state, not pre-edit), and a mention of the transcript path (which the model cannot practically read in JSONL format).

The model's recovery options depend on what was lost:

- **File edits** can be recovered reliably via `git diff` or `git log`, since git is the true source of truth for code changes.
- **Code it read but didn't edit** can simply be re-read with the `Read` tool — the content is still on disk.
- **Decisions or reasoning it explained** depend on how well the summary captured them. The summary is instructed to preserve all user messages and key decisions, but nuance may be lost.
- **Exact error messages or log output** are generally gone — the summary may note "there was an error" without the exact text.
- **Multi-step tool chain specifics** are reduced to narrative. The summary captures what happened but not the intermediate outputs.

There is no automatic mechanism that detects "the user is referencing something from before compaction" and fetches more detail. The model must use tools (git, Read, Bash) to reconstruct what it needs.

---

## Appendix A: Tool Result Budget Persistence

Replacement decisions from Step 2 are persisted via a fire-and-forget side channel that bypasses the normal QueryEngine message yield/push path:

```
query.ts Step 2:
  applyToolResultBudget(messages, state, callback)
      |
      |  when new replacements are made, fires the callback:
      |
      +-- records => void recordContentReplacement(records, agentId)
                            |
                            +-- getProject().insertContentReplacement(records)
                                  |
                                  +-- this.appendEntry({ type: 'content-replacement', ... })
                                        |
                                        v
                                  JSONL file (appended directly)
```

These are metadata records appended alongside conversation messages in the JSONL, not messages themselves:

```jsonl
{"type":"user", "uuid":"aaa", "message":{...}, "parentUuid":"...", ...}
{"type":"assistant", "uuid":"bbb", "message":{...}, "parentUuid":"aaa", ...}
{"type":"content-replacement", "sessionId":"...", "replacements":[{"kind":"tool-result","toolUseId":"xyz","replacement":"<persisted-output>..."}]}
{"type":"user", "uuid":"ccc", "message":{...}, "parentUuid":"bbb", ...}
```

On `--resume`, `loadTranscriptFile()` reads both the messages and these records, then passes the records to `reconstructContentReplacementState()` to rebuild the decision map — ensuring byte-identical replacements and prompt cache hits.

---

## Appendix B: Context Collapse Lifecycle

Context collapse uses a two-phase lifecycle to manage context incrementally without losing information.

### Phase 1: Staging

When the context reaches ~90% of the effective window, the system spawns a separate summarization agent (querySource `marble_origami`) to generate a summary for a span of old messages. The span is defined by UUID boundaries (`firstArchivedUuid` to `lastArchivedUuid`). The resulting summary is added to a staged queue with metadata including a risk score and timestamp. At this point the model still sees the original messages — staging is preparation, not application.

```
Context at ~90% of effective window
    |
    v
ctx-agent spawns, summarizes a span of messages
    |
    v
Staged queue: [{ startUuid, endUuid, summary, risk, stagedAt }, ...]
    |
    (model still sees original messages)
```

### Phase 2: Committing

When a staged collapse is applied (either proactively by `applyCollapsesIfNeeded` or reactively by `recoverFromOverflow`), it becomes a committed collapse. The commit is persisted as a `marble-origami-commit` entry in the JSONL transcript. From this point on, `projectView()` replays all committed collapses in order each iteration, replacing the archived span with a summary placeholder in the model's view.

```
Staged collapse is committed
    |
    v
JSONL: { type: "marble-origami-commit", collapseId, summaryUuid,
         summaryContent, firstArchivedUuid, lastArchivedUuid }
    |
    v
projectView() replaces [firstArchived...lastArchived] with summary placeholder
    |
    (model now sees summary instead of original messages)
```

### Overflow Recovery

When a 413 prompt-too-long is received after Step 8, `recoverFromOverflow()` can immediately commit all staged collapses without needing another API call — the summaries were already generated during the staging phase. This is the `collapse_drain_retry` continuation path in the query loop. If draining staged collapses still doesn't free enough space (or the staged queue is empty), the system falls through to reactive compact as a last resort.

### Persistence and Resume

Committed collapses are persisted as `marble-origami-commit` entries and the staged queue state is persisted as a `marble-origami-snapshot` entry (last-wins). On `--resume`, the commit log is replayed to reconstruct the collapsed view, and the snapshot restores the staged queue so the system can pick up where it left off.

The original messages are never deleted — they remain in the REPL array for UI scrollback and in the JSONL for session history. The collapsed view is purely a read-time projection.

