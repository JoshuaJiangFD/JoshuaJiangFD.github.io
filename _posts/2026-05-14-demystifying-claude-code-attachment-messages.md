---
title: "Demystifying Claude Code: Attachment Messages"
date: 2026-05-14 10:00:00 -0700
categories: [Claude Code]
tags: [Claude Code, LLM, Attachment Messages, Context Injection, System Reminders]
mermaid: true
---

This post covers how Claude Code uses attachment messages — a sideband context injection mechanism that lets the harness slip information into the conversation without the user typing it or the model requesting it. We trace the full lifecycle: from the 40+ attachment subtypes defined in `attachments.ts`, through the runtime producer pipeline in `getAttachments()`, the post-compaction context rebuild in `compact.ts`, to the `normalizeAttachmentForAPI()` function that converts each attachment into the API messages the model actually sees.

For where attachments sit in the 11-step message pipeline, see [Managing Message Context]({% post_url 2026-05-05-demystifying-claude-code-managing-message-context %}), Steps 9-10. For the skill listing attachment specifically, see [Skills]({% post_url 2026-05-13-demystifying-claude-code-skills %}), Section 4. For how attachments survive compaction, see [Message Compaction]({% post_url 2026-05-05-demystifying-claude-code-message-compaction %}). For the system prompt sections that reference `<system-reminder>` tags, see [The System Prompt]({% post_url 2026-05-07-demystifying-claude-code-system-prompt %}), Section 7.

---

## 1. What the User Sees

Attachment messages are invisible infrastructure. The user never types them, and the terminal doesn't render most of them as chat bubbles. But they are the reason the model knows:

- That a file changed externally while it was working (a linter auto-fixed something, the user edited in their IDE)
- What skills, agents, and MCP tools are currently available
- That it's in plan mode and shouldn't execute tools
- What memory files are relevant to the current task
- That a background task just finished
- That the date changed (the user coded past midnight)
- That the user queued a follow-up message while the model was mid-turn

From the user's perspective, these things "just work" — the model seems aware of its environment. From the implementation's perspective, each of these is an `AttachmentMessage` produced by a specific generator function, converted to an API message format, and injected into the conversation at the right moment.

---

## 2. The AttachmentMessage Type

An attachment message is a sideband context injection — a way for the harness to slip information into the conversation (file changes, available skills, mode reminders, queued user input) that isn't a direct user prompt, model response, or tool result. At API time, each attachment is converted into a `UserMessage` (usually wrapped in `<system-reminder>` tags) or silently consumed as a control signal.

The type itself is a thin wrapper around a discriminated union payload:

```typescript
// types/message.ts:285-291
interface AttachmentMessage<A extends Record<string, unknown> = Record<string, unknown>> {
  type: 'attachment'
  attachment: A & { type: string }   // the payload — discriminated by .type
  uuid: UUID
  timestamp: string
  isMeta?: true
}
```

The `attachment.type` field is the discriminant that drives the entire lifecycle. The `Attachment` union type (`attachments.ts:440-718`) defines 40+ subtypes — file contents, mode reminders, skill listings, hook results, task notifications, and more. The subtype determines three things:

1. **Where it comes from** — each producer function creates attachments of specific subtypes. `getChangedFiles()` produces `edited_text_file`, `getSkillListingAttachments()` produces `skill_listing`, `getPlanModeAttachments()` produces `plan_mode`, and so on (see [Section 3](#3-the-producer-pipeline)).
2. **How it reaches the model** — `normalizeAttachmentForAPI()` switches on the type to decide the API conversion: most become `<system-reminder>` text, file attachments become synthetic tool call pairs, and some produce nothing at all (see [Section 5](#5-how-attachments-reach-the-model)).
3. **What data it carries** — each subtype is a typed object with fields specific to its purpose. For example, `edited_text_file` carries a filename and a diff snippet; `skill_listing` carries the formatted content string and a skill count; `queued_command` carries the user's prompt text and origin metadata.

The full subtype reference with API conversion patterns is in [Appendix B](#appendix-b-attachment-subtype-reference).

### Creation

All attachment subtypes are created through the same factory:

```typescript
// attachments.ts:3201-3210
function createAttachmentMessage(attachment: Attachment): AttachmentMessage {
  return {
    attachment,
    type: 'attachment',
    uuid: randomUUID(),
    timestamp: new Date().toISOString(),
  }
}
```

This factory stamps each attachment with a unique UUID and timestamp, producing a message that can be yielded into the conversation history alongside `UserMessage`s and `AssistantMessage`s. The UUID enables session resume — when a session is replayed from the JSONL transcript, each attachment can be identified and its state reconstructed.

---

## 3. The Producer Pipeline

During normal operation, all attachment production flows through a single function — `getAttachments()` — called from two different sites with different inputs. At the inter-turn call site, this runs as part of the ATTACHMENTS step (Step 10 of the agentic loop). After `getAttachments()` returns within that same step, prefetch results (memory and skill discovery) that were running concurrently with the turn are also appended. Post-compaction attachment generation is a separate mechanism covered in [Section 4](#4-post-compaction-context-rebuild).

### 3.1 The Core Function: `getAttachments()`

All runtime attachment production flows through a single function — `getAttachments()`. It orchestrates 25+ independent producer functions in parallel and merges their results. The `input` parameter determines which producers actually fire.

The function is called from **two sites** with  different value of the `input` param:

| Call Site                             | `input` param      | When                                                                                        | Includes user input attachments?                          | Includes environment attachments? |
| ------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------- | --------------------------------------------------------- | --------------------------------- |
| Turn start (via `QueryEngine` → `processUserInput()`) | User's text string | Before the first API call of the turn, when the user submits a message | Yes                                                       | Yes                               |
| Inter-turn (via `query.ts`)           | `null`             | After each tool execution cycle (Step 10 of the agentic loop)                               | No (`input` is null, so user input producers are skipped) | Yes                               |


Internally, `getAttachments()` runs in two phases.  At the start of each user turn, it gets result from both phases as it pass a non-null user input: User Input Attachments and Environment Attachments . Every subsequent iteration within the same turn gets only environment attachments, because there's no new user input to parse.

#### User Input Attachments (when `input` is non-null)

This phase only runs when `input` is non-null (i.e., from the turn-start call site). It extracts context from the user's input text:

- **`file`** / **`directory`** — `@`-mentioned files and directories. The `extractAtMentionedFiles()` parser (`attachments.ts:2757`) handles both regular (`@path/to/file.txt`) and quoted (`@"path with spaces/file.txt"`) syntax, with optional line ranges (`@file.txt#L10-20`). For files, `generateFileAttachment()` reads the content through `FileReadTool.call()` and records the file state in `readFileState` for later change detection. For directories, `readdir()` lists entries (capped at 1,000).
- **`mcp_resource`** — `@server:uri` mentions that reference MCP resources
- **`agent_mention`** — `@agent-<type>` or `@"<type> (agent)"` mentions that signal intent to invoke a specific agent
- **`skill_discovery`** (experimental) — when skill search is enabled, the user's input text is analyzed for skill relevance

This phase must complete before environment attachments start because `@`-mentioned file paths populate `nestedMemoryAttachmentTriggers` — the set of file paths that `getNestedMemoryAttachments()` uses to discover nearby CLAUDE.md files. If both phases ran concurrently, the nested memory producer might see an empty trigger set.

#### Environment Attachments (always)

This phase runs on **every invocation** of `getAttachments()` — both at turn start and on every inter-turn iteration. It gathers context about the session's current state regardless of whether there's new user input.

Producers are split into two parallel groups:

```
Environment attachments (runs via Promise.all after user input phase completes):
    |
    +-- All-thread attachments (run in main + subagents)
    |     ├── getQueuedCommandAttachments()     → queued_command
    |     ├── getDateChangeAttachments()         → date_change
    |     ├── getUltrathinkEffortAttachment()    → ultrathink_effort
    |     ├── getDeferredToolsDeltaAttachment()   → deferred_tools_delta
    |     ├── getAgentListingDeltaAttachment()    → agent_listing_delta
    |     ├── getMcpInstructionsDeltaAttachment()  → mcp_instructions_delta
    |     ├── getChangedFiles()                   → edited_text_file, edited_image_file
    |     ├── getNestedMemoryAttachments()        → nested_memory
    |     ├── getDynamicSkillAttachments()        → dynamic_skill
    |     ├── getSkillListingAttachments()        → skill_listing
    |     ├── getPlanModeAttachments()            → plan_mode, plan_mode_reentry
    |     ├── getPlanModeExitAttachment()         → plan_mode_exit
    |     ├── getTodoReminderAttachments()        → todo_reminder / task_reminder
    |     └── getCriticalSystemReminderAttachment() → critical_system_reminder
    |
    +-- Main-thread-only attachments (skipped in subagents)
          ├── getSelectedLinesFromIDE()           → selected_lines_in_ide
          ├── getOpenedFileFromIDE()              → opened_file_in_ide
          ├── getOutputStyleAttachment()          → output_style
          ├── getDiagnosticAttachments()          → diagnostics
          ├── getUnifiedTaskAttachments()         → task_status
          ├── getAsyncHookResponseAttachments()   → async_hook_response
          ├── getTokenUsageAttachment()           → token_usage
          ├── getMaxBudgetUsdAttachment()         → budget_usd
          ├── getOutputTokenUsageAttachment()     → output_token_usage
          └── getVerifyPlanReminderAttachment()   → verify_plan_reminder
```

The all-thread / main-thread split exists because some attachments are semantically specific to the main conversation (IDE selections, diagnostics, token budgets) while others are needed by subagents too (queued commands, file changes, skill listings).

#### Error Isolation

With 25+ producer functions running in parallel, any one of them can fail — a file doesn't exist, an MCP server is unreachable, a timeout fires. Without isolation, a single crash would take down the entire attachment pipeline and the model would receive *no* context for that iteration.

The solution is a per-function error boundary: each producer is wrapped so that if it throws, the error is logged and the producer returns an empty array. The model never sees a partial failure; it simply doesn't receive that one attachment type on this iteration. A broken MCP server doesn't prevent file change detection from reaching the model; a crash in diagnostic tracking doesn't block skill listings.

The entire `getAttachments()` call also has a 1-second timeout. Producers that respect the abort signal bail out early; those that don't will complete but their results are discarded if the timeout already fired.

### 3.2 Prefetch Injection (Outside `getAttachments()`)

After `getAttachments()` returns within Step 10, `query.ts` appends two additional attachment sources. These live outside `getAttachments()` because they are long-running async operations that should not block the main attachment pipeline — they are started earlier and collected here only if they've settled.

1. **Memory prefetch** (`query.ts:301-304, 1599-1614`) — `startRelevantMemoryPrefetch()` fires **once per user turn**, before the `while(true)` loop begins, and produces `relevant_memories` attachments. Since the user's prompt doesn't change across loop iterations, repeating the prefetch would ask the same question multiple times. The consume point at Step 10 is checked on every iteration — if the prefetch hasn't settled yet, it's skipped and retried on the next iteration. Once consumed, it's marked as delivered and not consumed again.

2. **Skill discovery prefetch** (`query.ts:331-335, 1620-1628`) — `startSkillDiscoveryPrefetch()` fires **on every iteration** of the loop, but an internal `findWritePivot` guard returns early on non-write iterations — so it effectively only fires when the model performed a write operation. It produces `skill_discovery` attachments from a concurrent Haiku call that identifies relevant skills based on the model's recent tool activity. This replaces an earlier blocking approach where 97% of calls found nothing in production.

---

## 4. Conversation Start: The First Invocation

When the user sends their first message, `processUserInput()` calls `getAttachments()` with the user's text. Both phases run: user input attachments extract any `@`-mentioned files, and environment attachments fire for the first time. Since nothing has been announced yet — the delta tracking sets are all empty — the delta-based producers emit their full initial sets: all available skills (`skill_listing`), all agent types (`agent_listing_delta`), all deferred tools (`deferred_tools_delta`), and all MCP server instructions (`mcp_instructions_delta`).

`processTextPrompt()` returns these attachments *after* the user message: `[userMessage, ...attachments]`. Then `prependUserContext()` adds the user context (CLAUDE.md + current date) at the front: `[userContext, userMessage, ...attachments]`. But before the API call, the reordering step ([Section 6.1](#61-reordering)) bubbles the attachments to the top — neither `userContext` nor `userMessage` are stopping points (they're regular user messages, not assistant messages or tool results). The merging step ([Section 6.3](#63-merging)) then folds all consecutive user-role messages into a single API message. The model sees the content in order — initial attachment announcements, user context, then the user's prompt — but as content blocks within one message (see [The System Prompt]({% post_url 2026-05-07-demystifying-claude-code-system-prompt %}), Section 7 for the full layout).

On subsequent iterations within the same turn and across later turns, the delta-based producers only emit new additions or removals — the initial full set is already in the conversation history and visible to the model.

---

## 5. Post-Compaction Context Rebuild

When compaction fires (see [Message Compaction]({% post_url 2026-05-05-demystifying-claude-code-message-compaction %})), the entire conversation history is replaced with a summary. But the model needs more than a summary to continue working — it needs to know what files it was looking at, what mode it's in, what skills are active, and what tools are available. Compaction rebuilds this context through targeted attachments.

This path does not use `getAttachments()`. It has its own dedicated attachment generation in `compact.ts` because the context is fundamentally different — there are no messages to scan for prior announcements, no user input to parse, and the goal is reconstruction rather than incremental updates.

`compact.ts` (lines 531-585) generates these post-compaction attachments:

```
buildPostCompactMessages()
    |
    +-- createPostCompactFileAttachments()    → file (re-reads of recently accessed files, ≤5K each)
    +-- createAsyncAgentAttachmentsIfNeeded() → task_status (running background agents)
    +-- createPlanAttachmentIfNeeded()        → plan_file_reference (active plan content)
    +-- createPlanModeAttachmentIfNeeded()    → plan_mode (if in plan mode)
    +-- createSkillAttachmentIfNeeded()       → invoked_skills (skills loaded pre-compaction)
    +-- getDeferredToolsDeltaAttachment([], ...) → deferred_tools_delta (full re-announce)
    +-- getAgentListingDeltaAttachment([], ...)  → agent_listing_delta (full re-announce)
    +-- getMcpInstructionsDeltaAttachment([], ...) → mcp_instructions_delta (full re-announce)
```

The delta-based attachments (`deferred_tools_delta`, `agent_listing_delta`, `mcp_instructions_delta`) are called with an empty message array, which forces a diff against nothing — effectively re-announcing the full current state. This is necessary because compaction ate all prior delta announcements.

The file re-reads are bounded. `createPostCompactFileAttachments()` selects the most recently accessed files from `readFileState` (up to `POST_COMPACT_MAX_FILES_TO_RESTORE`) and re-reads each one, capped at 5K characters. Files too large for the cap get a `compact_file_reference` attachment instead — a lightweight pointer that tells the model the file exists but it needs to `Read` it again for full content.

---

## 6. How Attachments Reach the Model

An `AttachmentMessage` in the conversation history is not something the Anthropic API understands — the API expects `UserMessage`s and `AssistantMessage`s in alternating turns. The conversion from attachment to API message happens in `normalizeMessagesForAPI()` (`messages.ts:1989+`), which runs in `claude.ts` before every API call.

### 6.1 Reordering

Before normalization, `reorderAttachmentsForAPI()` (`messages.ts:1481-1527`) repositions attachments to ensure they land right after a "stopping point" — an assistant message or a tool_result user message. The algorithm scans backward from the end of the message array, collecting attachment messages into a buffer. When it hits a stopping point, it flushes the buffer so those attachments appear immediately after that stopping point in the final array. Attachments that aren't adjacent to any stopping point bubble all the way to the top.

This ensures that attachments don't break the tool_use → tool_result pairing that the API expects. In practice, attachments are usually already positioned after tool results (since Step 10 appends them there), so the reordering is a defensive guarantee rather than a frequent transformation.

### 6.2 Normalization

After reordering, each `AttachmentMessage` is converted into zero or more `UserMessage`s by `normalizeAttachmentForAPI()` (`messages.ts:3453-4286`). The conversion is subtype-specific — different attachment types map to different API representations depending on what the model needs to see. This function contains a large switch on `attachment.type` that maps each subtype to zero or more `UserMessage`s (see [Appendix B](#appendix-b-attachment-subtype-reference) for the full mapping). The conversion falls into four patterns:

**Pattern 1: `<system-reminder>` text.** A `UserMessage` with text wrapped in `<system-reminder>` tags, marked `isMeta: true` (hidden from the user's terminal). This is the default pattern for context injections that the model should read but the user shouldn't see in their chat history. The majority of subtypes use this pattern. For why this wrapper exists and what it enables, see [Appendix A](#appendix-a-the-system-reminder-wrapper).

**Pattern 2: Synthetic tool call pairs.** A synthetic `tool_use` + `tool_result` pair, as if the model had called `Read` or `ls` itself. This gives the model full file content in the same format it would see from its own tool calls. Used by `file` and `directory`.

**Pattern 3: Signal-only (empty array).** No API messages produced. These carry information for the harness only — they trigger side effects (like stopping the loop or capturing structured output) and the model never sees them.

| Subtype                     | Consumed By       | Effect                                                       |
| --------------------------- | ----------------- | ------------------------------------------------------------ |
| `max_turns_reached`         | `query.ts`        | Causes QueryEngine to exit the loop                          |
| `structured_output`         | SDK callers       | Captured as the return value for structured-output API calls |
| `hook_stopped_continuation` | `query.ts`        | Sets a flag that prevents the loop from continuing           |
| `hook_permission_decision`  | Permission system | Records an allow/deny decision from a hook                   |
| `command_permissions`       | Skill execution   | Carries allowed-tools metadata for context modification      |
| `already_read_file`         | Dedup logic       | Marks that a file was already read (no need to inject again) |
| `dynamic_skill`             | UI only           | Tells the terminal a new skill directory was discovered      |

These subtypes return `[]` from `normalizeAttachmentForAPI()`. They exist as `AttachmentMessage`s rather than being handled through a separate channel because: (1) they need to be persisted in the JSONL transcript for session resume, and (2) they need timestamps and UUIDs for the same lifecycle tracking as other messages.

**Pattern 4: Pass-through user message.** A `UserMessage` with the user's actual text, not wrapped in `<system-reminder>`. Unlike other attachments, these may be visible in the transcript (if the command came from human input rather than a system notification). The `isMeta` property is only set when the queued command was system-generated; human-typed messages drained mid-turn stay visible. Used by `queued_command`.

### 6.3 Merging

The Anthropic API requires strictly alternating user/assistant turns. But after normalization, multiple consecutive attachments each produce their own `UserMessage`, which would create adjacent user messages that violate this constraint. Merging solves this by folding attachment-produced `UserMessage`s into the preceding `UserMessage` via `mergeUserMessagesAndToolResults()` (`messages.ts:2280-2286`).

For example, suppose tool execution produces a tool result followed by two attachments:

```
[user: tool_result for Read]        ← already a UserMessage
[attachment: edited_text_file]      ← normalizes to a UserMessage
[attachment: plan_mode]             ← normalizes to another UserMessage
```

Without merging, the API would receive three consecutive user messages. With merging, the two attachment messages are folded into the tool_result message, producing a single `UserMessage` with all three content blocks combined.

---

## 7. Key Attachment Subtypes in Detail

### 7.1 File Change Detection (`edited_text_file`)

When the model reads a file (via `Read`, `Edit`, `Write`, or `@`-mention), the harness records the file's content and modification timestamp in `readFileState` — a per-session LRU cache. On every subsequent iteration, `getChangedFiles()` (`attachments.ts:2063-2161`) checks each cached file's current mtime against the recorded timestamp. If the file changed, it re-reads the file and computes a diff snippet via `getSnippetForTwoFileDiff()`.

The resulting `edited_text_file` attachment tells the model exactly what changed:

```
Note: /path/to/file.ts was modified, either by the user or by a linter.
This change was intentional, so make sure to take it into account as you
proceed (ie. don't revert it unless the user asks you to). Don't tell
the user this, since they are already aware. Here are the relevant changes
(shown with line numbers):
[diff snippet]
```

The instruction "don't tell the user this" prevents the model from narrating external modifications — the user already knows their linter ran, and a chatty notification would be noise.

Edge cases are handled carefully. Files deleted since the last read trigger an `ENOENT` error, which evicts the entry from `readFileState`. But transient stat failures — editor atomic-save races, network filesystem hiccups — do *not* evict, because false eviction would break subsequent `Edit` operations that depend on the cached content.

### 7.2 Queued Commands (`queued_command`)

Users can type follow-up messages while the model is mid-turn executing tools. These are captured in a message queue and drained by `getQueuedCommandAttachments()` (`attachments.ts:1046-1083`) at the end of each iteration. The attachment carries the user's text, optional image content blocks (for pasted images), and origin metadata.

The queue draining is agent-scoped: the main thread drains commands with `agentId === undefined`, while subagents drain only `task-notification` commands addressed to their specific `agentId`. This prevents a subagent from accidentally consuming the user's prompt.

After attachments are collected, `query.ts` removes the consumed commands from the queue and notifies the command lifecycle tracker. The ordering matters — only commands that were actually converted to attachments are removed, so a slash command in the queue (which is handled by a different path) stays queued.

### 7.3 Plan Mode Reminders (`plan_mode`)

Plan mode attachments implement a throttled reminder system that keeps the model in plan mode across long agentic sessions without consuming excessive context.

The throttle counts **human turns** (non-meta, non-tool-result user messages), not assistant messages — otherwise the reminder would fire every 5 tool calls instead of every 5 human turns. The turn counter scans backward through messages looking for the most recent `plan_mode` or `plan_mode_reentry` attachment.

Reminders alternate between full and sparse variants on a cycle:

```
Attachment #1  → full    (every FULL_REMINDER_EVERY_N_ATTACHMENTS = 5)
Attachment #2  → sparse
Attachment #3  → sparse
Attachment #4  → sparse
Attachment #5  → sparse
Attachment #6  → full    (cycle resets)
...
```

The cycle resets when the user exits and re-enters plan mode (`countPlanModeAttachmentsSinceLastExit` starts from zero after a `plan_mode_exit`). Re-entry also produces a special `plan_mode_reentry` attachment that instructs the model to read the existing plan file and decide whether to continue it or start fresh.

### 7.4 Delta-Based Announcements

Three attachment types use delta tracking to avoid redundant announcements: `skill_listing`, `agent_listing_delta`, and `deferred_tools_delta`. Each maintains a record of what has already been announced in the conversation, and only emits new additions or removals.

**Skill listings** use a per-agent `sentSkillNames` map (`Map<string, Set<string>>`). Each iteration, `getSkillListingAttachments()` (`attachments.ts:2661-2751`) gathers all available skills, checks which names are new (not in the sent set), and formats only the new ones via `formatCommandsWithinBudget()`. The sent set is keyed by `agentId` so sub-agents get their own fresh listings.

**Agent listings** reconstruct the announced set by scanning all prior `agent_listing_delta` attachments in the message history (`attachments.ts:1524-1529`). New agents are those in the current filtered pool but not in the announced set; removed agents are those in the announced set but no longer in the pool. This history-scanning approach means compaction naturally resets the delta — old attachments are gone from the compacted transcript, so the next call announces the full set.

**Deferred tools deltas** use the same history-scanning pattern. When an MCP server connects or disconnects mid-session, the delta picks up the change and announces it.

All three types produce empty arrays (no attachment) when nothing has changed — which is the common case on most iterations.

### 7.5 Nested Memory Discovery (`nested_memory`)

When the model touches a file (via `@`-mention, IDE selection, or file tool), the harness checks for CLAUDE.md instruction files in directories between cwd and the touched file. `getNestedMemoryAttachments()` (`attachments.ts:2167-2194`) consumes the `nestedMemoryAttachmentTriggers` set (populated by Phase 1 user input processing or by file tool operations), and for each trigger path runs a multi-phase directory walk:

1. **Managed/User conditional rules** — rules from `~/.claude/` and `/etc/claude-code/` that match the file path
2. **Nested directories** (cwd → target) — each intermediate directory's CLAUDE.md plus conditional rules
3. **CWD-level directories** (root → cwd) — conditional rules only (unconditional rules were already loaded at startup)

Each discovered file is deduplicated against `loadedNestedMemoryPaths` (a non-evicting `Set`) and `readFileState` (the LRU cache). Only files not already in context are included. When an `InstructionsLoaded` hook is configured, it fires for each newly loaded instruction file.

### 7.6 Relevant Memory Surfacing (`relevant_memories`)

The auto-memory system surfaces memory files relevant to the current conversation via an async prefetch mechanism. Unlike nested memories (which are path-triggered), relevant memories are content-triggered — a Sonnet classifier evaluates the user's input against memory file metadata and selects up to 5 files per turn.

The surfacing has three budget constraints:

| Constraint        | Value | Purpose                                                   |
| ----------------- | ----- | --------------------------------------------------------- |
| Per-file bytes    | 4,096 | Prevents a single memory from consuming excessive context |
| Per-turn files    | 5     | Caps a single injection at ~20KB (5 × 4KB)                |
| Per-session bytes | 60KB  | Stops prefetching entirely after ~3 full injections       |

The session-level cap scans messages (not a counter in `toolUseContext`) so that compaction naturally resets the budget — old memory attachments are gone from the compacted transcript, making re-surfacing valid again.

Each memory's header (age description + path) is computed once at attachment-creation time and stored in the attachment payload. This prevents a prompt cache bust that would occur if the age were recomputed on each turn — "saved 3 days ago" becoming "saved 4 days ago" changes the bytes, invalidating the cached prefix.

---

## 8. Design Decisions

**Attachments are a separate message type, not UserMessages from the start.** This is the foundational decision. Making attachments their own type enables three things: the normalization layer can selectively suppress signal-only subtypes, the REPL can render them differently from regular messages (badges and indicators vs. chat bubbles), and the JSONL persistence stores the structured payload cleanly for UI reconstruction on resume. If attachments were `UserMessage`s from creation, all three would require parsing free-form text to recover structure.

**Parallel production with error isolation.** The `maybe()` wrapper and `Promise.all()` structure mean that: (a) a slow MCP server doesn't block file change detection, (b) a crash in diagnostic tracking doesn't prevent skill listings from reaching the model, and (c) production latency data is sampled for monitoring. The 1-second timeout provides a hard upper bound.

**Phase ordering for dependency correctness.** Phase 1 (user input attachments) must complete before Phase 2 (inter-turn attachments) because `@`-mentioned file paths populate `nestedMemoryAttachmentTriggers`. This is the only cross-phase dependency; within each phase, all generators are independent and run concurrently.

**Delta tracking for context efficiency.** Skill listings, agent listings, and deferred tool announcements use delta tracking to avoid re-injecting the same information on every iteration. Once announced, a skill stays in the conversation history — visible to the model on every subsequent API call — so re-announcing wastes context tokens. The tracking is per-agent so sub-agents get independent announcements. Compaction naturally resets deltas by removing old attachment messages from the transcript.

**Throttled reminders with full/sparse cycling.** Plan mode and auto mode reminders count human turns, not assistant messages, to avoid over-firing during tool-heavy sessions. The full/sparse cycle ensures the model gets comprehensive instructions periodically while keeping most reminders lightweight. The cycle resets on mode exit/re-entry.

**History-scan over mutable state for agent listing deltas.** Agent listing deltas reconstruct the announced set by scanning `agent_listing_delta` attachments in the message history, rather than maintaining a `sentAgentTypes` map like skill listings do. This makes compaction implicitly reset the tracking (old attachments disappear from the compacted transcript), at the cost of an O(N) scan on each iteration. For the typical message count, the scan is negligible.

**Pre-computed headers for cache stability.** Relevant memory headers (containing age text like "saved 3 days ago") are computed once at attachment-creation time and frozen in the payload. If recomputed at render time, `Date.now()` would produce different age strings across turns — "3 days ago" becomes "4 days ago" — changing the message bytes and busting the prompt cache.

---

## 9. Telemetry

Attachment production is tracked through two main telemetry events:

**`tengu_attachment_compute_duration`** — Sampled at 5% rate, emitted per-generator by the `maybe()` wrapper:

| Field                   | Values                                                                     | Purpose                        |
| ----------------------- | -------------------------------------------------------------------------- | ------------------------------ |
| `label`                 | Generator name (e.g., `'changed_files'`, `'skill_listing'`, `'plan_mode'`) | Identifies which generator ran |
| `duration_ms`           | milliseconds                                                               | Tracks production latency      |
| `attachment_size_bytes` | bytes                                                                      | Tracks payload size            |
| `attachment_count`      | integer                                                                    | Number of attachments produced |
| `error`                 | boolean                                                                    | Whether the generator threw    |

**`tengu_attachments`** — Emitted once per `getAttachmentMessages()` call with the `attachment_types` array listing every attachment type produced on this iteration. This provides a per-iteration breakdown of what context the model is receiving.

Additional telemetry is emitted by specific generators: `tengu_at_mention_extracting_filename_success` / `error` for `@`-mention processing, `tengu_pdf_reference_attachment` for large PDF handling, `tengu_query_after_attachments` for the post-attachment state in `query.ts`, and `tengu_watched_file_compression_failed` when image change detection fails.

---

## Appendix A: The `<system-reminder>` Wrapper

Most text-based attachments are wrapped in `<system-reminder>` tags before reaching the model:

```
<system-reminder>
The date has changed. Today's date is now 2026-05-14.
DO NOT mention this to the user explicitly because they are already aware.
</system-reminder>
```

The `<system-reminder>` tag serves three purposes.

First, it **semantically separates injected context from user input**. The model sees the tag and understands that this information comes from the system, not from the user typing. This prevents the model from treating a plan mode reminder as a user request.

Second, it **enables stable caching**. These tags sit after the cached prefix in the message array. Because they're user messages (not system prompt content), they can change between turns without invalidating the system prompt cache. The system prompt says "look for `<system-reminder>` tags in the conversation" — that instruction is cached. The actual reminder content varies per turn.

Third, it **supports the smoosh optimization**. When `tengu_chair_sermon` is enabled, `ensureSystemReminderWrap()` guarantees every attachment-derived message has the wrapper, and `smooshSystemReminderSiblings()` can fold adjacent system-reminder text blocks into a single `tool_result` message — reducing the number of API messages and improving cache alignment.

---

## Appendix B: Attachment Subtype Reference

Complete mapping from attachment type to API conversion:

| Subtype                           | API Conversion                                                | Content Summary                                                 |
| --------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------- |
| `file` (text)                     | Synthetic `Read` tool_use + tool_result                       | Full file content, with truncation note if > max lines          |
| `file` (image/notebook/pdf)       | Synthetic `Read` tool_use + tool_result                       | Image data, notebook cells, or PDF pages                        |
| `directory`                       | Synthetic `ls` tool_use + tool_result                         | Directory listing (capped at 1,000 entries)                     |
| `compact_file_reference`          | `<system-reminder>` text                                      | Note that file was read before compaction; model should re-Read |
| `pdf_reference`                   | `<system-reminder>` text                                      | PDF metadata + instruction to use Read with pages parameter     |
| `edited_text_file`                | `<system-reminder>` text                                      | External modification notice with diff snippet                  |
| `selected_lines_in_ide`           | `<system-reminder>` text                                      | IDE selection (lines N-M of file, capped at 2,000 chars)        |
| `opened_file_in_ide`              | `<system-reminder>` text                                      | Notification that user opened a file in IDE                     |
| `nested_memory`                   | `<system-reminder>` text                                      | CLAUDE.md content from nested directory                         |
| `relevant_memories`               | Multiple `<system-reminder>` texts                            | One per memory file, each with age header                       |
| `plan_mode`                       | `<system-reminder>` instructions                              | Full or sparse plan mode reminder                               |
| `plan_mode_exit`                  | `<system-reminder>` text                                      | One-time notification of plan mode exit                         |
| `plan_mode_reentry`               | `<system-reminder>` text                                      | Re-entry guidance with existing plan reference                  |
| `auto_mode`                       | `<system-reminder>` instructions                              | Full or sparse auto mode reminder                               |
| `auto_mode_exit`                  | `<system-reminder>` text                                      | One-time notification of auto mode exit                         |
| `skill_listing`                   | `<system-reminder>` text                                      | Budget-formatted skill names and descriptions                   |
| `skill_discovery`                 | `<system-reminder>` text                                      | Relevant skills with invocation hint                            |
| `queued_command`                  | User message (may be visible)                                 | User's follow-up text, optionally with images                   |
| `task_status`                     | `<system-reminder>` text                                      | Task completion/running/stopped notification                    |
| `todo_reminder` / `task_reminder` | `<system-reminder>` text                                      | Gentle reminder to use task tools                               |
| `diagnostics`                     | `<system-reminder>` text                                      | New diagnostic issues (lint errors, type errors)                |
| `date_change`                     | `<system-reminder>` text                                      | Date rollover notification                                      |
| `deferred_tools_delta`            | `<system-reminder>` text                                      | Newly available or removed deferred tools                       |
| `agent_listing_delta`             | `<system-reminder>` text                                      | Newly available or removed agent types                          |
| `mcp_instructions_delta`          | `<system-reminder>` text                                      | MCP server instructions (added/removed)                         |
| `mcp_resource`                    | `<system-reminder>` text                                      | MCP resource content (text only; binary is summarized)          |
| `agent_mention`                   | `<system-reminder>` text                                      | Directive to invoke the mentioned agent                         |
| `plan_file_reference`             | `<system-reminder>` text                                      | Plan content for post-compaction context                        |
| `invoked_skills`                  | `<system-reminder>` text                                      | Previously invoked skill content for post-compaction            |
| `token_usage`                     | `<system-reminder>` text                                      | Token usage stats                                               |
| `budget_usd`                      | `<system-reminder>` text                                      | USD budget stats                                                |
| `output_token_usage`              | `<system-reminder>` text                                      | Output token stats (turn/session/budget)                        |
| `output_style`                    | `<system-reminder>` text                                      | Active output style reminder                                    |
| `ultrathink_effort`               | `<system-reminder>` text                                      | High reasoning effort directive                                 |
| `compaction_reminder`             | `<system-reminder>` text                                      | Auto-compact reassurance                                        |
| `context_efficiency`              | `<system-reminder>` text                                      | Snip-compact nudge                                              |
| `critical_system_reminder`        | `<system-reminder>` text                                      | SDK-provided critical reminder                                  |
| `verify_plan_reminder`            | `<system-reminder>` text                                      | Post-plan verification prompt                                   |
| `hook_success`                    | `<system-reminder>` text (SessionStart/UserPromptSubmit only) | Hook stdout                                                     |
| `hook_blocking_error`             | `<system-reminder>` text                                      | Hook blocking error message                                     |
| `hook_stopped_continuation`       | `<system-reminder>` text                                      | Hook stop message                                               |
| `hook_additional_context`         | `<system-reminder>` text                                      | Hook-provided context                                           |
| `async_hook_response`             | `<system-reminder>` text                                      | Async hook systemMessage + additionalContext                    |
| `teammate_mailbox`                | Formatted text                                                | Teammate messages                                               |
| `team_context`                    | `<system-reminder>` text                                      | Team coordination context                                       |
| `companion_intro`                 | `<system-reminder>` text                                      | Companion introduction                                          |
| `dynamic_skill`                   | *empty* (UI only)                                             | —                                                               |
| `already_read_file`               | *empty*                                                       | —                                                               |
| `command_permissions`             | *empty*                                                       | —                                                               |
| `edited_image_file`               | *empty*                                                       | —                                                               |
| `hook_cancelled`                  | *empty*                                                       | —                                                               |
| `hook_error_during_execution`     | *empty*                                                       | —                                                               |
| `hook_non_blocking_error`         | *empty*                                                       | —                                                               |
| `hook_system_message`             | *empty*                                                       | —                                                               |
| `hook_permission_decision`        | *empty*                                                       | —                                                               |
| `structured_output`               | *empty* (signal)                                              | —                                                               |
| `max_turns_reached`               | *empty* (signal)                                              | —                                                               |

---

## Appendix C: Legacy Attachment Types

The `normalizeAttachmentForAPI()` function includes a fallback for legacy attachment types that were removed but may still exist in old session transcripts loaded via `--resume`:

```typescript
// messages.ts:4268-4276
const LEGACY_ATTACHMENT_TYPES = [
  'autocheckpointing',
  'background_task_status',
  'todo',
  'task_progress',
  'ultramemory',
]
```

These return `[]` silently, preventing `--resume` from crashing on old transcripts. Unknown attachment types that aren't in this list trigger an `logAntError` diagnostic — this catches cases where a new attachment type was added to the `Attachment` union but its normalization case was forgotten.

---

## Appendix D: Key Source Files

| File                                                  | Lines  | Role                                                                                                                                                                                                                                                                                |
| ----------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/utils/attachments.ts`                            | ~3998  | Core attachment system: the `Attachment` union type (40+ subtypes), `getAttachments()` orchestrator, all generator functions, `getAttachmentMessages()` async generator, `createAttachmentMessage()` factory, `@`-mention parsing, file change detection, memory surfacing          |
| `src/utils/messages.ts`                               | ~5300+ | API conversion: `normalizeAttachmentForAPI()` (subtype → UserMessage switch), `reorderAttachmentsForAPI()` (bubbling), `normalizeMessagesForAPI()` (attachment case in main loop), `wrapMessagesInSystemReminder()`, `createToolUseMessage()` / `createToolResultMessage()` helpers |
| `src/types/message.ts`                                | ~416   | Type definitions: `AttachmentMessage<A>` interface, `Message` discriminated union                                                                                                                                                                                                   |
| `src/query.ts`                                        | ~1800+ | Integration point: Step 10 attachment collection (`getAttachmentMessages()` call), memory prefetch injection, skill discovery injection, queue draining                                                                                                                             |
| `src/services/compact/compact.ts`                     | ~700+  | Post-compaction attachment generation: `createPostCompactFileAttachments()`, plan/skill/delta re-announcements                                                                                                                                                                      |
| `src/components/messages/nullRenderingAttachments.ts` | ~72    | REPL rendering filter: `NULL_RENDERING_TYPES` list, `isNullRenderingAttachment()` check                                                                                                                                                                                             |
| `src/utils/processUserInput/processUserInput.ts`      | ~600+  | Turn-start attachment extraction: `getAttachmentMessages()` call for user input                                                                                                                                                                                                     |
