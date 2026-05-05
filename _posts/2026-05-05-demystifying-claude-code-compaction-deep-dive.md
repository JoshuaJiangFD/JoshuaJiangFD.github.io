---
title: "Demystifying Claude Code: Compaction Deep Dive"
date: 2026-05-05 12:00:00 -0700
categories: [Claude Code]
tags: [Claude Code, LLM, Context Management, Compaction]
mermaid: true
---

What happens when your Claude Code session crosses 167K tokens? A dedicated summarizer agent is forked, your entire conversation is condensed into a structured 9-section summary, and the model continues with a rebuilt context — carrying forward what matters, discarding what's recoverable via tools. This post traces the full lifecycle: the decision to compact, the summarization process, the context rebuild, and what gets lost along the way.

This is the companion deep-dive to [Demystifying Claude Code: Managing Message Context]({% post_url 2026-05-05-demystifying-claude-code-managing-message-context %}), expanding on Step 6 (AUTOCOMPACT) of the 11-step message pipeline.

---

## 1. Why Compaction Exists

Claude Code sessions accumulate context rapidly. A single tool call can inject 10-50K characters of file content. After a dozen tool uses, the conversation can hit 150K+ tokens — approaching the model's 200K context window. Without intervention, the session would hit a hard wall and stop working.

The naive alternatives each have problems:
- **Truncation** (drop oldest messages) loses semantic ordering and user intent. The model would forget *why* it's doing something.
- **RAG over the transcript** requires the model to know what to retrieve — a chicken-and-egg problem when the question is "what was I doing?"
- **Just stop** forces the user to start a new session, losing all accumulated understanding.

Compaction's approach: **summarize with structure, then rebuild working context**. A dedicated summarizer agent reads the full history and produces a structured summary preserving decisions, user messages, and current work state. Then the system re-injects the files and tools the model was actively using. The result is ~10-15K tokens that let the model continue working as if nothing happened — with graceful degradation for details it can recover via `git`, `Read`, or other tools.

---

## 2. The Compaction Modes

Claude Code has four distinct compaction paths, each serving a different trigger:

### 2.1 Proactive Compaction (autoCompact.ts → compact.ts)

The primary path. Fires **before** the API call when tokens exceed the 167K threshold. This is the "happy path" — the system notices context is growing large and preemptively summarizes. Section 3 details the full decision flow for this path.

### 2.2 Reactive Compaction (query.ts error recovery)

The fallback path. Fires **after** the API returns a 413 `prompt_too_long` error. This handles cases where:
- Proactive compaction was disabled or in reactive-only mode
- The token estimate was inaccurate (estimation vs. actual tokenization can differ)
- Context grew between the threshold check and the API call (rare race)

The reactive path uses the same `compactConversation()` internals but is gated by a single-shot guard: `hasAttemptedReactiveCompact`. After one reactive compact attempt per turn, if the retry still returns 413, the error surfaces to the user rather than retrying infinitely.

```
API returns 413 prompt_too_long
    |
    +-- Context collapse enabled?
    |   +-- Drain staged collapses (collapse_drain_retry)
    |   +-- If still 413 → fall through
    |
    +-- hasAttemptedReactiveCompact == false?
    |   +-- Run compactConversation()
    |   +-- Set hasAttemptedReactiveCompact = true
    |   +-- Retry API call with compacted messages (reactive_compact_retry)
    |
    +-- hasAttemptedReactiveCompact == true?
        +-- Surface error to user (no more retries)
```

The `hasAttemptedReactiveCompact` flag resets to `false` at the start of each new turn (after tool execution), so a fresh turn gets a fresh chance at reactive recovery.

### 2.3 Partial Compaction (compact.ts, REPL UI action)

User-initiated via the REPL UI — the user selects a specific message and chooses to summarize around it. Unlike full compaction which summarizes the entire history, partial compaction operates around a **pivot point** (the selected message's index):

- **`direction: 'up_to'`** — Summarize messages *before* the pivot, keep messages *after*. "Summarize everything before here."
- **`direction: 'from'`** — Summarize messages *after* the pivot, keep messages *before*. Less common; used when recent context is noise but earlier context is important.

This gives users precise control over what gets preserved at full fidelity versus what gets summarized. The function `partialCompactConversation()` in `compact.ts` handles this — it uses the same summarizer agent as full compaction but with a scoped prompt (`getPartialCompactPrompt`) and preserves the non-summarized side as `messagesToKeep` in the result.

Note: the **`/compact` slash command** does *not* invoke partial compaction. Its flow is: try session memory compaction first (if no custom instructions) → if the system is in reactive-only mode, route through `compactViaReactive` → otherwise run full `compactConversation()`.

---

## 3. The Proactive Decision Flow (autoCompact.ts)

`autoCompact.ts` is the gatekeeper for the primary compaction path (§2.1). Before every API call, it evaluates whether the conversation has grown too large.

### 3.1 The Threshold Math

```
contextWindow (e.g., 200K for Sonnet)
  - MAX_OUTPUT_TOKENS_FOR_SUMMARY (20K reserved for the summarizer to write)
  = effectiveContextWindow (180K)
    - AUTOCOMPACT_BUFFER_TOKENS (13K safety margin)
    = autoCompactThreshold (167K)

  Warning threshold: 147K (threshold - 20K)
  Blocking threshold: 177K (effectiveContextWindow - 3K)
```

The 167K trigger point (~93% of the effective window) balances two concerns: fire too early and you destroy context unnecessarily; fire too late and the summarizer itself may not have enough headroom to produce a quality summary.

### 3.2 Guards Before the Threshold Check

Even if the token count exceeds 167K, several guards can prevent compaction:

```
autoCompactIfNeeded(messages, toolUseContext, ...)
    |
    +-- DISABLE_COMPACT env? → skip
    |
    +-- Circuit breaker: consecutiveFailures >= 3? → skip
    |
    +-- shouldAutoCompact(messages, model, querySource, snipTokensFreed)
          |
          +-- querySource == 'compact' | 'session_memory' | 'marble_origami'?
          |   → false (recursion guard — see §3.3)
          |
          +-- isAutoCompactEnabled()?
          |   +-- DISABLE_COMPACT env → false
          |   +-- DISABLE_AUTO_COMPACT env → false
          |   +-- userConfig.autoCompactEnabled
          |
          +-- Reactive-only mode? → false
          |   (proactive disabled; let API 413 trigger reactive path)
          |
          +-- Context-collapse enabled? → false
          |   (collapse owns headroom management; autocompact would race it)
          |
          +-- Token check:
              tokenCount = tokenCountWithEstimation(messages) - snipTokensFreed
              → tokenCount >= threshold?
```

### 3.3 Recursion Guard

Compaction works by forking a temporary summarizer agent and injecting it with the full message history. This agent starts life with a context that's already over the threshold by definition. Without a guard, it would immediately trigger compaction on *itself*, forking another summarizer in an infinite recursive loop. The `querySource` check prevents this: if the current turn belongs to a `'compact'`, `'session_memory'`, or `'marble_origami'` source, `shouldAutoCompact` returns false immediately.

### 3.4 Circuit Breaker

If compaction fails three consecutive times (`MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3`), the system concludes the context is in a terminal state — likely a single message or tool result that's too large to summarize. The circuit breaker trips and all future compaction attempts are skipped for the session, preventing repeated API calls that are doomed to fail.

### 3.5 Session Memory: The Lighter First Attempt

Once the threshold is crossed and all guards pass, `autoCompactIfNeeded` doesn't immediately invoke full compaction. It first tries `trySessionMemoryCompaction()` — an experimental lighter-weight path that uses session-level memory to prune the history without a full summarization pass.

```
autoCompactIfNeeded (threshold crossed, guards passed)
    |
    +-- trySessionMemoryCompaction(messages, agentId, threshold)
    |   |
    |   +-- Success (under threshold) → return early, no full compact needed
    |   |
    |   +-- Fail/unavailable → fall through
    |
    +-- compactConversation(messages, ...) → full summarization
```

This preserves more granular context at lower cost — no summarizer agent is forked, no LLM call is needed. The tradeoff is that it can only free a limited amount of space (maximum 40K tokens per invocation), so sessions that are deeply over threshold will still need full compaction.

Configuration constraints: minimum 10K tokens, minimum 5 text-block messages, maximum 40K tokens per session memory compact. The system also ensures it never compacts below the last compact boundary (preserving chain continuity) and protects tool_use/tool_result pairs from being split.

---

## 4. Inside the Summarizer Agent

When full compaction fires (either proactively or reactively), `compact.ts` takes over.

### 4.1 Forking the Summarizer

The summarizer runs as a **forked agent** — a separate LLM call with its own system prompt and constraints. It receives the full message history to read and condense.

```
compactConversation(messages, context, cacheSafeParams, ...)
    |
    +-- preCompactTokenCount = tokenCountWithEstimation(messages)
    |
    +-- Execute PreCompact hooks (may add customInstructions)
    |
    +-- Build compact prompt (summaryRequest)
    |
    +-- SUMMARIZE via streamCompactSummary:
    |   |
    |   +-- PRIMARY: runForkedAgent (cache-sharing path)
    |   |   Shares main conversation's prompt cache prefix
    |   |   → avoids re-uploading the same tokens
    |   |
    |   +-- FALLBACK: queryModelWithStreaming (direct path)
    |       Sends full message history + summary request
    |       System: "You are a helpful AI assistant tasked with summarizing"
    |       Tools: [FileReadTool] (available in schema but usage forbidden)
    |
    +-- PTL retry loop (up to 3 retries)
    |
    +-- REBUILD post-compact context
    |
    +-- Return CompactionResult
```

### 4.2 Cache Sharing: The Cost Optimization

The summarizer needs to read the full conversation history — the same tokens the main agent has already sent and cached with the API. The forked agent exploits this by structuring its request so the prefix is **byte-identical** to what the main agent already cached:

```
Main agent's cached request:
  [system prompt] + [tools] + [conversation messages M1..Mn] + [user's latest prompt]

Summarizer's request (forked agent):
  [system prompt] + [tools] + [conversation messages M1..Mn] + [compact prompt]
                                                                 ↑ only new part
```

The fork concatenates `forkContextMessages` (the full conversation) with `promptMessages` (just the compact prompt as a single user message). Because the system prompt, tools, model, and message prefix are identical to the main agent's last request, the API recognizes the prefix as already-cached. The result: those ~170K tokens of conversation are **cache reads** (cheap) rather than fresh input tokens (expensive). The summarizer only pays full input cost for the compact prompt itself (~1K tokens), plus the output tokens for generating the summary.

This is why the code comment warns against setting `maxOutputTokens` on the fork — it would alter the thinking config (part of the cache key), creating a mismatch that invalidates the cache and forces a full re-upload of the entire conversation.

The fallback path (when cache sharing fails or is disabled) uses a different system prompt (`"You are a helpful AI assistant tasked with summarizing conversations."`) and sends the messages directly via `queryModelWithStreaming`. This path **cannot** share cache with the main agent because the system prompt differs, so it pays full input cost for all tokens.

Cache sharing is gated by the `tengu_compact_cache_prefix` feature flag (default enabled).

### 4.3 The Compact Prompt: Two-Pass Summarization

The summarizer follows a two-pass process designed to produce high-quality summaries.

**Pass 1: Analysis (chain-of-thought, discarded)**

```xml
<analysis>
Chronologically analyze each message and section of the conversation.
For each section thoroughly identify:
- The user's explicit requests and intents
- Your approach to addressing the user's requests
- Key decisions, technical concepts and code patterns
- Specific details (file names, code snippets, function signatures, file edits)
- Errors and how they were fixed
- Specific user feedback
</analysis>
```

This is the summarizer "thinking before summarizing." The chronological walkthrough forces it to attend to every part of the conversation rather than jumping to conclusions. But this analysis block is **stripped** before the summary enters context — it served its purpose as a reasoning scaffold.

```typescript
formattedSummary = formattedSummary.replace(/<analysis>[\s\S]*?<\/analysis>/, '')
```

**Pass 2: The 9-Section Structured Summary**

The output has 9 required sections, each preserving a specific type of information:

| # | Section | Why It Exists |
|---|---------|---------------|
| 1 | Primary Request and Intent | The model needs to know *what* it's working toward |
| 2 | Key Technical Concepts | Technologies, frameworks, patterns in use |
| 3 | Files and Code Sections | Specific files examined/modified/created, with snippets |
| 4 | Errors and Fixes | What went wrong and how it was resolved |
| 5 | Problem Solving | Ongoing troubleshooting context |
| 6 | All User Messages | Every non-tool-result user message (see below) |
| 7 | Pending Tasks | Work explicitly requested but not yet done |
| 8 | Current Work | Precise state of work-in-progress at summary time |
| 9 | Optional Next Step | Recommended continuation with direct quotes |

**Section 6 deserves special attention.** In the Anthropic API, tool results are sent as `role: 'user'` messages. Without an explicit instruction to list *actual* user messages separately, the summarizer might conflate user intent ("fix the login bug") with tool output (the contents of a file read). Section 6 ensures the user's voice is preserved distinctly from system responses.

### 4.4 The NO_TOOLS_PREAMBLE

Despite having `FileReadTool` in its schema, the summarizer receives an aggressive preamble forbidding tool use:

> "You are tasked with summarizing a conversation. Do NOT use any tools..."

Why have the tool in schema at all? The schema presence enables structured formatting in the output, but the prompt-level prohibition prevents the summarizer from actually reading files (which would add tokens to an already-full context and defeat the purpose of compaction).

### 4.5 PTL Retry Loop

If the summarizer's own request is too large for the model (the conversation being summarized is enormous), the system enters a retry loop:

1. Detect that the summary response starts with a `PROMPT_TOO_LONG` indicator
2. Call `truncateHeadForPTLRetry()` — groups messages by API round, drops the oldest rounds
3. Retry with the truncated history (up to `MAX_PTL_RETRIES = 3` times)

The truncation strategy is conservative: it drops the minimum number of complete API rounds needed to fit, and never drops the most recent round. On each retry, it calculates either the exact token gap or falls back to dropping 20% of rounds.

---

## 5. Post-Compact Rebuild: What Context Is Restored

After the summarizer produces its output, the system doesn't just leave the model with a bare summary. It rebuilds working context — the files, tools, and state the model needs to continue seamlessly.

### 5.1 File Attachments

The system re-reads up to **5 most recently accessed files**, with a budget of:
- 50K tokens total across all files
- 5K tokens maximum per file
- Skips: plan files, claude.md files, files already in preserved messages

Files are sorted by recency (most recently read = most likely still relevant). This ensures the model can reference code it was actively working with, even though the original `Read` tool results were summarized away.

### 5.2 Full Attachment List

```
Post-compact attachments:
    |
    +-- File attachments (up to 5 files, 50K total, 5K/file)
    +-- Async agent status (if background agents running)
    +-- Plan file (if a plan exists)
    +-- Plan mode reminder (if user is in plan mode)
    +-- Skill content (invoked skills, 5K/skill, 25K total)
    +-- Deferred tools delta (re-announce tool schemas)
    +-- Agent listing delta (available sub-agents)
    +-- MCP instructions delta (MCP server instructions)
    +-- SessionStart hook messages (hook-injected context)
```

### 5.3 The Final Message Array (buildPostCompactMessages)

```typescript
function buildPostCompactMessages(result: CompactionResult): Message[] {
  return [
    result.boundaryMarker,       // compact_boundary system message
    ...result.summaryMessages,   // the LLM-generated summary
    ...(result.messagesToKeep ?? []),  // preserved messages (partial compact)
    ...result.attachments,       // rebuilt working context
    ...result.hookResults,       // SessionStart hook output
  ]
}
```

| Index | Message | Purpose |
|-------|---------|---------|
| `[0]` | `boundaryMarker` — `type: 'system', subtype: 'compact_boundary'` | Marks where compaction happened; Step 1 of the pipeline slices from here on future iterations |
| `[1]` | `summaryMessages[0]` — `type: 'user', isCompactSummary: true` | The structured summary + transcript file path |
| `[2..n]` | `messagesToKeep` (partial compact only) | Original messages preserved from partial compact |
| `[n+1..m]` | `attachments` — file contents, plan, skills, tools, agents, MCP | Rebuilt working context |
| `[m+1..k]` | `hookResults` — SessionStart hook output | Hook-injected context |

This entire array replaces `messagesForQuery` — the old messages are gone from the model's view.

---

## 6. Post-Compact Cleanup: What Caches Are Invalidated

Compaction doesn't just replace messages — it invalidates several subsystems' assumptions about what the model has "seen." `postCompactCleanup.ts` resets:

| Cache/State | Why It's Cleared |
|-------------|-----------------|
| Microcompact state | Tool result tracking is meaningless after summary replaces them |
| Context-collapse state | Collapsed spans no longer exist in the new message array |
| getUserContext cache | CLAUDE.md files need re-evaluation (InstructionsLoaded hook must re-fire) |
| Memory file cache | Memory files may have changed since last load |
| System prompt sections | Sections derived from old context are stale |
| Classifier approvals | Permission decisions for old tool calls don't apply to new context |
| Speculative checks | Bash permission pre-checks based on old context |
| Session messages cache | Cached message computations are invalid |

**Intentionally NOT cleared:** `sentSkillNames`. Re-injecting the full skill listing (~4K tokens) post-compact is pure cache_creation overhead. The model still has `SkillTool` in its schema, `invoked_skills` attachment preserves used skill content, and dynamic additions are handled by a separate skill change detector.

This distinction — knowing what to reset vs. what to preserve — is important for understanding that compaction is a system-wide state transition, not just a message replacement.

---

## 7. Concrete End-to-End Example

### Starting State

A session has been running for 25 turns. The message array after Steps 1-5:

```
[0]  boundary (from earlier compact, or start of session)
[1]  summary: "User asked to refactor auth middleware..."
[2]  attachment: file_content of auth.ts (4.8K tokens)
[3]  user: "now fix the login bug in handleSession()"
[4]  assistant: "I'll read the session handler..." + tool_use(Read)
[5]  user: tool_result for Read (12K tokens — full file content)
[6]  assistant: "Found the bug — the token refresh is..." + tool_use(Edit)
[7]  user: tool_result for Edit (200 tokens — success confirmation)
[8]  assistant: "Fixed. Let me also update the tests..." + tool_use(Read)
[9]  user: tool_result for Read (8K tokens — test file)
[10] assistant: "I'll add a test case..." + tool_use(Edit)
[11] user: tool_result for Edit (200 tokens)
[12] attachment: edited_text_file diff notification
[13] user: "actually, also check if there's a race condition"
[14] assistant: "Good call. Let me trace the concurrent..." + tool_use(Bash)
[15] user: tool_result for Bash (25K tokens — grep output)
     ... (continues for many more turns)

Total estimated tokens: ~170K
```

### Step 1: Threshold Check

```
tokenCount = 170,000
threshold = 200,000 - 20,000 - 13,000 = 167,000
170,000 >= 167,000 → COMPACT
```

### Step 2: Session Memory Attempt

`trySessionMemoryCompaction()` is called first. In this example, suppose it fails (not enough session memory entries to sufficiently reduce context). Falls through to full compaction.

### Step 3: Summarizer Receives the History

The forked summarizer agent gets all messages from `[0]` to the end, plus the compact prompt asking for the 9-section summary. It shares the main agent's prompt cache, so only the compact prompt itself is "new" tokens.

### Step 4: Summarizer Produces Output

```xml
<analysis>
The conversation began with the user asking to refactor auth middleware
(from a previous compaction summary). The user then asked to fix a login
bug in handleSession(). I found a token refresh issue and fixed it.
The user then asked about race conditions. I investigated using grep
and found potential concurrent access to the session store...
</analysis>

<summary>
1. Primary Request: Fix login bug in handleSession() and investigate
   race conditions in the session middleware.

2. Key Technical Concepts: Token refresh flow, session store concurrency,
   Express middleware chain...

3. Files and Code: auth.ts (refactored), sessionHandler.ts (login bug
   fixed at line 42 — token refresh was using stale expiry),
   sessionHandler.test.ts (new test added)...

4. Errors and Fixes: Login bug — stale token expiry timestamp caused
   premature session invalidation. Fixed by reading expiry from refresh
   response instead of original token.

5. Problem Solving: Currently investigating potential race condition
   in concurrent session writes...

6. All user messages:
   - "now fix the login bug in handleSession()"
   - "actually, also check if there's a race condition"

7. Pending Tasks: Race condition investigation still in progress.

8. Current Work: Analyzing grep output for concurrent session store
   access patterns. Found 3 potential call sites that don't hold locks.

9. Next Step: "check if there's a race condition" — examine the 3
   identified call sites for proper mutex usage.
</summary>
```

### Step 5: Analysis Block Stripped, Context Rebuilt

The `<analysis>` block is removed. The system then:
1. Re-reads `sessionHandler.ts` (most recently edited, ≤5K tokens)
2. Re-reads `auth.ts` (recently accessed, ≤5K tokens)
3. Attaches tool schemas, agent listings, MCP instructions

### Step 6: Final Array

```
[0] NEW boundary (compact_boundary, preTokens: 170K)
[1] NEW summary (the 9-section output above, ~3K tokens)
[2] attachment: file_content of sessionHandler.ts (4.2K tokens)
[3] attachment: file_content of auth.ts (4.8K tokens)
[4] attachment: deferred_tools_delta (tool schemas, ~2K tokens)
[5] attachment: agent_listing_delta (~500 tokens)
[6] attachment: mcp_instructions_delta (~500 tokens)
[7] hook_result: SessionStart hook output (~200 tokens)

Total: ~15K tokens (down from 170K — a 91% reduction)
```

The model's next turn sees this 15K-token context and continues investigating the race condition, with full access to the current file contents and a clear summary of what happened before.

---

## 8. What Gets Lost (The Honest Accounting)

Compaction is lossy by design. Here's what survives, what degrades, and what's gone:

| Information Type | Post-Compact State | Recovery Path |
|-----------------|-------------------|---------------|
| User's intent and requests | Preserved (Section 6 lists all user messages) | — |
| What files were edited and why | Preserved in summary + git history | `git log`, `git diff` |
| Current file contents | Re-read and attached (up to 5 files) | `Read` tool |
| Exact code diffs applied | Lost from context | `git diff HEAD~N` |
| Intermediate tool outputs | Lost (grep results, bash output) | Re-run the command |
| Error messages | Partially preserved (summary notes "there was an error") | Check logs, re-run |
| Multi-step reasoning chains | Reduced to narrative | Not recoverable |
| Contradictory user instructions | Risk of flattening ("User wants Y" vs. "User first wanted X, then Y") | — |

### Order Preservation

The summary prompt instructs chronological analysis, and Section 6 lists user messages in order. But there is **no structural enforcement** — the summary is free-form text. Very long conversations (50+ user messages) face token budget pressure that forces the summarizer to compress.

### The Key Insight

There is **no automatic mechanism** that detects "the user is referencing something from before compaction" and fetches more detail. The model must actively use tools (`git diff`, `Read`, `Bash`) to reconstruct what it needs. The summary gives it enough context to know *what* to look for, but not necessarily the exact content.

---

## 9. Design Tradeoffs

### Why not just truncate old messages?

Truncation is simple but catastrophic for continuity. If the model loses the messages where the user explained their goal, it has no way to recover that intent — unlike file contents, which can be re-read. The summarizer preserves *why* the work is happening, which is the one thing that can't be reconstructed from the filesystem.

### Why not use RAG over the transcript?

The transcript is persisted as JSONL, so retrieval is technically possible. But effective retrieval requires knowing *what* to search for — and that's precisely what the model doesn't know when it has no context. The structured summary solves this by preemptively identifying what's important, rather than leaving the model to figure out what's missing.

### Why summarize everything rather than just the oldest N messages?

Full-context summarization produces better summaries because the summarizer can identify themes and connections across the entire conversation. Incremental summarization (only summarize the oldest chunk) risks losing cross-cutting concerns — a decision made early that constrains later work, for example.

### Why re-read files instead of preserving the original tool results?

Files change. The original `Read` result might be stale if the model (or an external process) edited the file since. A fresh re-read ensures the model sees current state. The 5K-per-file budget means these re-reads are truncated for very large files, but the model can always issue a full `Read` if needed.

### Why not preserve more than 5 files?

The 50K total / 5 file limit is a balance between context restoration and leaving room for the model to work. Post-compact context of 15K tokens leaves ~150K tokens of headroom. If file attachments consumed 100K, the model would be back near the threshold within a few turns. The model can always `Read` additional files it needs — the summary tells it which files are relevant.
