---
title: "Demystifying Claude Code: Skills"
date: 2026-05-13 15:00:00 -0700
categories: [Claude Code]
tags: [Claude Code, LLM, Skills, Slash Commands, Plugins, Extensibility]
mermaid: true
---

This post covers how Claude Code discovers, loads, lists, and executes skills — the prompt-based extensibility mechanism that lets users, plugins, and MCP servers teach the model new behaviors. We trace the full lifecycle from SKILL.md files on disk to the `<system-reminder>` that announces them to the model, through the SkillTool's validation, permission, and execution pipeline, to the context modification that constrains the model's subsequent behavior.

For how the SkillTool's prompt integrates into the system prompt, see [The System Prompt]({% post_url 2026-05-07-demystifying-claude-code-system-prompt %}), Section 6. For the tool execution pipeline that wraps the SkillTool (validation, hooks, permissions), see [Tool Execution]({% post_url 2026-05-12-demystifying-claude-code-tool-execution %}). For how forked skills spawn sub-agents, see [Spawned Agents]({% post_url 2026-05-07-demystifying-claude-code-spawned-agents %}).

---

## 1. What the User Sees

Skills surface in two observable ways.

**User-invoked.** The user types `/review` in the prompt. The harness looks up the skill by name, expands its prompt content (substituting arguments, executing inline shell commands), and injects the result as user messages into the conversation. The model sees these messages and acts on them — writing code, running tools, producing a review. From the user's perspective, `/review` is a shortcut that loads a specialized instruction set.

**Model-invoked.** The user asks "review this PR" without typing a slash command. The model sees the `review` skill listed in a `<system-reminder>` and calls the SkillTool proactively:

```
Skill(skill="review", args="123")
```

The model is instructed that when a skill matches the user's request, invoking it is a "BLOCKING REQUIREMENT" — it must call the SkillTool before generating any other response about the task.

Both paths converge on the same execution pipeline inside `SkillTool.ts`. The difference is only who initiates the call.

---

## 2. Skill Sources

Skills come from six sources, loaded in a specific priority order:

| Source      | Path / Mechanism                                                                                                | `loadedFrom` | Example                         |
| ----------- | --------------------------------------------------------------------------------------------------------------- | ------------ | ------------------------------- |
| **Managed** | SKILL.md files at `/etc/claude-code/skills/`, deployed by enterprise admins to enforce organization-wide skills | `'managed'`  | Enterprise-enforced skills      |
| **User**    | SKILL.md files at `~/.claude/skills/`, created by the user                                                      | `'skills'`   | Personal skills across projects |
| **Project** | SKILL.md files at `.claude/skills/` (walking up from cwd), checked into the repo                                | `'skills'`   | Repo-specific skills            |
| **Bundled** | Compiled into the CLI binary by Anthropic via `registerBundledSkill()` — no SKILL.md on disk                    | `'bundled'`  | `init`, `review`, `simplify`    |
| **Plugin**  | Installed plugins                                                                                               | `'plugin'`   | `ralph-loop:ralph-loop`         |
| **MCP**     | Model Context Protocol servers                                                                                  | `'mcp'`      | Prompt-type MCP commands        |

Additionally, an experimental seventh source exists behind the `EXPERIMENTAL_SKILL_SEARCH` feature gate: **remote skills** fetched from a GCS/AKI backend, identified by a `_canonical_<slug>` prefix. These are Anthropic-internal only.

### Two Loading Strategies

"Loading" a skill means parsing its SKILL.md file from disk (or extracting it from a bundled definition) into a `Command` object in the harness's Node.js process memory, and placing that object where `getCommands()` can find it. A loaded skill is findable by name and executable by the SkillTool — but it is not yet visible to the model. Visibility requires a separate step: the skill must pass through `getSkillToolCommands()`, get formatted by `formatCommandsWithinBudget()`, and be injected as a `<system-reminder>` attachment into the model's messages (covered in [Section 4](#4-how-skills-reach-the-model)). Loading is a prerequisite for visibility, not a synonym for it.

Not all skills are loaded at the same time. The system uses two strategies — eager startup loading and lazy mid-session activation — that merge at query time.

**Eager (startup).** `getSkills()` (`commands.ts:353`) loads from all sources in parallel:

```typescript
// commands.ts:353-398
async function getSkills(cwd: string): Promise<{
  skillDirCommands: Command[]
  pluginSkills: Command[]
  bundledSkills: Command[]
  builtinPluginSkills: Command[]
}>
```

These are combined by `loadAllCommands()` (`commands.ts:449`) in a fixed order: bundled, builtin-plugin, skill-dir, workflow, plugin-commands, plugin-skills, and finally the built-in COMMANDS. Both `loadAllCommands()` and the underlying `getSkillDirCommands()` are **memoized by cwd** (current working directory — the directory Claude Code was launched from). This means each function caches its return value after the first call, and subsequent calls with the same cwd return the cached result without re-executing the disk scans. The cache key is cwd because different directories have different `.claude/skills/` hierarchies. In practice, cwd doesn't change within a session, so this makes startup loading a one-time cost (unless a cache clear is triggered).

**Lazy (mid-session).** The startup scan misses two things. First, it walks *up* from cwd, so `.claude/skills/` directories nested *below* cwd in subdirectories are never found. Second, skills with `paths:` frontmatter (conditional skills) are parsed at startup but withheld — they sit dormant until relevant. Two mechanisms fill these gaps during the session: directory discovery (walking *down* from touched files to find nested skill directories) and conditional activation (promoting dormant skills when file operations match their `paths:` patterns). Both are described in [Section 7](#7-dynamic-mid-session-discovery).

**Merge point.** `getCommands()` (`commands.ts:476`) is **not** cached — it runs fresh on every call. It fetches `loadAllCommands()` (which returns the cached startup set), merges `getDynamicSkills()` (the current lazy set), deduplicates by name, and returns the combined result. This means any newly discovered or activated skill becomes visible to the model on the very next tool call.

---

## 3. The SKILL.md Format

Section 2 covered where skills come from — six sources across managed, user, project, bundled, plugin, and MCP directories. For the disk-based sources (managed, user, project), each skill is a directory containing a `SKILL.md` file. This section covers what that file looks like: the YAML frontmatter that controls a skill's behavior, and the markdown body that becomes the prompt the model sees.

### A Minimal Example

A skill on disk lives in a named directory under `.claude/skills/`:

```
.claude/skills/
└── review/
    └── SKILL.md
```

A typical SKILL.md has two parts — YAML frontmatter (between the `---` delimiters) that controls the skill's behavior, and a markdown body that becomes the prompt content the model receives when the skill is invoked:

```markdown
---                                              ┐
description: Review a pull request for issues     │
allowed-tools: Bash, Read, Grep                   │ frontmatter
argument-hint: <pr-number>                        │ (YAML metadata)
context: fork                                     │
---                                              ┘
                                                  ┐
Review the pull request #$1. Focus on:            │
- Security vulnerabilities                        │ markdown body
- Performance issues                              │ (becomes the prompt after
- Code style violations                           │  content expansion, see 5.3)
                                                  ┘
```

Most skills only need a handful of frontmatter fields — the example above uses four. The full set of supported fields is listed in [Appendix D](#appendix-d-frontmatter-fields-reference), grouped by purpose: identity and listing, execution behavior, and conditional activation.

---

## 4. How Skills Reach the Model

The model learns about available skills through three layers, split across two API channels with different caching properties (for background on the prompt caching mechanism, see [Prompt Caching]({% post_url 2026-05-05-demystifying-claude-code-prompt-caching %}) and [The System Prompt]({% post_url 2026-05-07-demystifying-claude-code-system-prompt %}), Section 7):

1. **System prompt** — says skills exist and references system-reminders. This text is identical for all users and lives in the globally-cached portion of the system prompt (before the cache boundary). It never changes between turns.
2. **SkillTool description** — says how to call skills; points the model to system-reminders for the actual list. This lives in the tools array, which is also cached across turns within a session.
3. **`<system-reminder>` messages** — provide skill names and one-line descriptions (not the full skill content). These are injected as user messages via the attachment system, which means they sit *after* the cached prefix and can change every turn without invalidating the cache. The full skill content — the markdown body with argument substitution, shell commands, and tool constraints — is only loaded when the model actually invokes the skill via the SkillTool (see [Section 5.3](#53-execution)).

The reason layer 3 is a separate channel: available skills change mid-session (MCP servers connect, plugins reload, conditional skills activate). If the skill list lived in the system prompt or tool description, every change would bust the API prompt cache — forcing the server to reprocess the entire prefix. By putting the volatile list in `<system-reminder>` messages and keeping only the stable "how to use skills" instructions in layers 1 and 2, the system avoids cache invalidation when the skill set changes.

### Skill Listing Generation (Layer 3 Internals)

The `<system-reminder>` skill listing described in layer 3 above is evaluated on **every iteration of the agentic loop** — not just once per user turn. Each iteration (each API call) runs `getAttachments()`, which calls `getSkillListingAttachments()` (`attachments.ts:2661`). However, it only produces an actual `<system-reminder>` when new skills exist, via the delta tracking described in step 4 below. New skills can appear between iterations via the dynamic discovery mechanisms described in [Section 7](#7-dynamic-mid-session-discovery) — if a file tool activates a skill on iteration N, the listing on iteration N+1 detects the new entry and announces it to the model.

```typescript
// attachments.ts:2661-2751
async function getSkillListingAttachments(
  toolUseContext: ToolUseContext,
): Promise<Attachment[]>
```

The function:

1. **Guard**: returns `[]` if the SkillTool is not in the agent's tool set (line 2669-2672)
2. **Gathers**: calls `getSkillToolCommands(cwd)` for local skills and `getMcpSkillCommands(...)` for MCP skills, deduplicates by name
3. **Filters** (when skill-search is enabled): calls `filterToBundledAndMcp()` — user/project/plugin skills go through the discovery pathway instead
4. **Computes delta**: uses a per-agent `sentSkillNames` map (`Map<string, Set<string>>`) to track which skills have already been announced. Only new skills are included — previously announced skills don't need re-announcing because their `<system-reminder>` messages are still in the conversation history, visible to the model on every subsequent API call
5. **Suppresses on resume**: when `--resume` replays a session, `suppressNextSkillListing()` prevents re-injecting listings already in the transcript
6. **Formats**: calls `formatCommandsWithinBudget(newSkills, contextWindowTokens)`
7. **Returns**: `[{ type: 'skill_listing', content, skillCount, isInitial }]`

#### The Budget System

The `<system-reminder>` listing from step 7 above contains skill names and descriptions — and with hundreds of MCP tools and dozens of plugins, it could grow large enough to crowd out actual conversation content. The budget system caps the total listing at **1% of the context window** (about 8,000 characters for a 200k-token context) and degrades gracefully when the listing doesn't fit:

1. If all skill descriptions fit within the budget, they're included in full.
2. If they don't fit, bundled skills (Anthropic-curated, like `init`, `review`, `simplify`) keep their full descriptions, and non-bundled skill descriptions are truncated proportionally to fit the remaining space.
3. In the extreme case where even truncated descriptions don't fit, non-bundled skills are reduced to names only, while bundled skills still keep their descriptions.

This ensures core functionality is always visible regardless of how many third-party skills are installed.

#### Conversion to API Message

The formatted listing is wrapped in a `<system-reminder>` tag and injected as a user message hidden from the user's terminal display (`isMeta: true`). The model sees:

```
<system-reminder>
The following skills are available for use with the Skill tool:

- review: Review a pull request for issues. When the user asks to review a PR or code changes
- simplify: Review changed code for reuse, quality, and efficiency
- init: Initialize a new CLAUDE.md file with codebase documentation
</system-reminder>
```

---

## 5. The SkillTool

Section 4 covered how the model learns about available skills via `<system-reminder>` listings. But those listings only contain names and one-line descriptions — the model knows a `review` skill exists, but it doesn't have the skill's full content yet. The SkillTool is the bridge: it takes a skill name (and optional arguments), validates that the skill exists and is allowed, and delivers the full expanded content to the model.

The SkillTool sits in the model's tool set alongside Bash, Read, Edit, and the other tools. It's called in two scenarios: when the user types a slash command like `/review 123` (the harness translates this into a SkillTool call), or when the model proactively decides a skill matches the user's request and emits a `Skill` tool call itself. Both paths go through the same pipeline, which this section traces: validation (Section 5.1), permissions (Section 5.2), and execution (Section 5.3). Context modification after execution is covered in [Section 6](#6-context-modification).

### 5.1 Input Validation (lines 354-430)

`validateInput()` determines whether the model's skill call is valid before any permission checks or execution. It normalizes the input, looks up the command, and rejects invalid calls with actionable error messages.

**Step 1: Normalize the input.** Two transformations run before any checks:

- **Leading slash strip** — if the model emits `skill: "/review"`, the leading `/` is stripped (line 366-372). A telemetry event `tengu_skill_tool_slash_prefix` fires when this happens, tracking how often the model adds an unnecessary prefix.
- **Remote canonical intercept** — when `EXPERIMENTAL_SKILL_SEARCH` is enabled and the skill name starts with `_canonical_`, it's routed to the remote skill pathway instead of the normal checks below.

**Step 2: Run five checks in sequence.** Each check can reject the call with a specific error code:

| #   | Check                                                                                                                                                                                                         | Error code            |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| 1   | Skill name is empty or invalid after normalization                                                                                                                                                            | Empty input           |
| 2   | No matching command found — `findCommand()` (`commands.ts:688`) searches by name, computed name, and aliases across the full command set from `getAllCommands()`, which merges local commands with MCP skills | Unknown skill         |
| 3   | Command has `disableModelInvocation: true`                                                                                                                                                                    | Disabled invocation   |
| 4   | Command is not `type: 'prompt'` (only prompt commands are skills)                                                                                                                                             | Not a prompt skill    |
| 5   | Remote canonical skill not found in session (experimental)                                                                                                                                                    | Remote not discovered |

---

### 5.2 Permission System

Every tool goes through the general permission system described in [Tool Execution]({% post_url 2026-05-12-demystifying-claude-code-tool-execution %}), Section 6. The SkillTool implements its own `checkPermissions` on top of that because skills are user-authored content with varying levels of risk. A skill that only declares a name and description is harmless — it just injects text. But a skill that declares `allowed-tools` (changes what tools the model can use), `hooks` (runs shell commands), or `shell` (configures the execution environment) alters the security posture of the session. The SkillTool's permission system distinguishes between these cases, auto-allowing safe skills while prompting the user for skills with security-relevant properties.

The check (`SkillTool.ts:432-578`) follows a three-tier decision protocol:

```
checkPermissions(skill, args, context)
    |
    +-- Check deny rules (exact + prefix match)
    |     → matched? deny
    |
    +-- Check allow rules (exact + prefix match)
    |     → matched? allow
    |
    +-- Check auto-allow for descriptive-only skills
    |     → skill has only descriptive properties? allow
    |
    +-- Fallback: ask user
```

#### Tier 1: Deny and Allow Rules

The first two tiers check the user's permission rules from `settings.json`. The `ruleMatches` helper (`SkillTool.ts:451-467`) supports two matching modes: **exact match** (`review` matches rule `review`) and **prefix match** (`ralph-loop:ralph-loop` matches rule `ralph-loop:*`). Prefix matching allows users to write a single rule like `Skill(ralph-loop:*)` to allow — or deny — all skills from a plugin. Deny rules are checked first; if a deny rule matches, the skill is rejected regardless of any allow rules.

#### Tier 2: Auto-Allow for Descriptive-Only Skills

A skill's frontmatter fields fall into two categories: **descriptive** fields that define what the skill *is* (name, description, version, model preference), and **capability** fields that change what the system *can do* (run shell hooks, restrict tool access, configure the execution environment). A skill that only has descriptive fields is no more dangerous than a user message — it just injects text. A skill with capability fields actively alters the security posture of the session.

`skillHasOnlySafeProperties()` (`SkillTool.ts:910`) implements this distinction. It checks every key on the command object against a hardcoded allowlist of descriptive properties (`SAFE_SKILL_PROPERTIES`, `SkillTool.ts:875-908`). If every non-empty field on the skill is in that allowlist, the skill is auto-allowed without a permission prompt. If any field falls outside the allowlist — specifically `allowedTools`, `hooks`, or `shell` — the skill requires explicit user approval.

#### Tier 3: Ask the User

When none of the above tiers resolve the decision, the system falls through to `{ behavior: 'ask' }` and shows a permission dialog. The dialog includes pre-filled suggestions — `Skill(review)` for an exact allow rule, or `Skill(ralph-loop:*)` for a prefix rule — so the user can one-click approve the skill or its entire plugin namespace. If the user approves, the rule is added to their settings and the same skill won't prompt again.

---

### 5.3 Execution

After validation and permission checks pass, `call()` takes the validated skill name, expands its content, and delivers it to the model. The skill author decides the execution mode by setting `context: 'fork'` in the SKILL.md frontmatter; without it, the default is inline.

#### Both Modes: Content Expansion

Regardless of execution mode, the skill's raw SKILL.md body is not delivered to the model as-is — it goes through a content expansion step first. The markdown body in a SKILL.md file can contain variables and inline commands that must be resolved at invocation time:

- **Argument substitution**: `$1`, `$2`, etc. are replaced with the positional arguments the user or model passed when invoking the skill. For example, if the body contains `Review the pull request #$1` and the skill is called with `args: "123"`, the model sees `Review the pull request #123`.
- **Variable replacement**: `${CLAUDE_SKILL_DIR}` is replaced with the skill's directory path on disk (so the skill can reference sibling files), and `${CLAUDE_SESSION_ID}` with the current session ID.
- **Inline shell commands**: lines prefixed with `!` are executed as shell commands, and their stdout replaces the line. This lets a skill dynamically include output like `!git log --oneline -5` in its prompt. As a security boundary, this is disabled for MCP skills — preventing a malicious MCP server from executing arbitrary commands via skill content.

After expansion, the content is registered for compaction preservation (so it survives context compression) and skill hooks are activated. What differs between the two modes is where the expanded content goes.

#### Inline Execution (Default)

In inline mode, the expanded content is injected directly into the main conversation as user messages via `newMessages`. The `tool_result` itself just contains a brief acknowledgment like `"Launching skill: review"` — the actual content goes in `newMessages` because a skill typically expands into 3–4 messages (metadata, main content, attachments, command permissions), which can't fit in a single `tool_result` block. Each injected message is tagged with `sourceToolUseID` (`tools/utils.ts:9`), which links it back to the parent tool call. From the model's perspective, the skill's instructions appear as if the user had typed them — they become part of the conversation context that the model reads and acts on across subsequent iterations. The model also receives the skill's metadata (allowed tools, model override, effort level) via the `contextModifier` described in [Section 6](#6-context-modification).

#### Forked Execution

Forked execution is designed for skills that are expensive (many tool calls) or that produce verbose intermediate output that would clutter the main conversation. A code review skill that reads dozens of files, runs linters, and synthesizes findings is a good candidate — the user cares about the final review, not the 30 intermediate Read tool calls.

In forked mode, the SkillTool spawns an isolated sub-agent via `runAgent()` (`SkillTool.ts:122-289`). The sub-agent receives the expanded skill content as its prompt, runs in its own token budget with its own tool context, and executes the full agentic loop independently. The parent conversation only sees the sub-agent's final result text, returned directly in the `tool_result` — all intermediate tool calls stay inside the fork. Progress is reported back to the parent via `onProgress`, so the user sees tool activity in the terminal, but the main conversation's message history stays clean.

The tradeoff: the forked agent cannot access the parent's in-progress context modifications, and its token budget is separate — it doesn't share the parent's remaining budget.

---

## 6. Context Modification

A skill's SKILL.md frontmatter can declare `allowed-tools`, `model`, and `effort` fields (see [Section 3](#3-the-skillmd-format)). These don't just describe the skill — they change how the harness behaves for the remainder of the turn after the skill loads. This is how a skill like `allowed-tools: Bash, Read` actually constrains the model: the SkillTool returns a `contextModifier` function (`SkillTool.ts:775-839`) that takes effect after the SkillTool completes, before the next tool in the turn runs.

Context modification only applies to **inline execution**. Forked skills run in an isolated sub-agent with its own tool context, so there is no parent context to modify — the allowed-tools, model, and effort settings are baked into the sub-agent's configuration at spawn time.

For inline skills, three modifications are applied:

| Modification    | Frontmatter Field | Effect                                                                                                                                        |
| --------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Allowed tools   | `allowed-tools`   | Adds tools to the permission context's `alwaysAllowRules`, so subsequent tool calls within the turn don't require user approval               |
| Model override  | `model`           | Changes the model for the remainder of the turn. `resolveSkillModelOverride` preserves the `[1m]` context window suffix from the parent model |
| Effort override | `effort`          | Changes the reasoning effort level for the remainder of the turn                                                                              |

---

## 7. Dynamic Mid-Session Discovery

As described in [Section 2](#2-skill-sources), the startup scan walks *up* from cwd, finding project-root and user-level skills. But in a large monorepo, teams may place `.claude/skills/` directories deep in subdirectories — a `frontend/` team's skills, a `backend/` team's skills — that the upward walk never reaches. Additionally, some skills are only relevant when the model is working in a specific part of the codebase — a deploy skill is noise when the model is editing UI components, but essential when it touches infrastructure files. Dynamic mid-session discovery solves both problems by activating skills lazily as the model navigates the codebase.

### How It's Triggered

The model doesn't trigger discovery directly — it happens as a side effect of file operations. Every time the model calls FileReadTool, FileEditTool, or FileWriteTool, the harness piggybacks on the file path to check for nearby skills. The model doesn't know this is happening; it just reads or edits a file, and the harness uses that file path as a signal to look for skills in the vicinity. This means discovery is opportunistic — it requires no separate scanning step and adds no extra tool calls.

### Two Discovery Mechanisms

**Directory discovery** covers the startup scan's gap with subdirectories. When the model touches a file like `frontend/components/Button.tsx`, the harness walks from that file's directory back up toward cwd, checking each intermediate directory for a `.claude/skills/` folder. If `frontend/.claude/skills/` exists, its skills are loaded and become available. Each directory is checked at most once per session — subsequent file operations in the same subtree skip the check. Skills closer to the touched file take precedence over those further up.

**Conditional activation** covers skills that are already known but deliberately withheld. Skills with a `paths:` field in their frontmatter (e.g., `paths: ["infrastructure/**", "terraform/**"]`) are parsed at startup but held dormant. When a file operation touches a path that matches the pattern (using gitignore-style matching), the skill is promoted from dormant to active. This enables a `deploy` skill that only appears when the model enters the infrastructure part of the codebase, reducing noise in unrelated contexts.

### How It Takes Effect

When either mechanism activates a skill, the harness clears its cached command lists so the next call to `getCommands()` includes the new skill. On the next iteration of the agentic loop, the skill listing logic (described in [Section 4](#4-how-skills-reach-the-model)) detects the new skill in its delta check and announces it to the model via a fresh `<system-reminder>`. From the model's perspective, a new skill simply appears between iterations — it can start using it immediately.

### Highlights

The two mechanisms complement each other. Directory discovery handles skills the startup scan *couldn't* find (nested below cwd). Conditional activation handles skills the startup scan *did* find but chose to withhold (because they're only relevant in certain contexts). Both feed into the same pool and both take effect through the same announcement pipeline, so the model sees no difference between a skill loaded at startup and one activated mid-session.

---

## 8. The SkillTool Prompt

Sections 4 through 7 describe how the harness handles skills — loading, listing, validating, executing, and discovering them. But none of that matters if the model doesn't know how to use the SkillTool correctly. The SkillTool's prompt is the instruction set that teaches the model when to invoke skills, how to call them, and what pitfalls to avoid. It lives in the tool's description field (part of layer 2 in the teaching chain from [Section 4](#4-how-skills-reach-the-model)), so the model sees it on every API call as part of the tools array.

The prompt solves three problems. First, the model needs to know that `/review` in a user message means "call the SkillTool," not "generate a response about reviewing." Second, the model must not invent skill names from its training data — it should only invoke skills that appear in the `<system-reminder>` listings. Third, when a user types `/review` directly (which the harness expands immediately), the model must not call the SkillTool again redundantly. The full prompt:

```
Execute a skill within the main conversation

When users ask you to perform tasks, check if any of the available skills match.
Skills provide specialized capabilities and domain knowledge.

When users reference a "slash command" or "/<something>", they are referring
to a skill. Use this tool to invoke it.

How to invoke:
- Set `skill` to the exact name of an available skill (no leading slash).
  For plugin-namespaced skills use the fully qualified `plugin:skill` form.
- Set `args` to pass optional arguments.

Important:
- Available skills are listed in system-reminder messages in the conversation
- Only invoke a skill that appears in that list, or one the user explicitly
  typed as `/<name>` in their message. Never guess or invent a skill name
  from training data; otherwise do not call this tool
- When a skill matches the user's request, this is a BLOCKING REQUIREMENT:
  invoke the relevant Skill tool BEFORE generating any other response
- NEVER mention a skill without actually calling this tool
- Do not invoke a skill that is already running
- Do not use this tool for built-in CLI commands (like /help, /clear, etc.)
- If you see a <command-name> tag in the current conversation turn, the skill
  has ALREADY been loaded - follow the instructions directly instead of
  calling this tool again
```

The "BLOCKING REQUIREMENT" is the strongest instruction in the prompt. Without it, the model tends to paraphrase or summarize a skill's purpose instead of actually invoking it — generating a response *about* the skill rather than running it. The `<command-name>` tag check addresses the double-invocation problem: when the user types `/review` directly, the harness expands the skill immediately and tags the content. The model sees the tag and knows the skill was already loaded, so calling `Skill("review")` again would be redundant.

---

## 9. Design Decisions

**Skills are prompt injection, not function calls.** Inline skills expand into user messages that the model reads and acts on. There is no special "skill mode" or constrained output format — the model simply receives additional instructions mid-conversation. This makes skills composable with every other feature (tools, memory, plans) without special integration work.

**Forked execution for expensive skills.** Skills that need many tool calls or produce verbose intermediate output can declare `context: 'fork'` to run in an isolated sub-agent. This keeps the main conversation clean and bounds the token cost. The tradeoff: the forked agent has its own tool budget and can't access the parent's in-progress context modifications.

**Budget-aware listings with progressive degradation.** The 1% context budget ensures skill metadata never crowds out conversation content. The three-tier fallback (full descriptions → truncated descriptions → names-only) degrades gracefully as the number of skills grows. Bundled skills are exempted from truncation — Anthropic-curated skills should always be fully visible.

**Per-agent delta tracking.** The `sentSkillNames` map prevents redundant skill listings across turns. When a skill has already been announced, it's not re-announced — saving context tokens. The tracking is per-agent, so sub-agents get their own fresh listings. Cache resets happen when skills genuinely change (plugin reload, file discovery).

**Safe-properties auto-allow.** Skills that only define metadata (name, description, model, effort) pass through without a permission prompt. This reduces permission fatigue for harmless skills while requiring explicit approval for skills that declare `allowed-tools`, `hooks`, or `shell` — the properties with security implications.

**MCP shell command prohibition.** Non-MCP skills can contain `!` shell commands that execute during expansion. MCP skills cannot — this prevents a malicious MCP server from executing arbitrary commands by injecting shell lines into skill content. The boundary is enforced in `getPromptForCommand`: the `loadedFrom === 'mcp'` check gates shell execution.

**Eager + lazy loading split.** Startup loads skills from known locations (up from cwd, user dir, managed dir, bundled, plugins) via memoized functions. This covers the common case fast. But a large monorepo may have `.claude/skills/` directories nested deep in subdirectories that the upward walk misses. Rather than scanning the entire tree at startup (expensive and slow), the system discovers these lazily when file tools touch nearby files. The non-memoized `getCommands()` merges both sets fresh on every call, so a newly discovered skill becomes visible on the very next tool invocation — no restart or cache flush needed from the user's perspective. Conditional skills with `paths:` frontmatter extend this further: they are parsed at startup (so no disk I/O on activation) but withheld until the model enters the relevant part of the codebase.

---

## 10. Telemetry

Skill invocations are tracked via the `tengu_skill_tool_invocation` analytics event with the following fields:

| Field                | Values                                                                | Purpose                                                       |
| -------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------- |
| `command_name`       | Sanitized: `builtin:<name>`, `bundled:<name>`, or `custom_skill`      | Avoids leaking user skill names                               |
| `execution_context`  | `'inline'`, `'fork'`, `'remote'`                                      | Which execution path was taken                                |
| `invocation_trigger` | `'claude-proactive'`, `'nested-skill'`                                | Whether the model called it vs. a skill calling another skill |
| `skill_source`       | `'bundled'`, `'skills'`, `'commands_DEPRECATED'`, `'plugin'`, `'mcp'` | Which source provided the skill                               |
| `was_discovered`     | boolean                                                               | Whether the skill came from skill search discovery            |
| `plugin_name`        | `'official'` or `'third_party'`                                       | For plugin skills — never leaks the actual plugin name        |
| `marketplace_name`   | string                                                                | Only for official marketplace plugins                         |
| `is_remote`          | boolean                                                               | Whether this was a remote (GCS/AKI) skill                     |

The `command_name` sanitization is noteworthy. Bundled and built-in skill names are logged directly (they're controlled by Anthropic). But custom and third-party plugin skill names are redacted to `'custom_skill'` or `'third_party'` — preventing user-created skill names from appearing in telemetry dashboards.

Additional telemetry events:
- `tengu_skill_tool_slash_prefix` — model added unnecessary `/` prefix (validation stripping)
- `tengu_skill_descriptions_truncated` — budget system truncated descriptions (with mode: `'names_only'` or `'description_trimmed'`)

---

## Appendix A: Bundled Skills

Bundled skills are compiled into the CLI binary and registered programmatically at startup. They use a different definition interface than disk-based skills:

* **File:** `src/skills/bundledSkills.ts`

```typescript
// bundledSkills.ts:15-41
export type BundledSkillDefinition = {
  name: string
  description: string
  aliases?: string[]
  whenToUse?: string
  argumentHint?: string
  allowedTools?: string[]
  model?: string
  disableModelInvocation?: boolean
  userInvocable?: boolean
  isEnabled?: () => boolean
  hooks?: HooksSettings
  context?: 'inline' | 'fork'
  agent?: string
  files?: Record<string, string>
  getPromptForCommand: (args: string, context: ToolUseContext) =>
    Promise<ContentBlockParam[]>
}
```

The `files` field is the distinguishing feature. A bundled skill can carry reference files (templates, schemas, etc.) that are lazily extracted to a secure temporary directory on first invocation:

```typescript
// bundledSkills.ts:53-100
export function registerBundledSkill(definition: BundledSkillDefinition): void {
  // If the definition has files, wrap getPromptForCommand with lazy extraction
  if (definition.files) {
    const originalGetPrompt = definition.getPromptForCommand
    definition.getPromptForCommand = async (args, context) => {
      // Extract files to getBundledSkillExtractDir(name) — memoized per-process
      const baseDir = await extractBundledSkillFiles(name, files)
      const blocks = await originalGetPrompt(args, context)
      return baseDir ? prependBaseDir(blocks, baseDir) : blocks
    }
  }
  bundledSkills.push(/* Command object */)
}
```

Extraction uses `O_NOFOLLOW | O_EXCL` flags to prevent symlink attacks on the final path component. Directories are created with mode `0o700`, files with `0o600`. Path traversal is rejected (`resolveSkillFilePath` rejects absolute paths and `..` components). On Windows, numeric `O_EXCL` can produce `EINVAL` through libuv, so the string flag `'wx'` is used instead.

`getBundledSkills()` (`bundledSkills.ts:106`) returns a shallow copy of the registry to prevent external mutation.

---

## Appendix B: MCP Skills

MCP (Model Context Protocol) servers can expose prompt-type commands that appear as skills. The integration uses a registry pattern to avoid circular dependencies:

* **File:** `src/skills/mcpSkillBuilders.ts`

```typescript
// mcpSkillBuilders.ts:26-29
export type MCPSkillBuilders = {
  createSkillCommand: typeof createSkillCommand
  parseSkillFrontmatterFields: typeof parseSkillFrontmatterFields
}
```

The problem: `mcpSkills.ts` needs `createSkillCommand` and `parseSkillFrontmatterFields` from `loadSkillsDir.ts`, but `loadSkillsDir.ts` transitively imports `client.ts`, which imports `mcpSkills.ts` — a cycle. A dynamic `import(variable)` avoids dep-cruiser detection but fails at runtime in Bun-bundled binaries (the specifier resolves against `/$bunfs/root/...` instead of the source tree).

The solution: `mcpSkillBuilders.ts` imports nothing but types. `loadSkillsDir.ts` calls `registerMCPSkillBuilders()` at module init (before any MCP server connects). `mcpSkills.ts` calls `getMCPSkillBuilders()` when it needs the functions. The getter throws with an explanatory error if called before registration.

MCP skills are filtered into the skill listing by `getMcpSkillCommands()` (`commands.ts:547`), which returns commands that are `type === 'prompt'`, `loadedFrom === 'mcp'`, and `!disableModelInvocation`. They appear alongside local skills in `<system-reminder>` listings.

One security restriction: MCP skills **never** execute inline shell commands (lines prefixed with `!` in the prompt body). This prevents a malicious MCP server from executing arbitrary commands on the user's machine via skill content injection.

---

## Appendix D: Frontmatter Fields Reference

The full set of SKILL.md frontmatter fields, grouped by purpose:

**Identity and listing** — these control how the skill appears in `<system-reminder>` listings and whether the model or user can invoke it:

| Field | Default | Purpose |
|-------|---------|---------|
| `name` | derived from directory name | Display name override |
| `description` | extracted from first paragraph of body | One-liner shown in skill listings |
| `when-to-use` | — | Guidance for the model on when to proactively invoke this skill; appended to description in listings |
| `argument-hint` | — | Shows users the expected argument format (e.g., `<pr-number>`) |
| `user-invocable` | `true` | Whether `/skill-name` works from the prompt |
| `disable-model-invocation` | `false` | Whether the model can proactively call it |

**Execution behavior** — these change what happens when the skill runs:

| Field | Default | Purpose |
|-------|---------|---------|
| `allowed-tools` | — | Restricts which tools the model can use after the skill loads (see [Section 6](#6-context-modification)) |
| `model` | — | Model override for the turn (`opus`, `sonnet`, `haiku`, or `inherit`) |
| `effort` | — | Reasoning effort level (0–100 or named: `low`, `high`) |
| `context` | inline | Set to `'fork'` to run in an isolated sub-agent (see [Section 5.3](#53-execution)) |
| `agent` | — | Custom agent type for forked execution |
| `arguments` | — | Named arguments for `$1`, `$2` substitution |
| `shell` | — | Shell directive (e.g., `bash-strict`) |
| `hooks` | — | Lifecycle hooks (shell commands that run before/after tool calls within the skill) |

**Conditional activation** — these control when the skill becomes available:

| Field | Default | Purpose |
|-------|---------|---------|
| `paths` | — | Gitignore-style patterns; skill stays dormant until a file operation matches (see [Section 7](#7-dynamic-mid-session-discovery)) |
| `version` | — | Semantic version |

---

## Appendix E: Key Source Files

| File                                                 | Lines  | Role                                                                                                            |
| ---------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------- |
| `src/tools/SkillTool/SkillTool.ts`                   | ~1109  | SkillTool definition: validation, permissions, inline/forked execution, context modification                    |
| `src/tools/SkillTool/prompt.ts`                      | ~242   | SkillTool prompt, budget-aware formatting, `formatCommandsWithinBudget()`                                       |
| `src/tools/SkillTool/UI.tsx`                         | ~128   | Terminal rendering: skill name, progress, success/error messages                                                |
| `src/skills/loadSkillsDir.ts`                        | ~1087  | Disk-based skill loading: SKILL.md parsing, frontmatter extraction, directory discovery, conditional activation |
| `src/skills/bundledSkills.ts`                        | ~222   | Bundled skill registry: registration, lazy file extraction, security                                            |
| `src/skills/mcpSkillBuilders.ts`                     | ~45    | Cycle-free bridge for MCP skill creation                                                                        |
| `src/commands.ts`                                    | ~700+  | Command aggregation: `getSkills()`, `getCommands()`, `getSkillToolCommands()`, `findCommand()`                  |
| `src/utils/attachments.ts`                           | ~2750+ | Skill listing attachment: `getSkillListingAttachments()`, delta tracking, resume suppression                    |
| `src/utils/messages.ts`                              | ~4500+ | Attachment-to-API conversion: `skill_listing` → `<system-reminder>`                                             |
| `src/utils/processUserInput/processSlashCommand.tsx` | ~850+  | Slash command processing: `processPromptSlashCommand()`, content expansion, hook registration                   |
