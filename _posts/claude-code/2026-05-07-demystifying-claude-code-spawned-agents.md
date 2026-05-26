---
title: "Demystifying Claude Code: Spawned Agents"
date: 2026-05-07 15:00:00 -0700
categories: [Claude Code]
tags: [Claude Code, Spawned Agents, Sub-Agents, Fork, AI Agents, CLI]
mermaid: true
---

This post covers how spawned agents (sub-agents and teammates) are defined, configured, and specialized in Claude Code. A spawned agent is any agent created by the `Agent` tool — whether a one-shot sub-agent like Explore or Plan, a fork that inherits the parent's full context, or a long-lived teammate in an agent team. This post focuses on the definition and configuration layer; for how agents coordinate as a team, see [Agent Team]({% post_url claude-code/2026-05-07-demystifying-claude-code-agent-team %}).

---

## 1. Agent Definition Structure

* **File:** `src/tools/AgentTool/loadAgentsDir.ts`

Every spawned agent is configured via a definition that determines its tools, model, permissions, system prompt, and behavioral constraints. Definitions are expressed as Markdown files with YAML frontmatter:

```typescript
export type BaseAgentDefinition = {
  agentType: string
  whenToUse: string
  tools?: string[]
  disallowedTools?: string[]
  skills?: string[]
  mcpServers?: AgentMcpServerSpec[]
  hooks?: HooksSettings
  color?: AgentColorName
  model?: string
  effort?: EffortValue
  permissionMode?: PermissionMode
  maxTurns?: number
  background?: boolean
  initialPrompt?: string
  memory?: AgentMemoryScope    // 'user' | 'project' | 'local'
  isolation?: 'worktree' | 'remote'
  omitClaudeMd?: boolean
}
```

The markdown file body (after frontmatter) becomes the agent's system prompt. If `memory` is enabled, a memory prompt is appended automatically.

---

## 2. Agent Sources

Three sources of agent definitions:

| Source | Type | Examples |
|--------|------|----------|
| **Built-in** | `BuiltInAgentDefinition` | Explore, Plan, code-reviewer |
| **Custom** | `CustomAgentDefinition` | User-authored `.md` files in `~/.claude/agents/` or `.claude/agents/` |
| **Plugin** | `PluginAgentDefinition` | Plugin-provided agents |

Built-in agents have a hardcoded `getSystemPrompt()` function. Custom agents derive their system prompt from the markdown body. Plugin agents are provided by installed plugins.

---

## 3. Built-in One-Shot Agents

* **File:** `src/tools/AgentTool/constants.ts`

```typescript
export const ONE_SHOT_BUILTIN_AGENT_TYPES: ReadonlySet<string> = new Set([
  'Explore',
  'Plan',
])
```

One-shot agents run once and return a report. The parent never sends follow-up messages via `SendMessage`. The system skips the agentId/SendMessage/usage trailer for these to save tokens (~135 chars per invocation).

---

## 4. Fork Sub-Agents

* **File:** `src/tools/AgentTool/forkSubagent.ts`

Fork is a special spawning mode where the child inherits the parent's **full conversation context**:

```typescript
export const FORK_AGENT = {
  agentType: FORK_SUBAGENT_TYPE,
  tools: ['*'],
  maxTurns: 200,
  model: 'inherit',
  permissionMode: 'bubble',
  source: 'built-in',
  getSystemPrompt: () => '',
}
```

**Key design principles:**

- **Cache optimization** — Fork children receive the same conversation prefix, maximizing prompt cache hits across parallel forks.
- **Recursive prevention** — Checks for `FORK_BOILERPLATE_TAG` in conversation history to prevent forks from forking.
- **Structured output** — Fork children are constrained to a strict output format:
  ```
  Scope: <echo back assigned scope>
  Result: <findings>
  Key files: <relevant paths>
  Files changed: <if any>
  Issues: <if any>
  ```

---

## 5. Worktree Isolation

* **File:** `src/tools/EnterWorktreeTool/EnterWorktreeTool.ts`

Any spawned agent — whether a one-shot sub-agent or a long-lived teammate — can be given its own isolated git worktree by passing `isolation: "worktree"` to the `Agent` tool. This prevents file conflicts when multiple agents edit the same repository concurrently.

**Lifecycle:**

1. **Creation** — `createAgentWorktree(slug)` calls `git worktree add` with a unique branch name derived from the agent's ID.
2. **Entry** — The agent's CWD changes to the worktree path. All caches (system prompt, memory files, plans directory) are cleared so the agent operates on the isolated copy.
3. **Work** — All file operations (Read, Edit, Write, Bash) happen in the isolated copy.
4. **Exit** — On completion, the worktree is either kept (path and branch returned in the result for the parent to merge) or automatically removed if the agent made no changes.

**In agent teams**, worktree isolation is particularly useful when multiple teammates edit overlapping files. The leader can spawn each teammate with `isolation: "worktree"` and later merge their branches. Worktrees are cleaned up during team deletion (see [Agent Team]({% post_url claude-code/2026-05-07-demystifying-claude-code-agent-team %}), Section 11.3).

---

## Appendix: Code Citations

| File | Purpose |
|------|---------|
| `src/tools/AgentTool/loadAgentsDir.ts` | Agent definition parsing from markdown |
| `src/tools/AgentTool/constants.ts` | Tool name constants, one-shot agent set |
| `src/tools/AgentTool/forkSubagent.ts` | Fork sub-agent implementation |
| `src/tools/AgentTool/agentColorManager.ts` | Color assignment for visual differentiation |
| `src/tools/AgentTool/agentMemory.ts` | Persistent memory support for agents |
| `src/tools/AgentTool/runAgent.ts` | Core agent execution loop |
| `src/tools/EnterWorktreeTool/EnterWorktreeTool.ts` | Worktree creation and cache clearing |
