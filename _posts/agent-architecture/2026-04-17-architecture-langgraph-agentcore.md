---
title: "Architecture: LangGraph + Amazon Bedrock AgentCore"
date: 2026-04-17 12:00:00 +0000
categories: [Agent Architecture]
tags: [Agent, LangGraph, Amazon Bedrock AgentCore]
---

## 1. Overview

This document describes the production architecture for deploying LangGraph-based AI agents on Amazon Bedrock AgentCore Runtime. It covers the concluded design, integration patterns, authentication, streaming, storage, and alternative approaches with trade-offs.

### Technology Stack

| Component | Technology | Role |
|---|---|---|
| Agent orchestration | LangGraph | Graph-based workflow definition, tool calling, routing, state management |
| Agent hosting | AgentCore Runtime | Serverless microVM hosting, session isolation, up to 8-hour execution |
| Memory | AgentCore Memory | Short-term (checkpoints) and long-term (preferences, facts) persistence |
| CRUD backend | API Gateway + Lambda + DynamoDB | Session listing, user metadata, non-agent business logic |
| Identity | Cognito / Okta (or internal IdP) | JWT token issuance for end users |
| Frontend | JavaScript SPA | Chat UI consuming SSE and WebSocket streams |

---

## 2. Concluded Architecture

```
                        ┌───────────────────────────────────┐
                        │        Frontend (JS/React)         │
                        │                                    │
                        │   JWT token from IdP (Cognito/Okta)│
                        └────┬─────────────────┬────────────┘
                             │                 │
               REST (CRUD)   │                 │  Agent interactions
                             │                 │  (streaming)
                             ▼                 ▼
                ┌────────────────┐   ┌──────────────────────────────────┐
                │ API Gateway    │   │  AgentCore Runtime (JWT auth)    │
                │ REST API       │   │                                  │
                │ + JWT Authorizer│  │  Frontend connects directly —    │
                └───────┬────────┘   │  no Lambda in between            │
                        │            │                                  │
                        ▼            │  ┌─────────────────────────┐     │
                ┌──────────────┐    │  │ POST /invocations (SSE)  │     │
                │ Lambda       │    │  │ GET  /ws (WebSocket)     │     │
                │              │    │  └─────────────────────────┘     │
                │ • list sessions│  └──────────────────────────────────┘
                │ • get history │           │
                │ • delete session│         ▼
                │ • user prefs  │   ┌──────────────┐
                └───────┬───────┘   │ AgentCore    │
                        │           │ Memory       │
                        ▼           │ (short+long) │
                ┌──────────────┐    └──────────────┘
                │  DynamoDB    │
                │  (metadata)  │
                └──────────────┘
```

### Key Design Decisions

1. **AgentCore Runtime configured with JWT auth** — the frontend talks to AgentCore directly for all agent interactions, eliminating Lambda as an intermediary.
2. **No Lambda proxy for agent calls** — avoids the 15-minute Lambda timeout bottleneck and streaming relay complexity.
3. **API Gateway + Lambda only for CRUD** — session management, user metadata, and business data. These are simple request-response JSON, no streaming needed.
4. **Both SSE and WebSocket supported** — same AgentCore runtime, same JWT, same session. The frontend chooses the protocol based on the use case.

---

## 3. LangGraph on AgentCore

### 3.1 What Each Layer Does

| Concern | LangGraph | AgentCore |
|---|---|---|
| Agent logic & orchestration | ✅ Graph-based workflow definition | — |
| Tool calling & routing | ✅ Nodes, edges, conditional routing | — |
| Hosting & scaling | — | ✅ Serverless Runtime with microVM isolation |
| Short-term memory | ✅ Checkpointer interface | ✅ `AgentCoreMemorySaver` backend |
| Long-term memory | ✅ Store interface | ✅ `AgentCoreMemoryStore` with auto-extraction |
| Identity & auth | — | ✅ Built-in JWT authorizer + Identity service |
| Tool gateway (MCP) | — | ✅ Gateway service |
| Observability | Via LangSmith | ✅ OpenTelemetry-based tracing |
| Evaluation | Via LangSmith | ✅ AgentCore Evaluations |

### 3.2 Agent Code Structure

The agent supports both SSE streaming (via `@app.entrypoint`) and WebSocket (via `@app.websocket`) on the same container, port 8080:

```python
import json
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

REGION = "us-west-2"
MEMORY_ID = "YOUR_MEMORY_ID"
MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

# Memory integrations
checkpointer = AgentCoreMemorySaver(MEMORY_ID, region_name=REGION)
store = AgentCoreMemoryStore(MEMORY_ID, region_name=REGION)

# LLM + agent graph
llm = init_chat_model(MODEL_ID, model_provider="bedrock_converse", region_name=REGION)
graph = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
    store=store,
)

# Shared agent logic
async def agent_stream(prompt, session_id, actor_id):
    config = {"configurable": {"thread_id": session_id, "actor_id": actor_id}}
    async for event in graph.astream(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
    ):
        yield json.dumps(event)

# SSE streaming via POST /invocations
@app.entrypoint
async def http_handler(payload, context):
    async for chunk in agent_stream(
        payload.get("prompt", ""),
        context.session_id,
        payload.get("actor_id", "default"),
    ):
        yield chunk

# WebSocket via GET /ws
@app.websocket
async def ws_handler(websocket, context):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            async for chunk in agent_stream(
                data.get("prompt", ""),
                context.session_id,
                data.get("actor_id", "default"),
            ):
                await websocket.send_text(chunk)
    except Exception:
        pass
    finally:
        await websocket.close()

app.run()
```

### 3.3 LangGraph Workflow Patterns

LangGraph supports several orchestration patterns, all deployable on AgentCore:

- **Prompt chaining** — sequential LLM calls with conditional gates
- **Parallelization** — fan-out to multiple nodes, fan-in to aggregator
- **Routing** — LLM-based classification routes to specialized handlers
- **Orchestrator-worker** — dynamic task decomposition via `Send` API
- **Evaluator-optimizer** — generation + evaluation loop until quality threshold met
- **Autonomous agent** — tool-calling loop with conditional termination

---

## 4. Authentication

### 4.1 AgentCore Runtime Auth Modes

AgentCore supports two inbound auth modes (one per runtime, not both simultaneously):

| Mode | How it works | Use case |
|---|---|---|
| **IAM SigV4** (default) | boto3 signs requests automatically | Backend-to-agent calls |
| **JWT Bearer Token** | Runtime validates JWT from an OIDC provider | Frontend-to-agent direct calls |

Our architecture uses **JWT auth** so the frontend can call AgentCore directly.

### 4.2 JWT Authorizer Configuration

```python
client.create_agent_runtime(
    agentRuntimeName="my-agent",
    authorizerConfiguration={
        "customJWTAuthorizer": {
            "discoveryUrl": "https://cognito-idp.us-west-2.amazonaws.com/POOL_ID/.well-known/openid-configuration",
            "allowedClients": ["YOUR_CLIENT_ID"],
            "allowedAudience": ["YOUR_AUDIENCE"],  # optional
            "allowedScopes": ["openid"],            # optional
        }
    },
    ...
)
```

### 4.3 Auth Flow per Path

**REST calls (CRUD via API Gateway):**
```
Frontend → Authorization: Bearer <JWT> → API Gateway (Cognito Authorizer) → Lambda → DynamoDB
```

**Agent calls (SSE, direct to AgentCore):**
```
Frontend → Authorization: Bearer <JWT> → AgentCore Runtime (built-in JWT authorizer)
```

**Agent calls (WebSocket, direct to AgentCore):**
```
Frontend → Sec-WebSocket-Protocol: base64UrlBearerAuthorization.<base64url(JWT)>
        → AgentCore Runtime (built-in JWT authorizer)
```

Both paths use the **same JWT from the same IdP**. The frontend authenticates once and uses the token everywhere.

### 4.4 Browser WebSocket Auth

Browsers cannot set custom headers on WebSocket. AgentCore accepts the JWT via the `Sec-WebSocket-Protocol` header:

```javascript
const base64url = btoa(jwtToken)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=/g, "");

const ws = new WebSocket(
    `wss://bedrock-agentcore.us-west-2.amazonaws.com/runtimes/${arn}/ws`,
    [`base64UrlBearerAuthorization.${base64url}`, "base64UrlBearerAuthorization"]
);
```

---

## 5. Streaming

### 5.1 SSE Streaming (Server → Client)

For standard chat interactions. The agent `yield`s chunks, AgentCore streams them as SSE events.

```javascript
const response = await fetch(
    `${AGENTCORE_ENDPOINT}/runtimes/${ESCAPED_ARN}/invocations?qualifier=DEFAULT`,
    {
        method: "POST",
        headers: {
            "Authorization": `Bearer ${token}`,
            "Content-Type": "application/json",
            "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": sessionId,
        },
        body: JSON.stringify({ prompt }),
    }
);

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) {
            outputDiv.textContent += line.slice(6);
        }
    }
}
```

### 5.2 WebSocket (Bidirectional)

For interactive/long-running sessions, real-time voice, or when the client needs to send data while receiving.

```javascript
const ws = connectWebSocket(sessionId, token); // see auth section above

ws.onopen = () => {
    ws.send(JSON.stringify({ prompt: "analyze this dataset" }));
};

ws.onmessage = (event) => {
    outputDiv.textContent += event.data;
};
```

### 5.3 When to Use Which

| Scenario | Protocol | Why |
|---|---|---|
| Standard chat (user sends, agent responds) | SSE | Simpler, stateless HTTP |
| Long-running tasks (> minutes) | WebSocket | No timeout, real-time updates |
| Voice / real-time audio | WebSocket / WebRTC | Bidirectional, low latency |
| User needs to interrupt mid-response | WebSocket | Client can send cancel signal |

The frontend can switch between SSE and WebSocket even within the same session — both use the same `runtimeSessionId` and hit the same microVM.

---

## 6. Storage & Memory

### 6.1 Storage Decision Matrix

```
Is the data only needed during this invocation?
  → Yes: Ephemeral session state (default, in-memory in microVM)

Does the agent need to resume work on the same files across stop/resume?
  → Yes: Session Storage (/mnt/workspace, Preview)

Is it conversational context (history, preferences, facts)?
  → Yes: AgentCore Memory

Does the data need to persist long-term, be shared across sessions/agents,
or be accessed by external systems?
  → Yes: S3 / DynamoDB (self-managed)
```

### 6.2 AgentCore Memory with LangGraph

The `langgraph-checkpoint-aws` package provides two integrations:

| Integration | Purpose | Persists across |
|---|---|---|
| `AgentCoreMemorySaver` | Short-term memory — conversation history, graph execution state | Session stops, microVM terminations |
| `AgentCoreMemoryStore` | Long-term memory — auto-extracted preferences, facts, summaries | Sessions, agents |

```python
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore

checkpointer = AgentCoreMemorySaver(MEMORY_ID, region_name="us-west-2")
store = AgentCoreMemoryStore(MEMORY_ID, region_name="us-west-2")

graph = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,  # short-term: conversation state
    store=store,                 # long-term: preferences, facts
)
```

With AgentCore Memory as the checkpointer, **session storage is not needed for conversational state**. Session storage is only needed if the agent produces filesystem artifacts (generated code, build outputs, etc.).

### 6.3 Alternative Checkpointer Backends

LangGraph supports multiple checkpointer backends via the same `langgraph-checkpoint-aws` package and others:

| Backend | Package | When to use |
|---|---|---|
| AgentCore Memory | `langgraph-checkpoint-aws` | Running on AgentCore, want managed long-term memory |
| DynamoDB | `langgraph-checkpoint-aws` | Need more control, custom query patterns |
| PostgreSQL | `langgraph-checkpoint-postgres` | Existing Postgres infrastructure |
| Redis | `langgraph-checkpoint-redis` | Low-latency requirements |
| MongoDB | `langgraph-checkpoint-mongodb` | Existing Mongo infrastructure |

### 6.4 When to Use Self-Managed Storage (S3/DynamoDB)

| Scenario | Why not built-in? |
|---|---|
| Large artifacts shared across sessions/agents | Session storage is per-session, has size limits |
| Data persisting beyond 14 days without invocation | Session storage auto-deletes after 14 days |
| Structured business data (audit logs, compliance) | AgentCore Memory is for conversational context |
| User-uploaded/downloaded files | Need addressable URLs (S3 presigned) |
| Data surviving agent runtime version updates | Session storage resets on version update |

---

## 7. Session Management

### 7.1 AgentCore Session Lifecycle

AgentCore does not manage session-to-user mappings. Your backend is responsible for:
- Generating unique session IDs (min 33 characters)
- Tracking which sessions belong to which users
- Managing session lifecycle (max sessions per user, cleanup)

```
Session states:
  Active → processing requests or background tasks ("HealthyBusy")
  Idle   → waiting for requests, 15-min timeout starts
  Stopped → microVM terminated, resumes on next invocation
```

### 7.2 CRUD API for Session Management

These go through API Gateway → Lambda → DynamoDB:

| Endpoint | Method | Description |
|---|---|---|
| `/sessions` | GET | List user's sessions |
| `/sessions/{id}` | GET | Get session metadata |
| `/sessions` | POST | Create new session (generate ID, store metadata) |
| `/sessions/{id}` | DELETE | Delete session (cleanup DynamoDB + call `StopRuntimeSession`) |

### 7.3 Long-Running Agents

For tasks exceeding typical request-response timeouts, AgentCore supports async processing:

1. Client sends request → agent returns immediately ("I've started working on this")
2. Agent processes in background, reports `HealthyBusy` via `/ping` (polled by AgentCore Runtime, not your client)
3. Client invokes same session later to retrieve results

```python
import threading

@app.entrypoint
def main(payload):
    task_id = app.add_async_task("processing")

    def background_work():
        # ... long-running agent work ...
        app.complete_async_task(task_id)

    threading.Thread(target=background_work, daemon=True).start()
    return {"status": "started", "task_id": task_id}
```

The `/ping` endpoint is polled by **AgentCore Runtime itself** (not your client) to determine session health. `HealthyBusy` keeps the session alive; `Healthy` starts the idle timeout.

---

## 8. Alternative Architecture: Lambda Proxy

### 8.1 Design

Instead of the frontend calling AgentCore directly, all requests go through API Gateway → Lambda → AgentCore:

```
Frontend → API Gateway (REST, stream mode) → Lambda → AgentCore Runtime (SigV4 or JWT)
```

### 8.2 Lambda Proxy Implementation

The Lambda forwards the JWT and relays the SSE stream. Requires Node.js runtime for Lambda response streaming:

```javascript
// Lambda (Node.js) — streams AgentCore SSE to frontend
export const handler = awslambda.streamifyResponse(
    async (event, responseStream, context) => {
        const body = JSON.parse(event.body);
        const bearerToken = event.headers["authorization"]?.replace("Bearer ", "");
        const sessionId = body.session_id || "default-session";

        responseStream = awslambda.HttpResponseStream.from(responseStream, {
            statusCode: 200,
            headers: {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
            },
        });

        const agentResponse = await fetch(
            `${ENDPOINT}/runtimes/${ESCAPED_ARN}/invocations?qualifier=DEFAULT`,
            {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${bearerToken}`,
                    "Content-Type": "application/json",
                    "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": sessionId,
                },
                body: JSON.stringify({ prompt: body.prompt }),
            }
        );

        const reader = agentResponse.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            responseStream.write(decoder.decode(value, { stream: true }));
        }
        responseStream.end();
    }
);
```

API Gateway must be configured with:
- REST API (not HTTP API — HTTP API doesn't support response streaming)
- Response transfer mode: `STREAM`
- Lambda proxy integration

### 8.3 Comparison: Direct vs Lambda Proxy

| Concern | Direct to AgentCore (concluded) | Lambda Proxy |
|---|---|---|
| **Timeout** | AgentCore: 8 hours | Lambda: 15 min max |
| **Streaming fidelity** | Native end-to-end | Relay through Lambda |
| **Latency** | Direct connection | Extra hop |
| **Cost** | AgentCore only | AgentCore + Lambda + API Gateway |
| **WebSocket** | Native AgentCore WebSocket | Must build WebSocket relay via API Gateway WebSocket API + Lambda |
| **Auth validation** | One point (AgentCore JWT) | Two points (API Gateway + AgentCore) |
| **Complexity** | `fetch()` or `new WebSocket()` | Lambda streaming (Node.js only), stream mode config |
| **Centralized control** | Less — agent calls bypass API Gateway | More — all traffic through API Gateway (throttling, WAF, logging) |
| **Custom domain** | Not supported on AgentCore endpoint | API Gateway provides custom domains |
| **Runtime auth mode** | Must be JWT | Can be SigV4 (Lambda signs) or JWT (Lambda forwards token) |

### 8.4 When to Use Lambda Proxy

- You need API Gateway features on agent calls: rate limiting, WAF, usage plans, custom domains
- You want a single API Gateway as the entry point for all traffic
- You need SigV4 auth on the runtime (no JWT authorizer setup)
- Agent interactions are short (< 15 min) and SSE-only (no WebSocket needed)

### 8.5 When to Use Direct (Recommended)

- Agent interactions may exceed 15 minutes
- You need WebSocket for real-time bidirectional streaming
- You want lower latency and simpler architecture
- You don't need API Gateway features on agent traffic

---

## 9. AgentCore Services Reference

| Service | Description | LangGraph Integration |
|---|---|---|
| **Runtime** | Serverless microVM hosting, session isolation, extended execution, bidirectional streaming | `BedrockAgentCoreApp` wraps LangGraph graph |
| **Memory** | Short-term (checkpoints) + long-term (auto-extracted insights) | `AgentCoreMemorySaver` + `AgentCoreMemoryStore` |
| **Gateway** | Converts APIs/Lambda into MCP-compatible tools | Agent tools can call Gateway endpoints |
| **Identity** | JWT authorizer, workload identities, OAuth credential providers | Inbound auth for runtime, outbound auth for 3P tools |
| **Code Interpreter** | Sandboxed code execution (Python, JS, TS) | Available as agent tool |
| **Browser** | Cloud-based browser for web interaction | Available as agent tool |
| **Observability** | OpenTelemetry tracing, CloudWatch integration | Supports LangGraph traces/spans |
| **Evaluations** | Automated agent quality assessment | Evaluates LangGraph sessions via OTEL/OpenInference |
| **Policy** | Deterministic guardrails (natural language or Cedar) | Integrates with Gateway to intercept tool calls |
| **Registry** | Centralized catalog for agents, MCP servers, tools | Discover/manage agents across org |

---

## 10. Testing & Local Development

### 10.1 Local Development

Run the agent locally — no auth, no deployment:

```bash
python agent.py
# Agent runs on localhost:8080

# Test HTTP
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello"}'

# Test WebSocket
python -c "
import asyncio, websockets, json
async def test():
    async with websockets.connect('ws://localhost:8080/ws') as ws:
        await ws.send(json.dumps({'prompt': 'hello'}))
        print(await ws.recv())
asyncio.run(test())
"
```

### 10.2 Testing Deployed Agent with JWT Auth

```bash
# Get token from your IdP
TOKEN=$(aws cognito-idp initiate-auth \
  --client-id "$CLIENT_ID" \
  --auth-flow USER_PASSWORD_AUTH \
  --auth-parameters USERNAME=testuser,PASSWORD=password \
  --region us-west-2 | jq -r '.AuthenticationResult.AccessToken')

# Invoke
curl -X POST "https://bedrock-agentcore.us-west-2.amazonaws.com/runtimes/${ESCAPED_ARN}/invocations?qualifier=DEFAULT" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -H "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: test-session" \
  -d '{"prompt": "hello"}'
```

### 10.3 Separate Runtime for CI/CD

You can create a separate runtime version with SigV4 auth for integration testing pipelines, while production uses JWT auth. Both run the same agent code.

---

## 11. Deployment

### 11.1 Deploy with AgentCore CLI

```bash
npm install -g @aws/agentcore
agentcore create    # scaffold project
agentcore deploy    # build, push container, create/update runtime
```

### 11.2 Deploy with SDK

```python
import boto3

client = boto3.client("bedrock-agentcore-control", region_name="us-west-2")

response = client.create_agent_runtime(
    agentRuntimeName="my-langgraph-agent",
    roleArn="arn:aws:iam::111122223333:role/AgentRuntimeRole",
    agentRuntimeArtifact={
        "containerConfiguration": {
            "containerUri": "111122223333.dkr.ecr.us-west-2.amazonaws.com/my-agent:latest"
        }
    },
    authorizerConfiguration={
        "customJWTAuthorizer": {
            "discoveryUrl": "https://cognito-idp.us-west-2.amazonaws.com/POOL_ID/.well-known/openid-configuration",
            "allowedClients": ["CLIENT_ID"],
        }
    },
    networkConfiguration={"networkMode": "PUBLIC"},
)
```

---

## 12. References

- [Amazon Bedrock AgentCore Developer Guide](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)
- [AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html)
- [AgentCore Memory + LangGraph Integration](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-integrate-lang.html)
- [Use Any Agent Framework (LangGraph example)](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/using-any-agent-framework.html)
- [AgentCore Auth (JWT + OAuth)](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-oauth.html)
- [AgentCore Bidirectional Streaming (WebSocket)](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-get-started-websocket.html)
- [AgentCore Long-Running Agents](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-long-run.html)
- [AgentCore Session Management](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html)
- [AgentCore Session Storage (Preview)](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-persistent-filesystems.html)
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph Workflows & Agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [LangGraph Checkpointer Integrations](https://docs.langchain.com/oss/python/integrations/checkpointers/index)
- [API Gateway Response Streaming](https://docs.aws.amazon.com/apigateway/latest/developerguide/response-transfer-mode.html)
- [langgraph-checkpoint-aws (PyPI)](https://pypi.org/project/langgraph-checkpoint-aws/)
- [AgentCore Samples (GitHub)](https://github.com/awslabs/amazon-bedrock-agentcore-samples)
