# API Reference

This document covers the FastAPI application setup, all endpoints, and the SSE event streaming protocol.

Base URL: `http://localhost:8000/api/v1`

---

## Table of Contents

1. [Application Setup](#1-application-setup)
2. [Session Endpoints](#2-session-endpoints)
3. [Message Endpoints](#3-message-endpoints)
4. [SSE Event Protocol](#4-sse-event-protocol)
5. [Document Endpoints](#5-document-endpoints)
6. [Why Direct SSE for Streaming](#6-why-direct-sse-for-streaming)

---

## 1. Application Setup

**File**: `app/main.py`

The FastAPI application uses a lifespan context manager for startup/shutdown:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    pool = AsyncConnectionPool(settings.DATABASE_URL_PSYCOPG)  # psycopg3 — for LangGraph
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()  # creates checkpoint tables if not exist
    app.state.graph = build_graph(checkpointer)  # compile StateGraph once

    yield

    # Shutdown
    await pool.close()
```

**Two PostgreSQL drivers**: psycopg3 (`psycopg`) for LangGraph's `AsyncPostgresSaver` (it uses psycopg3's native protocol features), asyncpg for all other application queries. They connect to the same database but use different connection pools.

**CORS**: `allow_origins=["*"]` — permits direct browser calls to `http://localhost:8000` from `http://localhost:3000`. Required for the SSE streaming path.

**Routers**:
| Router | Prefix | File |
|--------|--------|------|
| Health | `/health` | `api/v1/health.py` |
| Sessions | `/api/v1/sessions` | `api/v1/sessions.py` |
| Messages | `/api/v1/sessions/{id}/messages` | `api/v1/messages.py` |
| Documents | `/api/v1/sessions/{id}/documents` | `api/v1/documents.py` |

---

## 2. Session Endpoints

### `POST /api/v1/sessions`

Creates a new advisory session.

**Request**: No body

**Response**:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "langgraph_thread_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "created_at": "2026-06-13T10:30:00Z"
}
```

**What happens**:
1. Inserts a row into `advisory_sessions` with new UUID `id` and `langgraph_thread_id`
2. Calls `graph.aupdate_state(config, make_initial_state(session_id))` — writes the initial LangGraph checkpoint keyed by `langgraph_thread_id`
3. Returns the session data

The initial state includes `session_id`, `messages: []`, and all other fields at their zero values.

---

### `GET /api/v1/sessions`

Returns the 50 most recent sessions ordered by `created_at DESC`.

**Response**: Array of session objects (same shape as POST response)

---

### `GET /api/v1/sessions/{session_id}`

Returns a single session by UUID.

**Response**: Session object, or 404 if not found

---

## 3. Message Endpoints

### `GET /api/v1/sessions/{session_id}/messages`

Returns the conversation history for a session.

**Response**:
```json
[
  {"role": "user", "content": "Analyze my portfolio", "agent": null},
  {"role": "assistant", "content": "I'd be happy to help...", "agent": "intake"},
  {"role": "assistant", "content": "Based on your portfolio...", "agent": "advisor_copilot"}
]
```

**Implementation**: Calls `graph.aget_state(config)` to read the latest LangGraph checkpoint and extracts the `messages` field. Filters:
- Messages with empty string content are excluded
- Messages that are tool-call-only (have `tool_calls` but no text content) are excluded — e.g., intake's `collect_profile` tool invocation

---

### `POST /api/v1/sessions/{session_id}/messages`

Sends a message and streams the response via Server-Sent Events.

**Request body**:
```json
{"content": "What are the main risks in my portfolio?"}
```

**Response**: `text/event-stream` (SSE). See [SSE Event Protocol](#4-sse-event-protocol) for the full event sequence.

---

## 4. SSE Event Protocol

Every SSE event is formatted as:
```
event: <event-name>\n
data: <json-string>\n
\n
```

Events are emitted in two phases within a single request:

### Phase 1 — During graph execution

| Event | Data | When |
|-------|------|------|
| `node_start` | `{"node": "risk_assessment"}` | When a pipeline node begins (`on_chain_start`) |
| `node_complete` | `{"node": "risk_assessment"}` | When a pipeline node finishes (`on_chain_end`) |
| `token` | `{"content": "Based on"}` | Each streaming token from `advisor_copilot`'s LLM call |

**Nodes that emit events** (`_AGENT_NODES`): `guardrail_input`, `intake`, `document_intelligence`, `profile_builder`, `risk_assessment`, `strategy`, `scoring`, `advisor_copilot`

`supervisor` and `intent_classifier` do not emit events — they run too frequently (supervisor runs after every node) to be useful progress indicators.

`token` events are emitted **only** from `advisor_copilot` — the only node using `streaming=True`. Other nodes use non-streaming LLM calls.

### Phase 2 — After graph completes

| Event | Data | When |
|-------|------|------|
| `message` | `{"role": "assistant", "content": "...", "agent": "advisor_copilot"}` | One per new AI message added during this run |
| `state` | `{"risk_metrics": {...}, "allocation_result": {...}, "scoring_result": {...}}` | If any quant results were computed |
| `done` | `{"session_id": "..."}` | Stream end signal |
| `error` | `{"detail": "..."}` | If an exception occurred during graph execution |

**Message detection**: The endpoint snapshots `len(state.messages)` before the graph run. After it completes, new messages are `all_messages[msg_count_before + 1:]` (the `+1` skips the `HumanMessage` just sent).

**Why `state` comes after `message`**: The `state` event carries `risk_metrics`, `allocation_result`, and `scoring_result` — these are only present after the relevant pipeline nodes complete. The graph must finish before this data is accessible.

### Full event sequence example (full_analysis intent)

```
event: node_start
data: {"node": "guardrail_input"}

event: node_complete
data: {"node": "guardrail_input"}

event: node_start
data: {"node": "intake"}

event: node_complete
data: {"node": "intake"}

event: node_start
data: {"node": "risk_assessment"}

event: node_complete
data: {"node": "risk_assessment"}

event: node_start
data: {"node": "strategy"}

event: node_complete
data: {"node": "strategy"}

event: node_start
data: {"node": "scoring"}

event: node_complete
data: {"node": "scoring"}

event: node_start
data: {"node": "advisor_copilot"}

event: token
data: {"content": "Based on"}

event: token
data: {"content": " your portfolio"}

... (many more token events)

event: node_complete
data: {"node": "advisor_copilot"}

event: message
data: {"role": "assistant", "content": "Based on your portfolio...", "agent": "advisor_copilot"}

event: state
data: {"risk_metrics": {"sharpe_ratio": 0.82, ...}, "allocation_result": {...}, "scoring_result": {...}}

event: done
data: {"session_id": "550e8400-..."}
```

---

## 5. Document Endpoints

### `POST /api/v1/sessions/{session_id}/documents`

Uploads a PDF document to the session.

**Request**: `multipart/form-data` with `file` field

**Response**:
```json
{
  "doc_id": "64b8f3a2c9d1234567890abc",
  "filename": "portfolio_statement.pdf",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**What happens**:
1. Reads the uploaded file bytes
2. Inserts into MongoDB `raw_documents` collection with `session_id` and `filename`
3. Calls `graph.aupdate_state(config, {"documents_uploaded": [doc_id]})` — appends the new `doc_id` to the LangGraph state's `documents_uploaded` list using the `add_messages`-style reducer behavior
4. On the next user message, `supervisor` detects `documents_uploaded` non-empty and routes to `document_intelligence`

---

### `GET /api/v1/sessions/{session_id}/documents`

Lists all documents uploaded to a session.

**Response**:
```json
[
  {
    "doc_id": "64b8f3a2c9d1234567890abc",
    "filename": "portfolio_statement.pdf",
    "uploaded_at": "2026-06-13T10:30:00Z"
  }
]
```

Note: The response uses `doc_id` and `uploaded_at` (not `id` and `created_at`). The frontend `api.ts` maps these to the `Document` type's `id` and `created_at` fields.

---

## 6. Why Direct SSE for Streaming

The frontend calls `http://localhost:8000/api/v1/sessions/{id}/messages` directly for streaming, bypassing Next.js:

```typescript
// frontend/lib/api.ts
const STREAM_BASE = "http://localhost:8000/api/v1";

// Non-streaming calls go through Next.js proxy
const BASE = "/api/v1";  // proxied by next.config.ts rewrites
```

**Why not use the Next.js proxy for streaming?**

Next.js Route Handlers (`app/api/v1/.../route.ts`) and the `rewrites` proxy in `next.config.ts` both buffer response bodies before forwarding them. For SSE, this means:
- The proxy accumulates all events until the stream closes
- The browser receives nothing until the entire response is complete
- This completely defeats the purpose of streaming

Calling the FastAPI backend directly avoids this buffering. The CORS configuration on the backend (`allow_origins=["*"]`) permits cross-origin SSE connections from `localhost:3000`.

A Next.js Route Handler exists at `app/api/v1/sessions/[sessionId]/messages/route.ts` as a pass-through for server-component fetch calls (if ever needed), but `api.ts`'s `sendMessage` function deliberately uses `STREAM_BASE` to avoid buffering.
