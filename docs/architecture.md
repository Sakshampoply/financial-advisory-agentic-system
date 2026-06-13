# Architecture

This document covers the LangGraph multi-agent graph, the shared state schema, supervisor routing logic, session persistence, and the two-phase SSE streaming protocol.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Graph Topology](#2-graph-topology)
3. [GraphState Schema](#3-graphstate-schema)
4. [Supervisor Routing Logic](#4-supervisor-routing-logic)
5. [Intent-Gated Pipeline Stages](#5-intent-gated-pipeline-stages)
6. [Session Persistence](#6-session-persistence)
7. [SSE Streaming Architecture](#7-sse-streaming-architecture)

---

## 1. Overview

The system is built on **LangGraph** — a stateful graph execution framework built on top of LangChain. The graph is a `StateGraph` where each node is an async Python function that reads from a shared state dict and returns partial state updates.

**PostgreSQL checkpointing** means every graph invocation is committed to the database before moving to the next node. If a node fails midway, the session resumes from the last committed checkpoint rather than restarting. This also enables multi-turn conversations: when a user sends a second message, LangGraph loads the existing state (conversation history, analysis results, profile data) and continues from where it left off.

The framework for the graph is compiled once at application startup and stored on `app.state.graph`. All requests share the same compiled graph instance, isolated by `thread_id` (one per session).

---

## 2. Graph Topology

```
                    ┌─────────────────────────────────────────────────────┐
                    │                                                       │
  User message ──► guardrail_input ──► intent_classifier ──► supervisor ──┤
                         │                                        ▲        │
                         │ (injection detected)                   │        ├──► intake ──────────────────┐
                         ▼                                        │        │                              │
                   error_handler ──► END                          │        ├──► document_intelligence ───┤
                                                                  │        │                              │
                                                         (all nodes         ├──► profile_builder ─────────┤
                                                          route back        │                              │
                                                          to supervisor)    ├──► risk_assessment ──────────┤
                                                                  │        │                              │
                                                                  │        ├──► strategy ─────────────────┤
                                                                  │        │                              │
                                                                  └────────┤──► scoring ──────────────────┤
                                                                           │                              │
                                                                           └──► advisor_copilot ──► guardrail_output ──► END
```

**Entry sequence**: Every message enters at `guardrail_input`, passes through `intent_classifier`, then reaches `supervisor` which performs all routing decisions.

**Hub-and-spoke**: All pipeline nodes (`intake`, `document_intelligence`, `profile_builder`, `risk_assessment`, `strategy`, `scoring`) return control to `supervisor` after completing. The supervisor then decides what runs next based on updated state.

**Exit paths**:
- Normal: `advisor_copilot` → `guardrail_output` → `END`
- Error: `error_handler` → `END`
- Waiting for user input: `supervisor` routes to `END` (graph suspends; next user message resumes)

---

## 3. GraphState Schema

`GraphState` is a `TypedDict` defined in `agents/state.py`. The `messages` field uses LangChain's `add_messages` reducer (append-only); all other fields are last-write-wins.

| Field | Type | Reducer | Purpose |
|-------|------|:-------:|---------|
| `messages` | `list[BaseMessage]` | `add_messages` | Full conversation history including HumanMessage, AIMessage, ToolMessage |
| `session_id` | `str` | last-write | Links to the `advisory_sessions` PostgreSQL row |
| `intent` | `str \| None` | last-write | Current routing intent (`general`, `risk_analysis`, `score_portfolio`, `full_analysis`) |
| `intake_complete` | `bool` | last-write | Set by `intake` or `profile_builder`; gates the quantitative pipeline |
| `user_profile` | `UserProfile \| None` | last-write | Risk tolerance, investment horizon, amount, portfolio weights |
| `documents_uploaded` | `list[str]` | last-write | MongoDB ObjectId strings for uploaded PDFs |
| `documents_extracted` | `bool` | last-write | Prevents `document_intelligence` from running twice |
| `risk_metrics` | `RiskMetrics \| None` | last-write | Sharpe ratio, volatility, max drawdown, risk flags |
| `allocation_result` | `AllocationResult \| None` | last-write | Optimized portfolio weights, expected return/volatility, rationale |
| `scoring_result` | `ScoringResult \| None` | last-write | Composite score (0–100), Sharpe/drawdown/diversification sub-scores |
| `advisor_report_generated` | `bool` | last-write | Prevents `advisor_copilot` from running again without a new human message |
| `iteration_count` | `int` | last-write | Incremented by supervisor each pass; hard limit of 50 prevents infinite loops |
| `error` | `str \| None` | last-write | Set by `guardrail_input` on injection detection; triggers `error_handler` |

**`UserProfile` fields**: `risk_tolerance` (str), `investment_horizon_years` (int), `investment_amount_usd` (float), `annual_income_usd` (float, optional), `tax_bracket` (str, optional), `liquidity_needs` (str, optional), `portfolio` (dict[str, float] — ticker → weight)

---

## 4. Supervisor Routing Logic

`route_supervisor()` in `agents/supervisor.py` evaluates the current state and returns the name of the next node. Priority order (first matching condition wins):

```
1. error set?
      └─► error_handler

2. documents_uploaded non-empty AND documents_extracted == False?
      └─► document_intelligence

3. documents_extracted == True AND "portfolio" not in user_profile?
      └─► profile_builder

4. intent == "general" AND last message is from a human?
      └─► advisor_copilot  (skip all data collection)

5. intent requires intake AND effective_intake_complete == False?
      ├─ last message is human OR (docs extracted AND intake never started)?
      │     └─► intake
      └─ otherwise?
            └─► end  (wait — graph suspends until user responds)

6. risk_metrics absent AND intent needs risk?
      └─► risk_assessment

7. allocation_result absent AND intent needs strategy?
      └─► strategy

8. scoring_result absent AND intent needs scoring?
      └─► scoring

9. new human message after last advisor_copilot response OR no report yet?
      └─► advisor_copilot

10. default?
      └─► end
```

### `effective_intake_complete`

This derived boolean handles the **document bypass case**: a user who uploads a brokerage PDF instead of answering the intake questions. After `document_intelligence` extracts holdings and `profile_builder` normalizes them, the state has `portfolio` data and possibly `investment_amount_usd`, but `intake_complete` (set by the `intake` node) is still `False`.

`effective_intake_complete` returns `True` if either:
- `intake_complete` flag is `True`, OR
- `user_profile` contains all three required fields (`risk_tolerance`, `investment_horizon_years`, `investment_amount_usd`) **plus** the `portfolio` key

This allows the pipeline to proceed without waiting for the `intake` node to run.

### `has_new_human` logic

Before routing to `advisor_copilot` (step 9), the supervisor checks whether there is any `HumanMessage` in the message list that appeared **after** the last `AIMessage` from `advisor_copilot`. This prevents the advisor from producing a duplicate response when the graph resumes for an unrelated reason.

---

## 5. Intent-Gated Pipeline Stages

| Intent | Intake | Risk Assessment | Strategy | Scoring | Notes |
|--------|:------:|:---------------:|:--------:|:-------:|-------|
| `general` | — | — | — | — | Directly to `advisor_copilot` using only RAG context |
| `risk_analysis` | ✓ | ✓ | — | — | Profile + risk metrics only |
| `score_portfolio` | ✓ | ✓ | — | ✓ | Profile + risk + composite score (no optimization) |
| `full_analysis` | ✓ | ✓ | ✓ | ✓ | Complete pipeline |

The intent is set by `intent_classifier` and persists in state. It is re-classified on each new human message (unless intake is currently in progress).

---

## 6. Session Persistence

**Creating a session** (`POST /api/v1/sessions`):
1. Inserts a row into `advisory_sessions` with a new UUID `id` and `langgraph_thread_id`
2. Calls `graph.aupdate_state(config, make_initial_state(session_id))` — this writes the initial state as a LangGraph checkpoint, keyed by `langgraph_thread_id`
3. Returns the session UUID to the frontend

**Subsequent messages** pass `{"configurable": {"thread_id": session.langgraph_thread_id}}` to `graph.astream_events()`. LangGraph loads the last checkpoint for that thread, merges the new input (`HumanMessage`), and continues execution.

**Resuming history** (`GET /api/v1/sessions/{id}/messages`): calls `graph.aget_state(config)` which reads the latest checkpoint and returns the full `messages` list. Tool-call-only messages (e.g., `intake`'s `collect_profile` calls) are filtered out before returning to the frontend.

The `langgraph_thread_id` column is the bridge: the `advisory_sessions` row is the application's record of the session; the LangGraph checkpoint store (also in PostgreSQL) uses `thread_id` as its key.

---

## 7. SSE Streaming Architecture

Streaming happens in two phases, both within a single HTTP response from `POST /api/v1/sessions/{id}/messages`:

**Phase 1 — During graph execution** (`graph.astream_events()` is running):

| Event | Trigger |
|-------|---------|
| `node_start` | `on_chain_start` for any node in `_AGENT_NODES` |
| `node_complete` | `on_chain_end` for any node in `_AGENT_NODES` |
| `token` | `on_chat_model_stream` for `advisor_copilot` only — individual LLM token chunks |

`supervisor` and `intent_classifier` are excluded from `_AGENT_NODES` to reduce UI noise — they run too frequently to be meaningful progress indicators.

**Phase 2 — After graph completes** (graph run has finished, state is committed):

| Event | Content |
|-------|---------|
| `message` | Each new AI message added during this run (by diffing message count before/after) |
| `state` | `risk_metrics`, `allocation_result`, `scoring_result` if any were computed |
| `done` | Signals stream end |

**Why two phases?** Token streaming provides the live typing effect during `advisor_copilot`'s LLM call. But the final committed message content (including `guardrail_output`'s appended disclaimer) only exists after the graph fully completes. The `message` events in Phase 2 carry the canonical final content.

**`sep="\n"` on `EventSourceResponse`**: `sse_starlette` defaults to `\r\n` CRLF line endings. The frontend SSE parser normalizes `\r\n` → `\n` before parsing, and `sep="\n"` ensures the server uses LF-only separators for consistency.
