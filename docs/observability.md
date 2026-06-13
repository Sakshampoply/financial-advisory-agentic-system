# Observability

This document covers the Langfuse integration — setup, the `@traced_node` decorator, LangChain callback integration, span metadata, and eval score submission.

---

## Table of Contents

1. [Why Langfuse](#1-why-langfuse)
2. [Setup](#2-setup)
3. [Initialization Pattern](#3-initialization-pattern)
4. [The `@traced_node` Decorator](#4-the-traced_node-decorator)
5. [LangChain Callback Integration](#5-langchain-callback-integration)
6. [What You See in Langfuse](#6-what-you-see-in-langfuse)
7. [RAG Metadata on advisor_copilot Spans](#7-rag-metadata-on-advisor_copilot-spans)
8. [Eval Score Submission](#8-eval-score-submission)
9. [Langfuse Datasets](#9-langfuse-datasets)

---

## 1. Why Langfuse

Without observability, debugging LLM pipelines requires reading raw logs and guessing which node was slow or which prompt caused a bad response. Langfuse provides:

- **Trace timeline**: Shows every node that ran, in order, with start/end timestamps
- **LLM costs**: Token counts and USD cost per LLM call, nested under the enclosing node span
- **RAG inspection**: The exact chunks retrieved for any given query, visible in the span metadata tab
- **Eval trend tracking**: Named scores submitted from CI tests appear on traces and in dataset runs, enabling regression detection over time

The system works fully without Langfuse — all observability code is wrapped in try/except or conditional checks.

---

## 2. Setup

**Cloud** (easiest):
1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com)
2. Create a project → **Settings** → **API Keys** → copy Public and Secret keys
3. Set in `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

**Self-hosted**: Run the Langfuse Docker stack per their docs, then set `LANGFUSE_HOST` to your instance URL.

**Without Langfuse**: Leave the three keys unset (or empty). The system detects missing keys and disables all tracing gracefully.

---

## 3. Initialization Pattern

**File**: `observability/langfuse_setup.py`

```python
from langfuse import Langfuse
from app.config import settings

_langfuse: Langfuse | None = None

def get_langfuse_client() -> Langfuse | None:
    global _langfuse
    if _langfuse is None and settings.LANGFUSE_PUBLIC_KEY:
        _langfuse = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    return _langfuse
```

**Why explicit credentials instead of `os.environ`**: `pydantic_settings` reads `.env` and populates the `settings` object, but does **not** write values to `os.environ`. If code calls `Langfuse()` without arguments, it falls back to `os.getenv("LANGFUSE_PUBLIC_KEY")` which returns `None` — causing silent authentication failures. Always pass credentials explicitly from `settings`.

**Why not `get_client()`**: `langfuse.get_client()` is a convenience function that reads `os.environ`. Same issue as above — it will silently fail. Use the explicit constructor.

---

## 4. The `@traced_node` Decorator

**File**: `observability/langfuse_setup.py`

```python
from functools import wraps
from langfuse.decorators import observe

def traced_node(name: str):
    def decorator(fn):
        if not get_langfuse_client():
            return fn  # no-op if Langfuse is disabled

        @wraps(fn)
        async def wrapper(*args, **kwargs):
            return await observe(fn, name=name, as_type="agent")(*args, **kwargs)

        return wrapper
    return decorator
```

**Usage** on every agent node:
```python
@traced_node("risk_assessment")
async def risk_assessment_node(state: GraphState) -> dict:
    ...
```

**`as_type="agent"`**: Creates a Langfuse span of type "agent" in the trace timeline, visually distinct from LLM call spans.

**`@wraps(fn)`**: Preserves the original function's `__name__`, `__doc__`, and signature. LangGraph uses `fn.__name__` internally for some operations — without `@wraps`, introspection would break.

**No-op when disabled**: If `get_langfuse_client()` returns `None`, the decorator returns the original function unchanged. Zero overhead.

**Applied to**: `intent_classifier`, `intake`, `document_intelligence`, `profile_builder`, `risk_assessment`, `strategy`, `scoring`, `advisor_copilot`. Not applied to `supervisor` (runs too frequently, low per-call cost) or `guardrail_*` (short pure-Python, no LLM).

---

## 5. LangChain Callback Integration

**File**: `observability/langfuse_setup.py` + `llm/client.py`

```python
# langfuse_setup.py
from langfuse.callback import CallbackHandler

def get_langfuse_callbacks() -> list:
    client = get_langfuse_client()
    if client is None:
        return []
    return [CallbackHandler()]

# llm/client.py
from app.observability.langfuse_setup import get_langfuse_callbacks

def get_chat_model(streaming: bool = False, temperature: float = 0.7):
    return ChatOpenAI(
        model=settings.OPENROUTER_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.OPENROUTER_API_KEY,
        streaming=streaming,
        temperature=temperature,
        callbacks=get_langfuse_callbacks(),
    )
```

The `CallbackHandler` intercepts LangChain's LLM call lifecycle hooks (`on_llm_start`, `on_llm_end`, `on_llm_error`) and reports:
- **Prompt tokens** — number of input tokens sent
- **Completion tokens** — number of output tokens received
- **Total cost (USD)** — computed from OpenRouter's token pricing
- **Model name** — as reported by the API response
- **Latency** — time between `on_llm_start` and `on_llm_end`

These appear as child spans nested **inside** the enclosing `@traced_node` span. For example, `risk_assessment`'s LLM call for risk flags appears as a child of the `risk_assessment` agent span.

---

## 6. What You See in Langfuse

A typical full-analysis trace for one user message looks like:

```
[Trace] User message: "Analyze my portfolio and give me recommendations"
│
├── [Agent] guardrail_input         (3ms — no LLM)
├── [Agent] intent_classifier        (480ms)
│   └── [LLM] openrouter/llama-3.3  (480ms — 312 input tokens, 8 output tokens, $0.00004)
├── [Agent] supervisor               (1ms — pure routing, no LLM)
├── [Agent] intake                   (1,240ms)
│   └── [LLM] openrouter/llama-3.3  (1,240ms — tool-use, 892 tokens)
├── [Agent] supervisor               (1ms)
├── [Agent] risk_assessment          (3,800ms)
│   ├── [Tool] yfinance OHLCV fetch  (background — not a LangChain tool)
│   └── [LLM] openrouter/llama-3.3  (640ms — risk flags, 1,104 tokens, $0.00013)
├── [Agent] strategy                 (2,200ms)
│   └── [LLM] openrouter/llama-3.3  (580ms — rationale, 712 tokens)
├── [Agent] scoring                  (2ms — no LLM)
├── [Agent] supervisor               (1ms)
└── [Agent] advisor_copilot          (4,100ms)
    ├── RAG retrieval               (metadata: rag_query, chunks_retrieved=6, chunks=[...])
    └── [LLM] openrouter/llama-3.3  (3,800ms — full advisory response, 2,841 tokens, $0.00034)
```

**Total trace cost**: visible at the trace level, summing all LLM child spans. Useful for estimating per-user cost.

---

## 7. RAG Metadata on advisor_copilot Spans

The `advisor_copilot` node logs its RAG retrieval directly to the current Langfuse span:

```python
# In advisor_copilot.py
try:
    from langfuse import get_client as _get_langfuse
    _get_langfuse().update_current_span(
        metadata={
            "rag_query": rag_query,
            "rag_chunks_retrieved": len(rag_chunks),
            "rag_chunks": rag_chunks,  # full chunk text with [Source: ...] prefixes
        }
    )
except Exception:
    pass  # silently skip if Langfuse is disabled
```

In the Langfuse UI, navigate to the `advisor_copilot` span → **Metadata tab** to see:
- The exact query string sent to the retriever
- How many chunks were retrieved (0–6)
- The full text of each chunk (including which file it came from)

This is invaluable for debugging retrieval quality: if the advisor's response seems poorly grounded, you can inspect exactly what context it had available.

---

## 8. Eval Score Submission

Test files submit named scores to Langfuse after running evaluations, linking CI test results to production trace data.

**Pattern** (from `evals/conftest.py`):
```python
trace_id = lf.create_trace_id()  # new trace for this eval run

lf.create_score(
    trace_id=trace_id,
    name="advisor.faithfulness",
    value=faithfulness_metric.score,   # 0.0 – 1.0
    data_type="NUMERIC",
    comment=f"deepeval FaithfulnessMetric, threshold=0.7",
)
```

Scores appear in the Langfuse dashboard under **Scores** and can be charted over time to detect regressions.

**Langfuse SDK v4 notes**:
- `lf.trace()` was removed — use `lf.create_trace_id()` + `lf.create_score()`
- `lf.create_score()` requires `data_type` parameter for numeric scores

---

## 9. Langfuse Datasets

**File**: `evals/setup_langfuse_datasets.py`

Run once to create reusable evaluation datasets in Langfuse:

```bash
uv run python evals/setup_langfuse_datasets.py
```

| Dataset name | Contents | Used by |
|-------------|---------|---------|
| `rag_faithfulness_v1` | RAG query → expected source filenames | `test_rag_retrieval.py` |
| `rag_retrieval_quality_v2` | Queries with MRR ground truth rankings | `test_rag_retrieval.py` |
| `advisor_quality_v1` | 4 advisor intent test cases | `test_advisor_copilot.py` |
| `advisor_quality_v2` | 4 additional cases with stricter requirements | `test_advisor_copilot.py` |
| `intake_extraction_v1` | Field extraction accuracy test cases | `test_intake_extraction.py` |

These datasets appear in the Langfuse UI under **Datasets** and can be used for:
- Manual evaluation runs via the Langfuse playground
- Dataset experiments that track score trends across model changes
- A/B testing different prompts against the same benchmark cases
