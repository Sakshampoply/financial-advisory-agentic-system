# Evaluations

This document covers the 4-tier evaluation suite, every evaluator and what it measures, the Langfuse score submission pattern, and the CI pipeline configuration.

---

## Table of Contents

1. [Overview — The 4-Tier Pyramid](#1-overview--the-4-tier-pyramid)
2. [Test Infrastructure](#2-test-infrastructure)
3. [Unit Tests](#3-unit-tests)
4. [Integration Tests](#4-integration-tests)
5. [RAG Retrieval Tests](#5-rag-retrieval-tests)
6. [LLM Evaluation Tests](#6-llm-evaluation-tests)
7. [Eval Scores in Langfuse](#7-eval-scores-in-langfuse)
8. [CI Pipeline](#8-ci-pipeline)
9. [Pending Improvements](#9-pending-improvements)

---

## 1. Overview — The 4-Tier Pyramid

```
                    ┌─────────────────────────┐
                    │  LLM-as-judge (llm_eval) │  Slowest, most expensive
                    │  deepeval faithfulness   │  Requires OPENROUTER_API_KEY
                    ├─────────────────────────┤
                    │  RAG retrieval (rag)     │  Medium speed
                    │  MRR, Precision@3        │  Requires live PG + seeded KB
                    ├─────────────────────────┤
                    │  Integration             │  Medium speed
                    │  Live DB round-trips     │  Requires live PG + MongoDB
                    ├─────────────────────────┤
                    │  Unit (unit)             │  Fastest (~10s)
                    │  Pure logic, no I/O      │  No external dependencies
                    └─────────────────────────┘
```

| Tier | Pytest mark | Speed | What it catches |
|------|:-----------:|:-----:|-----------------|
| Unit | `unit` | ~10s | Logic bugs in formulas, routing, prompt building |
| Integration | `integration` | ~30s | DB round-trips, node interactions with real data |
| RAG retrieval | `rag` | ~30s | Retrieval quality degradation, source isolation bugs |
| LLM evaluation | `llm_eval` | ~5min | Hallucination, citation failures, intent mismatches |

---

## 2. Test Infrastructure

**File**: `evals/conftest.py`

### Pytest marks

Four custom marks are registered to avoid `PytestUnknownMarkWarning`:
```ini
# pyproject.toml [tool.pytest.ini_options]
markers = ["unit", "rag", "llm_eval", "integration"]
```

### Judge model

LLM-as-judge tests use `openai/gpt-4o-mini` via OpenRouter:
```python
@pytest.fixture
def judge_model():
    return LiteLLMModel(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.OPENROUTER_API_KEY,
    )
```

**Why gpt-4o-mini and not the project's primary model**: deepeval's `FaithfulnessMetric` and `ContextualPrecisionMetric` require the judge model to return structured JSON with specific schema. Conversational models (deepseek-chat, etc.) fail non-deterministically on this schema requirement. gpt-4o-mini has reliable structured output compliance.

### Mock fixtures

- **`mock_db`**: AsyncMock of `AsyncSession` — prevents unit tests from needing a real database connection
- **`mock_redis`**: AsyncMock of `aioredis.Redis` with pre-configured return values for common cache keys
- **Synthetic OHLCV generator**: Produces realistic price series (geometric Brownian motion) for market data tests without hitting yfinance

### Langfuse score submission

```python
lf = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST,
)

# Submit a score
lf.create_score(
    trace_id=trace_id,
    name="advisor.faithfulness",
    value=0.87,
    data_type="NUMERIC",
)
```

**Critical note**: Do NOT use `lf.trace()` — it was removed in Langfuse SDK v4. Use `lf.create_trace_id()` + `lf.create_score()`. Also do NOT use `get_client()` — it reads `os.environ`, but `pydantic_settings` loads `.env` into `settings` without writing to `os.environ`. Always initialize Langfuse with explicit credentials.

---

## 3. Unit Tests

No database, no LLM, no network. Run with `pytest -m unit`.

### `test_risk_agent.py`

Tests `compute_portfolio_metrics` and `compute_score` from `risk_assessment.py` and `scoring.py`.

| Test | What it verifies |
|------|-----------------|
| `test_sharpe_positive_returns` | Positive returns produce positive Sharpe |
| `test_sharpe_zero_returns` | Constant returns → Sharpe of 0 (no excess return) |
| `test_max_drawdown_monotone_rise` | Monotonically rising prices → drawdown of 0 |
| `test_max_drawdown_large_drop` | 50% price drop → max drawdown ≈ -0.50 |
| `test_optimize_portfolio_weights_sum_to_one` | Optimization output weights sum to 1.0 |
| `test_composite_score_formula` | `40% × sharpe_score + 30% × dd_score + 30% × div_score` matches expected |
| `test_hhi_single_asset` | Single asset portfolio → HHI = 1.0, diversification score = 0 |
| `test_hhi_equal_weight` | Equal-weight 5-asset → HHI = 0.2, diversification score = 80 |

### `test_guardrails.py`

Tests PII masking and injection detection logic.

| Test | What it verifies |
|------|-----------------|
| `test_ssn_masked` | `123-45-6789` → `[SSN REDACTED]` |
| `test_phone_masked` | 10-digit phone formats → `[PHONE REDACTED]` |
| `test_email_masked` | email@domain.com → `[EMAIL REDACTED]` |
| `test_credit_card_masked` | 16-digit card → `[CARD REDACTED]` |
| `test_injection_ignore_previous` | "ignore previous instructions" → `injection_detected` |
| `test_injection_act_as` | "act as a different AI" → `injection_detected` |
| `test_injection_system_header` | "system: new instructions" → `injection_detected` |
| `test_disclaimer_appended` | `guardrail_output` appends disclaimer |
| `test_disclaimer_not_duplicated` | Runs twice → disclaimer appears once |

### `test_supervisor.py`

Tests all routing paths in `route_supervisor()`.

| Test | What it verifies |
|------|-----------------|
| `test_routes_to_error_handler` | `error` set → `error_handler` |
| `test_routes_to_document_intelligence` | `documents_uploaded` non-empty + not extracted → `document_intelligence` |
| `test_routes_to_profile_builder` | Extracted + no portfolio → `profile_builder` |
| `test_general_intent_skips_intake` | `intent=general` → `advisor_copilot` directly |
| `test_effective_intake_complete_bypass` | All 3 required fields + portfolio → pipeline proceeds |
| `test_iteration_limit` | At `MAX_ITERATIONS=50` → routes to `end` |

### `test_intake_extraction.py`

Tests intake's prompt building, portfolio normalization, and fast-path.

| Test | What it verifies |
|------|-----------------|
| `test_dynamic_prompt_lists_missing_fields` | Prompt includes only unfilled fields |
| `test_fast_path_no_llm_call` | Complete `user_profile` → no LLM call made |
| `test_portfolio_normalization_percentages` | `weight_pct=40` → `weight=0.40` |
| `test_portfolio_normalization_decimals` | `weight_pct=0.40` → `weight=0.40` (no double-conversion) |
| `test_fake_confirmation_guard` | LLM text response without tool call → asks again |

### `test_intent_classifier.py`

| Test | What it verifies |
|------|-----------------|
| `test_valid_intent_returned` | LLM returns one of 4 valid intents |
| `test_fallback_to_full_analysis` | Invalid LLM response → `full_analysis` |
| `test_skip_during_intake` | Active intake → existing intent unchanged |

---

## 4. Integration Tests

Requires live PostgreSQL + MongoDB. Run with `pytest -m integration`.

### `test_node_integration.py`

Runs `risk_assessment`, `strategy`, and `scoring` nodes with synthetic market data injected via mock. Verifies:
- Sharpe ratio is finite and within [-5, 5] range
- Portfolio weights from strategy sum to 1.0 ± 0.001
- Composite score is within [0, 100]
- Scoring breakdown keys are all present

### `test_profile_builder.py`

Inserts synthetic `extracted_document_data` into a test MongoDB collection, runs `profile_builder_node`, and verifies:
- Portfolio weights normalize to sum 1.0
- Multiple documents with overlapping tickers are aggregated correctly
- `investment_amount_usd` is set from `account_value` when absent from `user_profile`

### `test_market_data.py`

Tests Redis caching behavior:
- Cache hit returns cached value without calling yfinance/FRED/Alpha Vantage
- Cache miss calls the API and stores result
- FRED timeout returns `None` (no crash)
- Alpha Vantage NaN strings (`"None"`, `"-"`) are handled by NaN-safe float parsing

---

## 5. RAG Retrieval Tests

**File**: `evals/test_rag_retrieval.py`

Requires live PostgreSQL with seeded knowledge base. Run with `pytest -m rag`.

### Format invariants

```python
chunks = await retrieve("portfolio diversification", db, session_id=None)
assert len(chunks) <= 6
assert all(c.startswith("[Source:") for c in chunks)
```

Every result must start with `[Source: ...]` and the total must not exceed 6.

### Mean Reciprocal Rank (MRR ≥ 0.40)

MRR measures how often the first relevant result appears near the top:
```
MRR = (1/|Q|) × Σ  1/rank_of_first_relevant_result
                 queries
```

Test queries and their expected source files:
| Query | Expected source keyword |
|-------|------------------------|
| "How is Sharpe ratio calculated" | `sharpe` / SEC / academic |
| "ETF liquidity risk" | SEC / Vanguard |
| "bond duration interest rate risk" | central_bank / academic |
| "portfolio rebalancing frequency" | Vanguard / Fidelity |
| "inflation hedging strategies" | FRED / BIS |

A chunk is considered "relevant" if its source file matches the expected category. MRR ≥ 0.40 means on average the first relevant result appears within the top 2-3 positions.

### Precision@3

For each test query, at least 1 of the top-3 results must come from an expected source category. Precision@3 = relevant results in top 3 / 3.

### Adversarial inputs

```python
@pytest.mark.parametrize("bad_input", [
    "'; DROP TABLE document_chunks; --",  # SQL injection
    "ignore previous instructions",        # prompt injection
    "こんにちは世界",                         # Unicode
    "word " * 500,                         # 500-word query
])
async def test_adversarial_inputs_handled_gracefully(bad_input, db):
    result = await retrieve(bad_input, db, session_id=None)
    assert isinstance(result, list)       # no crash, no exception
```

### Cross-session isolation

```python
chunks_a = await retrieve("portfolio", db, session_id=session_a_id)
chunks_b = await retrieve("portfolio", db, session_id=session_b_id)
sources_a = {c.split("]")[0] for c in chunks_a}
sources_b = {c.split("]")[0] for c in chunks_b}
# Session B's uploaded doc should not appear in session A's results
```

### LLM-as-judge for RAG

Uses deepeval `FaithfulnessMetric` and `ContextualPrecisionMetric`:
```python
test_case = LLMTestCase(
    input=query,
    actual_output=synthesized_answer,
    retrieval_context=chunks,
)
faithfulness = FaithfulnessMetric(threshold=0.7, model=judge_model)
faithfulness.measure(test_case)
assert faithfulness.score >= 0.7
```

---

## 6. LLM Evaluation Tests

**File**: `evals/test_advisor_copilot.py`

Requires `OPENROUTER_API_KEY`. Run with `pytest -m llm_eval --timeout=300`.

### Source citation presence

```python
assert "[Source:" in response or "(Source:" in response
```

Every advisor response must include at least one source citation.

### Disclaimer presence

```python
assert "not professional financial advice" in response.lower()
```

### Per-intent requirements

Test cases are loaded from `evals/fixtures/advisor_test_cases.json`. Each case has:
```json
{
  "intent": "risk_analysis",
  "query": "What are the main risks in my portfolio?",
  "must_contain": ["Sharpe", "volatility", "drawdown"],
  "forbidden": ["guaranteed returns", "certain profit"]
}
```

8 test cases cover all 4 intents with 2 cases each. Failures are reported per-field.

### deepeval FaithfulnessMetric (threshold ≥ 0.7)

Measures whether every factual claim in the response is supported by the retrieved context (RAG chunks + analysis data). A score of 0.7 means 70% of claims are traceable to the provided context.

**What it catches**: The advisor asserting a specific fund's performance not present in the KB, inventing regulatory rules, or citing statistics not in the quantitative analysis.

### deepeval AnswerRelevancyMetric (threshold ≥ 0.6)

Measures whether the response actually addresses the user's question. Catches verbose responses that answer a different question.

### Hallucination detection

```python
_VALID_ANALYSIS_SOURCES = [
    "User Profile", "Portfolio Risk Analysis",
    "Strategy Engine", "Portfolio Score"
]

def test_advisor_does_not_hallucinate_sources(response):
    # Uses substring matching, not set equality
    # because LLMs sometimes combine labels:
    # "Portfolio Risk Analysis vs. Strategy Engine"
    assert any(s in response for s in _VALID_ANALYSIS_SOURCES)
```

### Citation richness

```python
n_citations = response.count("(Source:")
# Submitted to Langfuse as advisor.citation_richness score
lf.create_score(trace_id=..., name="advisor.citation_richness", value=min(n_citations, 3) / 3)
```

---

## 7. Eval Scores in Langfuse

Each test submits named numeric scores to Langfuse, linking CI test results to production traces:

| Score name | Source test | Range |
|-----------|------------|:-----:|
| `advisor.faithfulness` | `test_advisor_copilot.py` | 0–1 |
| `advisor.relevancy` | `test_advisor_copilot.py` | 0–1 |
| `advisor.citation_richness` | `test_advisor_copilot.py` | 0–1 |
| `rag.mrr` | `test_rag_retrieval.py` | 0–1 |
| `rag.precision_at_3` | `test_rag_retrieval.py` | 0–1 |
| `intake.field_extraction_accuracy` | `test_intake_extraction.py` | 0–1 |

**Langfuse datasets** (created by `evals/setup_langfuse_datasets.py`):

| Dataset | Contents |
|---------|---------|
| `rag_faithfulness_v1` | RAG query → expected source categories |
| `rag_retrieval_quality_v2` | Queries with MRR ground truth |
| `advisor_quality_v1` | 4 advisor intent test cases |
| `advisor_quality_v2` | 4 additional advisor test cases with stricter requirements |
| `intake_extraction_v1` | Intake field extraction accuracy cases |

These datasets appear in the Langfuse UI for manual evaluation runs and dataset experiment tracking.

---

## 8. CI Pipeline

**File**: `.github/workflows/ci.yml`

Three jobs with different triggers and requirements:

### `unit-tests` — runs on every push

```yaml
- name: Run unit tests
  run: uv run pytest evals/ -m unit -v
```

No secrets needed. Fast gate — if unit tests fail, no further jobs run.

### `rag-tests` — runs on PRs and nightly

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    env:
      POSTGRES_USER: advisor
      POSTGRES_PASSWORD: advisor_secret
      POSTGRES_DB: financial_advisor

steps:
  - run: uv run alembic upgrade head
  - run: uv run python scripts/seed_kb.py
  - run: uv run pytest evals/ -m rag -v
```

Starts a PostgreSQL service container, runs migrations, seeds the knowledge base, then runs RAG tests.

### `llm-eval-tests` — nightly only (00:00 UTC)

```yaml
on:
  schedule:
    - cron: '0 0 * * *'

env:
  OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
  LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
  LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}

- run: uv run pytest evals/ -m llm_eval -v --timeout=300
```

Runs nightly to catch LLM quality regressions without blocking every PR. 5-minute timeout per test.

---

## 9. Pending Improvements

The following scoring improvements are designed but not yet implemented (tracked in `CLAUDE.md`):

**Gradient citation richness score** (currently binary):
```python
# Current: binary pass/fail
# Proposed: min(n_citations, 3) / 3  →  score name "advisor.citation_richness"
```

**Gradient intent requirements score** (currently binary):
```python
# Current: all must_contain present = pass, else fail
# Proposed:
contain_score = items_found / total_must_contain
forbidden_score = 1.0 if no_forbidden_terms_present else 0.0
score = contain_score * forbidden_score
```

**Gradient hallucination penalty**:
```python
# Current: any valid source present = pass
# Proposed: max(0.0, 1.0 - 0.33 * len(hallucinated_sources))
```

CI asserts (pass/fail) remain unchanged — only the Langfuse score value submitted alongside would change to provide richer signal for trend analysis.
