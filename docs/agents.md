# Agents

Detailed breakdown of all 12 agents in the system. Each section covers purpose, trigger condition, key implementation details, and state outputs.

For routing logic and how agents connect, see [architecture.md](architecture.md).

---

## Table of Contents

1. [guardrail_input](#1-guardrail_input)
2. [intent_classifier](#2-intent_classifier)
3. [supervisor](#3-supervisor)
4. [intake](#4-intake)
5. [document_intelligence](#5-document_intelligence)
6. [profile_builder](#6-profile_builder)
7. [risk_assessment](#7-risk_assessment)
8. [strategy](#8-strategy)
9. [scoring](#9-scoring)
10. [advisor_copilot](#10-advisor_copilot)
11. [guardrail_output](#11-guardrail_output)
12. [error_handler](#12-error_handler)

---

## 1. guardrail_input

**File**: `agents/guardrails/input_guard.py`

**Purpose**: Sanitizes every incoming user message before it reaches any LLM. Masks PII and blocks prompt injection attempts.

**Trigger**: Entry point — runs on every message before anything else.

**PII masking** (regex substitution, replaced with placeholder text):
- SSN: `\b\d{3}-\d{2}-\d{4}\b` → `[SSN REDACTED]`
- Phone: 10-digit patterns with separators → `[PHONE REDACTED]`
- Email: standard email pattern → `[EMAIL REDACTED]`
- Credit card: 16-digit groups → `[CARD REDACTED]`

**Prompt injection detection** (11 patterns checked against the lowercased message):
- `ignore previous instructions` / `disregard previous` / `forget previous`
- `act as` / `you are now` / `pretend you are`
- `system:` header patterns
- `new instructions` / `override`
- Attempts to close a system prompt: `` ``` `` followed by instruction keywords

**Idempotent message replacement**: When PII is found or injection is detected, the node replaces the original `HumanMessage` with a new one carrying the **same `id`**. LangChain's `add_messages` reducer deduplicates by ID, so the sanitized message overwrites the original rather than appending a duplicate.

**On injection detection**: Sets `error: "injection_detected"` in state, which causes supervisor to route to `error_handler` instead of continuing.

**Outputs**: Updated `messages` (with sanitized content), optionally `error: "injection_detected"`

---

## 2. intent_classifier

**File**: `agents/intent_classifier.py`

**Purpose**: Classifies the user's latest message into one of 4 intents. The intent gates which pipeline stages run.

**Trigger**: After `guardrail_input`, before `supervisor`, on every message.

**Intents**:
| Intent | When to classify |
|--------|-----------------|
| `general` | General financial knowledge question, no portfolio analysis needed |
| `risk_analysis` | User wants to understand portfolio risk (Sharpe, drawdown, volatility) |
| `score_portfolio` | User wants their portfolio scored or evaluated |
| `full_analysis` | Complete investment plan, allocation recommendations, or general "analyze my portfolio" |

**Skip condition**: If intake is currently in progress (the user is answering intake's questions, not expressing a new intent), re-classification is skipped to avoid overwriting the active intent mid-conversation. Detection: intake is in progress when `intake_complete` is False AND the last `AIMessage` (if any) came from the `intake` node.

**Temperature**: `0.0` — classification should be deterministic, not creative.

**Fallback**: If the LLM returns an unrecognized string, falls back to `full_analysis`. This is the safest default because it runs the complete pipeline.

**Outputs**: `intent` (str)

---

## 3. supervisor

**File**: `agents/supervisor.py`

**Purpose**: Pure routing node — reads state and returns the next node name. Does no LLM work.

**Trigger**: After `intent_classifier` (first pass) and after every pipeline node returns.

**Only state change**: Increments `iteration_count` by 1 each pass. The `MAX_ITERATIONS = 50` guard prevents infinite loops in pathological cases.

**All routing decisions** are in `route_supervisor()` — see [architecture.md → Supervisor Routing Logic](architecture.md#4-supervisor-routing-logic) for the full priority order.

**Outputs**: `iteration_count` (incremented). The routing decision itself is returned as the LangGraph conditional edge value, not a state update.

---

## 4. intake

**File**: `agents/intake.py`

**Purpose**: Collects the user's financial profile via a structured conversation. Uses LLM tool-use to extract fields rather than free-form parsing.

**Trigger**: When `effective_intake_complete` is False and the intent requires data collection.

**Tool schema** — `collect_profile`:
```
risk_tolerance: str          (required) — "conservative", "moderate", "aggressive"
investment_horizon_years: int (required) — e.g. 10
investment_amount_usd: float  (required) — e.g. 50000
annual_income_usd: float      (optional)
tax_bracket: str              (optional)
liquidity_needs: str          (optional)
existing_portfolio: list      (optional) — [{ticker, weight_pct}, ...]
wants_new_portfolio: bool     (optional)
```

**Dynamic system prompt**: Builds a prompt that lists only the fields still missing from the current `user_profile`. Never asks the user to re-confirm data already extracted from uploaded documents.

**Fast-path**: If all 4 required fields (risk_tolerance, investment_horizon_years, investment_amount_usd, plus portfolio choice) are already in `user_profile` when `intake_node` is called (e.g., pre-populated by `profile_builder`), the node returns `intake_complete: True` immediately without making any LLM call.

**Fake-confirmation guard**: If the LLM responds with text like "I've captured your profile" but does NOT call the `collect_profile` tool, intake detects this (no `ToolCall` in the response) and asks the user again. This prevents the LLM from prematurely ending the intake conversation.

**Portfolio gate**: If the LLM calls `collect_profile` with risk/horizon/amount but without portfolio info (and `wants_new_portfolio` is also absent), intake saves the partial profile and explicitly asks "What is your current portfolio?" Sets `intake_complete: True` only when all required fields plus portfolio choice are confirmed.

**`_build_portfolio`**: Normalizes `existing_portfolio` from the tool call format `[{ticker, weight_pct}]` to a weights dict `{ticker: weight}` that sums to 1.0. Handles both percentage (e.g., 40.0) and decimal (e.g., 0.40) formats.

**Outputs**: Updated `user_profile`, `intake_complete: True` (when complete), `messages` (intake's conversational response)

---

## 5. document_intelligence

**File**: `agents/document_intelligence.py`

**Purpose**: Extracts structured portfolio data (holdings, account value, institution) from uploaded PDF brokerage statements.

**Trigger**: When `documents_uploaded` is non-empty and `documents_extracted` is False.

**Process**:
1. Reads each PDF binary from MongoDB `raw_documents` collection by ObjectId
2. Extracts text with `pypdf`, limited to 12,000 characters (covers most brokerage statement pages)
3. Calls LLM with `extract_document_data` tool schema:
   ```
   holdings: [{ticker, shares, current_value}, ...]
   account_value: float
   account_type: str    (e.g., "Individual Brokerage", "IRA")
   institution: str     (e.g., "Fidelity")
   ```
4. Checks MongoDB `extracted_document_data` collection — skips any `doc_id` already extracted (idempotent)
5. Saves extraction result to `extracted_document_data` with the `session_id`

**Idempotency**: The node processes only `doc_id`s not yet in `extracted_document_data`. Running twice on the same session is safe.

**Outputs**: `documents_extracted: True`, `messages` (confirmation to user)

---

## 6. profile_builder

**File**: `agents/profile_builder.py`

**Purpose**: Converts extracted document data into a normalized portfolio weight dict and merges it into `user_profile`.

**Trigger**: When `documents_extracted` is True and `portfolio` key is absent from `user_profile`.

**Aggregation logic**:
1. Reads all `extracted_document_data` documents for the current `session_id` from MongoDB
2. Aggregates holdings across multiple documents by summing values per ticker: `{ticker: total_value}`
3. Computes total portfolio value and normalizes to weights: `{ticker: value / total}`
4. Uses the largest `account_value` across all documents as `investment_amount_usd` if not already set in `user_profile`

**`intake_complete` promotion**: If the merged `user_profile` now contains all 3 required fields (`risk_tolerance`, `investment_horizon_years`, `investment_amount_usd`), `profile_builder` sets `intake_complete: True`. This handles the document bypass case — intake collected the non-portfolio fields via conversation, and the document provided the portfolio. Without this, the supervisor would loop back to `intake` indefinitely even though all data is present.

**Always sets `portfolio` key**: Even if no holdings were found (empty document), sets `user_profile["portfolio"] = {}` to prevent the supervisor from routing to `profile_builder` repeatedly.

**Outputs**: Updated `user_profile` (with `portfolio` key), conditionally `intake_complete: True`

---

## 7. risk_assessment

**File**: `agents/risk_assessment.py`

**Purpose**: Computes quantitative portfolio risk metrics using real market data. The only LLM call is for qualitative risk flag descriptions.

**Trigger**: When `risk_metrics` is absent and intent requires risk data.

**Math isolation rule**: All numeric calculations use pandas/numpy/PyPortfolioOpt only. The LLM is called exclusively for `risk_flags: list[str]` (qualitative descriptions).

**Quantitative pipeline**:

1. **Fetch OHLCV**: Gets 2-year daily price history for each portfolio ticker via yfinance. Falls back to `DEFAULT_UNIVERSE = [SPY, QQQ, BND, GLD, VNQ]` if fewer than 2 valid tickers are retrieved (e.g., all user tickers are invalid or delisted).

2. **Compute daily returns**: `prices.pct_change().dropna()`

3. **Sharpe ratio** (annualized):
   ```
   excess_daily = daily_returns - (risk_free_rate / 252)
   sharpe = mean(excess_daily) / std(excess_daily) × √252
   ```
   Risk-free rate from FRED DGS3MO (3-month Treasury yield).

4. **Annualized volatility**:
   ```
   vol = portfolio_daily_std × √252
   ```
   Where `portfolio_daily_std` uses portfolio weights and the covariance matrix.

5. **Max drawdown**:
   ```
   cumulative = (1 + daily_returns).cumprod()
   running_max = cumulative.cummax()
   drawdown = (cumulative - running_max) / running_max
   max_drawdown = drawdown.min()  (negative value)
   ```

6. **Concurrent macro data** (asyncio.gather): FRED risk-free rate, yield curve spread (T10Y2Y), CPI inflation (YoY CPIAUCSL).

7. **Fundamentals**: Top-5 portfolio holdings by weight — weighted-average beta, per-ticker P/E ratio via Alpha Vantage.

8. **News sentiment**: Top-3 holdings — relevance-weighted sentiment score from Alpha Vantage NEWS_SENTIMENT endpoint.

9. **LLM tool-use** for `report_risk_flags`: Passes the computed metrics as context; LLM returns 3-5 qualitative flag strings (e.g., "High concentration in technology sector", "Negative Sharpe ratio suggests underperformance vs. risk-free rate").

**Outputs**: `risk_metrics` (RiskMetrics typed dict with sharpe_ratio, volatility, max_drawdown, risk_flags, beta, yield_curve, inflation)

---

## 8. strategy

**File**: `agents/strategy.py`

**Purpose**: Computes an optimized portfolio allocation using mean-variance optimization.

**Trigger**: When `allocation_result` is absent and intent is `full_analysis`.

**Math isolation rule**: Portfolio optimization uses PyPortfolioOpt only. The LLM call is for `strategy_rationale` (2-3 sentence explanation).

**Optimization pipeline**:

1. **Expected returns**: `expected_returns.mean_historical_return(prices)` — annualized mean return from 2-year price history

2. **Covariance matrix**: `risk_models.sample_cov(prices)` — sample covariance of daily returns, annualized

3. **Max-Sharpe optimization**:
   ```python
   ef = EfficientFrontier(mu, S)
   ef.max_sharpe(risk_free_rate=rf)
   weights = ef.clean_weights(cutoff=1e-4)  # removes negligible allocations
   ```

4. **Fallback**: If optimization fails (e.g., singular covariance matrix, all-zero returns), falls back to equal weights across all tickers.

5. **Performance metrics**: `ef.portfolio_performance()` returns expected annual return and volatility for the optimized portfolio.

6. **LLM tool-use** for `report_rationale`: Explains the allocation decision in 2-3 sentences based on the weights, expected return, and volatility.

**Outputs**: `allocation_result` (AllocationResult with weights dict, expected_return, expected_volatility, strategy_rationale)

---

## 9. scoring

**File**: `agents/scoring.py`

**Purpose**: Computes a composite 0–100 portfolio health score from the risk metrics. Fully deterministic — no LLM call.

**Trigger**: When `scoring_result` is absent and intent requires scoring (`score_portfolio` or `full_analysis`).

**Scoring formulas**:

**Sharpe score** — maps the Sharpe ratio range [-2, 2] linearly to [0, 100]:
```
sharpe_score = clamp((sharpe_ratio + 2) / 4 × 100, 0, 100)
```
A Sharpe of 0.0 scores 50. A Sharpe of 1.0 scores 75.

**Drawdown score** — maps max drawdown range [-0.5, 0] linearly to [0, 100]:
```
drawdown_score = clamp((1 + max_drawdown / 0.5) × 100, 0, 100)
```
A drawdown of 0% scores 100. A drawdown of -50% or worse scores 0.

**Diversification score** — Herfindahl–Hirschman Index (HHI) based:
```
HHI = Σ(wᵢ²)  for all portfolio weights wᵢ
diversification_score = (1 − HHI) × 100
```
A perfectly equal-weight 10-asset portfolio: HHI = 0.10, score = 90. A single-asset portfolio: HHI = 1.0, score = 0.

**Composite score** (weighted average):
```
composite = 0.40 × sharpe_score + 0.30 × drawdown_score + 0.30 × diversification_score
```

**Weight rationale**:
- 40% Sharpe — risk-adjusted return is the primary signal of portfolio quality
- 30% Drawdown — protects against catastrophic loss scenarios
- 30% Diversification — a hygiene check; heavily concentrated portfolios carry uncompensated idiosyncratic risk

**Outputs**: `scoring_result` (ScoringResult with composite_score and breakdown dict)

---

## 10. advisor_copilot

**File**: `agents/advisor_copilot.py`

**Purpose**: Generates the final advisory response, grounded in RAG-retrieved knowledge base excerpts and the quantitative analysis results.

**Trigger**: When supervisor determines a new advisory response is needed (new human message after last advisor response, or first run).

**`_build_context`**: Assembles analysis results into labelled text blocks for injection into the system prompt:
```
[Source: User Profile]
User profile:
  Risk tolerance: moderate
  Horizon: 10 years
  ...

[Source: Portfolio Risk Analysis]
Risk analysis:
  Sharpe: 0.82 | Vol: 18.3% | Max drawdown: -24.1%
  ...
```
The `[Source: ...]` labels allow the LLM to cite data provenance in its response.

**`_retrieve_context`**: Runs hybrid RAG retrieval (see [rag.md](rag.md)) for the latest `HumanMessage` query. Returns up to 6 chunks, each prefixed with `[Source: filename]`.

**Langfuse metadata logging**: Logs `rag_query`, `rag_chunks_retrieved`, and full `rag_chunks` array to the current Langfuse span's metadata — visible under the span's Metadata tab in the Langfuse UI.

**`_GROUNDING_RULE`**: A mandatory instruction block appended to every system prompt:
- Response MUST be grounded in KB excerpts and quantitative analysis
- No financial statistics from training data that aren't in the KB
- Explicitly state when KB doesn't cover a topic
- Every factual claim must cite its source as `(Source: filename.txt)` in plain parenthetical text

**No-KB warning**: When retrieval returns no chunks, appends `⚠️ Note: No relevant knowledge base excerpts were retrieved` to the response.

**4 system prompt variants** by intent:
- `general` — answer from KB only, no profile context
- `risk_analysis` — present risk metrics with KB-grounded interpretation
- `score_portfolio` — present composite score with sub-score breakdown
- `full_analysis` — complete analysis with allocation recommendations

**Streaming**: Uses `get_chat_model(streaming=True)` so individual tokens flow as `on_chat_model_stream` events during graph execution.

**Outputs**: `messages` (AIMessage with `name="advisor_copilot"`), `advisor_report_generated: True`

---

## 11. guardrail_output

**File**: `agents/guardrails/output_guard.py`

**Purpose**: Appends a standard disclaimer to every advisor response.

**Trigger**: After `advisor_copilot`, before `END`.

**Disclaimer**: `*This is not professional financial advice.*`

**Idempotency**: Checks whether `_DISCLAIMER` string is already present in the message content before appending. If the content already ends with the disclaimer (e.g., the LLM included it), the node returns an empty dict and makes no state change. This prevents duplicate disclaimers.

**Message update**: Returns an `AIMessage` with the same `id` as the original advisor message but with the disclaimer appended. The `add_messages` reducer deduplicates by ID, replacing the original.

**Outputs**: Updated `messages` (with disclaimer appended), or empty dict if already present

---

## 12. error_handler

**File**: `agents/error_handler.py`

**Purpose**: Handles errors detected by `guardrail_input` (and potentially other nodes) by returning a user-friendly message and clearing the error state.

**Trigger**: When `error` field is non-None in state (set by `guardrail_input` on injection detection).

**Behavior**: Returns a fixed message like "I'm sorry, I detected content in your message that I'm unable to process. Please rephrase your question." and clears `error: None`.

**Outputs**: `messages` (AIMessage with error explanation), `error: None`
