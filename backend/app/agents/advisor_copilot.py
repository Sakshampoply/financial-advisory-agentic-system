import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.state import GraphState
from app.db.postgres import AsyncSessionLocal
from app.llm.client import get_chat_model
from app.observability.langfuse_setup import traced_node
from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)


def _build_context(state: GraphState) -> str:
    """Assemble available analysis context to inject into the system prompt."""
    profile = state.get("user_profile") or {}
    risk = state.get("risk_metrics") or {}
    alloc = state.get("allocation_result") or {}
    scoring = state.get("scoring_result") or {}

    parts = []

    if profile:
        portfolio = profile.get("portfolio") or {}
        top = ", ".join(f"{t} {w*100:.0f}%" for t, w in list(portfolio.items())[:6])
        parts.append(
            f"[Source: User Profile]\n"
            f"User profile:\n"
            f"  Risk tolerance: {profile.get('risk_tolerance', 'unknown')}\n"
            f"  Horizon: {profile.get('investment_horizon_years', '?')} years\n"
            f"  Amount: ${profile.get('investment_amount_usd', 0):,.0f}\n"
            f"  Portfolio: {top or 'not provided'}"
        )

    if risk:
        flags = "; ".join(risk.get("risk_flags") or [])
        parts.append(
            f"[Source: Portfolio Risk Analysis]\n"
            f"Risk analysis:\n"
            f"  Sharpe: {risk.get('sharpe_ratio', 0):.2f} | "
            f"Vol: {risk.get('volatility', 0):.1%} | "
            f"Max drawdown: {risk.get('max_drawdown', 0):.1%}\n"
            f"  Flags: {flags or 'none'}"
        )

    if alloc:
        weights = alloc.get("weights") or {}
        top_w = ", ".join(f"{t} {w*100:.0f}%" for t, w in list(weights.items())[:5])
        parts.append(
            f"[Source: Strategy Engine]\n"
            f"Recommended allocation:\n"
            f"  {top_w}\n"
            f"  Expected return: {alloc.get('expected_return', 0):.1%} | "
            f"Vol: {alloc.get('expected_volatility', 0):.1%}\n"
            f"  Rationale: {alloc.get('strategy_rationale', '')}"
        )

    if scoring:
        score = scoring.get("composite_score", 0)
        bd = scoring.get("breakdown") or {}
        parts.append(
            f"[Source: Portfolio Score]\n"
            f"Portfolio score: {score:.0f}/100 "
            f"(Sharpe {bd.get('sharpe_score', 0):.0f}, "
            f"Drawdown {bd.get('drawdown_score', 0):.0f}, "
            f"Diversification {bd.get('diversification_score', 0):.0f})"
        )

    return "\n\n".join(parts)


async def _retrieve_context(state: GraphState) -> list[str]:
    """Fetch RAG chunks for the latest user message; returns empty list on failure."""
    messages = state.get("messages") or []
    query = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage) and isinstance(m.content, str)),
        "",
    )
    if not query:
        return []
    try:
        async with AsyncSessionLocal() as db:
            chunks = await retrieve(query, db, session_id=state.get("session_id"))
        return chunks or []
    except Exception as exc:
        logger.warning("RAG retrieval failed (%s) — proceeding without context", exc)
    return []


_GROUNDING_RULE = """

IMPORTANT — KNOWLEDGE BASE GROUNDING RULES:
- Your response MUST be grounded in the Knowledge Base excerpts and quantitative analysis provided above.
- Do NOT use financial information, statistics, or recommendations from your training \
data that are not reflected in the Knowledge Base or analysis context.
- If the Knowledge Base does not contain sufficient information to answer a specific \
part of the question, explicitly say so: "The available knowledge base does not cover \
[topic] in detail."
- Never fabricate figures, fund names, historical returns, or regulatory rules not \
present in the Knowledge Base or the quantitative analysis context.
- Topics outside the scope (e.g., cryptocurrency, tax-specific advice, specific \
retirement account rules) should be flagged as such.
- CITATIONS: Every factual claim must cite its source in parentheses:
  * Knowledge Base documents: use the exact filename shown in [Source: ...] labels, \
e.g. "(Source: SEC_ETF_Guide.txt)" — write as plain parenthetical text, never in backticks or code formatting
  * Quantitative analysis data: use one of these exact labels:
    - "(Source: User Profile)" — risk tolerance, horizon, investment amount, portfolio holdings
    - "(Source: Portfolio Risk Analysis)" — Sharpe ratio, volatility, max drawdown, risk flags
    - "(Source: Strategy Engine)" — recommended allocation weights and expected return/volatility
    - "(Source: Portfolio Score)" — composite score and Sharpe/drawdown/diversification sub-scores
  * If a claim draws on multiple sources, list all of them."""

_NO_KB_WARNING = (
    "\n\n⚠️ Note: No relevant knowledge base excerpts were retrieved for this query. "
    "The response below is based solely on the quantitative analysis data and general "
    "financial principles — treat it with appropriate caution."
)


@traced_node("advisor_copilot")
async def advisor_copilot_node(state: GraphState) -> dict:
    intent = state.get("intent") or "full_analysis"
    context = _build_context(state)
    rag_chunks = await _retrieve_context(state)

    # Log retrieved chunks as structured metadata on the Langfuse span so they are
    # visible under the advisor_copilot span → Metadata tab without reading the full prompt.
    try:
        from langfuse import get_client as _get_langfuse
        messages_list = state.get("messages") or []
        rag_query = next(
            (m.content for m in reversed(messages_list)
             if isinstance(m, HumanMessage) and isinstance(m.content, str)),
            "",
        )
        _get_langfuse().update_current_span(
            metadata={
                "rag_query": rag_query,
                "rag_chunks_retrieved": len(rag_chunks),
                "rag_chunks": rag_chunks,
            }
        )
    except Exception:
        pass

    if rag_chunks:
        rag_block = "\n\n## Knowledge Base\n" + "\n---\n".join(rag_chunks) + _GROUNDING_RULE
        no_kb_suffix = ""
    else:
        rag_block = _GROUNDING_RULE
        no_kb_suffix = _NO_KB_WARNING

    if intent == "general" or not context:
        system_content = (
            "You are a financial advisor assistant. Answer the user's question "
            "using ONLY the Knowledge Base excerpts provided. "
            "If the question requires personal data you don't have, explain what would be needed. "
            "End with: '*This is not professional financial advice.*'"
            + rag_block
        )
    elif intent == "risk_analysis":
        system_content = (
            "You are a financial advisor assistant. "
            "You have analyzed the user's portfolio risk profile. "
            "Present the risk metrics (Sharpe ratio, volatility, max drawdown) and risk flags clearly, "
            "grounding your interpretation in the Knowledge Base excerpts. "
            "Give actionable guidance on improving risk management. "
            "End with: '*This is not professional financial advice.*'\n\n"
            f"Analysis context:\n{context}"
            + rag_block
        )
    elif intent == "score_portfolio":
        system_content = (
            "You are a financial advisor assistant. "
            "You have scored the user's portfolio on a 0–100 composite scale. "
            "Present the overall score and each sub-score (Sharpe, drawdown, diversification) "
            "with interpretation grounded in the Knowledge Base excerpts. "
            "Suggest concrete steps to improve the score. "
            "End with: '*This is not professional financial advice.*'\n\n"
            f"Analysis context:\n{context}"
            + rag_block
        )
    else:
        system_content = (
            "You are a financial advisor assistant. "
            "You have completed a full quantitative analysis of the user's portfolio. "
            "Answer the user's question using the analysis context and Knowledge Base excerpts ONLY. "
            "Be specific, reference the numbers, and give actionable recommendations. "
            "End with: '*This is not professional financial advice.*'\n\n"
            f"Analysis context:\n{context}"
            + rag_block
        )

    llm = get_chat_model(streaming=True, temperature=0.3)
    messages = [SystemMessage(content=system_content)] + list(state.get("messages") or [])
    response = await llm.ainvoke(messages)

    content = response.content + no_kb_suffix
    return {
        "messages": [AIMessage(content=content, name="advisor_copilot")],
        "advisor_report_generated": True,
    }
