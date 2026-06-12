from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import GraphState
from app.llm.client import get_chat_model
from app.observability.langfuse_setup import traced_node

_SYSTEM = """Classify the user's message into exactly ONE of these four categories. Reply with ONE word only.

"general" — educational or factual financial question, no personal portfolio data needed.
Examples: "What is a Sharpe ratio?", "How do ETFs work?", "Best strategies for retirement",
          "Explain diversification", "Difference between stocks and bonds", "What is inflation?",
          "Tell me about index funds", "How does compound interest work?"

"risk_analysis" — wants to understand their portfolio's risk profile: volatility, drawdown, risk flags,
concentration, or safety. Does NOT need an optimized allocation or a score.
Examples: "How risky is my portfolio?", "What's my volatility?", "Show me my risk flags",
          "Am I over-concentrated?", "What is my Sharpe ratio?", "What's my max drawdown?",
          "Is my portfolio safe?", "How much risk am I taking?"

"score_portfolio" — wants a numeric score, grade, or rating of their current portfolio quality.
Examples: "Score my portfolio", "Rate my allocation", "How well am I doing?",
          "Grade my portfolio", "What's my portfolio score?", "How good is my portfolio?",
          "Give me a rating out of 100"

"full_analysis" — wants a complete investment plan, optimization, rebalancing recommendation,
or comprehensive advice. This is the default for any personalized advice not covered above.
Examples: "Optimize my portfolio", "Build me an investment plan", "Should I rebalance?",
          "Suggest an allocation", "Analyze my portfolio", "What should I invest in?",
          "Help me build a portfolio", "Give me financial advice", "What stocks should I buy?"

When in doubt, reply: full_analysis

Reply with ONLY one of: general  risk_analysis  score_portfolio  full_analysis"""

_INTENT_MAP = {
    "general": "general",
    "risk_analysis": "risk_analysis",
    "score_portfolio": "score_portfolio",
    "full_analysis": "full_analysis",
}


@traced_node("intent_classifier")
async def intent_classifier_node(state: GraphState) -> dict:
    messages = state.get("messages") or []
    if not messages or not isinstance(messages[-1], HumanMessage):
        return {}

    # Don't re-classify while intake is in progress — the user is answering intake's
    # questions, not expressing a new intent. Re-classifying "moderate risk, 5 years"
    # as full_analysis would override the original intent (e.g., risk_analysis).
    if (
        not state.get("intake_complete")
        and state.get("intent") is not None
        and any(getattr(m, "name", None) == "intake" for m in messages)
    ):
        return {}

    content = messages[-1].content if isinstance(messages[-1].content, str) else ""

    llm = get_chat_model(temperature=0.0)
    resp = await llm.ainvoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=content),
    ])

    raw = (resp.content or "").strip().lower().split()[0] if resp.content else ""
    intent = _INTENT_MAP.get(raw, "full_analysis")
    return {"intent": intent}
