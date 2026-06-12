from langchain_core.messages import HumanMessage

from app.agents.state import GraphState

MAX_ITERATIONS = 50


def supervisor_node(state: GraphState) -> dict:
    return {"iteration_count": state["iteration_count"] + 1}


def route_supervisor(state: GraphState) -> str:
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "end"

    if state.get("error"):
        return "error_handler"

    messages = state.get("messages") or []
    last_is_human = bool(messages and isinstance(messages[-1], HumanMessage))
    intent = state.get("intent")

    # --- Document pipeline — always runs when docs are waiting ---
    uploaded = state.get("documents_uploaded") or []
    if uploaded and not state.get("documents_extracted"):
        return "document_intelligence"

    profile = state.get("user_profile") or {}
    if state.get("documents_extracted") and "portfolio" not in profile:
        return "profile_builder"

    # --- General questions: skip all data collection, answer directly ---
    if intent == "general" and last_is_human:
        return "advisor_copilot"

    # --- Determine which pipeline stages this intent requires ---
    # risk_analysis: intake + risk only (no strategy, no scoring)
    # score_portfolio: intake + risk + scoring (no strategy)
    # full_analysis: intake + risk + strategy + scoring (complete pipeline)
    needs_intake = intent in ("risk_analysis", "score_portfolio", "full_analysis")
    needs_risk = intent in ("risk_analysis", "score_portfolio", "full_analysis")
    needs_strategy = intent == "full_analysis"
    needs_scoring = intent in ("score_portfolio", "full_analysis")

    # Defensive: profile_builder may have completed the profile (document bypass) without
    # explicitly setting the flag. Treat intake as done if all required fields + portfolio
    # key are present — covers the case where the user uploaded a document for portfolio
    # info instead of answering intake's portfolio question.
    effective_intake_complete = state.get("intake_complete") or (
        bool(profile.get("risk_tolerance")) and
        bool(profile.get("investment_horizon_years")) and
        bool(profile.get("investment_amount_usd")) and
        "portfolio" in profile
    )

    # --- Profile collection gate ---
    if needs_intake and not effective_intake_complete:
        intake_started = any(getattr(m, "name", None) == "intake" for m in messages)
        if last_is_human or (state.get("documents_extracted") and not intake_started):
            return "intake"
        return "end"

    # --- Quantitative pipeline — only runs the stages this intent requires ---
    if needs_risk and state.get("risk_metrics") is None:
        return "risk_assessment"

    if needs_strategy and state.get("allocation_result") is None:
        return "strategy"

    if needs_scoring and state.get("scoring_result") is None:
        return "scoring"

    # --- Pipeline complete: generate report once, then on every new user message ---
    # last_is_human is unreliable here — intake may have appended its own AIMessage in the
    # same graph execution, making messages[-1] an AIMessage even though the user just sent
    # a message. Instead, check whether any HumanMessage appears after the last advisor response.
    last_advisor_pos = -1
    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "name", None) == "advisor_copilot":
            last_advisor_pos = i
            break
    has_new_human = any(isinstance(m, HumanMessage) for m in messages[last_advisor_pos + 1:])
    if not state.get("advisor_report_generated") or has_new_human:
        return "advisor_copilot"

    return "end"
