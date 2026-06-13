from app.agents.state import GraphState, UserProfile
from app.db.mongo import get_mongo_db
from app.observability.langfuse_setup import traced_node


@traced_node("profile_builder")
async def profile_builder_node(state: GraphState) -> dict:
    mongo = get_mongo_db()
    session_id = state["session_id"]

    holdings_by_ticker: dict[str, float] = {}
    total_account_value: float | None = None

    async for record in mongo["extracted_document_data"].find(
        {"session_id": session_id}
    ):
        extraction = record.get("extraction") or {}

        acct_val = extraction.get("account_value")
        if acct_val and isinstance(acct_val, (int, float)) and acct_val > 0:
            # Use the largest single account value if multiple docs
            if total_account_value is None or float(acct_val) > total_account_value:
                total_account_value = float(acct_val)

        for holding in extraction.get("holdings") or []:
            ticker = (holding.get("ticker") or "").upper().strip()
            value = holding.get("value") or 0
            if ticker and float(value) > 0:
                holdings_by_ticker[ticker] = holdings_by_ticker.get(
                    ticker, 0.0
                ) + float(value)

    # Build normalised portfolio weights
    portfolio: dict[str, float] = {}
    if holdings_by_ticker:
        total = sum(holdings_by_ticker.values())
        if total > 0:
            portfolio = {t: round(v / total, 4) for t, v in holdings_by_ticker.items()}

    profile: UserProfile = dict(state.get("user_profile") or {})
    # Always set "portfolio" key so supervisor doesn't loop back to profile_builder
    profile["portfolio"] = portfolio

    if total_account_value and "investment_amount_usd" not in profile:
        profile["investment_amount_usd"] = total_account_value

    result: dict = {"user_profile": profile}
    # If intake already collected the 3 required fields (risk/horizon/amount) and
    # profile_builder just filled in the portfolio from documents, the profile is now
    # complete — set the flag so supervisor can proceed past the intake gate.
    if (
        profile.get("risk_tolerance")
        and profile.get("investment_horizon_years")
        and profile.get("investment_amount_usd")
    ):
        result["intake_complete"] = True
    return result
