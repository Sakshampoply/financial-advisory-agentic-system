from langchain_core.messages import AIMessage, SystemMessage

from app.agents.state import GraphState, UserProfile
from app.llm.client import get_chat_model
from app.observability.langfuse_setup import traced_node

_COLLECT_PROFILE_TOOL = {
    "type": "function",
    "function": {
        "name": "collect_profile",
        "description": "Save the user's complete investment profile. Call only when all required fields are known.",
        "parameters": {
            "type": "object",
            "properties": {
                "risk_tolerance": {
                    "type": "string",
                    "enum": ["low", "moderate", "high"],
                    "description": "User's risk tolerance",
                },
                "investment_horizon_years": {
                    "type": "integer",
                    "description": "Investment horizon in years",
                },
                "investment_amount_usd": {
                    "type": "number",
                    "description": "Amount to invest in USD",
                },
                "existing_portfolio": {
                    "type": "array",
                    "description": "Existing holdings. Provide if user listed specific tickers.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string", "description": "Ticker symbol e.g. 'AAPL'"},
                            "weight_pct": {"type": "number", "description": "Allocation percentage e.g. 30 for 30%"},
                        },
                        "required": ["ticker", "weight_pct"],
                    },
                },
                "wants_new_portfolio": {
                    "type": "boolean",
                    "description": "True if the user wants a brand-new portfolio built from scratch.",
                },
                "annual_income_usd": {
                    "type": "number",
                    "description": "Annual income in USD (optional)",
                },
                "tax_bracket": {
                    "type": "string",
                    "description": "Tax bracket e.g. '22%' (optional)",
                },
                "liquidity_needs": {
                    "type": "string",
                    "description": "Liquidity needs description (optional)",
                },
            },
            "required": ["risk_tolerance", "investment_horizon_years", "investment_amount_usd"],
        },
    },
}


def _build_portfolio(holdings: list[dict]) -> dict[str, float]:
    """Normalise a list of {ticker, weight_pct} into a weights dict summing to 1.0."""
    total = sum(h["weight_pct"] for h in holdings)
    if total <= 0:
        return {}
    return {h["ticker"].upper(): round(h["weight_pct"] / total, 4) for h in holdings}


def _build_system_prompt(profile: dict) -> str:
    """Dynamic prompt that lists only the fields still missing from the profile."""
    known: dict[str, str] = {}
    if profile.get("risk_tolerance"):
        known["risk_tolerance"] = str(profile["risk_tolerance"])
    if profile.get("investment_horizon_years"):
        known["investment_horizon_years"] = str(profile["investment_horizon_years"])
    if profile.get("investment_amount_usd"):
        known["investment_amount_usd"] = f"${profile['investment_amount_usd']:,.0f}"

    # portfolio is "known" only if it's a non-empty dict (empty means extraction found nothing)
    portfolio_known = bool(profile.get("portfolio"))

    missing = []
    if "risk_tolerance" not in known:
        missing.append("risk_tolerance: exactly \"low\", \"moderate\", or \"high\"")
    if "investment_horizon_years" not in known:
        missing.append("investment_horizon_years: integer number of years")
    if "investment_amount_usd" not in known:
        missing.append("investment_amount_usd: dollar amount to invest")

    lines = [
        "You are a financial advisory intake specialist. Collect the user's investment profile "
        "through friendly, natural conversation.",
        "",
    ]

    if known:
        lines.append("Already known (extracted from uploaded documents — do NOT ask the user for these):")
        for k, v in known.items():
            lines.append(f"  - {k}: {v}")
        lines.append("")

    if missing:
        lines.append("Still needed from the user:")
        for m in missing:
            lines.append(f"  - {m}")
        lines.append("")

    if not portfolio_known:
        lines += [
            "After collecting all required fields above, ask ONE question about portfolio:",
            '  "Do you have existing investments you\'d like me to analyze? If so, list the tickers '
            'and rough percentages (e.g. AAPL 30%, VTI 50%, BND 20%). Or say \'build me a new '
            'portfolio\' if you\'d like recommendations from scratch."',
            "  - If they list holdings → populate existing_portfolio",
            "  - If they say build from scratch → set wants_new_portfolio: true",
            "",
        ]
    else:
        lines += [
            "Portfolio is already known from uploaded documents — do NOT ask the user about it.",
            "",
        ]

    lines += [
        "Optional fields (collect only if the user mentions them voluntarily): annual_income_usd, tax_bracket, liquidity_needs.",
        "Do NOT ask about optional fields — only record them if the user brings them up.",
        "",
        "Rules:",
        "- Ask naturally — no rigid forms.",
        "- NEVER assume, invent, or guess a value. Use only what the user explicitly states.",
        "- For pre-populated fields, use those exact values when calling collect_profile.",
        "- Call collect_profile the INSTANT all required pieces are known — do not ask follow-up questions first.",
        "- Do NOT generate a confirmation or summary message as plain text. The confirmation is produced automatically when you call collect_profile.",
        "- Keep responses concise.",
    ]

    return "\n".join(lines)


def _build_confirmation(profile: UserProfile, intent: str | None = None) -> str:
    if profile.get("portfolio"):
        items = ", ".join(f"{t} ({w * 100:.0f}%)" for t, w in profile["portfolio"].items())
        portfolio_line = f"\n- Existing portfolio: {items}"
    else:
        portfolio_line = "\n- Portfolio: build new recommendations from scratch"

    next_step = {
        "risk_analysis":  "I'll now analyze your portfolio's risk profile.",
        "score_portfolio": "I'll now score your portfolio.",
        "full_analysis":   "I'll now run a full risk assessment and build a strategy for you.",
    }.get(intent or "full_analysis", "I'll now run a full risk assessment and build a strategy for you.")

    return (
        f"Thanks! I've captured your profile:\n"
        f"- Risk tolerance: {profile['risk_tolerance']}\n"
        f"- Investment horizon: {profile['investment_horizon_years']} years\n"
        f"- Amount: ${profile['investment_amount_usd']:,.0f}"
        f"{portfolio_line}\n\n"
        f"{next_step}"
    )


@traced_node("intake")
async def intake_node(state: GraphState) -> dict:
    existing_profile: dict = dict(state.get("user_profile") or {})

    # --- Step 6 fast-path: all required fields + portfolio already pre-populated ---
    has_risk = bool(existing_profile.get("risk_tolerance"))
    has_horizon = bool(existing_profile.get("investment_horizon_years"))
    has_amount = bool(existing_profile.get("investment_amount_usd"))
    portfolio_known = bool(existing_profile.get("portfolio"))  # non-empty portfolio dict

    if has_risk and has_horizon and has_amount and portfolio_known:
        return {
            "messages": [AIMessage(content=_build_confirmation(existing_profile, state.get("intent")), name="intake")],
            "user_profile": existing_profile,
            "intake_complete": True,
        }

    # --- Normal conversational path ---
    llm = get_chat_model().bind_tools([_COLLECT_PROFILE_TOOL])
    system_prompt = _build_system_prompt(existing_profile)
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = await llm.ainvoke(messages)

    if not response.tool_calls:
        content = response.content or ""
        # Guard: if the LLM generated a fake confirmation instead of calling the tool, replace it
        if "i've captured your profile" in content.lower() or "i have captured your profile" in content.lower():
            missing = [
                f for f, val in [
                    ("risk tolerance (low/moderate/high)", existing_profile.get("risk_tolerance")),
                    ("investment horizon in years", existing_profile.get("investment_horizon_years")),
                    ("investment amount in USD", existing_profile.get("investment_amount_usd")),
                ]
                if not val
            ]
            ask = (
                "I still need a couple of details to complete your profile. "
                + ("Could you tell me your " + " and ".join(missing) + "?" if missing
                   else "Could you confirm your investment details?")
            )
            return {"messages": [AIMessage(content=ask, name="intake")]}
        return {"messages": [AIMessage(content=content, name="intake")]}

    args = response.tool_calls[0]["args"]

    # If the LLM called collect_profile without portfolio info, save partial profile and
    # ask the portfolio question explicitly. Enforced in code because the LLM reliably
    # ignores the prompt instruction once the three required fields are known.
    if not args.get("existing_portfolio") and not args.get("wants_new_portfolio"):
        partial: UserProfile = {
            "risk_tolerance": args.get("risk_tolerance") or existing_profile.get("risk_tolerance", ""),
            "investment_horizon_years": int(
                args.get("investment_horizon_years") or existing_profile.get("investment_horizon_years", 0)
            ),
            "investment_amount_usd": float(
                args.get("investment_amount_usd") or existing_profile.get("investment_amount_usd", 0)
            ),
        }
        # Portfolio was already extracted from uploaded documents — carry it forward and
        # complete intake without asking. Without this check, the return below would set
        # user_profile to the partial dict (no portfolio key), making supervisor loop back
        # to profile_builder and sending the user a spurious portfolio question.
        if existing_profile.get("portfolio"):
            partial["portfolio"] = existing_profile["portfolio"]
            return {
                "messages": [AIMessage(content=_build_confirmation(partial, state.get("intent")), name="intake")],
                "user_profile": partial,
                "intake_complete": True,
            }
        ask = (
            "Do you have existing investments you'd like me to analyze? "
            "If so, list the tickers and rough percentages (e.g. AAPL 30%, VTI 50%, BND 20%). "
            "Or say 'build me a new portfolio' if you'd like recommendations from scratch."
        )
        return {
            "messages": [AIMessage(content=ask, name="intake")],
            "user_profile": partial,  # saved — next intake call won't re-ask risk/horizon/amount
        }

    # Merge tool args with pre-populated profile values (tool args take precedence)
    profile = UserProfile(
        risk_tolerance=args.get("risk_tolerance") or existing_profile.get("risk_tolerance", ""),
        investment_horizon_years=int(
            args.get("investment_horizon_years") or existing_profile.get("investment_horizon_years", 0)
        ),
        investment_amount_usd=float(
            args.get("investment_amount_usd") or existing_profile.get("investment_amount_usd", 0)
        ),
    )
    if args.get("annual_income_usd") is not None:
        profile["annual_income_usd"] = float(args["annual_income_usd"])
    if args.get("tax_bracket"):
        profile["tax_bracket"] = args["tax_bracket"]
    if args.get("liquidity_needs"):
        profile["liquidity_needs"] = args["liquidity_needs"]

    # Portfolio: prefer tool args, fall back to pre-populated
    holdings = args.get("existing_portfolio") or []
    if holdings:
        profile["portfolio"] = _build_portfolio(holdings)
    elif existing_profile.get("portfolio"):
        profile["portfolio"] = existing_profile["portfolio"]
    # wants_new_portfolio → leave portfolio absent (strategy will use DEFAULT_UNIVERSE)

    return {
        "messages": [AIMessage(content=_build_confirmation(profile, state.get("intent")), name="intake")],
        "user_profile": profile,
        "intake_complete": True,
    }
