from app.agents.state import GraphState, ScoringResult
from app.observability.langfuse_setup import traced_node


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_score(
    sharpe: float,
    max_drawdown: float,
    weights: dict[str, float],
) -> ScoringResult:
    """
    Deterministic composite score 0–100.
      Sharpe       40% — maps [-2, 2] linearly to [0, 100]
      Max drawdown 30% — maps [-0.5, 0] linearly to [0, 100]
      Diversification 30% — (1 - HHI) * 100, HHI = sum of squared weights
    """
    sharpe_score = _clamp((sharpe + 2.0) / 4.0 * 100, 0, 100)

    drawdown_score = _clamp((1.0 + max_drawdown / 0.5) * 100, 0, 100)

    if weights:
        hhi = sum(w ** 2 for w in weights.values())
        diversification_score = _clamp((1.0 - hhi) * 100, 0, 100)
    else:
        diversification_score = 0.0

    composite = (
        0.40 * sharpe_score
        + 0.30 * drawdown_score
        + 0.30 * diversification_score
    )

    return ScoringResult(
        composite_score=round(composite, 2),
        breakdown={
            "sharpe_score": round(sharpe_score, 2),
            "drawdown_score": round(drawdown_score, 2),
            "diversification_score": round(diversification_score, 2),
        },
    )


@traced_node("scoring")
async def scoring_node(state: GraphState) -> dict:
    risk = state.get("risk_metrics") or {}
    alloc = state.get("allocation_result") or {}

    result = compute_score(
        sharpe=float(risk.get("sharpe_ratio", 0)),
        max_drawdown=float(risk.get("max_drawdown", 0)),
        weights=dict(alloc.get("weights") or {}),
    )
    return {"scoring_result": result}
