from langchain_core.messages import AIMessage

from app.agents.state import GraphState
from app.observability.langfuse_setup import traced_node

_DISCLAIMER = "\n\n*Disclaimer: This is not professional financial advice. Consult a licensed financial advisor before making investment decisions.*"


@traced_node("guardrail_output")
async def guardrail_output_node(state: GraphState) -> dict:
    messages = list(state.get("messages") or [])
    if not messages:
        return {}

    last = messages[-1]
    if not isinstance(last, AIMessage):
        return {}

    content = last.content if isinstance(last.content, str) else str(last.content)

    # Idempotent — don't double-append
    if _DISCLAIMER.strip() in content:
        return {}

    updated = AIMessage(content=content + _DISCLAIMER, id=last.id, name=getattr(last, "name", None))
    return {"messages": [updated]}
