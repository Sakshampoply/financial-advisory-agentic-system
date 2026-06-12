from langchain_core.messages import AIMessage

from app.agents.state import GraphState


async def error_handler_node(state: GraphState) -> dict:
    return {
        "messages": [
            AIMessage(
                content="I'm sorry, I wasn't able to process that request. Please rephrase and try again.",
                name="error_handler",
            )
        ],
        "error": None,
    }
