import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.db.postgres import get_db
from app.models.session import AdvisorySession

router = APIRouter(tags=["messages"])

# Nodes that should emit node_start / node_complete events
_AGENT_NODES = {
    "guardrail_input",
    "intake",
    "document_intelligence",
    "profile_builder",
    "risk_assessment",
    "strategy",
    "scoring",
    "advisor_copilot",
}


class MessageRequest(BaseModel):
    content: str


class MessageHistoryItem(BaseModel):
    role: str
    content: str
    agent: str | None = None


@router.get("/sessions/{session_id}/messages", response_model=list[MessageHistoryItem])
async def get_message_history(
    session_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AdvisorySession).where(AdvisorySession.id == uuid.UUID(session_id))
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    thread_id = str(session.langgraph_thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    state = await request.app.state.graph.aget_state(config)
    messages = state.values.get("messages", []) if state.values else []

    history = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else ""
        if not content:
            continue
        if getattr(msg, "tool_calls", None):
            continue
        role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
        agent = getattr(msg, "name", None) if role == "assistant" else None
        history.append(MessageHistoryItem(role=role, content=content, agent=agent))

    return history


@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: str,
    body: MessageRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AdvisorySession).where(AdvisorySession.id == uuid.UUID(session_id))
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    thread_id = str(session.langgraph_thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    graph = request.app.state.graph

    async def event_generator():
        # Snapshot message count before so we can find new AI messages after
        state_before = await graph.aget_state(config)
        msg_count_before = len(state_before.values.get("messages", []))

        seen_nodes: set[str] = set()

        try:
            async for event in graph.astream_events(
                {"messages": [HumanMessage(content=body.content)]},
                config=config,
                version="v2",
            ):
                kind = event["event"]
                node_name = event.get("metadata", {}).get("langgraph_node", "")

                if kind == "on_chain_start" and node_name in _AGENT_NODES and node_name not in seen_nodes:
                    seen_nodes.add(node_name)
                    yield {"event": "node_start", "data": json.dumps({"node": node_name})}

                elif kind == "on_chain_end" and node_name in _AGENT_NODES:
                    yield {"event": "node_complete", "data": json.dumps({"node": node_name})}

                elif kind == "on_chat_model_stream" and node_name == "advisor_copilot":
                    chunk = event["data"].get("chunk")
                    if chunk and isinstance(chunk.content, str) and chunk.content:
                        yield {"event": "token", "data": json.dumps({"content": chunk.content})}

        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"detail": str(exc)})}
            return

        # Emit new AI messages added during this graph run
        state_after = await graph.aget_state(config)
        all_messages = state_after.values.get("messages", [])
        # skip the HumanMessage we just sent (+1) to find new AI messages
        new_messages = all_messages[msg_count_before + 1:]

        for msg in new_messages:
            content = msg.content if isinstance(msg.content, str) else ""
            if content and not getattr(msg, "tool_calls", None):
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "role": "assistant",
                        "content": content,
                        "agent": getattr(msg, "name", None) or "advisor",
                    }),
                }

        state_vals = state_after.values
        risk_m = state_vals.get("risk_metrics")
        alloc = state_vals.get("allocation_result")
        score = state_vals.get("scoring_result")
        if risk_m or alloc or score:
            yield {
                "event": "state",
                "data": json.dumps({
                    "risk_metrics": jsonable_encoder(risk_m) if risk_m else None,
                    "allocation_result": jsonable_encoder(alloc) if alloc else None,
                    "scoring_result": jsonable_encoder(score) if score else None,
                }),
            }

        yield {"event": "done", "data": json.dumps({"session_id": session_id})}

    return EventSourceResponse(event_generator(), sep="\n")
