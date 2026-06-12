import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.state import make_initial_state
from app.db.postgres import get_db
from app.models.session import AdvisorySession

router = APIRouter(tags=["sessions"])


class SessionResponse(BaseModel):
    id: str
    langgraph_thread_id: str
    status: str
    created_at: datetime


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    session_id = uuid.uuid4()
    thread_id = uuid.uuid4()

    row = AdvisorySession(id=session_id, langgraph_thread_id=thread_id)
    db.add(row)
    await db.commit()
    await db.refresh(row)

    config = {"configurable": {"thread_id": str(thread_id)}}
    initial_state = make_initial_state(str(session_id))
    await request.app.state.graph.aupdate_state(config, initial_state)

    return SessionResponse(
        id=str(row.id),
        langgraph_thread_id=str(row.langgraph_thread_id),
        status=row.status,
        created_at=row.created_at,
    )


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AdvisorySession).order_by(AdvisorySession.created_at.desc()).limit(50)
    )
    rows = result.scalars().all()
    return [
        SessionResponse(
            id=str(r.id),
            langgraph_thread_id=str(r.langgraph_thread_id),
            status=r.status,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AdvisorySession).where(AdvisorySession.id == uuid.UUID(session_id))
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(
        id=str(row.id),
        langgraph_thread_id=str(row.langgraph_thread_id),
        status=row.status,
        created_at=row.created_at,
    )
