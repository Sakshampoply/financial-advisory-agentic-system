import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.mongo import get_mongo_db
from app.db.postgres import get_db
from app.models.session import AdvisorySession

router = APIRouter(tags=["documents"])


class DocumentResponse(BaseModel):
    doc_id: str
    filename: str


class DocumentListItem(BaseModel):
    doc_id: str
    filename: str
    uploaded_at: str


@router.post("/sessions/{session_id}/documents", response_model=DocumentResponse, status_code=201)
async def upload_document(
    session_id: str,
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AdvisorySession).where(AdvisorySession.id == uuid.UUID(session_id))
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    mongo = get_mongo_db()
    insert_result = await mongo["raw_documents"].insert_one({
        "session_id": session_id,
        "filename": filename,
        "content": pdf_bytes,
        "uploaded_at": datetime.now(timezone.utc),
    })
    doc_id_str = str(insert_result.inserted_id)

    # Append doc_id to graph state (last-write-wins field — must read current list first)
    config = {"configurable": {"thread_id": str(session.langgraph_thread_id)}}
    graph = request.app.state.graph
    current_state = await graph.aget_state(config)
    current_docs = list(current_state.values.get("documents_uploaded") or [])
    current_docs.append(doc_id_str)
    await graph.aupdate_state(config, {"documents_uploaded": current_docs}, as_node="__start__")

    return DocumentResponse(doc_id=doc_id_str, filename=filename)


@router.get("/sessions/{session_id}/documents", response_model=list[DocumentListItem])
async def list_documents(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AdvisorySession).where(AdvisorySession.id == uuid.UUID(session_id))
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Session not found")

    mongo = get_mongo_db()
    docs = []
    async for doc in mongo["raw_documents"].find(
        {"session_id": session_id}, {"content": 0}
    ):
        uploaded_at = doc.get("uploaded_at", datetime.now(timezone.utc))
        docs.append(DocumentListItem(
            doc_id=str(doc["_id"]),
            filename=doc.get("filename", "document.pdf"),
            uploaded_at=uploaded_at.isoformat() if hasattr(uploaded_at, "isoformat") else str(uploaded_at),
        ))
    return docs
