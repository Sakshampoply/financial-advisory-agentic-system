"""Hybrid RAG retriever: pgvector cosine + PostgreSQL BM25, merged by RRF.

Usage:
    chunks = await retrieve("what is duration risk?", db, session_id=uuid_or_none)

Returns up to 6 plain-text strings ready for injection into a system prompt.
"""
import logging
import uuid as uuid_mod
from typing import Any

from fastembed import TextEmbedding
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Singleton — fastembed downloads the model on first use and caches it locally.
# Thread-safe: TextEmbedding is stateless after init.
_embed_model: TextEmbedding | None = None


def _get_embed_model() -> TextEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _embed_model


def _embed(query: str) -> list[float]:
    model = _get_embed_model()
    vectors = list(model.embed([query]))
    return vectors[0].tolist()


async def retrieve(
    query: str,
    db: AsyncSession,
    session_id: str | uuid_mod.UUID | None = None,
) -> list[str]:
    """Return up to 6 relevant text chunks for the given query.

    Searches:
      - Global knowledge base chunks (session_id IS NULL) — always included
      - Session-uploaded document chunks (session_id = <sid>) — included when provided
    """
    if not query.strip():
        return []

    try:
        vec = _embed(query)
    except Exception as exc:
        logger.warning("Embedding failed (%s) — skipping RAG retrieval", exc)
        return []

    # Normalise session_id
    sid: str | None = str(session_id) if session_id is not None else None

    vec_str = f"[{','.join(str(v) for v in vec)}]"

    # --- Vector search (cosine): top-20 ---
    if sid:
        vector_sql = text("""
            SELECT id::text, content, source_filename,
                   1 - (embedding <=> CAST(:vec AS vector)) AS score
            FROM document_chunks
            WHERE session_id IS NULL OR session_id = CAST(:sid AS uuid)
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT 20
        """)
        vector_rows: list[Any] = (
            await db.execute(vector_sql, {"vec": vec_str, "sid": sid})
        ).fetchall()
    else:
        vector_sql = text("""
            SELECT id::text, content, source_filename,
                   1 - (embedding <=> CAST(:vec AS vector)) AS score
            FROM document_chunks
            WHERE session_id IS NULL
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT 20
        """)
        vector_rows = (
            await db.execute(vector_sql, {"vec": vec_str})
        ).fetchall()

    # --- BM25 full-text search: top-20 ---
    if sid:
        bm25_sql = text("""
            SELECT id::text, content, source_filename,
                   ts_rank(content_tsv, plainto_tsquery('english', :q)) AS score
            FROM document_chunks
            WHERE (session_id IS NULL OR session_id = CAST(:sid AS uuid))
              AND content_tsv @@ plainto_tsquery('english', :q)
            ORDER BY score DESC
            LIMIT 20
        """)
        bm25_rows: list[Any] = (
            await db.execute(bm25_sql, {"q": query, "sid": sid})
        ).fetchall()
    else:
        bm25_sql = text("""
            SELECT id::text, content, source_filename,
                   ts_rank(content_tsv, plainto_tsquery('english', :q)) AS score
            FROM document_chunks
            WHERE session_id IS NULL
              AND content_tsv @@ plainto_tsquery('english', :q)
            ORDER BY score DESC
            LIMIT 20
        """)
        bm25_rows = (
            await db.execute(bm25_sql, {"q": query})
        ).fetchall()

    if not vector_rows and not bm25_rows:
        return []

    # --- Reciprocal Rank Fusion ---
    rrf: dict[str, float] = {}
    content_map: dict[str, str] = {}
    source_map: dict[str, str] = {}

    for rank, row in enumerate(vector_rows):
        chunk_id, content, source = row[0], row[1], row[2]
        rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (60 + rank)
        content_map[chunk_id] = content
        source_map[chunk_id] = source or "unknown"

    for rank, row in enumerate(bm25_rows):
        chunk_id, content, source = row[0], row[1], row[2]
        rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (60 + rank)
        content_map[chunk_id] = content
        source_map[chunk_id] = source or "unknown"

    top6_ids = sorted(rrf, key=lambda k: rrf[k], reverse=True)[:6]
    # Embed the source filename in each chunk so the LLM can cite it.
    return [
        f"[Source: {source_map[cid]}]\n{content_map[cid]}"
        for cid in top6_ids
    ]
