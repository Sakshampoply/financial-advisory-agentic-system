"""Seed the RAG knowledge base from rag_knowledge_base/ (all subdirectories).

Usage:
    uv run python scripts/seed_kb.py              # seed (skip if already seeded)
    uv run python scripts/seed_kb.py --force      # drop existing global KB chunks first

Before running, download PDFs:
    cd rag_knowledge_base/ && python download_pdfs.py
"""
import argparse
import asyncio
import sys
import uuid
from pathlib import Path

# Resolve repo root so this script works from any working directory
_REPO_ROOT = Path(__file__).resolve().parents[2]
_KB_DIR = _REPO_ROOT / "rag_knowledge_base"
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from fastembed import TextEmbedding
from pypdf import PdfReader
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import AsyncSessionLocal

# Match the exact model used in retriever.py
_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_CHUNK_WORDS = 512
_OVERLAP_WORDS = 64


def _extract_text(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        raw = path.read_text(encoding="utf-8")
    else:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                pages.append(txt.strip())
        raw = "\n\n".join(pages)
    # PostgreSQL cannot store null bytes in text columns; strip them.
    return raw.replace("\x00", "")


def _chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_words - overlap_words
    return chunks


async def _clear_global_chunks(db: AsyncSession) -> int:
    result = await db.execute(
        text("DELETE FROM document_chunks WHERE session_id IS NULL")
    )
    await db.commit()
    return result.rowcount


async def _seed(db: AsyncSession, force: bool) -> None:
    pdf_files = sorted([
        *_KB_DIR.rglob("*.pdf"),
        *[f for f in _KB_DIR.rglob("*.txt") if f.name not in ("README.md",)],
    ])
    if not pdf_files:
        print(f"No PDFs or .txt files found under {_KB_DIR}.")
        print("Run: cd rag_knowledge_base && python download_pdfs.py")
        return

    if force:
        deleted = await _clear_global_chunks(db)
        print(f"Cleared {deleted} existing global KB chunks.")

    print(f"Loading embedding model '{_MODEL_NAME}'...")
    model = TextEmbedding(_MODEL_NAME)

    total_chunks = 0
    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name} ... ", end="", flush=True)

        if not force:
            existing = await db.execute(
                text("SELECT 1 FROM document_chunks WHERE session_id IS NULL AND source_filename = :fn LIMIT 1"),
                {"fn": pdf_path.name},
            )
            if existing.fetchone() is not None:
                print("SKIP (already seeded)")
                continue

        try:
            text_content = _extract_text(pdf_path)
        except Exception as exc:
            print(f"SKIP (extraction failed: {exc})")
            continue

        chunks = _chunk_text(text_content, _CHUNK_WORDS, _OVERLAP_WORDS)
        if not chunks:
            print("SKIP (no text extracted)")
            continue

        # Embed all chunks for this PDF in one batch
        embeddings = list(model.embed(chunks))

        rows = []
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vec_str = f"[{','.join(str(float(v)) for v in emb)}]"
            rows.append({
                "id": str(uuid.uuid4()),
                "session_id": None,
                "content": chunk,
                "embedding": vec_str,
                "chunk_index": idx,
                "source_filename": pdf_path.name,
            })

        await db.execute(
            text("""
                INSERT INTO document_chunks (id, session_id, content, embedding, chunk_index, source_filename)
                VALUES (:id, :session_id, :content, CAST(:embedding AS vector), :chunk_index, :source_filename)
                ON CONFLICT DO NOTHING
            """),
            rows,
        )
        await db.commit()
        print(f"{len(chunks)} chunks")
        total_chunks += len(chunks)

    print(f"\nDone: {total_chunks} chunks from {len(pdf_files)} file(s) seeded into document_chunks.")


async def main(force: bool) -> None:
    async with AsyncSessionLocal() as db:
        await _seed(db, force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed RAG knowledge base from PDFs.")
    parser.add_argument("--force", action="store_true", help="Delete existing global KB chunks before seeding.")
    args = parser.parse_args()
    asyncio.run(main(force=args.force))
