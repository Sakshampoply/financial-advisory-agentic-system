# RAG System

This document covers the knowledge base structure, chunking strategy, embedding model, hybrid retrieval algorithm, and how source citations flow into advisor responses.

---

## Table of Contents

1. [Knowledge Base Structure](#1-knowledge-base-structure)
2. [Chunking Strategy](#2-chunking-strategy)
3. [Embedding Model](#3-embedding-model)
4. [Database Schema](#4-database-schema)
5. [Seeding the Knowledge Base](#5-seeding-the-knowledge-base)
6. [Hybrid Retrieval — Vector Search](#6-hybrid-retrieval--vector-search)
7. [Hybrid Retrieval — BM25 Full-Text Search](#7-hybrid-retrieval--bm25-full-text-search)
8. [Reciprocal Rank Fusion (RRF)](#8-reciprocal-rank-fusion-rrf)
9. [Score Weighting Explained](#9-score-weighting-explained)
10. [Source Citation Injection](#10-source-citation-injection)
11. [Session-Specific Document Retrieval](#11-session-specific-document-retrieval)
12. [Retrieval Pipeline Diagram](#12-retrieval-pipeline-diagram)

---

## 1. Knowledge Base Structure

The knowledge base lives in `rag_knowledge_base/` and is the **single authoritative source** for financial domain knowledge. It contains approximately 41 documents across four categories:

```
rag_knowledge_base/
├── download_pdfs.py              ← automated downloader for BIS, Fed, arXiv PDFs
├── README.md                     ← document inventory + manual download instructions
│
├── regulatory/                   ← Highest authority — rule of law
│   ├── SEC_*.txt                 ← SEC investor education documents
│   ├── FINRA_*.txt               ← FINRA suitability and disclosure rules
│   └── *.pdf                     ← regulatory PDFs where available
│
├── central_bank/                 ← Monetary policy and macro research
│   ├── FRED_*.txt                ← Federal Reserve economic commentary
│   ├── BIS_*.pdf                 ← Bank for International Settlements papers
│   └── IMF_*.pdf                 ← International Monetary Fund research
│
├── institutional_research/       ← Practitioner-grade investment research
│   ├── Vanguard_*.txt / *.pdf    ← Vanguard investment research
│   ├── Fidelity_*.txt            ← Fidelity portfolio guidance
│   └── JPM_*.pdf                 ← JPMorgan Asset Management research
│
└── academic/                     ← Peer-reviewed quantitative finance
    └── *.pdf                     ← arXiv q-fin papers on portfolio theory, risk models
```

**Current baseline**: ~1429 chunks from 41 files (as of initial seeding). Re-run `seed_kb.py --force` after adding new documents.

---

## 2. Chunking Strategy

**File**: `scripts/seed_kb.py`, `_chunk_text()` function

The seeder uses a **word-based sliding window** — not token-based. This is intentional:

| Parameter | Value | Rationale |
|-----------|:-----:|-----------|
| Chunk size | 512 words | Fits within bge-small-en-v1.5's 512-token context window while preserving semantic units like an entire paragraph or definition |
| Overlap | 64 words | 12.5% overlap prevents splitting a concept that straddles a chunk boundary |
| Strategy | Word-based | Consistent behaviour across different tokenizers; ensures chunks are human-readable and debuggable |

**`_chunk_text` walkthrough**:
```python
def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap   # slide forward by (chunk_size - overlap)
    return chunks
```

**Null byte stripping**: `pypdf` extracts `\x00` null bytes from some encrypted or malformed PDFs. These cause PostgreSQL insert failures. The seeder strips them: `text.replace("\x00", "")` before chunking.

---

## 3. Embedding Model

**Model**: BAAI/bge-small-en-v1.5 via `fastembed`
**Dimensions**: 384
**Inference**: Local — no API key required, no network call

The model is downloaded on first use (~130MB to `.fastembed_cache/`) and cached as a singleton:

```python
_embedder: TextEmbedding | None = None

def _get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _embedder
```

**Why bge-small-en-v1.5**:
- Strong retrieval performance (BEIR benchmark top-tier for its size class)
- 384 dimensions — small enough for fast cosine similarity in pgvector, large enough for nuanced financial text
- No external dependency — works offline, no API rate limits, no cost per embedding

---

## 4. Database Schema

**Table**: `document_chunks`

| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID | Primary key |
| `session_id` | UUID nullable | NULL for global KB; session UUID for user-uploaded documents |
| `source_file` | text | Original filename — used in `[Source: filename]` citation labels |
| `content` | text | Raw chunk text |
| `content_tsv` | TSVECTOR | Precomputed full-text search vector (English stemming + stop words) |
| `embedding` | vector(384) | fastembed BAAI/bge-small-en-v1.5 embedding |

**Indexes**:

```sql
-- HNSW index for approximate nearest-neighbour vector search
CREATE INDEX document_chunks_embedding_idx
  ON document_chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- GIN index for fast full-text search
CREATE INDEX document_chunks_tsv_idx
  ON document_chunks USING gin (content_tsv);
```

**Why HNSW over IVFFlat**: HNSW (Hierarchical Navigable Small World) provides better recall/speed tradeoff than IVFFlat for collections under ~1M vectors. It doesn't require a training step (IVFFlat requires building cluster centroids), and `ef_construction=64` / `m=16` are standard parameters for high recall at this scale.

**Unique constraint**: `(source_file, session_id)` — prevents re-seeding the same file. `ON CONFLICT DO NOTHING` on insert.

---

## 5. Seeding the Knowledge Base

**File**: `scripts/seed_kb.py`

```bash
uv run python scripts/seed_kb.py          # seed new files only
uv run python scripts/seed_kb.py --force  # drop all global chunks and re-seed
```

**Process**:

1. **Discovery**: `rglob("**/*.pdf")` + `rglob("**/*.txt")` from `rag_knowledge_base/`, excluding `README.md`

2. **Skip check**: Queries `document_chunks` — if any row exists with `source_file = filename` and `session_id IS NULL`, skip this file (already seeded). Bypassed by `--force`.

3. **Text extraction**:
   - `.pdf`: `pypdf.PdfReader` page by page, concatenate all pages, strip null bytes
   - `.txt`: plain `open().read()`

4. **Chunking**: `_chunk_text(text, chunk_size=512, overlap=64)` produces a list of strings

5. **Batch embedding**: `list(embedder.embed(chunks))` — fastembed embeds in batches internally

6. **Insert**:
   ```sql
   INSERT INTO document_chunks (id, session_id, source_file, content, embedding)
   VALUES (:id, NULL, :source_file, :content, CAST(:embedding AS vector))
   ON CONFLICT DO NOTHING
   ```
   **Important**: Uses `CAST(:embedding AS vector)` not `::vector`. The `::` cast syntax is part of PostgreSQL's native protocol, but `asyncpg` tokenizes query strings before sending them — `::vector` confuses the tokenizer and causes a parse error. The `CAST()` form is SQL standard and works with asyncpg.

7. **`content_tsv` auto-update**: A database trigger updates the `content_tsv` column on every insert/update using `to_tsvector('english', content)`. This is handled at the DB level — the seeder doesn't need to compute it.

---

## 6. Hybrid Retrieval — Vector Search

**File**: `rag/retriever.py`

Vector search finds semantically similar chunks using pgvector's cosine distance operator.

```sql
SELECT
    source_file,
    content,
    1 - (embedding <=> CAST(:vec AS vector)) AS score
FROM document_chunks
WHERE session_id IS NULL OR session_id = :sid
ORDER BY score DESC
LIMIT 20
```

- `<=>` is pgvector's cosine distance operator (returns distance, not similarity — hence `1 - ...`)
- `CAST(:vec AS vector)` — same asyncpg reason as seeding
- Returns top-20 candidates for RRF input
- Searches both global KB (`session_id IS NULL`) and session-specific chunks (`session_id = :sid`) in a single query

---

## 7. Hybrid Retrieval — BM25 Full-Text Search

**File**: `rag/retriever.py`

BM25 keyword search uses PostgreSQL's built-in full-text search engine via `ts_rank`.

```sql
SELECT
    source_file,
    content,
    ts_rank(content_tsv, plainto_tsquery('english', :query)) AS score
FROM document_chunks
WHERE
    content_tsv @@ plainto_tsquery('english', :query)
    AND (session_id IS NULL OR session_id = :sid)
ORDER BY score DESC
LIMIT 20
```

- `content_tsv @@ query` — boolean match filter (uses GIN index for fast lookup)
- `ts_rank` — relevance score based on term frequency and position
- `plainto_tsquery('english', ...)` — converts plain text to a tsquery, handling arbitrary user input safely (no special characters required, unlike `to_tsquery`)
- English text search configuration handles stemming (`"investing"` matches `"investment"`) and stop words (`"the"`, `"a"` ignored)
- Returns top-20 candidates for RRF input

---

## 8. Reciprocal Rank Fusion (RRF)

**File**: `rag/retriever.py`

RRF merges the two ranked result lists into a single ranking without requiring score normalization.

**Formula**:
```
RRF_score(document) = Σ  1 / (k + rank(document, list))
                     lists
```

Where:
- `k = 60` (smoothing constant)
- `rank` is 1-based position in each result list
- Sum is over both the vector search list and the BM25 list

**Python implementation**:
```python
k = 60
rrf_scores: dict[str, float] = {}

for rank, (chunk_id, _) in enumerate(vector_results, start=1):
    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank)

for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank)

# Sort by RRF score descending, take top 6
final = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:6]
```

**Why k=60**: The smoothing constant prevents documents ranked #1 from completely dominating the merge. With k=60, the gap between rank 1 and rank 10 is `1/61 - 1/70 ≈ 0.002` — significant but not overwhelming.

**Why RRF over score normalization**: Vector similarity scores and BM25 ts_rank scores have completely different distributions and scales. Normalizing them to [0,1] before combining would require knowing the global max, which changes per query. RRF only uses rank positions, making it robust to score distribution differences.

**Returns top 6**: Empirically balances coverage (enough context) against prompt length (too many chunks bloat the system prompt and dilute the most relevant content).

---

## 9. Score Weighting Explained

Vector search and BM25 have **equal weight** in the RRF merge — each list contributes `1/(60+rank)` independently.

| Scenario | Vector score | BM25 score | RRF total |
|----------|:-----------:|:----------:|:---------:|
| Rank 1 in both lists | 1/61 ≈ 0.0164 | 1/61 ≈ 0.0164 | **0.0328** |
| Rank 1 in vector only | 1/61 ≈ 0.0164 | 0 | 0.0164 |
| Rank 5 in both lists | 1/65 ≈ 0.0154 | 1/65 ≈ 0.0154 | **0.0308** |
| Rank 1 vector, rank 20 BM25 | 0.0164 | 1/80 ≈ 0.0125 | 0.0289 |

A document ranked at the top of both lists is strongly preferred — it gets roughly double the score of a document appearing in only one list.

This makes the hybrid approach complementary:
- **Vector search** excels at semantic similarity ("what is duration risk" → returns chunks about bond price sensitivity even if they don't use the word "duration")
- **BM25** excels at exact terminology ("HNSW index" → returns chunks containing that exact phrase)

---

## 10. Source Citation Injection

Each retrieved chunk is returned with a `[Source: filename]` prefix prepended in the retriever:

```python
return [f"[Source: {row['source_file']}]\n{row['content']}" for row in final_chunks]
```

These labelled chunks are injected into the advisor's system prompt under `## Knowledge Base`.

The `_GROUNDING_RULE` in `advisor_copilot.py` instructs the LLM to:
- Cite every factual claim using the **exact filename** shown in the `[Source: ...]` label
- Format citations as plain parenthetical text: `(Source: SEC_ETF_Guide.txt)`
- Never use backtick or code formatting for source citations

The frontend's `MarkdownRenderer` then converts `(Source: filename.txt)` patterns into amber `SourceCitation` badge components — see [frontend.md](frontend.md#markdownrenderer).

---

## 11. Session-Specific Document Retrieval

When a user uploads a brokerage PDF:
1. The binary is stored in MongoDB `raw_documents`
2. `document_intelligence` extracts holdings and saves to `extracted_document_data`
3. The text chunks are stored in `document_chunks` with `session_id = <user_session_id>` (not NULL)

The retriever's WHERE clause includes both global and session-specific chunks:
```sql
WHERE session_id IS NULL OR session_id = :sid
```

This means the advisor's responses for that session draw from both the global financial knowledge base AND the user's own uploaded document, enabling personalized advice grounded in their actual portfolio.

Cross-session isolation is enforced: session A's documents are never visible to session B.

---

## 12. Retrieval Pipeline Diagram

```
User query
     │
     ▼
fastembed BAAI/bge-small-en-v1.5
(384-dim vector)
     │
     ├──────────────────────────┐
     ▼                          ▼
pgvector cosine search    PostgreSQL ts_rank
(HNSW index, top-20)      (GIN index, top-20)
     │                          │
     └──────────┬───────────────┘
                ▼
     Reciprocal Rank Fusion
     score = Σ 1/(60 + rank)
     k=60, both lists equal weight
                │
                ▼
          top-6 chunks
     (each prefixed [Source: filename])
                │
                ▼
     Injected into advisor_copilot
     system prompt as ## Knowledge Base
```
