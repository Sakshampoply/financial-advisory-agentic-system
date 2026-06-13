# Setup Guide

Complete local development setup for the Financial Advisory Agentic System.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone and Configure](#2-clone-and-configure)
3. [Start Infrastructure](#3-start-infrastructure)
4. [Backend Setup](#4-backend-setup)
5. [Database Migrations](#5-database-migrations)
6. [RAG Knowledge Base](#6-rag-knowledge-base)
7. [Start the Backend](#7-start-the-backend)
8. [Frontend Setup](#8-frontend-setup)
9. [Obtaining API Keys](#9-obtaining-api-keys)
10. [Running Tests](#10-running-tests)
11. [Docker Networking Note](#11-docker-networking-note)

---

## 1. Prerequisites

| Tool | Minimum Version | Purpose |
|------|:--------------:|---------|
| Docker Desktop | 24+ | PostgreSQL, MongoDB, Redis containers |
| Python | 3.12+ | Backend runtime |
| Node.js | 20+ | Frontend runtime |
| uv | 0.5+ | Python package and environment manager |
| git | any | Source control |

Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 2. Clone and Configure

```bash
git clone <repo-url>
cd financial-advisory-agentic-system
cp .env.example .env
```

Open `.env` and fill in your API keys. See [Obtaining API Keys](#9-obtaining-api-keys) for where to get each one. The `.env.example` uses `localhost` for all database URLs — leave these as-is for local development.

---

## 3. Start Infrastructure

```bash
docker compose up -d
```

This starts three services:

| Service | Port | Role |
|---------|:----:|------|
| PostgreSQL 16 + pgvector | 5432 | Session storage, RAG document chunks, LangGraph checkpoints |
| MongoDB 7 | 27017 | Raw PDF binary storage, extracted document data |
| Redis 7 | 6379 | Market data cache (yfinance, Alpha Vantage, FRED) |

Verify all services are healthy:

```bash
docker compose ps
# All services should show "healthy" status

# Confirm PostgreSQL
docker exec financial_postgres psql -U advisor -d financial_advisor -c '\dt' 2>/dev/null || echo "DB not yet migrated (normal)"

# Confirm MongoDB
docker exec financial_mongo mongosh -u advisor -p advisor_secret \
  --authenticationDatabase admin financial_advisor --eval "db.stats().ok" --quiet

# Confirm Redis
docker exec financial_redis redis-cli ping
# Expected: PONG
```

---

## 4. Backend Setup

```bash
cd backend
uv sync
```

`uv sync` reads `pyproject.toml` and installs all dependencies into a local `.venv`. Key packages:

- **FastAPI + uvicorn** — HTTP server and ASGI runtime
- **LangGraph + LangChain** — multi-agent orchestration and LLM abstractions
- **fastembed** — local BAAI/bge-small-en-v1.5 embedding model (downloads on first use, ~130MB)
- **PyPortfolioOpt** — max-Sharpe portfolio optimization
- **deepeval** — LLM-as-judge evaluation metrics
- **psycopg3 + asyncpg** — PostgreSQL drivers (psycopg3 for LangGraph checkpointer, asyncpg for app queries)

---

## 5. Database Migrations

```bash
# From backend/
uv run alembic upgrade head
```

This runs `alembic/versions/001_initial.py`, which creates:

**`advisory_sessions` table**
- `id` UUID primary key
- `langgraph_thread_id` UUID — bridges this row to LangGraph's checkpoint store
- `created_at` timestamp

**`document_chunks` table** (RAG storage)
- `id` UUID primary key
- `session_id` UUID nullable — `NULL` for global knowledge base, session UUID for uploaded documents
- `source_file` text — original filename, used for `[Source: filename]` citation labels
- `content` text — raw chunk text
- `content_tsv` TSVECTOR — precomputed full-text search vector, auto-updated via trigger
- `embedding vector(384)` — fastembed BAAI/bge-small-en-v1.5 embedding

Two performance indexes:
- **HNSW index** on `embedding` using cosine distance operator (`vector_cosine_ops`), parameters `m=16, ef_construction=64` — approximate nearest-neighbour search with better recall/speed tradeoff than IVFFlat for under 1M vectors
- **GIN index** on `content_tsv` — enables sub-10ms BM25 full-text search without sequential scan

---

## 6. RAG Knowledge Base

The knowledge base lives in `rag_knowledge_base/` at the repo root. It contains ~41 authoritative financial documents across four categories.

**Step 1 — Download PDFs** (from the repo root):

```bash
cd rag_knowledge_base
python download_pdfs.py
cd ..
```

This automatically downloads BIS working papers, Federal Reserve publications, and arXiv q-fin papers. Some sources (SEC investor.gov, CFPB, FINRA, IMF) are blocked by CDN protection — their content is fully covered by the `.txt` files already in the repository. See `rag_knowledge_base/README.md` for manual download instructions for those sources.

**Step 2 — Seed the database** (from `backend/`):

```bash
uv run python scripts/seed_kb.py
```

Expected output: approximately **1429 chunks from 41 files**. The seeder is idempotent — re-running skips already-seeded files based on `(source_file, session_id)` uniqueness.

To force a complete re-seed (drops all global KB chunks and re-inserts):
```bash
uv run python scripts/seed_kb.py --force
```

Run `--force` after adding new documents to the knowledge base.

---

## 7. Start the Backend

```bash
# From backend/
uv run uvicorn app.main:app --reload
```

Server starts on `http://localhost:8000`. The FastAPI lifespan handler performs three startup actions:

1. Opens an `AsyncConnectionPool` (psycopg3) for the LangGraph checkpointer — this is separate from the asyncpg pool used for application queries because LangGraph requires psycopg3's native protocol
2. Initializes `AsyncPostgresSaver` and calls `.setup()` to create the LangGraph checkpoint tables if they don't exist
3. Compiles the LangGraph `StateGraph` with the checkpointer attached and stores it on `app.state.graph`

API documentation: `http://localhost:8000/docs`

---

## 8. Frontend Setup

```bash
# From frontend/
npm install
npm run dev
```

Frontend starts on `http://localhost:3000`.

Next.js is configured with a `rewrites` rule in `next.config.ts` that proxies `/api/v1/**` to `http://localhost:8000/api/v1/**` for non-streaming calls. Streaming SSE calls bypass this proxy and connect to `http://localhost:8000` directly — see [docs/frontend.md](frontend.md) for the reason.

---

## 9. Obtaining API Keys

### OpenRouter (required — all LLM calls)

1. Register at [openrouter.ai](https://openrouter.ai)
2. Go to **Keys** → **Create Key**
3. Set `OPENROUTER_API_KEY` in `.env`
4. Choose a model for `OPENROUTER_MODEL`. Free options: `meta-llama/llama-3.3-70b-instruct:free`

### Alpha Vantage (required — fundamentals and news sentiment)

1. Register at [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
2. Free tier: 25 requests/day, sufficient for development
3. Set `ALPHA_VANTAGE_API_KEY`

### FRED — Federal Reserve Economic Data (required — macro indicators)

1. Register at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Free, no meaningful rate limits
3. Set `FRED_API_KEY`

### Langfuse (optional — observability)

1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com) or self-host with Docker
2. Create a project → **Settings** → **API Keys**
3. Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`
4. Without these keys, all `@traced_node` decorators and `get_langfuse_callbacks()` calls no-op silently — the system works fully without observability

---

## 10. Running Tests

All test commands run from `backend/`:

```bash
# Fast unit tests — no database, no LLM (~10 seconds)
uv run pytest evals/ -m unit -v

# RAG retrieval quality — requires live PostgreSQL + seeded knowledge base
uv run pytest evals/ -m rag -v

# LLM-as-judge evaluations — requires OPENROUTER_API_KEY; slow (~5 minutes)
uv run pytest evals/ -m llm_eval -v --timeout=300

# Specific test files
uv run pytest evals/test_guardrails.py -v
uv run pytest evals/test_risk_agent.py -v
```

| Pytest mark | Speed | External dependencies |
|-------------|:-----:|----------------------|
| `unit` | ~10s | None |
| `integration` | ~30s | Live PostgreSQL + MongoDB |
| `rag` | ~30s | Live PostgreSQL + seeded KB |
| `llm_eval` | ~5min | OPENROUTER_API_KEY |

See [docs/evals.md](evals.md) for a full explanation of what each test verifies.

---

## 11. Docker Networking Note

When running the backend with `uv run uvicorn` (outside Docker), the backend process runs on your host machine. It connects to database containers via `localhost` on the mapped ports, not the Docker Compose internal service names (`postgres`, `mongo`, `redis`).

The `.env.example` uses `localhost` — do not change these to service names unless you're running the backend inside Docker too. If you do run everything inside Docker, update the URLs to use service names and port 5432/27017/6379 directly (the internal container ports, not the host-mapped ports).
