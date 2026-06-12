from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.api.v1.documents import router as documents_router
from app.api.v1.health import router as health_router
from app.api.v1.messages import router as messages_router
from app.api.v1.sessions import router as sessions_router
from app.config import settings
from app.db.mongo import close_mongo
from app.db.redis_client import close_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    # PostgreSQL connection pool for LangGraph checkpointer (uses psycopg3, not asyncpg)
    pg_url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    pool = AsyncConnectionPool(
        conninfo=pg_url,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,
    )
    await pool.open()

    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    from app.agents.graph import create_graph
    app.state.graph = create_graph(checkpointer)
    app.state.pool = pool

    yield

    await pool.close()
    await close_mongo()
    await close_redis()


app = FastAPI(
    title="Financial Advisory API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(sessions_router, prefix="/api/v1")
app.include_router(messages_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
