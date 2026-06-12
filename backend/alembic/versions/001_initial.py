"""initial schema: advisory_sessions and document_chunks

Revision ID: 001
Revises:
Create Date: 2026-05-17
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable required extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # advisory_sessions
    op.create_table(
        "advisory_sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("langgraph_thread_id", UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_advisory_sessions_langgraph_thread_id", "advisory_sessions", ["langgraph_thread_id"], unique=True)

    # document_chunks — content_tsv is a GENERATED ALWAYS AS STORED column
    op.execute("""
        CREATE TABLE document_chunks (
            id UUID PRIMARY KEY,
            session_id UUID REFERENCES advisory_sessions(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            embedding vector(384) NOT NULL,
            content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            chunk_index INTEGER NOT NULL,
            source_filename VARCHAR(500) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        )
    """)

    # HNSW index for cosine similarity search
    op.execute(
        "CREATE INDEX ix_document_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )
    # GIN index for full-text search
    op.execute(
        "CREATE INDEX ix_document_chunks_content_tsv ON document_chunks USING gin (content_tsv)"
    )
    op.execute("CREATE INDEX ix_document_chunks_session_id ON document_chunks (session_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS document_chunks")
    op.drop_table("advisory_sessions")
