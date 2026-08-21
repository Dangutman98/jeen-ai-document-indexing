"""Postgres + pgvector storage and similarity search."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector
from pgvector.psycopg.vector import Vector

from .errors import DatabaseConnectionError


@contextmanager
def get_connection():
    """Yield a connected, pgvector-registered psycopg connection.

    Raises DatabaseConnectionError with a clear message on any connection
    failure -- wrong host, wrong credentials, or the container not running.
    """
    conn_str = os.environ.get("POSTGRES_URL")
    if not conn_str:
        raise DatabaseConnectionError(
            "POSTGRES_URL is not set. Add it to your .env file (see .env.example)."
        )
    try:
        conn = psycopg.connect(conn_str, connect_timeout=5)
    except psycopg.OperationalError as e:
        raise DatabaseConnectionError(
            f"Could not connect to Postgres: {e}\n"
            "Is the database running? Try: docker compose up -d"
        ) from e

    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def insert_chunks(
    conn,
    *,
    filename: str,
    split_strategy: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> int:
    """Insert chunk rows. Returns the number of rows inserted."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO document_chunks
                (chunk_text, embedding, filename, split_strategy, chunk_index)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (text, embedding, filename, split_strategy, idx)
                for idx, (text, embedding) in enumerate(zip(chunks, embeddings))
            ],
        )
    conn.commit()
    return len(chunks)


@dataclass
class SearchResult:
    id: int
    chunk_text: str
    filename: str
    split_strategy: str
    created_at: object
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity in [0, 1], derived from cosine distance."""
        return 1 - self.distance


def search(conn, query_embedding: list[float], *, limit: int = 5) -> list[SearchResult]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, chunk_text, filename, split_strategy, created_at,
                   embedding <=> %s AS distance
            FROM document_chunks
            ORDER BY distance ASC
            LIMIT %s
            """,
            (Vector(query_embedding), limit),
        )
        rows = cur.fetchall()
    return [
        SearchResult(
            id=r[0], chunk_text=r[1], filename=r[2], split_strategy=r[3],
            created_at=r[4], distance=r[5],
        )
        for r in rows
    ]
