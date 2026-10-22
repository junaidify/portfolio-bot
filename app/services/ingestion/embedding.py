"""
Ingestion — embedding + pgvector storage layer.

Embeds document chunks via NVIDIA NIM and stores them in a
``document_chunks`` table with a pgvector column.
"""

import logging
import time

import psycopg2
from openai import OpenAI
from pgvector.psycopg2 import register_vector
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.services.ingestion.chunking import ingest_file

logger = logging.getLogger(__name__)

# ── NVIDIA embedding client (OpenAI-compatible) ──────────────────

client = OpenAI(
    base_url=settings.nvidia_base_url,
    api_key=settings.nvidia_api_key,
)

EMBED_MODEL = settings.nvidia_embed_model
EMBED_DIM = settings.nvidia_embed_dim


# ── Database helpers ──────────────────────────────────────────────

def get_connection_db():
    """Return a fresh psycopg2 connection (without registering the vector type)."""
    connection = psycopg2.connect(
        dbname=settings.pgdatabase,
        user=settings.pguser,
        password=settings.pgpassword,
        host=settings.pghost,
        port=settings.pgport,
    )
    return connection


def setup_table(db_connection):
    """Create the vector extension and ``document_chunks`` table if needed."""
    with db_connection.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                source TEXT,
                chunk_index INT,
                content TEXT,
                embedding VECTOR({EMBED_DIM})
            )
        """)
    db_connection.commit()
    # Now that the extension exists, register the type on this connection
    register_vector(db_connection)


# ── Embedding ─────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def embed_batch(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """
    Embed a batch of texts in one API call.

    *input_type* must be ``"passage"`` when indexing documents and
    ``"query"`` when embedding a search question.
    Retries up to 3× on transient failures (rate-limits, timeouts).
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        encoding_format="float",
        extra_body={"input_type": input_type},
    )
    return [item.embedding for item in response.data]


# ── Chunk CRUD ────────────────────────────────────────────────────

def delete_existing(db_connection, sources):
    """Remove old rows so re-ingesting a file replaces, not duplicates."""
    with db_connection.cursor() as cur:
        cur.execute(
            "DELETE FROM document_chunks WHERE source = ANY(%s)",
            (list(sources),),
        )


def insert_chunks(db_connection, chunks, embeddings):
    """Insert chunk rows with their embedding vectors."""
    with db_connection.cursor() as cur:
        for chunk, vector in zip(chunks, embeddings):
            cur.execute(
                """
                INSERT INTO document_chunks (source, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (chunk["source"], chunk["chunk_index"], chunk["text"], vector),
            )


# ── Public: ingest a single file end-to-end ──────────────────────

def run_file(file_path):
    """Chunk → embed → store a single file in pgvector."""
    batch_size = settings.nvidia_embed_batch_size
    sleep_seconds = settings.nvidia_embed_sleep_seconds

    chunks = ingest_file(file_path)
    logger.info("Chunked %s → %d chunks", file_path, len(chunks))

    if not chunks:
        logger.warning("No chunks to ingest: %s", file_path)
        return

    db_connection = get_connection_db()

    try:
        setup_table(db_connection)

        source = chunks[0]["source"]
        delete_existing(db_connection, {source})

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors = embed_batch(
                [c["text"] for c in batch], input_type="passage",
            )
            insert_chunks(db_connection, batch, vectors)

            logger.info(
                "Embedded %d/%d chunks for %s",
                i + len(batch), len(chunks), file_path,
            )
            time.sleep(sleep_seconds)

        db_connection.commit()
        logger.info("✓ Ingested %s (%d chunks)", file_path, len(chunks))

    except Exception:
        db_connection.rollback()
        raise

    finally:
        db_connection.close()
