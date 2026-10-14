import os
import time

import psycopg2
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .chunking import ingest_folder

EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
EMBED_DIM = 1024  

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)


def get_connection_db():
    return psycopg2.connect(
        dbname=os.getenv("PGDATABASE", "postgres"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
    )


# retrieve.py imports this name
get_connection = get_connection_db


def setup_table(db_connection):
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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def embed_batch(texts, input_type="passage"):
    """
    Embeds a batch of texts in one API call.
    input_type must be "passage" when indexing documents, "query" when embedding
    a search question. Retries up to 3 times on failures like rate limits.
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        encoding_format="float",
        extra_body={"input_type": input_type},
    )
    return [item.embedding for item in response.data]


def delete_existing(db_connection, sources):
    # Re-ingesting a file replaces its old chunks instead of duplicating them
    with db_connection.cursor() as cur:
        cur.execute("DELETE FROM document_chunks WHERE source = ANY(%s)", (list(sources),))


def insert_chunks(db_connection, chunks, embeddings):
    with db_connection.cursor() as cur:
        for chunk, vector in zip(chunks, embeddings):
            cur.execute(
                """
                INSERT INTO document_chunks (source, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (chunk["source"], chunk["chunk_index"], chunk["text"], vector),
            )


def run(folder, batch_size: int = 16, sleep_seconds: float = 1.5):
    chunks = ingest_folder(folder)
    if not chunks:
        print("No chunks found.")
        return

    db_connection = get_connection_db()
    try:
        setup_table(db_connection)
        # One transaction: delete old rows + insert new ones, committed together.
        # If anything fails midway, rollback leaves the old data untouched.
        delete_existing(db_connection, {c["source"] for c in chunks})

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectors = embed_batch([c["text"] for c in batch], input_type="passage")
            insert_chunks(db_connection, batch, vectors)
            print(f"Embedded {i + len(batch)}/{len(chunks)}")
            time.sleep(sleep_seconds)

        db_connection.commit()
    except Exception:
        db_connection.rollback()
        raise
    finally:
        db_connection.close()