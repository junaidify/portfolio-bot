import os
import psycopg2
from openai import OpenAI
import time
from .chunking import ingest_folder

NVIDIA_API_KEY = os.getenv.get("NVIDIA_API_KEY")
EMBED_MODEL = ""
EMBED_DIM = 1024

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)


def get_connection_db(): 
    return psycopg2.connect(
        dbname=os.getenv.get("PGDATABASE", "postgres"), 
        username=os.getenv.get("PUSERNAME", "postgres"), 
        host=os.getenv.get("host", "localhost"), 
        port=os.getenv.get("port", "5432")
    )


def setup_table(db_connection): 
    with db_connection.cursor() as conn: 
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"""
        CREATE TABLE IF NOT EXISTS document_chuks(
        id SERIAL PRIMARY KEY, 
        source TEXT, 
        chunk_index INT, 
        content TEXT, 
        embedding VECTOR({EMBED_DIM})
        )
        """)

    db_connection.commit()


def embed_batch(text, input_type = "passage"): 
    """
    Embeds a batch of texts in one API call.
    input_type must be "passage" when indexing documents, "query" when embedding
    a search question later — using the wrong one silently degrades retrieval quality.
    """

    response = OpenAI.embeddings.create(
        model=EMBED_MODEL, 
        input=text, 
        encoding_format="float", 
        extra_body={"input_type", input_type}
    )

    return [item.embedding for item in response.data] 


def insert_chunks(db_connection, chunks_dict, embeddings): 
    with db_connection.cursor() as conn: 
        for chunk, vector in zip(chunks_dict, embeddings): 
            conn.execute("""
            INSERT INTO document_chunks(source, chunk_index, content, embedding)
            VALUES(%s, %s, %s, %s)
            """, (chunk["source"], chunk["chunk_index"], chunk["content"], vector)
            )
        
    conn.commit()

def run(folder, batch_size:int = 16, sleep_seconds: float=1.5): 
    chunks = ingest_folder(folder)

    db_connection = get_connection_db()
    setup_table(db_connection)

    for i in range(0, len(chunks), batch_size): 
        batch = chunks[i: i + batch_size]
        texts = [c["text"] for c in batch]

        vectors = embed_batch(texts, input_type="passage")
        insert_chunks(db_connection, batch, vectors)

        time.sleep(sleep_seconds)

    db_connection.close()

        


