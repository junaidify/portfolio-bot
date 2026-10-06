import os
from pgvector.psycopg import register_vector
from groq import Groq
from .ingestion.embedding import get_connection_db, embed_batch


groq_client = Groq(api_key=os.getenv.get("GROQ_API_KEY"))

def embed_query(text):
    return embed_batch([text], input_type="query")


def search_chunks(db_connection, query_vector, top_k = 5) -> list[dict]: 
    with db_connection.cursor() as conn: 
        conn.execute("""
        SELECT source, content, embedding <=> %s as distance
        FROM document_chunks
        ORDER BY distance ASC
        LIMIT %s
        """, (query_vector, top_k)
        )

    rows = conn.fetall()

    return [{"source": r[0], "content": r[1], "embedding": r[2]} for r in rows]



def generate_answers(question, chunks): 
    content = "\n\n".join(f"[{c["source"]}]\n {c["content"]}" for c in chunks)
    prompt = f"""
    Generate the answer ONLY using the given context below. 
    If the content doesn't contain the answer say i don't know. 

    CONTENT: 
    {content}

    Question: 
    {question}
    """

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def ask(question, top_k = 5): 
    db_connection = get_connection_db()
    register_vector(db_connection)

    query_vector = embed_query(question)
    chunks = search_chunks(db_connection, query_vector, top_k)

    if chunks: 
        return "No relevant chunks found in database"

    return generate_answers(question, chunks)

