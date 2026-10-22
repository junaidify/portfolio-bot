# Portfolio Bot 🤖

A **RAG-powered portfolio assistant** that ingests your personal data (PDF, DOCX, TXT, Markdown) and answers questions about you through a sleek streaming chat UI.

## Architecture

```
my_data/  ──▶  Chunking  ──▶  NVIDIA Embeddings  ──▶  pgvector (PostgreSQL)
                                                            │
User Question  ──▶  Embed Query  ──▶  Cosine Search  ──────┘
                                           │
                                     Scope Guard  ──▶  (reject off-topic)
                                           │
                                     Groq LLM (streaming)  ──▶  Chat UI
```

## Quick Start

### 1. Prerequisites

- **Python 3.11+**
- **PostgreSQL** with the `pgvector` extension installed
- **NVIDIA NIM API key** (for embeddings) — [build.nvidia.com](https://build.nvidia.com)
- **Groq API key** (for LLM) — [console.groq.com](https://console.groq.com)

### 2. Set up PostgreSQL

```sql
-- In psql:
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3. Configure environment

Edit `.env` in the project root with your credentials:

```env
NVIDIA_API_KEY=nvapi-...
GROQ_API_KEY=gsk_...
PGPASSWORD=your_pg_password
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add your data

Drop your files into `app/my_data/`:

```
app/my_data/
├── resume.pdf
├── about_me.md
├── projects.txt
└── experience.docx
```

Supported formats: `.pdf`, `.docx`, `.txt`, `.md`

### 6. Run

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000** in your browser.

## How It Works

1. **Ingestion** — On startup (and when files change), documents are chunked and embedded via NVIDIA NIM, then stored in pgvector.
2. **Query** — User questions are embedded and cosine-searched against stored chunks.
3. **Scope Guard** — If the closest chunk is too distant (configurable threshold), the bot rejects the question as out-of-scope. The system prompt further constrains answers.
4. **Streaming** — Groq streams the answer token-by-token via SSE to the browser.

## API

### `POST /api/chat`

```json
{
  "question": "What are their skills?",
  "top_k": 5
}
```

Returns `text/event-stream`:
```
data: {"token": "They "}
data: {"token": "have "}
data: {"token": "experience..."}
data: [DONE]
```

## Project Structure

```
portfolio-bot/
├── app/
│   ├── main.py              # FastAPI app + lifespan
│   ├── config.py             # Pydantic Settings
│   ├── watcher.py            # Watchdog auto-ingest
│   ├── my_data/              # YOUR data goes here
│   ├── api/routes/chat.py    # SSE streaming endpoint
│   ├── services/
│   │   ├── retrieval.py      # RAG pipeline + scope guard
│   │   └── ingestion/
│   │       ├── chunking.py   # File readers + text splitting
│   │       └── embedding.py  # NVIDIA embeddings + pgvector storage
│   └── static/index.html     # Chat UI
├── .env                      # Configuration
├── requirements.txt
└── README.md
```
