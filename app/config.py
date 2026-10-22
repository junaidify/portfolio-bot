"""
Centralised configuration — loads .env once via pydantic-settings.
Every other module imports `settings` from here instead of calling
os.getenv() / load_dotenv() on its own.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── NVIDIA Embeddings ─────────────────────────────────────────
    nvidia_api_key: str
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_embed_model: str = "nvidia/nv-embedqa-e5-v5"
    nvidia_embed_dim: int = 1024
    nvidia_embed_batch_size: int = 16
    nvidia_embed_sleep_seconds: float = 1.5

    # ── PostgreSQL / pgvector ─────────────────────────────────────
    pgdatabase: str = "postgres"
    pguser: str = "postgres"
    pgpassword: str = ""
    pghost: str = "localhost"
    pgport: int = 5432

    # ── Groq LLM ─────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # ── RAG tuning ────────────────────────────────────────────────
    scope_distance_threshold: float = 0.75


settings = Settings()
