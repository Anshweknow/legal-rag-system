"""Environment-based application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    openrouter_api_key: str
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    pdf_dir: Path
    embedding_model_name: str
    chunk_size: int
    chunk_overlap: int


def get_settings() -> Settings:
    """Load settings from the environment and `.env` if present."""
    load_dotenv()
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "legal_documents"),
        pdf_dir=Path(os.getenv("PDF_DIR", "data/pdfs")),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "700")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
    )
