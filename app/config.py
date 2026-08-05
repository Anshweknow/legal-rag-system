"""Environment-based application configuration."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OPENROUTER_MODEL = "google/gemma-4-26b-a4b-it:free"


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    openrouter_api_key: str
    openrouter_model: str
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    pdf_dir: Path
    embedding_model_name: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    min_retrieval_score: float


def _load_dotenv() -> None:
    if importlib.util.find_spec("dotenv") is not None:
        dotenv = __import__("dotenv")
        dotenv.load_dotenv()


def _get_int(name: str, default: str) -> int:
    value = os.getenv(name, default)
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc


def _get_float(name: str, default: str) -> float:
    value = os.getenv(name, default)
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc


def get_settings() -> Settings:
    """Load settings from the environment and `.env` if present."""
    _load_dotenv()
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        openrouter_model=os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip(),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333").strip(),
        qdrant_api_key=(os.getenv("QDRANT_API_KEY") or None),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "legal_documents").strip(),
        pdf_dir=Path(os.getenv("PDF_DIR", "data/pdfs")),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"
        ).strip(),
        chunk_size=_get_int("CHUNK_SIZE", "700"),
        chunk_overlap=_get_int("CHUNK_OVERLAP", "120"),
        retrieval_top_k=_get_int("RETRIEVAL_TOP_K", "5"),
        min_retrieval_score=_get_float("MIN_RETRIEVAL_SCORE", "0.25"),
    )


def validate_settings(settings: Settings, *, require_openrouter: bool = True) -> None:
    """Fail fast when required configuration is missing or internally inconsistent."""
    missing = []
    if require_openrouter and not settings.openrouter_api_key:
        missing.append("OPENROUTER_API_KEY")
    if not settings.openrouter_model:
        missing.append("OPENROUTER_MODEL")
    if not settings.qdrant_url:
        missing.append("QDRANT_URL")
    if not settings.qdrant_collection:
        missing.append("QDRANT_COLLECTION")
    if not settings.embedding_model_name:
        missing.append("EMBEDDING_MODEL_NAME")
    if missing:
        raise ConfigurationError(f"Missing required configuration: {', '.join(missing)}")
    if settings.chunk_size <= 0:
        raise ConfigurationError("CHUNK_SIZE must be greater than 0")
    if settings.chunk_overlap < 0 or settings.chunk_overlap >= settings.chunk_size:
        raise ConfigurationError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
    if settings.retrieval_top_k <= 0:
        raise ConfigurationError("RETRIEVAL_TOP_K must be greater than 0")
    if not 0 <= settings.min_retrieval_score <= 1:
        raise ConfigurationError("MIN_RETRIEVAL_SCORE must be between 0 and 1")
