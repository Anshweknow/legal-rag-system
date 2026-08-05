"""FastAPI application entrypoint and ingestion CLI hook."""

from __future__ import annotations

from fastapi import FastAPI

from app.ingestion import run_ingestion
from app.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

app = FastAPI(title="Legal RAG System")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return service health information."""
    return {"status": "ok"}


if __name__ == "__main__":
    result = run_ingestion()
    logger.info(
        "Ingestion finished: %s PDFs, %s pages, %s chunks",
        result.pdf_count,
        result.page_count,
        result.chunk_count,
    )
