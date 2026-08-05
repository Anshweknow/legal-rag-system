"""Shared data models for document ingestion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentPage:
    """Text extracted from a single PDF page."""

    document: str
    page: int
    text: str


@dataclass(frozen=True)
class DocumentChunk:
    """Chunked text with source metadata."""

    document: str
    page: int
    text: str

    def payload(self) -> dict[str, str | int]:
        """Return the Qdrant payload representation for this chunk."""
        return {"document": self.document, "page": self.page, "text": self.text}
