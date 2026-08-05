"""Text chunking utilities for page-level PDF content."""

from __future__ import annotations

import re

from app.models import DocumentChunk, DocumentPage

_SENTENCE_END_RE = re.compile(r"[.!?]\s+")


class TextChunker:
    """Split page text into overlapping chunks while preserving metadata."""

    def __init__(self, chunk_size: int = 700, overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_page(self, page: DocumentPage) -> list[DocumentChunk]:
        """Split one parsed page into document chunks."""
        text = self._normalize(page.text)
        chunks: list[DocumentChunk] = []
        start = 0
        while start < len(text):
            end = self._choose_end(text, start)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(document=page.document, page=page.page, text=chunk_text)
                )
            if end >= len(text):
                break
            start = max(end - self.overlap, start + 1)
            while start < len(text) and text[start].isspace():
                start += 1
        return chunks

    def chunk_pages(self, pages: list[DocumentPage]) -> list[DocumentChunk]:
        """Split many parsed pages into chunks."""
        return [chunk for page in pages for chunk in self.chunk_page(page)]

    def _choose_end(self, text: str, start: int) -> int:
        hard_end = min(start + self.chunk_size, len(text))
        if hard_end == len(text):
            return hard_end
        window = text[start:hard_end]
        matches = list(_SENTENCE_END_RE.finditer(window))
        if not matches:
            return hard_end
        sentence_end = start + matches[-1].end()
        minimum_end = start + int(self.chunk_size * 0.6)
        return sentence_end if sentence_end >= minimum_end else hard_end

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
