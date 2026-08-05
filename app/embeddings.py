"""Embedding service using sentence-transformers."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbeddingService:
    """Reusable sentence-transformers embedding wrapper."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self.model = _load_sentence_transformer(model_name)

    @property
    def dimension(self) -> int:
        """Return the vector dimension produced by the embedding model."""
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Embedding model did not report a vector dimension")
        return int(dimension)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate document/passage embeddings for a sequence of texts."""
        return self._encode(texts)

    def embed_query(self, question: str) -> list[float]:
        """Generate a query embedding using the model's retrieval instruction."""
        text = question.strip()
        if self._uses_bge_instruction():
            text = f"{BGE_QUERY_PREFIX}{text}"
        return self._encode([text])[0]

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def _uses_bge_instruction(self) -> bool:
        return "bge-" in self.model_name.lower()
