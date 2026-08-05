"""Embedding service using sentence-transformers."""

from __future__ import annotations

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Reusable sentence-transformers embedding wrapper."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Return the vector dimension produced by the embedding model."""
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Embedding model did not report a vector dimension")
        return int(dimension)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for a sequence of texts."""
        if not texts:
            return []
        embeddings = self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()
