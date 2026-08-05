"""Retrieval layer for question answering over indexed legal documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
from app.embeddings import EmbeddingService
from app.logger import get_logger
from app.qdrant_client import QdrantVectorStore

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A document chunk retrieved from the vector store."""

    document: str
    page: int
    retrieved_text: str
    score: float


class RetrievalService:
    """Embed user queries and retrieve relevant chunks from Qdrant."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: EmbeddingService | None = None,
        vector_store: QdrantVectorStore | None = None,
        top_k: int = 5,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or EmbeddingService(self.settings.embedding_model_name)
        self.vector_store = vector_store or QdrantVectorStore(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            collection_name=self.settings.qdrant_collection,
        )
        self.top_k = top_k

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Return the top relevant chunks for a question."""
        try:
            query_vector = self.embedder.embed_texts([question])[0]
            results = self.vector_store.search(query_vector=query_vector, limit=self.top_k)
        except Exception:
            logger.exception("retrieval_failed", extra={"top_k": self.top_k})
            raise

        return [self._to_retrieved_chunk(result) for result in results]

    @staticmethod
    def _to_retrieved_chunk(result: Any) -> RetrievedChunk:
        payload = getattr(result, "payload", None) or {}
        return RetrievedChunk(
            document=str(payload.get("document", "")),
            page=int(payload.get("page", 0)),
            retrieved_text=str(payload.get("text", "")),
            score=float(getattr(result, "score", 0.0) or 0.0),
        )
