"""Qdrant vector-store integration isolated from application logic."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.logger import get_logger
from app.models import DocumentChunk

logger = get_logger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when Qdrant operations fail."""


class QdrantVectorStore:
    """Create collections, upload vectors, and search Qdrant."""

    def __init__(self, url: str, collection_name: str, api_key: str | None = None) -> None:
        from qdrant_client import QdrantClient

        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key, timeout=10.0)

    def ensure_collection(self, vector_size: int) -> None:
        """Create the configured collection when it does not already exist."""
        from qdrant_client.http import models as qmodels

        try:
            collections = self.client.get_collections().collections
            if any(collection.name == self.collection_name for collection in collections):
                return
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        except Exception as exc:
            logger.exception(
                "qdrant_ensure_collection_failed",
                extra={"collection": self.collection_name},
            )
            raise VectorStoreError("Qdrant collection setup failed") from exc

    def upload_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        batch_size: int = 64,
    ) -> None:
        """Upload chunk embeddings and metadata payloads to Qdrant."""
        from qdrant_client.http import models as qmodels

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        try:
            for start in range(0, len(chunks), batch_size):
                batch_chunks = chunks[start : start + batch_size]
                batch_embeddings = embeddings[start : start + batch_size]
                points = [
                    qmodels.PointStruct(
                        id=str(uuid4()),
                        vector=vector,
                        payload=chunk.payload(),
                    )
                    for chunk, vector in zip(batch_chunks, batch_embeddings, strict=True)
                ]
                self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:
            logger.exception(
                "qdrant_upload_failed",
                extra={"collection": self.collection_name},
            )
            raise VectorStoreError("Qdrant upload failed") from exc

    def is_empty(self) -> bool:
        """Return whether the collection has no indexed points or does not yet exist."""
        try:
            count = self.client.count(collection_name=self.collection_name, exact=False)
            return int(count.count) == 0
        except Exception as exc:
            message = str(exc).lower()
            if "not found" in message or "doesn't exist" in message:
                return True
            logger.exception(
                "qdrant_count_failed",
                extra={"collection": self.collection_name},
            )
            raise VectorStoreError("Qdrant count failed") from exc

    def search(self, query_vector: list[float], limit: int = 5) -> list[Any]:
        """Search the collection for the most similar vectors."""
        try:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=limit,
                    with_payload=True,
                )
                return list(getattr(response, "points", response))
            return list(
                self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                )
            )
        except Exception as exc:
            logger.exception(
                "qdrant_search_failed",
                extra={"collection": self.collection_name},
            )
            raise VectorStoreError("Qdrant search failed") from exc
