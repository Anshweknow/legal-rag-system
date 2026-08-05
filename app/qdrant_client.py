"""Qdrant vector-store integration isolated from application logic."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.models import DocumentChunk


class QdrantVectorStore:
    """Create collections, upload vectors, and search Qdrant."""

    def __init__(self, url: str, collection_name: str, api_key: str | None = None) -> None:
        from qdrant_client import QdrantClient

        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self, vector_size: int) -> None:
        """Create the configured collection when it does not already exist."""
        from qdrant_client.http import models as qmodels

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

    def search(self, query_vector: list[float], limit: int = 5) -> list[Any]:
        """Search the collection for the most similar vectors."""
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
