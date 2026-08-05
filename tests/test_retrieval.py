from types import SimpleNamespace

from app.retrieval import RetrievalService


class FakeEmbedder:
    def embed_texts(self, texts):
        assert texts == ["question"]
        return [[0.1, 0.2]]


class FakeVectorStore:
    collection_name = "test"

    def is_empty(self):
        return False

    def search(self, query_vector, limit):
        assert query_vector == [0.1, 0.2]
        assert limit == 5
        return [
            SimpleNamespace(
                payload={"document": "doc.pdf", "page": 2, "text": "snippet"},
                score=0.92,
            )
        ]


def test_retrieval_returns_top_chunks_with_metadata() -> None:
    service = RetrievalService(embedder=FakeEmbedder(), vector_store=FakeVectorStore())

    chunks = service.retrieve("question")

    assert len(chunks) == 1
    assert chunks[0].document == "doc.pdf"
    assert chunks[0].page == 2
    assert chunks[0].retrieved_text == "snippet"
    assert chunks[0].score == 0.92


class LowScoreVectorStore(FakeVectorStore):
    def search(self, query_vector, limit):
        return [
            SimpleNamespace(
                payload={"document": "doc.pdf", "page": 2, "text": "snippet"},
                score=0.05,
            )
        ]


class EmptyVectorStore(FakeVectorStore):
    def is_empty(self):
        return True

    def search(self, query_vector, limit):  # pragma: no cover - should not be called
        raise AssertionError("search should not run for empty collections")


def test_retrieval_returns_empty_for_low_similarity() -> None:
    service = RetrievalService(
        embedder=FakeEmbedder(),
        vector_store=LowScoreVectorStore(),
        min_score=0.25,
    )

    assert service.retrieve("question") == []


def test_retrieval_skips_embedding_for_empty_collection() -> None:
    class FailingEmbedder:
        def embed_texts(self, texts):  # pragma: no cover - should not be called
            raise AssertionError("embedding should not run for an empty collection")

    service = RetrievalService(embedder=FailingEmbedder(), vector_store=EmptyVectorStore())

    assert service.retrieve("question") == []
