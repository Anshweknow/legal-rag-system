from types import SimpleNamespace

from app.retrieval import RetrievalService


class FakeEmbedder:
    def embed_texts(self, texts):
        assert texts == ["question"]
        return [[0.1, 0.2]]


class FakeVectorStore:
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
