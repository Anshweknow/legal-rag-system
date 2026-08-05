from app import embeddings
from app.embeddings import BGE_QUERY_PREFIX, EmbeddingService


class FakeModel:
    def __init__(self) -> None:
        self.encoded_texts = []

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, normalize_embeddings, show_progress_bar):
        self.encoded_texts.append(list(texts))

        class FakeEmbeddings:
            def tolist(self):
                return [[0.1, 0.2] for _ in texts]

        return FakeEmbeddings()


def test_bge_query_embedding_adds_retrieval_instruction(monkeypatch) -> None:
    model = FakeModel()
    monkeypatch.setattr(embeddings, "_load_sentence_transformer", lambda model_name: model)

    service = EmbeddingService("BAAI/bge-small-en-v1.5")

    assert service.embed_query("What is personal liberty?") == [0.1, 0.2]
    assert model.encoded_texts[-1] == [
        f"{BGE_QUERY_PREFIX}What is personal liberty?"
    ]


def test_document_embeddings_do_not_add_query_instruction(monkeypatch) -> None:
    model = FakeModel()
    monkeypatch.setattr(embeddings, "_load_sentence_transformer", lambda model_name: model)

    service = EmbeddingService("BAAI/bge-small-en-v1.5")

    assert service.embed_texts(["passage text"]) == [[0.1, 0.2]]
    assert model.encoded_texts[-1] == ["passage text"]
