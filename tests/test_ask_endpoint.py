import os

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from fastapi.testclient import TestClient

from app.main import app, get_llm_client, get_retrieval_service
from app.prompts import UNKNOWN_ANSWER
from app.retrieval import RetrievedChunk


class FakeRetrievalService:
    def retrieve(self, question: str):
        return [
            RetrievedChunk(
                document="policy.pdf",
                page=17,
                retrieved_text="Employees may take leave with approval.",
                score=0.92,
            )
        ]


class FakeLLMClient:
    def generate(self, prompt: str) -> str:
        assert "Employees may take leave with approval." in prompt
        return (
            "Leave requires approval. Document: policy.pdf. Page: 17. "
            "Retrieved text snippet: Employees may take leave with approval."
        )


def test_ask_endpoint_rejects_blank_question() -> None:
    client = TestClient(app)

    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_ask_endpoint_returns_answer_and_citations() -> None:
    app.dependency_overrides.clear()
    get_retrieval_service.cache_clear()
    get_llm_client.cache_clear()
    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService()
    app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient()
    client = TestClient(app)

    response = client.post("/ask", json={"question": "What is the leave policy?"})

    assert response.status_code == 200
    data = response.json()
    assert "Leave requires approval" in data["answer"]
    assert data["citations"] == [
        {
            "document": "policy.pdf",
            "page": 17,
            "retrieved_text": "Employees may take leave with approval.",
            "score": 0.92,
        }
    ]


def test_ask_endpoint_unknown_answer_has_no_citations() -> None:
    class UnknownLLMClient:
        def generate(self, prompt: str) -> str:
            return UNKNOWN_ANSWER

    app.dependency_overrides.clear()
    app.dependency_overrides[get_retrieval_service] = lambda: FakeRetrievalService()
    app.dependency_overrides[get_llm_client] = lambda: UnknownLLMClient()
    client = TestClient(app)

    response = client.post("/ask", json={"question": "Unknown?"})

    assert response.status_code == 200
    assert response.json() == {"answer": UNKNOWN_ANSWER, "citations": []}


def test_openrouter_client_sends_bearer_authorization(monkeypatch) -> None:
    from app import llm
    from app.llm import OpenRouterClient

    captured_headers = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"answer"}}]}'

    def fake_urlopen(request, timeout):
        captured_headers.update(dict(request.header_items()))
        return FakeResponse()

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

    answer = OpenRouterClient(api_key="sk-or-v1-test-key").generate("Question?")

    assert answer == "answer"
    assert captured_headers["Authorization"] == "Bearer sk-or-v1-test-key"
