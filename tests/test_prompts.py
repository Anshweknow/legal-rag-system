from app.prompts import UNKNOWN_ANSWER, build_rag_prompt
from app.retrieval import RetrievedChunk


def test_build_rag_prompt_includes_grounding_rules_and_sources() -> None:
    prompt = build_rag_prompt(
        "What did the court hold?",
        [
            RetrievedChunk(
                document="case.pdf",
                page=7,
                retrieved_text="The court held that notice was required.",
                score=0.91,
            )
        ],
    )

    assert "Use ONLY the retrieved context" in prompt
    assert "Never use outside knowledge" in prompt
    assert UNKNOWN_ANSWER in prompt
    assert "Document: case.pdf" in prompt
    assert "Page: 7" in prompt
    assert "The court held that notice was required." in prompt
