"""Prompt construction for grounded legal question answering."""

from __future__ import annotations

from app.retrieval import RetrievedChunk

UNKNOWN_ANSWER = "The information is not available in the supplied documents."


def build_rag_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """Build a prompt that constrains the LLM to retrieved document context."""
    context_blocks = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"Source {index}:",
                    f"Document: {chunk.document}",
                    f"Page: {chunk.page}",
                    f"Retrieved text snippet: {chunk.retrieved_text}",
                    f"Similarity score: {chunk.score:.4f}",
                ]
            )
        )
    context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."

    return f"""You are a retrieval-augmented question-answering assistant for legal documents.
Use ONLY the retrieved context below to answer the user's question.
Never use outside knowledge, invent facts, infer unsupported details, or guess.
If the answer is missing from the retrieved context, respond exactly:
{UNKNOWN_ANSWER}

Every answer found in the context must be concise and factual, and must include the supporting document name, page number, and retrieved text snippet.

Retrieved context:
{context}

Question:
{question}

Answer:"""
