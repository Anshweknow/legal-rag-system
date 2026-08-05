"""FastAPI application entrypoint and ingestion CLI hook."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.config import ConfigurationError, get_settings, validate_settings
from app.ingestion import run_ingestion
from app.llm import LLMError, OpenRouterClient
from app.logger import configure_logging, get_logger
from app.prompts import UNKNOWN_ANSWER, build_rag_prompt
from app.qdrant_client import VectorStoreError
from app.retrieval import RetrievalError, RetrievalService

configure_logging()
logger = get_logger(__name__)

app = FastAPI(title="Legal RAG System")


class AskRequest(BaseModel):
    """Request body for the question-answering endpoint."""

    question: str = Field(..., min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        """Reject empty or whitespace-only questions."""
        if not value.strip():
            raise ValueError("question must not be empty")
        return value.strip()


class Citation(BaseModel):
    """Source citation returned with an answer."""

    document: str
    page: int
    retrieved_text: str
    score: float


class AskResponse(BaseModel):
    """Response body for the question-answering endpoint."""

    answer: str
    citations: list[Citation]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return consistent validation errors without internal details."""
    logger.info("request_validation_failed", extra={"path": request.url.path})
    return JSONResponse(status_code=422, content={"detail": "Invalid request body"})


@app.on_event("startup")
def validate_configuration_on_startup() -> None:
    """Validate required configuration before serving requests."""
    try:
        validate_settings(get_settings())
    except ConfigurationError as exc:
        logger.error("configuration_invalid", extra={"error": str(exc)})
        raise RuntimeError(str(exc)) from exc


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    """Return a cached retrieval service to reuse model and client instances."""
    return RetrievalService()


@lru_cache(maxsize=1)
def get_llm_client() -> OpenRouterClient:
    """Return a cached OpenRouter client."""
    settings = get_settings()
    return OpenRouterClient(
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return service health information."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    llm_client: OpenRouterClient = Depends(get_llm_client),
) -> AskResponse:
    """Answer a question using only chunks retrieved from Qdrant."""
    try:
        retrieved_chunks = retrieval_service.retrieve(request.question)
    except (RetrievalError, VectorStoreError):
        logger.exception("ask_retrieval_failed")
        raise HTTPException(status_code=503, detail="Retrieval service unavailable")

    if not retrieved_chunks:
        return AskResponse(answer=UNKNOWN_ANSWER, citations=[])

    prompt = build_rag_prompt(request.question, retrieved_chunks)
    try:
        answer = llm_client.generate(prompt)
    except LLMError:
        logger.exception("ask_llm_failed")
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    citations = [Citation(**chunk.__dict__) for chunk in retrieved_chunks]
    if answer.strip() == UNKNOWN_ANSWER:
        citations = []
    return AskResponse(answer=answer, citations=citations)


if __name__ == "__main__":
    settings = get_settings()
    validate_settings(settings, require_openrouter=False)
    result = run_ingestion()
    logger.info(
        "ingestion_finished",
        extra={
            "pdf_count": result.pdf_count,
            "page_count": result.page_count,
            "chunk_count": result.chunk_count,
        },
    )
