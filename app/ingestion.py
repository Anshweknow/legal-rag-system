"""End-to-end document ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from app.chunking import TextChunker
from app.config import Settings, get_settings
from app.embeddings import EmbeddingService
from app.logger import get_logger
from app.parser import PDFParser
from app.qdrant_client import QdrantVectorStore
from app.utils import find_pdf_files

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """Summary of an ingestion run."""

    pdf_count: int
    page_count: int
    chunk_count: int


class IngestionPipeline:
    """Parse PDFs, chunk text, embed chunks, and upload them to Qdrant."""

    def __init__(
        self,
        settings: Settings,
        parser: PDFParser | None = None,
        chunker: TextChunker | None = None,
        embedder: EmbeddingService | None = None,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.parser = parser or PDFParser()
        self.chunker = chunker or TextChunker(settings.chunk_size, settings.chunk_overlap)
        self.embedder = embedder or EmbeddingService(settings.embedding_model_name)
        self.vector_store = vector_store or QdrantVectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection,
        )

    def run(self) -> IngestionResult:
        """Execute the full ingestion pipeline."""
        pdf_files = find_pdf_files(self.settings.pdf_dir)
        logger.info("Found %s PDF files in %s", len(pdf_files), self.settings.pdf_dir)

        pages = []
        for pdf_file in pdf_files:
            parsed_pages = self.parser.parse_file(pdf_file)
            logger.info("Parsed %s non-empty pages from %s", len(parsed_pages), pdf_file.name)
            pages.extend(parsed_pages)

        chunks = self.chunker.chunk_pages(pages)
        logger.info("Created %s chunks from %s pages", len(chunks), len(pages))

        if chunks:
            self.vector_store.ensure_collection(self.embedder.dimension)
            embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
            self.vector_store.upload_chunks(chunks, embeddings)
            logger.info("Uploaded %s chunks to Qdrant", len(chunks))

        return IngestionResult(
            pdf_count=len(pdf_files), page_count=len(pages), chunk_count=len(chunks)
        )


def run_ingestion() -> IngestionResult:
    """Create and run the default ingestion pipeline from environment settings."""
    return IngestionPipeline(get_settings()).run()
