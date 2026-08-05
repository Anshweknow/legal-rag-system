"""PDF parsing utilities based on PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import fitz

from app.logger import get_logger
from app.models import DocumentPage

logger = get_logger(__name__)


class PDFParser:
    """Parse PDF files into page-level text records."""

    def parse_file(self, pdf_path: Path) -> list[DocumentPage]:
        """Extract non-empty text pages from a PDF file.

        Parsing errors are logged and converted to an empty result so a single
        bad PDF does not stop the complete ingestion run.
        """
        pages: list[DocumentPage] = []
        try:
            with fitz.open(pdf_path) as document:
                for index, page in enumerate(document, start=1):
                    text = page.get_text("text").strip()
                    if not text:
                        logger.debug("Skipping empty page %s in %s", index, pdf_path.name)
                        continue
                    pages.append(DocumentPage(document=pdf_path.name, page=index, text=text))
        except Exception as exc:  # PyMuPDF raises multiple exception types.
            logger.exception("Failed to parse PDF %s: %s", pdf_path, exc)
        return pages
