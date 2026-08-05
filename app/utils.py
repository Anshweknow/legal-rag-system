"""Small filesystem utilities used by ingestion."""

from __future__ import annotations

from pathlib import Path


def find_pdf_files(pdf_dir: Path) -> list[Path]:
    """Return all PDF files in a directory sorted by filename."""
    if not pdf_dir.exists():
        return []
    return sorted(path for path in pdf_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
