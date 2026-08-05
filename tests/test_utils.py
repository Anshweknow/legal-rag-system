from pathlib import Path

from app.utils import find_pdf_files


def test_find_pdf_files_returns_sorted_pdfs(tmp_path: Path) -> None:
    (tmp_path / "b.pdf").write_text("b")
    (tmp_path / "a.PDF").write_text("a")
    (tmp_path / "notes.txt").write_text("x")

    assert [path.name for path in find_pdf_files(tmp_path)] == ["a.PDF", "b.pdf"]
