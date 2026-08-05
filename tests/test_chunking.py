from app.chunking import TextChunker
from app.models import DocumentPage


def test_chunk_page_preserves_metadata() -> None:
    page = DocumentPage(
        document="case.pdf",
        page=3,
        text="First sentence. Second sentence. Third sentence. Fourth sentence.",
    )
    chunks = TextChunker(chunk_size=35, overlap=8).chunk_page(page)

    assert chunks
    assert all(chunk.document == "case.pdf" for chunk in chunks)
    assert all(chunk.page == 3 for chunk in chunks)
    assert all(chunk.text for chunk in chunks)


def test_chunker_rejects_invalid_overlap() -> None:
    try:
        TextChunker(chunk_size=100, overlap=100)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
