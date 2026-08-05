# Legal RAG System

Foundation for a Retrieval-Augmented Generation application that ingests legal PDF documents into Qdrant.

## Current scope

This version implements project setup and document ingestion only:

1. Scan `data/pdfs/` for PDF files.
2. Parse each PDF page-by-page with PyMuPDF.
3. Split extracted page text into overlapping chunks.
4. Generate embeddings with `BAAI/bge-small-en-v1.5` via `sentence-transformers`.
5. Store vectors and source metadata in Qdrant.

Question answering and OpenRouter integration are intentionally not implemented yet.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Update `.env` for your Qdrant instance if needed.

## Run ingestion

Start Qdrant, then run:

```bash
python -m app.main
```

The ingestion pipeline is also reusable from Python:

```python
from app.ingestion import run_ingestion

result = run_ingestion()
```
