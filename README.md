# Legal RAG System

## Project Overview
A compact Retrieval-Augmented Generation (RAG) backend for answering questions from supplied legal PDF documents. The service parses PDFs, chunks page text, embeds chunks, stores vectors in Qdrant, retrieves relevant context, and generates grounded answers through OpenRouter. If retrieved evidence is unavailable or weak, the API returns: `The information is not available in the supplied documents.`

## Architecture
`PDFs → PyMuPDF parsing → page-aware chunking → sentence-transformer embeddings → Qdrant vector storage → top-k retrieval → OpenRouter answer generation → cited API response`

## Technology Stack and Libraries
Python, FastAPI, Uvicorn, PyMuPDF, sentence-transformers, Qdrant Client, OpenRouter Chat Completions API, Pydantic, pytest.

## Models
- **Embedding model:** `BAAI/bge-small-en-v1.5` by default (`EMBEDDING_MODEL_NAME`).
- **OpenRouter model:** `google/gemma-4-26b-a4b-it:free` by default (`OPENROUTER_MODEL`).

## Project Structure
```text
app/                 Core API, ingestion, retrieval, LLM, and configuration modules
data/pdfs/           Source legal PDF documents
tests/               Unit/API tests with external services mocked
.env.example         Environment variable template
requirements.txt     Python dependencies
```

## Setup Instructions
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Environment Variables
Required for the API: `OPENROUTER_API_KEY`. Common settings: `OPENROUTER_MODEL`, `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `PDF_DIR`, `EMBEDDING_MODEL_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K`, `MIN_RETRIEVAL_SCORE`.

## Running the Project
Start Qdrant locally, for example:
```bash
docker run -p 6333:6333 qdrant/qdrant
```
## Running Ingestion
```bash
python -m app.main
```
This reads PDFs from `PDF_DIR`, creates embeddings, and uploads chunks with `document`, `page`, and `text` metadata.

## Running the API
```bash
uvicorn app.main:app --reload
```
## Example Request
```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What did the court say about personal liberty?"}'
```

## Example Response
```json
{
  "answer": "The answer must be grounded in the retrieved legal text and include document name, page number, and snippet details.",
  "citations": [
    {
      "document": "02_Maneka_Gandhi_vs_Union_of_India_p0683-p0794.pdf",
      "page": 12,
      "retrieved_text": "Relevant retrieved text snippet...",
      "score": 0.82
    }
  ]
}
```

## Known Assumptions
PDF page numbers are the physical PDF page indices returned during parsing. Qdrant must be reachable before ingestion or question answering. External services are mocked in tests; a real OpenRouter key is required only when running the live API.
