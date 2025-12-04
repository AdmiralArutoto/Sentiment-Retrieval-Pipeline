# Sentiment RAG Retrieval

This project implements the retrieval layer of a Retrieval-Augmented Generation (RAG) system. It ingests a small Kaggle sentiment dataset, chunks each entry, creates OpenAI embeddings, stores them in a Chroma vector DB, and exposes a FastAPI endpoint plus a tiny HTML UI for querying.

## Key decisions

### Dataset selection
- **Source:** Trimmed subset of the Kaggle dataset “Sentiment Analysis Dataset (Multi-Source)” – a lightweight collection of short user opinions that include sentiment labels, source, timestamps, and confidence scores.
- **Why this dataset?** Each row already contains rich, text-based context plus metadata that is useful for retrieval. It is tiny (98 rows, 7 columns, <15k characters) so it satisfies the assignment limits and keeps embedding cost well below $1.
- **Supported question types:** The curated dataset is suited for queries such as:
  - “Show confident negative experiences about customer support.”
  - “Find positive travel or restaurant reviews.”
  - “What are people saying about product quality issues?”

### Vector DB & embeddings
- **Vector DB:** [Chroma](https://docs.trychroma.com/) PersistentClient. It’s easy to run locally, requires zero external services, and effortlessly fits this small dataset. The persistence directory (`vector_store/`) lets us reuse embeddings between runs.
- **Embeddings:** `text-embedding-3-small` from OpenAI. It balances quality and price (≈$0.0001 for the entire dataset). The model ID is configurable through `EMBEDDING_MODEL`.

### Chunking & retrieval parameters
- **Chunk size:** 260 characters with **40 character overlap.** The dataset is mostly short snippets, but the overlap avoids cutting longer entries mid-thought and keeps metadata aligned with the text.
- **Retrieval:** Default Top-K = 4 and a minimum cosine similarity score of 0.62 (computed as `1 - distance` from Chroma). Users can override both values in the UI to explore broader or stricter matches.

## Project structure

```
rag-retrieval-assignment/
├── pyproject.toml
├── backend/
│   └── app/
│       ├── main.py          # FastAPI wiring + routes
│       ├── ingestion.py     # dataset loading + chunking
│       ├── embeddings.py    # OpenAI embedding helper
│       └── retrieval.py     # Chroma PersistentClient + queries
├── frontend/
│   └── index.html
├── data/
│   └── dataset.csv
└── README.md
```

## Prerequisites
- [uv](https://github.com/astral-sh/uv) (>=0.4) for dependency + virtualenv management.
- Python 3.12+
- An OpenAI API key with access to the embeddings endpoint.

## Setup & usage
1. **Install dependencies**
   ```bash
   uv sync
   ```
2. **Configure secrets**
   ```bash
   cp .env.example .env
   # set OPENAI_API_KEY inside .env
   ```
   `CHUNK_SIZE`, `CHUNK_OVERLAP`, `DEFAULT_TOP_K`, and `DEFAULT_MIN_SCORE` can be overridden via env vars to test different retrieval behaviors.
3. **Run the API + UI**
   ```bash
   uv run uvicorn backend.app.main:app --reload
   ```
   Open http://127.0.0.1:8000/ to load the UI.
4. **Interact**
   - Submit a question (e.g., “Which users in Paris had positive experiences?”).
   - Adjust Top-K or the similarity threshold to broaden/narrow matches.
   - Each result shows chunk id, source, sentiment, and metadata so you can copy the relevant context into another system.

## API reference

`POST /query`

```json
{
  "query": "positive travel experiences",
  "top_k": 4,
  "min_score": 0.6
}
```

Returns the retrieved chunks sorted by similarity. Errors include validation issues (empty query, invalid numbers) or missing embeddings due to an unset `OPENAI_API_KEY`.

Helper endpoints:
- `GET /health` – quick status check.
- `GET /config` – exposes current chunking + retrieval settings for the UI or debugging.

## UI details
- Plain HTML/CSS/JS hosted by FastAPI (`frontend/`).
- Fetches `/query` and renders chunk cards with similarity scores and metadata.
- Highlights empty states, errors, and the number of returned chunks.

## Reasoning summary
- **Dataset**: multi-domain sentiment snippets keep the scope tight yet realistic for customer-experience queries.
- **Vector DB**: Chroma keeps everything local, eliminating external dependencies and cost.
- **Embeddings**: OpenAI `text-embedding-3-small` offers strong quality for narrative text while staying far below the $1 budget.
- **Chunking**: character-based chunking plus overlap keeps metadata and sentiment context intact and is trivial to explain/document.

## Future improvements
1. Store chunk hashes to skip re-embedding when the dataset is unchanged.
2. Add lightweight analytics (e.g., sentiment filters) on top of the retrieved results.
3. Containerize the API/UI for easier deployment.
