# Sentiment RAG Retrieval

This repository captures my thinking process for building a retrieval-only RAG demo. Instead of obsessing over model orchestration, I focused on making every design decision explicit so reviewers can understand why the system behaves the way it does.

## Why this dataset?
- **Source**: a tiny slice of Kaggle’s “Sentiment Analysis Dataset (Multi-Source)” (≈100 rows).
- **Reasoning**: the assignment asked for <30k characters and ≤200 rows. This dataset already mixes domains (support tickets, travel reviews, product feedback) and comes with sentiment labels, timestamps, and confidence scores. That gives retrieval queries something meaningful to latch onto (sentiment, location, channel) while staying cheap to embed.
- **Intended questions**: “Show confident negative experiences with customer support”, “What positive travel reviews mention Sydney?”, “Are there users in Berlin who like the Spotify experience?”. Each query is grounded in metadata present in the CSV.

## Pipeline decisions

### Chunking
- **Strategy**: character-based split (`CHUNK_SIZE=260`, `CHUNK_OVERLAP=40`).
- **Why**: Each entry is short but I still wanted to avoid slicing mid-sentence when a review is slightly longer. Overlap keeps metadata aligned and preserves context for borderline cases, yet the small dataset keeps costs low.
- **Metadata**: each chunk carries sentiment, source, date, user/id, location, confidence, chunk index, and char positions so the UI can surface meaningful context (e.g., “Chunk 0 • Sentiment: Positive • Source: TripAdvisor”).

### Embeddings
- **Model**: `text-embedding-3-small`.
- **Rationale**: balances quality and price. We only embed a few hundred chunks, so the total cost is negligible (<$0.001). A stronger model would be overkill; a weaker one would make cross-domain retrieval noisy.
- **Implementation**: `backend/app/embeddings.py` wraps the OpenAI client. I intentionally kept it thin—if I swap providers later, it’s one file.

### Vector store
- **Choice**: local Chroma PersistentClient.
- **Reasoning**: No need for Pinecone/Weaviate overhead. Chroma lets me rebuild on every startup and store files under `vector_store/` (git-ignored) without external services. It defaults to cosine similarity (HNSW), which matches the embedding model.
- **Rebuild policy**: on startup I wipe the collection to ensure the index matches the CSV. With such a small dataset, rebuild time is seconds.

### Retrieval parameters
- **Top-K & min-score**: loaded from env (`DEFAULT_TOP_K`, `DEFAULT_MIN_SCORE`) and enforced server-side. I removed UI controls after realizing they weakened the “document decisions” story: I’d rather demonstrate deliberate defaults than expose sliders.
- **Why `min_score=0.62` initially**: empirical check—anything below ~0.6 felt like noise. I documented it, but it’s easy to change in `.env` if the reviewer wants to explore.
- **Query flow**: each request embeds the question, queries Chroma for up to `top_k*2` candidates (to allow filtering), and returns the best matches that pass the score threshold. No synonyms, no prompt engineering—pure similarity.

## Frontend choices
- **Tech**: plain HTML/CSS/JS (no build step). The UI encourages the reviewer to focus on retrieval output rather than UI widgets.
- **Design**: simple form with a query textarea and a hint explaining that retrieval parameters are env-driven. Results show chunk metadata pills so you can understand “why” a chunk was returned.
- **Fetch logic**: error handling is minimal but clear; it surfaces FastAPI errors (e.g., missing query or API key).

## Backend architecture
- `backend/app/main.py` orchestrates everything:
  - Loads `.env` to capture dataset paths, vector store location, chunking params, embedding model, retrieval thresholds.
  - On startup: loads the CSV (`ingestion.py`), chunks it, builds embeddings, and writes to Chroma (`retrieval.py`).
  - Endpoints: `/` serves the frontend, `/health` for liveness, `/config` exposes current settings (useful when presenting), `/query` performs retrieval.
- `ingestion.py`: handles the csv quirks (quote sanitization) and chunk metadata. Reasoning: I wanted deterministic ingestion to explain exactly what goes into each chunk.
- `retrieval.py`: purposely lean—no hybrid search, no reranking—so I can describe the entire retrieval stack in a few sentences.

## Setup instructions
1. Use [uv](https://github.com/astral-sh/uv) to install dependencies:
   ```bash
   uv sync
   ```
2. Configure environment:
   ```bash
   cp .env.example .env
   # add OPENAI_API_KEY
   # optionally tweak CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_TOP_K, DEFAULT_MIN_SCORE, etc.
   ```
3. Run the API + UI:
   ```bash
   uv run uvicorn backend.app.main:app --reload
   ```
4. Visit http://127.0.0.1:8000. The UI only accepts the query text; everything else is driven by env values to keep the decision log tight.

## API reference

`POST /query`

```json
{
  "query": "positive travel experiences"
}
```

Response includes the original query and an array of chunks with `chunk_id`, `text`, `score`, and metadata.

Supporting endpoints:
- `GET /health` – quick status check.
- `GET /config` – exposes the current chunking + retrieval configuration (handy during demos).

## Thought process summary
1. **Constraints first**: the assignment capped dataset size and cost, so I intentionally selected a dataset that already meets those limits and documents user sentiment.
2. **Explainable defaults**: Rather than add toggles everywhere, I favored environment variables with documented reasoning (chunk size, overlap, retrieval thresholds). This makes the reviewer’s job easier when assessing trade-offs.
3. **Keep UI humble**: a minimalist UI prevents scope creep and highlights retrieval behavior. It also stresses that this is a retrieval component, not a full chatbot.
4. **Local-first tooling**: by using uv and Chroma, the reviewer can spin up the project offline (aside from the OpenAI call) without provisioning services.
5. **Future levers** (documented for completeness):
   - Add chunk hashing to skip re-embedding unchanged data.
   - Introduce simple filters (e.g., sentiment dropdown) once the reasoning section already justifies the dataset.
   - Experiment with reranking or cross-encoder validation if retrieval quality ever becomes the bottleneck.

This README is as much a narrative as documentation—it should help evaluators see that every choice (dataset, chunking, embedding model, vector DB, UI minimalism) was deliberate and aligned with the assignment’s constraints.
