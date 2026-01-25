
## Customer Sentiment RAG

Small demo that runs a full retrieval-augmented generation loop: chunk + embed a sentiment dataset, retrieve similar chunks, and generate grounded answers with citations.



## Project Structure
```bash
Sentiment-Retrieval-Pipeline/
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/
│   └── dataset.csv
├── backend/
│   └── app/
│       ├── main.py          # FastAPI bootstrap + app state
│       ├── routes.py        # Routes: /, /config, /query, /generate
│       ├── schemas.py       # Pydantic models + defaults
│       ├── ingestion.py     # CSV sanitation + chunking
│       ├── embeddings.py    # OpenAI embedding helper
│       ├── retrieval.py     # Chroma persistence + querying
│       └── generation.py    # Grounded generation with citations
├── frontend/
│   ├── index.html           # UI shell and styles
│   ├── styles.css           # (minimal) layout/typography
│   └── app.js               # Fetch /generate, render answer + chunks
└── vector_store/           
    
```




## Dataset

Sentiment Analysis Dataset CSV.
Mixed domains (support tickets, travel reviews, product feedback) with sentiment labels, timestamps, and confidence scores. Lightweight (<30k chars, ≤200 rows) so rebuilds are fast. Metadata like sentiment, location, and source make retrieval demos informative.



  
## RAG Pipeline (demo defaults)

1) **Chunking**  
- 260-char windows with 40-char overlap to avoid cutting sentences too harshly.  
- Each chunk keeps metadata (sentiment, source, location, etc.).

2) **Embeddings**  
- OpenAI `text-embedding-3-small`.  
- All chunks embedded once on startup.

3) **Vector DB**  
- Chroma persistent client (local directory).  
- Collection dropped/rebuilt on startup (keeps demo clean; dataset is tiny).

4) **Retrieval**  
- Defaults: `DEFAULT_TOP_K=4`, `DEFAULT_MIN_SCORE=0.62` (configurable via env and UI).  
- Filters out low-score chunks.

5) **Generation**  
- OpenAI Responses API (`GENERATION_MODEL`, default `gpt-4.1-mini`).  
- Prompt forces grounding and chunk-id citations (`[record-XX-chunk-YY]`).  
- If context is empty, responds that it cannot answer from provided context.





## Frontend

Plain HTML/CSS/JS. Users enter a query and tweak `top_k`, `min_score`, and max output tokens.  
The UI calls `/generate`, shows the grounded answer, then the supporting chunks with scores and metadata.




## Backend

- **main.py**: loads env, builds chunks/embeddings/vector store on startup, wires app state + router.  
- **routes.py**: `/`, `/config`, `/query`, `/generate`.  
- **schemas.py**: Pydantic models + defaults for retrieval/generation.  
- **ingestion.py**: CSV cleanup + chunking + metadata.  
- **embeddings.py**: OpenAI embedding helper.  
- **retrieval.py**: Chroma persistence + similarity search.  
- **generation.py**: Grounded generation via OpenAI Responses API with citations.




## How to Run

1. Install dependencies (via uv or pip):
   ```bash
   uv sync
   # or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ```
2. Configure environment:
   ```bash
   cp .env.example .env
   # set OPENAI_API_KEY
   # optional: CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_TOP_K, DEFAULT_MIN_SCORE, GENERATION_MODEL, DEFAULT_MAX_OUTPUT_TOKENS
   ```
3. Run API + UI:
   ```bash
   uv run uvicorn backend.app.main:app --reload
   ```
4. Open http://127.0.0.1:8000





## API Refrence

`POST /query`

```json
{
  "query": "positive travel experiences"
}
```

Response includes the original query and an array of chunks with `chunk_id`, `text`, `score`, and metadata.

Supporting endpoints:

- `GET /config` – exposes current chunking + retrieval + generation config (handy during demos).

`POST /generate`

```json
{
  "query": "positive travel experiences",
  "top_k": 4,
  "min_score": 0.62,
  "max_output_tokens": 256
}
```

Returns the grounded `answer` plus `citations` (array of chunks with ids/metadata/scores).




## Examples

**Intended questions**: “Show confident negative experiences with customer support”, “What positive travel reviews mention Sydney?”, “Are there users in Berlin who like the Spotify experience?”.
