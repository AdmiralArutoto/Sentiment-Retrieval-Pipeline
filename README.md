
## Customer Sentiment Retrieval

This project is a small demo of how the retrieval part of a RAG system works. There is no LLM generation, only searching for the most relevant text chunks using embeddings.



## Project Structure
```bash
rag_assignment/
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/
│   └── dataset.csv
├── backend/
│   └── app/
│       ├── main.py          # FastAPI bootstrap + routes
│       ├── ingestion.py     # CSV sanitation + chunking
│       ├── embeddings.py    # OpenAI embedding helper
│       └── retrieval.py     # Chroma persistence + querying
├── frontend/
│   ├── index.html           # Minimal UI shell
│   ├── styles.css           # Layout/typography
│   └── app.js               # Fetch /query and render results
└── vector_store/           
    
```




## Dataset

 Sentiment Analysis Dataset CSV. 
 I used this dataset because it has mixed domains (support tickets, travel reviews, product feedback) and comes with sentiment labels, timestamps, and confidence scores, and is relativly lightweight ( <30k characters and ≤200 rows).
  The mixed Domains and lables give the retrieval queries something to latch onto (location, sentiment, etc..).
  



  
## RAG Pipeline

How the pipeline works
1. **Chunking**

- I split each review into small text “chunks.”

- Chunk size is 260 characters, overlap 40 characters.

- The overlap helps avoid cutting sentences too aggressively.

- Every chunk keeps metadata (sentiment, source, location, etc.) so we can show it later.

2. **Embeddings**

- Uses the OpenAI model text-embedding-3-small.

- It’s good enough for a demo project like this one.

- All chunks are embedded once during app startup.

3. **Vector database**

- I chose Chroma because it works locally and is simple to use.

- On each restart, the database is cleared and rebuilt. (This keeps things clean and the dataset is small anyway).

4. **Retrieval settings**

- The system is configured with defaults (DEFAULT_TOP_K = 4, DEFAULT_MIN_SCORE = 0.55) that should balance precision and recall on this dataset. The UI exposes them so we can change and see how different settings affect retrieval behavior.

- Ive picked Top-K neighbors to be 4 because it gives enough variety given the size of the dataset. Despire for the dataset being relativley small, for the min_score ive chosen 0.55 because its the middle ground fpr producing chunks that  match user's querry well enoguht.

**How retrieval works**

1. User types a question.

2. The question is embedded.

3. Chroma returns the most similar chunks.

4. The backend filters out chunks below the minimum score.

5. The results are shown in the UI.





## Frontend

Plain HTML, CSS, and a little JavaScript.

User enters the query and can adjust retrieval params (for the purpose of this demo of course).

Results appear as cards that show:
- the chunk text
- similarity score
- metadata (sentiment, user id, source, etc.)




## Backend

**Main.py**
- Loads environment variables
- Loads the dataset, chunks it, builds embeddings
- Sets up the vectorDB
- Provides /query, /config, and /health endpoints

**ingestion.py**
- cleans the CSV
- Splits text into chunks
- Prepares metadata

**embeddings.py**
- Wrapper around OpenAI embeddings

**retrieval.py**
- Handles Chroma collection
- Runs similarity search




## How to Run

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
4. Visit http://127.0.0.1:8000.





## API Refrence

`POST /query`

```json
{
  "query": "positive travel experiences"
}
```

Response includes the original query and an array of chunks with `chunk_id`, `text`, `score`, and metadata.

Supporting endpoints:

- `GET /config` – exposes the current chunking + retrieval configuration (handy during demos).




## Examples

**Intended questions**: “Show confident negative experiences with customer support”, “What positive travel reviews mention Sydney?”, “Are there users in Berlin who like the Spotify experience?”.
