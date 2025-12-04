from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .embeddings import EmbeddingService
from .ingestion import chunk_records, load_dataset
from .retrieval import VectorRetriever

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = Path(os.getenv("DATASET_PATH", PROJECT_ROOT / "data" / "dataset.csv"))
VECTOR_DIR = Path(os.getenv("VECTOR_DIR", PROJECT_ROOT / "vector_store"))
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", PROJECT_ROOT / "frontend"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "260"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.62"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sentiment_rag")

embedding_service: EmbeddingService | None = None
vector_retriever: VectorRetriever | None = None
chunk_total = 0


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question to search for.")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=10)
    min_score: float = Field(DEFAULT_MIN_SCORE, ge=0.0, le=1.0)


class ChunkResponse(BaseModel):
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    query: str
    results: List[ChunkResponse]


class ConfigResponse(BaseModel):
    dataset_path: str
    vector_store_path: str
    chunk_size: int
    chunk_overlap: int
    chunk_total: int
    default_top_k: int
    default_min_score: float
    vector_db: str
    embedding_model: str


app = FastAPI(title="Sentiment RAG Retrieval", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    global embedding_service, vector_retriever, chunk_total

    records = load_dataset(DATASET_PATH)
    chunks = chunk_records(records, CHUNK_SIZE, CHUNK_OVERLAP)
    chunk_total = len(chunks)

    embedding_service = EmbeddingService(model=EMBEDDING_MODEL)
    vector_retriever = VectorRetriever(
        vector_dir=VECTOR_DIR,
        collection_name=COLLECTION_NAME,
        embedding_service=embedding_service,
    )
    vector_retriever.build(chunks)


def _ensure_frontend() -> Path:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI not found.")
    return index_file


@app.get("/", response_class=HTMLResponse)
def serve_index() -> HTMLResponse:
    index_file = _ensure_frontend()
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/config", response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    return ConfigResponse(
        dataset_path=str(DATASET_PATH),
        vector_store_path=str(VECTOR_DIR),
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        chunk_total=chunk_total,
        default_top_k=DEFAULT_TOP_K,
        default_min_score=DEFAULT_MIN_SCORE,
        vector_db="Chroma (local persistent)",
        embedding_model=EMBEDDING_MODEL,
    )


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest) -> QueryResponse:
    if vector_retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized yet.")

    try:
        results = vector_retriever.query(request.query, request.top_k, request.min_score)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        query=request.query,
        results=[ChunkResponse(**result) for result in results],
    )
