from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .embeddings import EmbeddingService
from .generation import GenerationService
from .ingestion import chunk_records, load_dataset
from .retrieval import VectorRetriever
from .routes import router
from .schemas import DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MIN_SCORE, DEFAULT_TOP_K

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = Path(os.getenv("DATASET_PATH", PROJECT_ROOT / "data" / "dataset.csv"))
VECTOR_DIR = Path(os.getenv("VECTOR_DIR", PROJECT_ROOT / "vector_store"))
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", PROJECT_ROOT / "frontend"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4.1-mini")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "260"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sentiment_rag")

app = FastAPI(title="Sentiment RAG Retrieval", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.dataset_path = DATASET_PATH
app.state.vector_dir = VECTOR_DIR
app.state.frontend_dir = FRONTEND_DIR
app.state.chunk_size = CHUNK_SIZE
app.state.chunk_overlap = CHUNK_OVERLAP
app.state.chunk_total = 0
app.state.default_top_k = DEFAULT_TOP_K
app.state.default_min_score = DEFAULT_MIN_SCORE
app.state.default_max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
app.state.embedding_model = EMBEDDING_MODEL
app.state.generation_model = GENERATION_MODEL
app.state.collection_name = COLLECTION_NAME
app.state.embedding_service = None
app.state.vector_retriever = None
app.state.generation_service = None

app.include_router(router)


@app.on_event("startup")
def _startup() -> None:
    records = load_dataset(app.state.dataset_path)
    chunks = chunk_records(records, app.state.chunk_size, app.state.chunk_overlap)
    app.state.chunk_total = len(chunks)

    embedding_service = EmbeddingService(model=app.state.embedding_model)
    generation_service = GenerationService(model=app.state.generation_model)
    vector_retriever = VectorRetriever(
        vector_dir=app.state.vector_dir,
        collection_name=app.state.collection_name,
        embedding_service=embedding_service,
    )
    vector_retriever.build(chunks)

    app.state.embedding_service = embedding_service
    app.state.vector_retriever = vector_retriever
    app.state.generation_service = generation_service
