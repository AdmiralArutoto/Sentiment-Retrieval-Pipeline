from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.62"))


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
