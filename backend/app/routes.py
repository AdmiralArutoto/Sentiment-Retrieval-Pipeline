from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from .schemas import ChunkResponse, ConfigResponse, QueryRequest, QueryResponse

router = APIRouter()


def get_app_state(request: Request) -> Any:
    return request.app.state


def _ensure_frontend(frontend_dir: Path) -> Path:
    index_file = frontend_dir / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI not found.")
    return index_file


@router.get("/", response_class=HTMLResponse)
def serve_index(state: Any = Depends(get_app_state)) -> HTMLResponse:
    index_file = _ensure_frontend(state.frontend_dir)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@router.get("/config", response_model=ConfigResponse)
def get_config(state: Any = Depends(get_app_state)) -> ConfigResponse:
    return ConfigResponse(
        dataset_path=str(state.dataset_path),
        vector_store_path=str(state.vector_dir),
        chunk_size=state.chunk_size,
        chunk_overlap=state.chunk_overlap,
        chunk_total=state.chunk_total,
        default_top_k=state.default_top_k,
        default_min_score=state.default_min_score,
        vector_db="Chroma (local persistent)",
        embedding_model=state.embedding_model,
    )


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest, state: Any = Depends(get_app_state)) -> QueryResponse:
    vector_retriever = state.vector_retriever
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
