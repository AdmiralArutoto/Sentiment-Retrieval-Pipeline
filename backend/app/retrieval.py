from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import chromadb

from .embeddings import EmbeddingService
from .ingestion import Chunk


class VectorRetriever:
    """Stores embeddings inside Chroma and exposes similarity search."""

    def __init__(
        self,
        vector_dir: Path,
        collection_name: str,
        embedding_service: EmbeddingService,
    ) -> None:
        self.vector_dir = vector_dir
        self.collection_name = collection_name
        self.embedding_service = embedding_service

        self._client: chromadb.PersistentClient | None = None
        self.collection: chromadb.Collection | None = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise RuntimeError("Chunk list is empty. Cannot build vector store.")

        self.chunks = list(chunks)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.vector_dir))
        # Drop collection if it exists so previous embeddings do not linger.
        try:
            self._client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        embeddings = self.embedding_service.embed([chunk.text for chunk in self.chunks])
        self.collection.upsert(
            ids=[chunk.id for chunk in self.chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in self.chunks],
            metadatas=[chunk.metadata for chunk in self.chunks],
        )

    def query(self, query_text: str, top_k: int, min_score: float) -> List[Dict[str, Any]]:
        if not query_text.strip():
            raise ValueError("Query text must not be empty.")
        if self.collection is None:
            raise RuntimeError("Vector store is not ready yet.")

        query_embedding = self.embedding_service.embed([query_text])[0]
        n_results = min(max(top_k * 2, top_k), len(self.chunks))

        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "documents", "distances"],
        )

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: List[Dict[str, Any]] = []
        for chunk_id, doc, metadata, distance in zip(ids, documents, metadatas, distances):
            score = 1 - distance if distance is not None else 1.0
            if score < min_score:
                continue
            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "score": round(score, 4),
                    "metadata": metadata,
                }
            )
            if len(results) >= top_k:
                break

        return results
