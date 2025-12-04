from __future__ import annotations

import os
from typing import List, Sequence

from openai import OpenAI


class EmbeddingService:
    """Thin wrapper over the OpenAI embeddings API."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Please configure it before running the API.")
        self._client = OpenAI(api_key=self.api_key)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in response.data]
