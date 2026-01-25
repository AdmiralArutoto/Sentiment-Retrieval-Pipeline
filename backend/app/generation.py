from __future__ import annotations

import os
from typing import Any, Iterable, List

from openai import OpenAI


class GenerationService:
    """Grounded text generation backed by the OpenAI Responses API."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Please configure it before running the API.")
        self._client = OpenAI(api_key=self.api_key)

    def generate(
        self,
        question: str,
        contexts: Iterable[dict[str, Any]],
        max_output_tokens: int,
    ) -> str:
        """Generate an answer grounded strictly on provided contexts."""
        context_block = "\n\n".join(
            f"[{ctx['chunk_id']}] {ctx['text']}" for ctx in contexts if ctx.get("text")
        )

        if not context_block.strip():
            return "No context available to answer the question."

        prompt = (
            "You are a helpful assistant that answers using ONLY the provided context. "
            "Cite the chunk ids you rely on using [chunk-id] notation. "
            "If the answer cannot be found in the context, say you do not know based on the provided context."
        )

        response = self._client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer with citations.",
                },
            ],
            max_output_tokens=max_output_tokens,
        )

        return self._extract_text(response)

    @staticmethod
    def _extract_text(response: Any) -> str:
        # The Responses API exposes output_text for convenience; fall back to parsing blocks.
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        try:
            outputs: List[Any] = getattr(response, "output", [])
            if outputs:
                content = outputs[0].content  # type: ignore[attr-defined]
                parts: List[str] = []
                for block in content:
                    value = getattr(block, "text", None) if not isinstance(block, dict) else block.get("text")
                    if isinstance(value, str):
                        parts.append(value)
                if parts:
                    return "".join(parts).strip()
        except Exception:
            pass

        return ""
