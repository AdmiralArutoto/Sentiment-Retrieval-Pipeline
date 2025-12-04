from __future__ import annotations

import re
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from io import StringIO


@dataclass
class Chunk:
    """Represents a single chunk of context stored inside the vector DB."""

    id: str
    text: str
    metadata: Dict[str, Any]


def load_dataset(dataset_path: Path) -> List[Dict[str, str]]:
    """Load rows from a lightweight CSV file and return dict records."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    sanitized_lines: List[str] = []
    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        sanitized = _sanitize_line(raw_line)
        if sanitized:
            sanitized_lines.append(sanitized)

    if not sanitized_lines:
        raise RuntimeError(f"{dataset_path} is empty.")

    reader = csv.reader(StringIO("\n".join(sanitized_lines)))
    rows = list(reader)
    header = [col.strip() for col in rows[0]]

    records: List[Dict[str, str]] = []
    for row_index, row in enumerate(rows[1:], start=1):
        if len(row) != len(header):
            raise ValueError(
                f"Row {row_index} has {len(row)} columns but expected {len(header)}."
            )
        record = {header[i]: row[i].strip() for i in range(len(header))}
        record["row_index"] = str(row_index)
        records.append(record)

    return records


def chunk_records(
    records: Sequence[Dict[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Chunk]:
    """Chunk dataset records into overlapping character windows."""
    if chunk_size <= chunk_overlap:
        raise ValueError("Chunk size must be greater than the overlap.")

    chunks: List[Chunk] = []
    for record in records:
        row_index = record.get("row_index", "0")
        text = _build_chunk_text(record)
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end]

            chunk_id = f"record-{row_index}-chunk-{chunk_idx}"
            metadata = {
                "row_index": int(row_index),
                "sentiment": record.get("Sentiment", ""),
                "source": record.get("Source", ""),
                "date": record.get("Date/Time", ""),
                "user_id": record.get("User ID", ""),
                "location": record.get("Location", ""),
                "confidence_score": _safe_float(record.get("Confidence Score")),
                "chunk_index": chunk_idx,
                "char_start": start,
                "char_end": end,
            }

            chunks.append(Chunk(id=chunk_id, text=chunk_text, metadata=metadata))
            chunk_idx += 1

            if end >= len(text):
                break
            start = max(0, end - chunk_overlap)

    return chunks


def _build_chunk_text(record: Dict[str, str]) -> str:
    template = (
        '{text} (Sentiment: {sentiment}, Source: {source}, Date: {date}, '
        'User: {user}, Location: {location}, Confidence: {confidence}).'
    )
    rendered = template.format(
        text=record.get("Text", "").strip(),
        sentiment=record.get("Sentiment", "").strip(),
        source=record.get("Source", "").strip(),
        date=record.get("Date/Time", "").strip(),
        user=record.get("User ID", "").strip(),
        location=record.get("Location", "").strip(),
        confidence=record.get("Confidence Score", "").strip(),
    )
    return re.sub(r"\s+", " ", rendered).strip()


def _safe_float(value: str | None) -> float:
    try:
        cleaned = value.strip() if value is not None else None
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0


def _sanitize_line(line: str) -> str:
    """Fix odd quoting issues present in the lightweight CSV export."""
    stripped = line.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        stripped = stripped[1:-1]
    return stripped.replace('""', '"')
