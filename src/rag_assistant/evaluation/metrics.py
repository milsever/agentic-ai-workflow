"""Metrics for a small retrieval benchmark."""

from collections.abc import Sequence
from langchain_core.documents import Document

def is_relevant(
        doc: Document,
        expected_records: Sequence[int],
) -> bool:
    """Match retrieved documents against gold JSONL record numbers."""
    record = doc.metadata.get("record")

    if record is None:
        return False

    return int(record) in expected_records

def recall_at_k(
        results: Sequence[Document],
        expected_records: Sequence[int],
        k: int,
) -> float:
    """Binary Recall@k for the expected record set."""
    if not expected_records:
        return float("nan")

    found = {
        int(doc.metadata["record"])
        for doc in results[:k]
        if doc.metadata.get("record") is not None
    }

    return len(found.intersection(expected_records)) / len(expected_records)

def reciprocal_rank_at_k(
        results: Sequence[Document],
        expected_records: Sequence[int],
        k: int,
) -> float:
    """Reciprocal rank of the first expected record."""
    if not expected_records:
        return float("nan")

    for rank, doc in enumerate(results[:k], start=1):
        if is_relevant(doc, expected_records):
            return 1.0 / rank

    return 0.0
