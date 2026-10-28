
from langchain_core.documents import Document

from rag_assistant.retrieval.hybrid_retriever import (
    document_key,
    reciprocal_rank_fusion,
)


def make_document(
    content: str,
    source: str = "guide.pdf",
    page: int = 1,
    start_index: int = 0,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "page": page,
            "start_index": start_index,
        },
    )


def test_document_key_identifies_same_document():
    doc1 = make_document("Transfer fees")
    doc2 = make_document("Transfer fees")

    assert document_key(doc1) == document_key(doc2)


def test_document_key_distinguishes_different_content():
    doc1 = make_document("Transfer fees")
    doc2 = make_document("Account fees")

    assert document_key(doc1) != document_key(doc2)


def test_rrf_ranks_document_found_by_both_retrievers_first():
    doc_a = make_document("Transfer fees")
    doc_b = make_document("Account fees")
    doc_c = make_document("Card fees")

    results = reciprocal_rank_fusion([
        [doc_a, doc_b],
        [doc_a, doc_c],
    ])

    assert results[0] == doc_a
    assert len(results) == 3


def test_rrf_does_not_count_duplicate_within_one_result_list():
    doc_a = make_document("Transfer fees")
    doc_b = make_document("Account fees")

    results = reciprocal_rank_fusion([
        [doc_a, doc_a, doc_b],
    ])

    assert len(results) == 2
    assert results[0] == doc_a
