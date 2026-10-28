import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from rag_assistant.retrieval.bm25_retriever import BM25DocumentRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "banking_knowledge"
EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

DENSE_K = 10
BM25_K = 10
FINAL_K = 10
RRF_CONSTANT = 60


def document_key(doc: Document) -> str | tuple:
    """Return a stable chunk identifier, with a legacy fallback."""
    chunk_id = doc.metadata.get("chunk_id")
    if chunk_id:
        return chunk_id

    # Compatibility with chunks indexed before the chunk_id metadata was added.
    return (
        doc.metadata.get("source"),
        doc.metadata.get("page"),
        doc.metadata.get("start_index"),
    )


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
) -> list[tuple[Document, float]]:
    scores = {}
    documents = {}

    for results in result_lists:
        seen = set()

        for rank, doc in enumerate(results, start=1):
            key = document_key(doc)

            if key in seen:
                continue
            seen.add(key)

            scores[key] = (
                scores.get(key, 0.0)
                + 1.0 / (RRF_CONSTANT + rank)
            )
            documents[key] = doc

    return sorted(
        [(documents[key], score) for key, score in scores.items()],
        key=lambda item: item[1],
        reverse=True,
    )


def main() -> None:

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True,}
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Read indexed chunks from the existing ChromaDB collection.
    stored = vector_store.get(include=["documents", "metadatas"])

    if not stored["documents"]:
        raise RuntimeError(
            "ChromaDB is empty. Run ingest.py first."
        )

    documents = [
        Document(
            page_content=text,
            metadata=metadata or {},
        )
        for text, metadata in zip(
            stored["documents"],
            stored["metadatas"],
        )
    ]

    # Sparse retrieval: exact terms and lexical relevance.
    bm25 = BM25DocumentRetriever(documents=documents, k=BM25_K)

    # Dense retrieval: semantic similarity.
    dense = vector_store.as_retriever(
        search_kwargs={"k": DENSE_K}
    )

    question = input("Search query: ").strip()

    if not question:
        print("Please enter a search query.")
        return

    dense_results = dense.invoke(question)
    bm25_results = bm25.invoke(question)

    fused_results = reciprocal_rank_fusion(
        [dense_results, bm25_results]
    )[:FINAL_K]

    print(f"\nHybrid search results for: {question}")

    for rank, (doc, score) in enumerate(fused_results, start=1):
        print(f"\n--- Result {rank} | RRF: {score:.5f} ---")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Page: {doc.metadata.get('page')}")
        print(doc.page_content[:1200])


if __name__ == "__main__":
    main()

# Logo şirketinin faturalanan gelirlerinin 2026 yılındaki durumunu özetler misin.