from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from rag_assistant.retrieval.bm25_retriever import BM25DocumentRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "banking_knowledge"
EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

DENSE_K = 10
BM25_K = 10
FINAL_K = 10
RRF_CONSTANT = 60

def document_key(doc: Document) -> tuple:
    return (
        doc.metadata.get("source"),
        doc.metadata.get("page"),
        doc.metadata.get("start_index"),
        doc.page_content,
    )

def reciprocal_rank_fusion(
        result_lists: list[list[Document]],
) -> list[Document]:
    scores: dict[tuple, float] = {}
    documents: dict[tuple, Document] = {}

    for results in result_lists:
        seen = set()

        for rank, doc in enumerate(results, start=1):
            key = document_key(doc)

            if key in seen:
                continue

            seen.add(key)
            scores[key] = scores.get(key, 0.0) + (
                    1.0 / (RRF_CONSTANT + rank)
            )
            documents[key] = doc

    ranked = sorted(
        scores,
        key=lambda key: scores[key],
        reverse=True,
    )

    return [documents[key] for key in ranked]

class HybridRetriever:
    """Combine Chroma dense retrieval and BM25 using RRF."""

    def __init__(self, final_k: int = FINAL_K) -> None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

        stored = self.vector_store.get(
            include=["documents", "metadatas"]
        )

        texts = stored.get("documents") or []
        metadatas = stored.get("metadatas") or []

        if not texts:
            raise RuntimeError(
                "ChromaDB is empty. Run ingest.py first."
            )

        documents = [
            Document(
                page_content=text,
                metadata=metadata or {},
            )
            for text, metadata in zip(texts, metadatas)
        ]

        self.bm25 = BM25DocumentRetriever(
            documents=documents,
            k=BM25_K,
        )
        self.dense = self.vector_store.as_retriever(
            search_kwargs={"k": DENSE_K}
        )
        self.final_k = final_k

    def invoke(self, question: str) -> list[Document]:
        dense_results = self.dense.invoke(question)
        bm25_results = self.bm25.invoke(question)

        fused = reciprocal_rank_fusion(
            [dense_results, bm25_results]
        )

        return fused[:self.final_k]
