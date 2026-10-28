
import re

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class BM25DocumentRetriever:
    """Lexical document retrieval using the BM25 algorithm."""

    def __init__(
        self,
        documents: list[Document],
        k: int = 10,
    ) -> None:
        if not documents:
            raise ValueError("Documents cannot be empty.")

        self.documents = documents
        self.k = k

        tokenized_corpus = [
            self._tokenize(doc.page_content)
            for doc in documents
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    def invoke(
        self,
        query: str,
        k: int | None = None,
    ) -> list[Document]:
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        limit = self.k if k is None else k

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for index in ranked_indices:
            if scores[index] <= 0:
                continue

            results.append(self.documents[index])

            if len(results) >= limit:
                break

        return results
