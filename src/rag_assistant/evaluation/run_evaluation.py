"""Compare dense-only, BM25-only, and hybrid RRF retrieval.

Run from the project root:
    $env:PYTHONPATH = "src"
    python -m rag_assistant.evaluation.run_evaluation_comparison
"""
import csv
from pathlib import Path
from statistics import mean

from rag_assistant.evaluation.dataset import TEST_CASES
from rag_assistant.evaluation.metrics import recall_at_k, reciprocal_rank_at_k
from rag_assistant.retrieval.hybrid_retriever import (
    HybridRetriever,
    reciprocal_rank_fusion,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = PROJECT_ROOT / "data" / "retrieval_comparison.csv"
K_VALUES = (1, 3, 5, 10)
STRATEGIES = ("dense", "bm25", "hybrid_rrf")


def describe_top(results, limit=3):
    return " | ".join(
        f'{doc.metadata.get("source", "unknown")} '
        f'[page={doc.metadata.get("page", "")}, '
        f'record={doc.metadata.get("record", "")}]'
        for doc in results[:limit]
    )


def main() -> None:
    retriever = HybridRetriever()
    rows = []
    metric_values = {
        strategy: {
            f"{metric}@{k}": []
            for metric in ("recall", "mrr")
            for k in K_VALUES
        }
        for strategy in STRATEGIES
    }

    for case in TEST_CASES:
        question = case["question"]
        expected_records = case.get("expected_records", [])

        # Run the two underlying retrievers once, then fuse those same results.
        dense_results = retriever.dense.invoke(question)
        bm25_results = retriever.bm25.invoke(question)
        hybrid_results = reciprocal_rank_fusion(
            [dense_results, bm25_results]
        )[:retriever.final_k]

        results_by_strategy = {
            "dense": dense_results,
            "bm25": bm25_results,
            "hybrid_rrf": hybrid_results,
        }

        print(f'\n[{case["id"]}] {question}')
        for strategy, results in results_by_strategy.items():
            row = {
                "id": case["id"],
                "question": question,
                "strategy": strategy,
                "expected_records": str(expected_records),
                "retrieved_count": len(results),
                "top_sources": describe_top(results),
                "notes": case.get("notes", ""),
            }

            for rank, doc in enumerate(results[:3], start=1):
                snippet = " ".join(doc.page_content.split())[:180]
                print(
                    f'  {strategy} #{rank}: '
                    f'record={doc.metadata.get("record")} '
                    f'source={doc.metadata.get("source")}'
                )
                print(f"    {snippet}")

            for k in K_VALUES:
                if expected_records:
                    recall = recall_at_k(results, expected_records, k)
                    mrr = reciprocal_rank_at_k(results, expected_records, k)
                    row[f"recall@{k}"] = recall
                    row[f"mrr@{k}"] = mrr
                    metric_values[strategy][f"recall@{k}"].append(recall)
                    metric_values[strategy][f"mrr@{k}"].append(mrr)
                else:
                    row[f"recall@{k}"] = ""
                    row[f"mrr@{k}"] = ""
            rows.append(row)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "question", "strategy", "expected_records",
        "retrieved_count", "top_sources", "notes",
        *[f"{metric}@{k}" for k in K_VALUES for metric in ("recall", "mrr")],
    ]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Aggregate metrics (answerable cases only) ===")
    for strategy in STRATEGIES:
        print(f"\n{strategy}:")
        for name, values in metric_values[strategy].items():
            print(f"  {name}: {mean(values):.3f}" if values else f"  {name}: n/a")

    print(f"\nDetailed comparison written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
