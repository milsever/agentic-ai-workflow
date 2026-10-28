RAG retrieval evaluation starter

1. Copy `src/rag_assistant/evaluation` into your project's `src/rag_assistant/` directory.
2. Check/edit `expected_terms` in dataset.py to match the exact text in your indexed chunks.
3. From the project root, using your project virtual environment:
   $env:PYTHONPATH = "src"
   python -m rag_assistant.evaluation.run_evaluation
4. Inspect `data/evaluation_results.csv` and terminal snippets.

Limitations:
- Bootstrap benchmark, not a definitive evaluation set.
- A result is relevant only if all expected_terms occur in the same chunk; chunk boundaries or formatting may cause misses.
- Recall@k is binary per question (at least one relevant chunk in top k); MRR@k uses the first relevant rank.
- The no-answer test is excluded from retrieval metrics; evaluate abstention in answer generation separately.
- Fee records are dated demo data; check source dates before presenting values as current.
