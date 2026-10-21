# DocumentAssistant evaluation

Run commands from the project root with the project's existing virtual environment activated.

```powershell
python evaluation/evaluate.py --pdf "data/raw_pdfs/RAG_Testing.pdf" --limit 3
python evaluation/evaluate.py --pdf "data/raw_pdfs/RAG_Testing.pdf"
```

The first command is a smoke test; the second runs all 20 cases. Results are written to `evaluation/results/` as CSV and JSON.

These are custom diagnostic measures, not Ragas metrics:
- `expected_page_hit_rate`: whether any manually specified expected page was retrieved.
- `retrieved_context_answer_overlap`: lexical overlap between expected-answer terms and selected context.
- `generated_answer_lexical_overlap`: lexical overlap between expected-answer terms and generated answer.
- `mean_latency_seconds`: average end-to-end latency per question.

Lexical overlap is only a rough diagnostic. It does not prove semantic correctness, faithfulness, or citation support. Review the answer and cited passages manually before reporting results.
