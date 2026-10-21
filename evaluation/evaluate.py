"""
Lightweight evaluation for DocumentAssistant.
Metrics are transparent custom checks, not Ragas metrics.
Run from the project root:
  python evaluation/evaluate.py --pdf "data/raw_pdfs/RAG_Testing.pdf" --limit 3
  python evaluation/evaluate.py --pdf "data/raw_pdfs/RAG_Testing.pdf"
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_DIR))

from config import RAW_PDF_DIR  # noqa: E402
from rag_pipeline import DocumentAssistant  # noqa: E402

EVALUATION_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALUATION_DIR / "results"
CASES_PATH = EVALUATION_DIR / "test_cases.json"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "what", "when", "which", "with", "why", "system", "document",
}

def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {word for word in words if word not in STOPWORDS and len(word) > 1}

def lexical_overlap(expected_answer: str, retrieved_text: str) -> float:
    """Fraction of expected-answer content words present in retrieved context."""
    expected = tokens(expected_answer)
    if not expected:
        return 0.0
    return len(expected & tokens(retrieved_text)) / len(expected)

def resolve_pdf(pdf_arg: str | None) -> Path:
    if pdf_arg:
        path = Path(pdf_arg)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        return path
    candidates = sorted(RAW_PDF_DIR.glob("*.pdf"))
    if not candidates:
        raise FileNotFoundError(
            f"No PDF found in {RAW_PDF_DIR}. Pass --pdf path/to/file.pdf."
        )
    preferred = next((p for p in candidates if p.name.lower() == "rag_testing.pdf"), None)
    return preferred or candidates[0]

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DocumentAssistant on fixed test cases.")
    parser.add_argument("--pdf", help="PDF path; defaults to RAG_Testing.pdf in data/raw_pdfs.")
    parser.add_argument("--limit", type=int, help="Run only the first N cases as a smoke test.")
    args = parser.parse_args()

    with CASES_PATH.open(encoding="utf-8") as file:
        cases = json.load(file)
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("No test cases selected.")

    pdf_path = resolve_pdf(args.pdf)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"PDF: {pdf_path}")
    print(f"Test cases: {len(cases)}")
    print("Initializing DocumentAssistant (model and embeddings may take a moment)...")
    assistant = DocumentAssistant()
    assistant.load_pdf(pdf_path)

    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}: {case['question']}")
        start = time.perf_counter()
        try:
            # Retrieve once, then generate from the same question using the app's public API.
            # ask() performs retrieval internally; returned sources represent selected context.
            result = assistant.ask(case["question"])
            elapsed = time.perf_counter() - start
            sources = result.get("sources", [])
            retrieved_text = "\n".join(str(source.get("content", "")) for source in sources)
            retrieved_pages = sorted({
                int(source["page"]) for source in sources
                if str(source.get("page", "")).isdigit()
            })
            expected_pages = case.get("expected_pages", [])
            page_hit = any(page in retrieved_pages for page in expected_pages) if expected_pages else None
            overlap = lexical_overlap(case["expected_answer"], retrieved_text)
            answer = str(result.get("answer", ""))
            answer_overlap = lexical_overlap(case["expected_answer"], answer)
            rows.append({
                "id": case["id"],
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "generated_answer": answer,
                "expected_pages": ";".join(map(str, expected_pages)),
                "retrieved_pages": ";".join(map(str, retrieved_pages)),
                "expected_page_hit": page_hit if page_hit is not None else "",
                "retrieved_context_answer_overlap": round(overlap, 4),
                "generated_answer_lexical_overlap": round(answer_overlap, 4),
                "source_count": len(sources),
                "latency_seconds": round(elapsed, 2),
                "error": "",
            })
        except Exception as exc:
            elapsed = time.perf_counter() - start
            rows.append({
                "id": case["id"],
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "generated_answer": "",
                "expected_pages": ";".join(map(str, case.get("expected_pages", []))),
                "retrieved_pages": "",
                "expected_page_hit": False,
                "retrieved_context_answer_overlap": 0.0,
                "generated_answer_lexical_overlap": 0.0,
                "source_count": 0,
                "latency_seconds": round(elapsed, 2),
                "error": repr(exc),
            })
            print(f"  ERROR: {exc}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"evaluation_{timestamp}.csv"
    summary_path = RESULTS_DIR / f"summary_{timestamp}.json"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    valid = [row for row in rows if not row["error"]]
    page_checks = [row for row in valid if row["expected_page_hit"] != ""]
    summary = {
        "pdf": pdf_path.name,
        "test_case_count": len(cases),
        "successful_runs": len(valid),
        "failed_runs": len(rows) - len(valid),
        "expected_page_hit_rate": (
            round(sum(bool(row["expected_page_hit"]) for row in page_checks) / len(page_checks), 4)
            if page_checks else None
        ),
        "mean_retrieved_context_answer_overlap": (
            round(sum(row["retrieved_context_answer_overlap"] for row in valid) / len(valid), 4)
            if valid else None
        ),
        "mean_generated_answer_lexical_overlap": (
            round(sum(row["generated_answer_lexical_overlap"] for row in valid) / len(valid), 4)
            if valid else None
        ),
        "mean_latency_seconds": (
            round(sum(row["latency_seconds"] for row in valid) / len(valid), 2)
            if valid else None
        ),
        "notes": [
            "Custom diagnostic measures only; these are not Ragas metrics.",
            "Lexical overlap is a rough diagnostic and does not establish semantic correctness or faithfulness.",
            "expected_page_hit_rate checks whether any manually specified expected page was among the returned source pages.",
            "Review generated answers and cited passages manually before claiming answer correctness."
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n--- EVALUATION SUMMARY ---")
    print(json.dumps(summary, indent=2))
    print(f"\nCSV results: {csv_path}")
    print(f"Summary:     {summary_path}")

if __name__ == "__main__":
    main()
