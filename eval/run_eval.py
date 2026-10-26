"""Run the evaluation set and report accuracy, refusals, citations and latency.

Grading is automatic and deliberately strict:
- answerable question -> correct if it is not a refusal AND every regex in
  ``must_match`` is found in the answer (the key fact is present);
- unanswerable question -> correct if the answer is the refusal sentence;
- citation ok -> the answer cites the expected file as [file, p.X];
- retrieval hit -> the expected file is among the k retrieved chunks.

Usage:  python eval/run_eval.py [--search-type mmr|similarity|hybrid] [--rerank] [--delay 4]
Writes eval/results_<tag>.json and eval/results_<tag>.md
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import get_settings  # noqa: E402
from rag_chain import build_pipeline, extract_citations, is_refusal  # noqa: E402

EVAL_DIR = ROOT / "eval"


def grade(item: dict[str, Any], answer: str, retrieved_files: list[str]) -> dict[str, Any]:
    refused = is_refusal(answer)
    cited_files = {name for name, _ in extract_citations(answer)}
    if item["answerable"]:
        facts_ok = all(re.search(p, answer, re.IGNORECASE) for p in item["must_match"])
        correct = facts_ok and not refused
        citation_ok = item["expected_source"] in cited_files
        retrieval_hit = item["expected_source"] in retrieved_files
    else:
        correct = refused
        citation_ok = retrieval_hit = None
    return {"refused": refused, "correct": correct, "citation_ok": citation_ok,
            "retrieval_hit": retrieval_hit, "cited_files": sorted(cited_files)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-type", choices=["similarity", "mmr", "hybrid"])
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between questions (free-tier rate limit)")
    parser.add_argument("--tag", default=None, help="suffix of the result files")
    args = parser.parse_args()

    settings = get_settings()
    if args.search_type:
        settings.search_type = args.search_type
    if args.rerank:
        settings.rerank = True
    tag = args.tag or f"{settings.search_type}{'_rerank' if settings.rerank else ''}"

    pipeline = build_pipeline(settings)
    items = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    for i, item in enumerate(items):
        if i:
            time.sleep(args.delay)
        start = time.perf_counter()
        try:
            result = pipeline.chain.invoke({"question": item["question"], "chat_history": []})
            answer, docs, error = result["answer"].strip(), result["docs"], None
        except Exception as exc:  # keep going: a failed call is graded as wrong
            answer, docs, error = "", [], f"{exc.__class__.__name__}: {exc}"
        latency_ms = int((time.perf_counter() - start) * 1000)
        retrieved = [f"{d.metadata['source']} p.{d.metadata['page']}" for d in docs]
        row = {"id": item["id"], "question": item["question"], "answerable": item["answerable"],
               "answer": answer, "retrieved": retrieved, "latency_ms": latency_ms, "error": error,
               **grade(item, answer, [d.metadata["source"] for d in docs])}
        rows.append(row)
        print(f"{item['id']} {'OK ' if row['correct'] else 'BAD'} {latency_ms:>6} ms  {answer[:90]!r}")

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    latencies = [r["latency_ms"] for r in rows if not r["error"]]
    summary = {
        "config": {"llm_model": settings.llm_model, "fallbacks": settings.llm_fallback_models,
                   "search_type": settings.search_type, "rerank": settings.rerank, "top_k": settings.top_k,
                   "chunk_size": settings.chunk_size, "chunk_overlap": settings.chunk_overlap},
        "overall_accuracy": sum(r["correct"] for r in rows) / len(rows),
        "answerable_accuracy": f"{sum(r['correct'] for r in answerable)}/{len(answerable)}",
        "correct_refusals": f"{sum(r['correct'] for r in unanswerable)}/{len(unanswerable)}",
        "false_refusals": sum(r["refused"] for r in answerable),
        "citation_accuracy": f"{sum(bool(r['citation_ok']) for r in answerable)}/{len(answerable)}",
        "retrieval_hit_at_k": f"{sum(bool(r['retrieval_hit']) for r in answerable)}/{len(answerable)}",
        "avg_latency_ms": int(statistics.mean(latencies)) if latencies else None,
        "max_latency_ms": max(latencies) if latencies else None,
        "errors": sum(bool(r["error"]) for r in rows),
    }

    (EVAL_DIR / f"results_{tag}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [f"# Evaluation results ({tag})", "", "| Metric | Value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in summary.items() if k != "config"]
    md += ["", f"Config: `{json.dumps(summary['config'])}`", "",
           "| id | correct | refused | citation | retrieval hit | latency (ms) | answer |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        short = (r["error"] or r["answer"]).replace("\n", " ").replace("|", "/")[:160]
        md.append(f"| {r['id']} | {r['correct']} | {r['refused']} | {r['citation_ok']} | {r['retrieval_hit']} "
                  f"| {r['latency_ms']} | {short} |")
    (EVAL_DIR / f"results_{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
