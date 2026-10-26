"""Compare plain similarity search, MMR and hybrid search (top-k = 4) on 3 questions.

No LLM is needed: this only looks at what each retriever returns.
Usage:  python eval/compare_retrieval.py   -> prints and writes eval/retrieval_comparison.md
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import get_settings  # noqa: E402
from rag_chain import build_retriever  # noqa: E402

QUESTIONS = [
    "What is the maximum height before a guardrail is mandatory on a scaffold?",
    "Which PPE is required for concrete cutting?",
    "What must a worker do in the first 24 hours after an incident?",
]
SEARCH_TYPES = ["similarity", "mmr", "hybrid"]


def main() -> None:
    settings = get_settings()
    settings.rerank = False
    retrievers = {name: build_retriever(settings, search_type=name, k=4) for name in SEARCH_TYPES}
    lines = ["# Retrieval comparison (k = 4)", ""]

    for question in QUESTIONS:
        lines += [f"## {question}", "", "| rank | " + " | ".join(SEARCH_TYPES) + " |",
                  "|---|" + "---|" * len(SEARCH_TYPES)]
        results = {name: r.invoke(question) for name, r in retrievers.items()}
        for rank in range(4):
            cells = []
            for name in SEARCH_TYPES:
                docs = results[name]
                if rank < len(docs):
                    d = docs[rank]
                    snippet = " ".join(d.page_content.split())[:70].replace("|", "/")
                    cells.append(f"{d.metadata['source']} p.{d.metadata['page']} - {snippet}...")
                else:
                    cells.append("")
            lines.append(f"| {rank + 1} | " + " | ".join(cells) + " |")
        for name in SEARCH_TYPES:
            distinct = len({(d.metadata['source'], d.metadata['page'], d.metadata.get('start_index'))
                            for d in results[name]})
            files = sorted({d.metadata['source'] for d in results[name]})
            lines.append(f"\n- **{name}**: {distinct} distinct chunks from {len(files)} file(s): {', '.join(files)}")
        lines.append("")

    report = "\n".join(lines)
    (ROOT / "eval" / "retrieval_comparison.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
