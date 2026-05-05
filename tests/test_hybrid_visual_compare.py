"""可视化对比：Base RAG vs Hybrid RAG（Web + 本地 + Rerank）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.ragsummarize import HybridRAG, RAGSummarize


def _score_bar(score: float, width: int = 20) -> str:
    clamped = max(0.0, min(1.0, float(score)))
    filled = int(round(clamped * width))
    return "█" * filled + "░" * (width - filled)


def _preview(text: str, limit: int = 72) -> str:
    one_line = (text or "").replace("\n", " ").strip()
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit] + "..."


def _print_block(title: str, rows: list[tuple[str, float, str]]) -> None:
    print(f"\n{title}")
    print("-" * 110)
    print("Rank | Channel | Score  | Visualization          | Snippet")
    print("-" * 110)
    for idx, (channel, score, snippet) in enumerate(rows, 1):
        print(
            f"{idx:>4} | {channel:<7} | {score:>0.3f} | {_score_bar(score):<22} | {snippet}"
        )
    print("-" * 110)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visual compare Base RAG and Hybrid RAG")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="Cursor Agent Mode 和 Plan Mode 区别是什么？",
        help="用于对比的具体问题",
    )
    args = parser.parse_args()

    query = args.query.strip()
    print("\n=== Multi-Retrieval Visual Compare ===")
    print(f"Query: {query}")

    base = RAGSummarize()
    hybrid = HybridRAG()

    base_docs = base.retrieve_docs_with_scores(query)
    hybrid_docs = hybrid.retrieve_docs_with_scores(query)

    base_rows = [
        ("local", score, _preview(doc.page_content)) for doc, score in base_docs
    ]
    hybrid_rows = [
        (doc.metadata.get("source_channel", "?"), score, _preview(doc.page_content))
        for doc, score in hybrid_docs
    ]

    _print_block("Base RAG (local only)", base_rows)
    _print_block("Hybrid RAG (local + web)", hybrid_rows)

    base_local = sum(1 for ch, _, _ in base_rows if ch == "local")
    hybrid_local = sum(1 for ch, _, _ in hybrid_rows if ch == "local")
    hybrid_web = sum(1 for ch, _, _ in hybrid_rows if ch == "web")

    print("\nSummary")
    print("-" * 110)
    print(f"Base docs:   {len(base_rows)} (local={base_local})")
    print(f"Hybrid docs: {len(hybrid_rows)} (local={hybrid_local}, web={hybrid_web})")
    print("-" * 110)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
