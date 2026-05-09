"""FinSight RAG 冲烟测试：在重建后的索引上跑几个典型查询，验证召回质量。"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag.ragsummarize import RAGSummarize  # noqa: E402


# 每个查询给一个"应当出现的关键词"做粗判（命中即视为通过）。
QUERIES = [
    ("什么是市盈率 PE？怎么计算？", ["每股收益", "市盈率"]),
    ("DCF 估值的核心假设有哪些？", ["WACC", "永续"]),
    ("半导体产业链上游有哪些环节？", ["EDA", "光刻", "材料"]),
    ("新能源车的三电系统是什么？", ["电池", "电机", "电控"]),
    ("HBM 是什么？谁是主要供应商？", ["HBM", "海力士"]),
    ("星核能源 999001 的 2024 年营收是多少？", ["星核", "312"]),
    ("AI 算力推理侧的需求驱动是什么？", ["推理", "计算"]),
]


def _truncate(text: str, n: int = 80) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n] + "..."


def main() -> int:
    print("=" * 70)
    print("FinSight · RAG 冲烟测试")
    print("=" * 70)

    rag = RAGSummarize()

    passed = 0
    for i, (q, expected_kw) in enumerate(QUERIES, 1):
        print(f"\n[{i}/{len(QUERIES)}] Q: {q}")
        try:
            docs = rag.retrieve_docs(q)
        except Exception as e:
            print(f"  [FAIL] retrieve raised: {e}")
            continue

        if not docs:
            print("  [FAIL] no docs returned")
            continue

        joined_text = "\n".join(d.page_content for d in docs)
        hit_kw = [k for k in expected_kw if k.lower() in joined_text.lower()]
        ok = len(hit_kw) >= 1
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {len(docs)} 条候选；期望关键词 {expected_kw}，命中 {hit_kw}")

        for j, d in enumerate(docs[:3], 1):
            score = d.metadata.get("rerank_score", 0.0)
            section = d.metadata.get("section", "-")
            source = d.metadata.get("source", "-")
            src = Path(source).name if source != "-" else "-"
            print(
                f"    Top-{j}  rerank={score:.3f}  src={src}  section={_truncate(section, 30)}"
            )
            print(f"           text=[{_truncate(d.page_content, 100)}]")

    print("\n" + "=" * 70)
    print(f"通过 {passed}/{len(QUERIES)}")
    print("=" * 70)
    return 0 if passed == len(QUERIES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
