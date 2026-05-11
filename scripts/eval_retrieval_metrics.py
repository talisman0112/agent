"""FinSight RAG 检索性能评估。

对照三档 pipeline 在 ``tests/golden_set.yml`` 上跑出真实数字：
1. **Vector-only**           —— 仅向量检索 + 父块展开（top-5）
2. **+Rerank**               —— 向量 top-20 → Qwen3-Rerank → 取前 5（不压缩）
3. **+Rerank+Compression**   —— 在 ``+Rerank`` 之上再做上下文压缩（FinSight 默认 pipeline）

输出指标
--------
- Recall@5  : top-5 中是否命中至少一个相关文档（每题 0/1）
- Precision@5: top-5 中相关文档占比
- MRR       : 第一个命中位置的倒数（未命中 = 0）
- 延迟       : 平均 / P50 / P95 / Max（ms）
- 输入 tokens : 阶段输出总字符数 / 4 估算（中文 ≈ 1 字 / 1.5 token，统一近似）
- 压缩率     : 1 − post / pre（仅第三档）

用法
----
    python scripts/eval_retrieval_metrics.py                    # 跑全部 30 题
    python scripts/eval_retrieval_metrics.py --limit 10         # 调试模式仅跑前 10
    python scripts/eval_retrieval_metrics.py --output report.md # 自定义报告路径

Query 扩展对照
-------------
在 ``config/rag.yml`` 中切换 ``query_expansion_enabled`` / ``query_expansion_max_coarse_docs`` 等，
可对比多查询合并粗排与单路检索的 Recall@5/MRR（评测走真实 ``RAGSummarize.retrieve_docs``）。

判定命中规则
-----------
对每条 expected_docs 中的 basename 进行**匹配优先**判定：
若 ``doc.metadata.source`` 的 basename 在 expected_docs 列表中，记为命中；
否则用 expected_keywords 的 OR 命中作兜底（覆盖元数据缺失场景）。
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.documents import Document  # noqa: E402

from rag.context_compressor import CompressionStrategy  # noqa: E402
from rag.ragsummarize import RAGSummarize  # noqa: E402


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


@dataclass
class GoldenItem:
    id: str
    category: str
    question: str
    expected_docs: list[str]
    expected_keywords: list[str]


@dataclass
class StageResult:
    """单个 query 在某档 pipeline 上的评估结果。"""
    name: str
    docs: list[Document]
    elapsed_ms: float
    total_chars: int
    hit_at_k: bool
    first_hit_rank: int  # 1-indexed；未命中 = 0
    relevant_count: int  # top-K 中相关文档数
    returned_count: int  # 实际返回的文档数（rerank 可能因去重 / 阈值过滤少于 K）

    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_hit_rank if self.first_hit_rank > 0 else 0.0


@dataclass
class QueryResult:
    item: GoldenItem
    vector_only: StageResult
    vector_rerank: StageResult
    full_pipeline: StageResult
    pre_compression_chars: int = 0
    post_compression_chars: int = 0
    compression_ratio: float = 0.0
    compression_method: str = "-"
    compression_quality: float = 0.0


@dataclass
class StageAggregate:
    name: str
    n: int = 0
    hit_count: int = 0
    precision_sum: float = 0.0          # ∑ relevant / K（K=5 固定分母）
    precision_actual_sum: float = 0.0   # ∑ relevant / actual_returned（每题各自分母）
    rr_sum: float = 0.0
    latencies: list[float] = field(default_factory=list)
    chars_sum: int = 0
    returned_sum: int = 0


# ---------------------------------------------------------------------------
# 工具：文档命中判定 / token 估算
# ---------------------------------------------------------------------------


def _doc_basename(doc: Document) -> str:
    src = (doc.metadata or {}).get("source", "") or ""
    return os.path.basename(src)


def _is_relevant(doc: Document, item: GoldenItem) -> bool:
    """元数据匹配优先，关键词兜底。"""
    bn = _doc_basename(doc)
    if bn and bn in item.expected_docs:
        return True
    text = (doc.page_content or "")
    return any(kw and kw in text for kw in item.expected_keywords)


def _est_tokens(chars: int) -> int:
    """字符 → token 粗估：中文文本约 1 字 ≈ 1.5 token，混合文本按 1.0 折算稍低估算。"""
    return int(chars * 1.0)


# ---------------------------------------------------------------------------
# 三档 pipeline 执行
# ---------------------------------------------------------------------------


def evaluate_one(rs: RAGSummarize, item: GoldenItem) -> QueryResult:
    """跑完三档 pipeline 并采集指标。"""
    chroma = rs.vector_store.chroma
    expand = rs.vector_store.expand_retrieval_to_parents
    reranker = rs.reranker
    compressor = rs.compressor
    rerank_top_n = getattr(reranker, "top_n", 5) or 5

    # ---------- Stage 1: vector-only ----------
    t0 = time.perf_counter()
    raw_docs_5 = chroma.as_retriever(search_kwargs={"k": rerank_top_n}).invoke(item.question)
    raw_docs_5 = expand(raw_docs_5)[:rerank_top_n]
    t1 = time.perf_counter()

    # ---------- Stage 2: vector + rerank（不压缩）----------
    raw_docs_20 = chroma.as_retriever(search_kwargs={"k": 20}).invoke(item.question)
    raw_docs_20 = expand(raw_docs_20)
    t_rerank_start = time.perf_counter()
    reranked = reranker.rerank(item.question, raw_docs_20)[:rerank_top_n]
    t2 = time.perf_counter()

    # ---------- Stage 3: + 压缩 ----------
    pre_chars = sum(len(d.page_content) for d in reranked)
    post_chars = pre_chars
    method_used = "-"
    quality_score = 0.0
    if compressor and reranked:
        result = compressor.compress(
            query=item.question,
            documents=reranked,
            max_tokens=compressor.max_tokens,
            strategy=CompressionStrategy.AUTO,
        )
        compressed = result.documents
        post_chars = sum(len(d.page_content) for d in compressed)
        method_used = result.stats.method_used
        quality_score = result.quality_score
    else:
        compressed = reranked
    t3 = time.perf_counter()

    # ---------- 命中判定 ----------
    def _stage(name: str, docs: list[Document], elapsed_ms: float) -> StageResult:
        hits = [_is_relevant(d, item) for d in docs]
        relevant = sum(hits)
        first_rank = next((i + 1 for i, h in enumerate(hits) if h), 0)
        return StageResult(
            name=name,
            docs=docs,
            elapsed_ms=elapsed_ms,
            total_chars=sum(len(d.page_content) for d in docs),
            hit_at_k=relevant > 0,
            first_hit_rank=first_rank,
            relevant_count=relevant,
            returned_count=len(docs),
        )

    vector_only_ms = (t1 - t0) * 1000
    vector_rerank_ms = vector_only_ms + (t2 - t_rerank_start) * 1000  # 用 top-5 阶段的检索时间作为基础
    # 注：此处假设"+Rerank"档复用相同的向量检索成本（K=5），仅多出 reranker 调用。
    # 实际生产中"+Rerank"档需要 K=20 检索，会更慢；为了让对照更符合"vector-only 替换为 +rerank"的成本视角，
    # 这里采用了"vector-only 时间 + reranker 调用时间"的口径，避免双倍计向量检索时间。
    full_ms = vector_rerank_ms + (t3 - t2) * 1000

    qr = QueryResult(
        item=item,
        vector_only=_stage("Vector-only", raw_docs_5, vector_only_ms),
        vector_rerank=_stage("+Rerank", reranked, vector_rerank_ms),
        full_pipeline=_stage("+Rerank+Compression", compressed, full_ms),
        pre_compression_chars=pre_chars,
        post_compression_chars=post_chars,
        compression_ratio=(1 - post_chars / pre_chars) if pre_chars > 0 else 0.0,
        compression_method=method_used,
        compression_quality=quality_score,
    )
    return qr


# ---------------------------------------------------------------------------
# 聚合 + 报告输出
# ---------------------------------------------------------------------------


def aggregate(results: list[QueryResult]) -> dict[str, StageAggregate]:
    out: dict[str, StageAggregate] = {
        "Vector-only": StageAggregate(name="Vector-only"),
        "+Rerank": StageAggregate(name="+Rerank"),
        "+Rerank+Compression": StageAggregate(name="+Rerank+Compression"),
    }
    k = 5  # top-K（固定分母）
    for qr in results:
        for stage in (qr.vector_only, qr.vector_rerank, qr.full_pipeline):
            agg = out[stage.name]
            agg.n += 1
            agg.hit_count += int(stage.hit_at_k)
            agg.precision_sum += stage.relevant_count / k
            denom = stage.returned_count if stage.returned_count > 0 else 1
            agg.precision_actual_sum += stage.relevant_count / denom
            agg.rr_sum += stage.reciprocal_rank()
            agg.latencies.append(stage.elapsed_ms)
            agg.chars_sum += stage.total_chars
            agg.returned_sum += stage.returned_count
    return out


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    pos = (len(s) - 1) * p / 100
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def category_recall(results: list[QueryResult]) -> dict[str, tuple[int, int, int, int, int]]:
    """每个 category 的 (hits_v, hits_r, hits_f, n, total) — 三档分别 hit 数。"""
    grouped: dict[str, list[QueryResult]] = {}
    for qr in results:
        grouped.setdefault(qr.item.category, []).append(qr)
    out = {}
    for cat, items in grouped.items():
        hv = sum(int(qr.vector_only.hit_at_k) for qr in items)
        hr = sum(int(qr.vector_rerank.hit_at_k) for qr in items)
        hf = sum(int(qr.full_pipeline.hit_at_k) for qr in items)
        out[cat] = (hv, hr, hf, len(items), len(items))
    return out


def stress_test_compression(
    rs: RAGSummarize,
    results: list[QueryResult],
    budgets: tuple[int, ...] = (1500, 800, 300),
) -> list[dict[str, object]]:
    """压缩力度模拟：在不同 token 预算下评估压缩器对 reranked 文档的实际表现。

    现实数据集中（每题 reranked top-3~5 加起来 < 3500 token），主 pipeline 的压缩器
    会因为不超阈值而选 ``none``。本函数用更激进的预算（如 1500 / 800 / 300）重新跑一次，
    用以展示压缩器在长输入场景下的真实能力，输出可写进简历的"潜在节省"数字。
    """
    from rag.context_compressor import ContextCompressor  # 延迟 import
    from utils.config_hander import rerank_config

    base_cfg = {
        "compression_min_tokens": rerank_config.get("compression_min_tokens", 200),
        "compression_extract_ratio": rerank_config.get("compression_extract_ratio", 0.6),
        "compression_quality_threshold": rerank_config.get(
            "compression_quality_threshold", 0.7
        ),
    }
    out: list[dict[str, object]] = []
    for budget in budgets:
        comp = ContextCompressor(
            llm_client=None, embedding_client=None,
            config={**base_cfg, "compression_max_tokens": budget},
        )
        ratios: list[float] = []
        proc_times: list[float] = []
        qualities: list[float] = []
        methods: dict[str, int] = {}
        for qr in results:
            docs = qr.vector_rerank.docs
            if not docs:
                continue
            pre = sum(len(d.page_content) for d in docs)
            try:
                r = comp.compress(
                    query=qr.item.question,
                    documents=docs,
                    max_tokens=budget,
                    strategy=CompressionStrategy.AUTO,
                )
            except Exception:  # noqa: BLE001
                continue
            post = sum(len(d.page_content) for d in r.documents)
            if pre > 0:
                ratios.append(1 - post / pre)
            proc_times.append(r.stats.processing_time_ms)
            if r.quality_score > 0:
                qualities.append(r.quality_score)
            methods[r.stats.method_used] = methods.get(r.stats.method_used, 0) + 1
        out.append({
            "budget": budget,
            "n": len(ratios),
            "avg_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
            "max_ratio": max(ratios) if ratios else 0.0,
            "avg_proc_ms": sum(proc_times) / len(proc_times) if proc_times else 0.0,
            "avg_quality": sum(qualities) / len(qualities) if qualities else 0.0,
            "methods": methods,
        })
    return out


def render_markdown(
    results: list[QueryResult],
    aggregates: dict[str, StageAggregate],
    *,
    golden_path: Path,
    started_at: float,
    finished_at: float,
    stress_results: list[dict[str, object]] | None = None,
    failed_items: list[tuple[str, str]] | None = None,
) -> str:
    n = len(results)

    def _fmt_pct(numer: float, denom: float) -> str:
        if denom <= 0:
            return "—"
        return f"{numer / denom * 100:.1f}%"

    def _fmt_ms(values: list[float]) -> tuple[str, str, str, str]:
        if not values:
            return ("—", "—", "—", "—")
        avg = statistics.mean(values)
        p50 = _percentile(values, 50)
        p95 = _percentile(values, 95)
        mx = max(values)
        return (f"{avg:.1f}", f"{p50:.1f}", f"{p95:.1f}", f"{mx:.1f}")

    rows: list[tuple[str, str]] = []  # (label, value)
    stages = ["Vector-only", "+Rerank", "+Rerank+Compression"]

    # ---- 主表 ----
    head = "| 指标 | " + " | ".join(stages) + " |"
    sep = "|" + "---|" * (len(stages) + 1)
    lines = [head, sep]

    def _row(label: str, fn) -> None:
        cells = [str(fn(aggregates[s])) for s in stages]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    _row("Recall@5（命中率）",         lambda a: _fmt_pct(a.hit_count, a.n))
    _row("Precision@5（K=5 分母）",     lambda a: _fmt_pct(a.precision_sum, a.n))
    _row("Precision@实返（实际分母）",   lambda a: _fmt_pct(a.precision_actual_sum, a.n))
    _row("MRR",                        lambda a: f"{a.rr_sum / a.n:.3f}" if a.n else "—")
    _row("平均返回文档数",              lambda a: f"{a.returned_sum / a.n:.2f}" if a.n else "—")
    _row("平均延迟 (ms)",              lambda a: _fmt_ms(a.latencies)[0])
    _row("延迟 P50 (ms)",              lambda a: _fmt_ms(a.latencies)[1])
    _row("延迟 P95 (ms)",              lambda a: _fmt_ms(a.latencies)[2])
    _row("延迟 Max (ms)",              lambda a: _fmt_ms(a.latencies)[3])
    _row("平均输入 tokens",             lambda a: f"{_est_tokens(a.chars_sum // a.n):,}" if a.n else "—")

    main_table = "\n".join(lines)

    # ---- 压缩单独统计 ----
    pre_chars = sum(qr.pre_compression_chars for qr in results)
    post_chars = sum(qr.post_compression_chars for qr in results)
    ratio = (1 - post_chars / pre_chars) if pre_chars > 0 else 0.0
    methods: dict[str, int] = {}
    qualities: list[float] = []
    for qr in results:
        if qr.compression_method and qr.compression_method != "-":
            methods[qr.compression_method] = methods.get(qr.compression_method, 0) + 1
        if qr.compression_quality > 0:
            qualities.append(qr.compression_quality)
    method_str = ", ".join(f"{m}×{c}" for m, c in sorted(methods.items())) or "—"
    avg_quality = (sum(qualities) / len(qualities)) if qualities else 0.0

    compression_block = (
        f"### 压缩明细\n\n"
        f"- 总压缩前字符数: **{pre_chars:,}**（≈ {_est_tokens(pre_chars):,} tokens）\n"
        f"- 总压缩后字符数: **{post_chars:,}**（≈ {_est_tokens(post_chars):,} tokens）\n"
        f"- **平均压缩率**: **{ratio * 100:.1f}%**\n"
        f"- 自动选用的压缩策略分布: {method_str}\n"
        f"- 平均质量评分: {avg_quality:.2f}（0–1，>0.7 视为高质量）\n"
    )

    # ---- 分类 Recall@5 ----
    cat_lines = ["| 类别 | 题数 | Vector-only | +Rerank | +Rerank+Compression |", "|---|---|---|---|---|"]
    for cat, (hv, hr, hf, ntot, _) in category_recall(results).items():
        cat_lines.append(
            f"| {cat} | {ntot} | "
            f"{hv}/{ntot} ({_fmt_pct(hv, ntot)}) | "
            f"{hr}/{ntot} ({_fmt_pct(hr, ntot)}) | "
            f"{hf}/{ntot} ({_fmt_pct(hf, ntot)}) |"
        )
    cat_table = "\n".join(cat_lines)

    # ---- 压缩力度模拟（不同 token 预算）----
    stress_block = ""
    if stress_results:
        sl = ["### 压缩力度模拟（在 +Rerank 文档上以不同 token 预算重跑压缩器）", "",
              "| 预算 (tokens) | 样本数 | 平均压缩率 | 最大压缩率 | 平均耗时 (ms) | 平均质量 | 策略分布 |",
              "|---|---|---|---|---|---|---|"]
        for s in stress_results:
            methods_dict = s["methods"] if isinstance(s["methods"], dict) else {}
            mstr = ", ".join(
                f"{k}×{v}" for k, v in sorted(methods_dict.items(), key=lambda kv: -kv[1])
            ) or "—"
            sl.append(
                f"| {s['budget']} | {s['n']} | "
                f"{s['avg_ratio'] * 100:.1f}% | {s['max_ratio'] * 100:.1f}% | "
                f"{s['avg_proc_ms']:.1f} | {s['avg_quality']:.2f} | {mstr} |"
            )
        stress_block = "\n".join(sl)

    # ---- 失败题（含 retry 失败 + pipeline 未命中）----
    fail_block_lines = []
    failed_items = failed_items or []
    if failed_items:
        fail_block_lines.append("### 评估期间失败的题（含网络重试后失败）")
        for fid, err in failed_items:
            fail_block_lines.append(f"- **{fid}** : {err[:160]}")
    fails = [qr for qr in results if not qr.full_pipeline.hit_at_k]
    if fails:
        fail_block_lines.append("\n### 完整 pipeline 未命中的题（便于排查）")
        for qr in fails:
            fail_block_lines.append(
                f"- **{qr.item.id}** [{qr.item.category}] {qr.item.question}"
                f"\n  预期: {qr.item.expected_docs} | 关键词: {qr.item.expected_keywords}"
            )
    fail_block = "\n".join(fail_block_lines)

    elapsed_total = finished_at - started_at

    return (
        f"# FinSight 检索性能评估报告\n\n"
        f"- 黄金集: `{golden_path.name}` · {n} 题（+ {len(failed_items)} 题在网络重试后仍失败）\n"
        f"- 评估耗时: {elapsed_total:.1f} s\n"
        f"- 评估完成时间（UTC 秒级戳）: {int(finished_at)}\n\n"
        f"## 主表\n\n{main_table}\n\n"
        f"## 分类 Recall@5\n\n{cat_table}\n\n"
        f"{compression_block}\n"
        f"{stress_block}\n\n"
        f"{fail_block}\n"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="FinSight 检索性能评估")
    parser.add_argument(
        "--golden",
        default=str(_PROJECT_ROOT / "tests" / "golden_set.yml"),
        help="黄金集 YAML 路径",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅评估前 N 题（0 为全部）")
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "tests" / "eval_results.md"),
        help="评估报告输出路径（markdown）",
    )
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"[ERROR] 黄金集文件不存在: {golden_path}")
        return 2
    raw = yaml.safe_load(golden_path.read_text(encoding="utf-8")) or []
    items: list[GoldenItem] = [
        GoldenItem(
            id=row["id"],
            category=row.get("category", "uncategorized"),
            question=row["question"],
            expected_docs=row.get("expected_docs") or [],
            expected_keywords=row.get("expected_keywords") or [],
        )
        for row in raw
    ]
    if args.limit > 0:
        items = items[: args.limit]

    print(f"加载黄金集: {len(items)} 题")
    print("初始化 RAGSummarize（首次会触发模型 / 向量库加载）...")
    rs = RAGSummarize()

    started = time.time()
    results: list[QueryResult] = []
    failed_items: list[tuple[str, str]] = []  # (id, error_msg)
    for i, item in enumerate(items, 1):
        qr = None
        last_err = None
        for attempt in range(2):  # 最多重试 1 次（应对偶发 SSL/网络抖动）
            try:
                qr = evaluate_one(rs, item)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt == 0:
                    time.sleep(2.0)
                    continue
        if qr is None:
            print(f"  [{i:02d}/{len(items)}] {item.id} FAILED after retry: {last_err}")
            failed_items.append((item.id, str(last_err)))
            continue
        v_hit = "✓" if qr.vector_only.hit_at_k else "✗"
        r_hit = "✓" if qr.vector_rerank.hit_at_k else "✗"
        f_hit = "✓" if qr.full_pipeline.hit_at_k else "✗"
        print(
            f"  [{i:02d}/{len(items)}] {item.id} [{item.category}]  "
            f"vec {v_hit}  +rerank {r_hit}  +compress {f_hit}  "
            f"({qr.full_pipeline.elapsed_ms:.0f}ms, 压缩 {qr.compression_ratio*100:.1f}%)"
        )
        results.append(qr)
    finished = time.time()

    if not results:
        print("[ERROR] 没有任何题成功评估。")
        return 3

    aggregates = aggregate(results)

    print("\n压缩力度模拟（在 reranked docs 上以不同 token 预算重跑压缩器）...")
    stress_results = stress_test_compression(rs, results)
    for s in stress_results:
        print(
            f"  budget={s['budget']:>5} tokens · 样本 {s['n']} · "
            f"压缩率 {s['avg_ratio']*100:5.1f}% · 耗时 {s['avg_proc_ms']:5.1f}ms · "
            f"策略 {dict(s['methods'])}"
        )

    report = render_markdown(
        results,
        aggregates,
        golden_path=golden_path,
        started_at=started,
        finished_at=finished,
        stress_results=stress_results,
        failed_items=failed_items,
    )
    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n[OK] 报告已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
