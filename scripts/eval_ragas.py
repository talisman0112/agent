"""RAGAS 端到端评估：读取 YAML/JSONL 评测集、复用 ``RAGSummarize`` 跑通检索+生成，输出 CSV。

依赖（需单独安装）::

    pip install ragas

建议使用项目可选依赖::

    pip install -r requirements.txt -r requirements-optional.txt

环境变量与日常 RAG 相同（如 ``DASHSCOPE_API_KEY``），需可用的向量库与 Chat 模型。

数据集格式
----------
- **YAML**（默认 ``tests/golden_set.yml``）：每行字段 ``id``、``question`` 必填；
  ``category``、``expected_docs`` 等会原样写入明细 CSV；
  若每题都有非空 ``reference``（或 ``ground_truth`` / ``expected_answer``），则额外计算 ``context_recall``。
- **JSONL**：每行一个 JSON，至少含 ``question``；可选 ``id``、``category``、``reference``。

指标（与当前 RAGAS 内置默认不同，避免要求 ``reference`` 的 context_precision 等）：

- ``faithfulness``
- ``answer_relevancy``（需 embedding，复用项目 ``embedding_model``）

用法::

    python scripts/eval_ragas.py
    python scripts/eval_ragas.py --dataset tests/golden_set.yml --limit 5

提升 ``faithfulness`` 时可将 ``config/rag.yml`` 中 ``rag_strict_grounding: true``，以在统一提示词后追加强约束尾注（禁止通识兜底段）。

超时：若控制台出现 ``TimeoutError()``，可提高 ``ragas_eval_timeout_seconds`` 或传 ``--ragas-timeout``（RAGAS 默认单任务 180s，``faithfulness`` 易超限）。

示例::

    python scripts/eval_ragas.py --dataset ./data/eval.jsonl --summary-out reports/ragas_summary.csv
    python scripts/eval_ragas.py --ragas-timeout 600 --ragas-max-workers 4
    python scripts/eval_ragas.py --rebuild-summary

输出
----
- ``--rows-out``：每题一行（含耗时、上下文条数、各指标得分、错误信息）。
- ``--summary-out``：各指标均值/标准差/有效样本数（汇总）。
- Windows 下若 Excel/WPS 正在打开汇总 CSV，写入可能 ``PermissionError``；脚本会优先写明细，
  并可用 ``--rebuild-summary`` 仅根据已生成的明细重算汇总（无需重跑模型）。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import statistics
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:  # pragma: no cover
    print("[ERROR] 需要 PyYAML：pip install PyYAML", file=sys.stderr)
    raise SystemExit(2) from e

try:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_recall import context_recall
    from ragas.metrics._faithfulness import faithfulness
    from ragas.run_config import RunConfig
except ImportError as e:  # pragma: no cover
    print(
        "[ERROR] 需要 ragas：pip install ragas\n"
        "或：pip install -r requirements.txt -r requirements-optional.txt",
        file=sys.stderr,
    )
    raise SystemExit(2) from e

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _int_from_config(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def build_ragas_run_config(args: argparse.Namespace) -> RunConfig:
    """RAGAS ``RunConfig``。

    优先级：CLI → ``RAGAS_EVAL_TIMEOUT_SECONDS`` / ``RAGAS_EVAL_MAX_WORKERS`` → ``rag.yml``。
    （RAGAS 库若不传 ``run_config`` 则单任务仅 180s，``faithfulness`` 常会 ``TimeoutError``。）
    """
    from utils.config_hander import rag_config  # noqa: E402 — 在项目根已在 path 之后

    # 单样本×单指标超时（秒）；官方默认 180，faithfulness 内多跳 LLM 易触发 TimeoutError
    timeout_s: int | None = args.ragas_timeout
    if timeout_s is None:
        env_t = os.environ.get("RAGAS_EVAL_TIMEOUT_SECONDS", "").strip()
        if env_t.isdigit():
            timeout_s = int(env_t)
        else:
            timeout_s = _int_from_config(
                rag_config.get("ragas_eval_timeout_seconds"),
                480,
            )
    timeout_s = max(60, min(timeout_s, 7200))

    workers: int | None = args.ragas_max_workers
    if workers is None:
        env_w = os.environ.get("RAGAS_EVAL_MAX_WORKERS", "").strip()
        if env_w.isdigit():
            workers = int(env_w)
        else:
            workers = _int_from_config(
                rag_config.get("ragas_eval_max_workers"),
                6,
            )
    workers = max(1, min(workers, 32))

    return RunConfig(timeout=int(timeout_s), max_workers=int(workers))


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    suf = path.suffix.lower()
    if suf in {".yml", ".yaml"}:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if not isinstance(raw, list):
            raise ValueError(f"YAML 评测集应为列表: {path}")
        return raw
    if suf == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    raise ValueError(f"不支持的评测集格式: {path}（使用 .yml / .yaml / .jsonl）")


def _row_question(row: dict[str, Any]) -> str:
    q = row.get("question") or row.get("query") or row.get("user_input")
    if not q or not str(q).strip():
        raise KeyError("记录缺少 question / query / user_input")
    return str(q).strip()


def _row_reference(row: dict[str, Any]) -> str:
    ref = row.get("reference") or row.get("ground_truth") or row.get("expected_answer")
    if ref is None:
        return ""
    return str(ref).strip()


def _row_id(row: dict[str, Any], index: int) -> str:
    rid = row.get("id")
    if rid is not None and str(rid).strip():
        return str(rid).strip()
    return f"row_{index:04d}"


def _is_nan(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float):
        return math.isnan(x)
    try:
        return bool(math.isnan(float(x)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _nanmean_std(values: list[float]) -> tuple[float, float, int]:
    xs = [float(v) for v in values if not _is_nan(v)]
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    if n == 1:
        return (xs[0], float("nan"), 1)
    return (statistics.mean(xs), statistics.stdev(xs), n)


# 明细 CSV 中非 RAGAS 指标的列（用于 --rebuild-summary）
_ROWS_NON_METRIC_COLS = frozenset({
    "id",
    "category",
    "question",
    "reference",
    "n_contexts",
    "rag_llm_ms",
    "retrieve_to_answer_total_ms",
    "error",
    "response",
})


def _write_csv_atomic(path: Path, writer: Any) -> None:
    """经临时文件再 replace，降低 Windows 上「文件被占用」导致的写入失败。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        suffix=".csv",
        prefix=f"{path.stem}_",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            writer(fh)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def rebuild_summary_from_rows_file(
    rows_path: Path,
    summary_path: Path,
    *,
    dataset_label: str,
    metrics_used: str = "",
) -> None:
    """根据已生成的 eval_ragas 明细 CSV 重新计算汇总（不重跑 RAG / RAGAS）。"""
    text = rows_path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError(f"{rows_path} 无表头或为空")
    metric_keys = [k for k in reader.fieldnames if k not in _ROWS_NON_METRIC_COLS]
    rows = list(reader)
    n_total = len(rows)

    def write_summary(fh: Any) -> None:
        sw = csv.DictWriter(
            fh,
            fieldnames=[
                "metric",
                "mean",
                "std",
                "n_valid",
                "dataset",
                "n_total",
                "n_evaluated",
                "metrics_used",
            ],
        )
        sw.writeheader()
        n_eval = sum(1 for r in rows if not (r.get("error") or "").strip())
        for key in metric_keys:
            vals: list[float] = []
            for r in rows:
                cell = (r.get(key) or "").strip()
                if not cell:
                    continue
                try:
                    vals.append(float(cell))
                except ValueError:
                    continue
            mean_v, std_v, n_v = _nanmean_std(vals)
            sw.writerow({
                "metric": key,
                "mean": f"{mean_v:.6f}" if not math.isnan(mean_v) else "",
                "std": f"{std_v:.6f}" if not math.isnan(std_v) else "",
                "n_valid": str(n_v),
                "dataset": dataset_label,
                "n_total": str(n_total),
                "n_evaluated": str(n_eval),
                "metrics_used": metrics_used or ";".join(metric_keys),
            })

    try:
        _write_csv_atomic(summary_path, write_summary)
    except (PermissionError, OSError) as e:
        fb = summary_path.with_name(
            f"{summary_path.stem}_fallback_{int(time.time())}{summary_path.suffix}"
        )
        _write_csv_atomic(fb, write_summary)
        print(
            f"[WARN] 无法写入 {summary_path}（{e}），已写入 {fb}",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS 评估（RAGSummarize + CSV 输出）")
    parser.add_argument(
        "--dataset",
        default=str(_PROJECT_ROOT / "tests" / "golden_set.yml"),
        help="评测集路径（.yml / .yaml / .jsonl）",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅评前 N 条（0 表示全部）")
    parser.add_argument(
        "--pipeline",
        default="local",
        choices=["local", "hybrid"],
        help="RAG pipeline: local or hybrid",
    )
    parser.add_argument(
        "--rows-out",
        default=str(_PROJECT_ROOT / "tests" / "eval_ragas_rows.csv"),
        help="每题明细 CSV",
    )
    parser.add_argument(
        "--summary-out",
        default=str(_PROJECT_ROOT / "tests" / "eval_ragas_summary.csv"),
        help="指标汇总 CSV",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭 RAGAS 进度条",
    )
    parser.add_argument(
        "--rebuild-summary",
        action="store_true",
        help="仅根据已有 --rows-out 明细重算 --summary-out（不重跑模型与 RAGAS）",
    )
    parser.add_argument(
        "--ragas-timeout",
        type=int,
        default=None,
        metavar="SEC",
        help=(
            "RAGAS RunConfig.timeout：单样本×单指标异步等待上限（秒）；"
            "缺省读 rag.yml 的 ragas_eval_timeout_seconds（无则 480）；"
            "或环境变量 RAGAS_EVAL_TIMEOUT_SECONDS（RAGAS 库自带默认仅 180）"
        ),
    )
    parser.add_argument(
        "--ragas-max-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "RAGAS RunConfig.max_workers：评估并行度；过低慢、过高易排队像超时。"
            "缺省读 rag.yml（无则 6）或环境变量 RAGAS_EVAL_MAX_WORKERS"
        ),
    )
    args = parser.parse_args()

    rows_out = Path(args.rows_out)
    summary_out = Path(args.summary_out)

    if args.rebuild_summary:
        if not rows_out.is_file():
            print(f"[ERROR] 找不到明细 CSV: {rows_out}", file=sys.stderr)
            return 2
        try:
            rebuild_summary_from_rows_file(
                rows_out,
                summary_out,
                dataset_label=str(rows_out.resolve()),
            )
        except (OSError, ValueError) as e:
            print(f"[ERROR] 重算汇总失败: {e}", file=sys.stderr)
            return 2
        print(f"[OK] 汇总已写入: {summary_out}")
        return 0

    ds_path = Path(args.dataset)
    try:
        raw_rows = _load_dataset(ds_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    if args.limit > 0:
        raw_rows = raw_rows[: args.limit]

    from model.model import chat_model, embedding_model  # noqa: E402

    if chat_model is None:
        print(
            "[ERROR] chat_model 为 None。请安装 langchain-community / dashscope 并设置 "
            "DASHSCOPE_API_KEY 或 TONGYI_API_KEY。",
            file=sys.stderr,
        )
        return 3
    if embedding_model is None:
        print(
            "[ERROR] embedding_model 为 None（answer_relevancy 需要）。"
            "请配置 DashScope Embeddings。",
            file=sys.stderr,
        )
        return 3

    print(f"加载评测集: {ds_path} · {len(raw_rows)} 条")
    pipeline = args.pipeline
    if pipeline == "hybrid":
        from rag.ragsummarize import HybridRAG  # noqa: E402
        rag = HybridRAG()
        print("初始化 HybridRAG（多路召回）...")
    else:
        from rag.ragsummarize import RAGSummarize  # noqa: E402
        rag = RAGSummarize()
        print("初始化 RAGSummarize...")

    samples: list[dict[str, Any]] = []
    row_meta: list[dict[str, Any]] = []
    first_pipeline_err: str | None = None

    for i, row in enumerate(raw_rows):
        rid = _row_id(row, i)
        q = _row_question(row)
        ref = _row_reference(row)
        cat = row.get("category", "") or ""

        err = ""
        contexts: list[str] = []
        answer = ""
        t_llm_ms = float("nan")

        t0 = time.perf_counter()
        try:
            docs = rag.retrieve_docs(q)
            contexts = [d.page_content or "" for d in docs if (d.page_content or "").strip()]
            t1 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    answer = rag.summarize_with_docs(q, docs)
            t2 = time.perf_counter()
            t_llm_ms = (t2 - t1) * 1000.0
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            if first_pipeline_err is None:
                first_pipeline_err = err

        sample: dict[str, Any] = {
            "user_input": q,
            "retrieved_contexts": contexts if contexts else [""],
            "response": answer or "",
        }
        if ref:
            sample["reference"] = ref
        samples.append(sample)

        row_meta.append({
            "id": rid,
            "category": cat,
            "question": q,
            "reference": ref,
            "n_contexts": len(contexts),
            "rag_llm_ms": f"{t_llm_ms:.1f}" if not math.isnan(t_llm_ms) else "",
            "retrieve_to_answer_total_ms": f"{(time.perf_counter() - t0) * 1000.0:.1f}",
            "error": err,
        })

        status = "ERR" if err else "OK"
        print(f"  [{i + 1:03d}/{len(raw_rows)}] {rid} [{cat!s}] {status}  ctx={len(contexts)}")

    evaluable_indices = [i for i, m in enumerate(row_meta) if not m["error"]]
    if not evaluable_indices:
        print("[ERROR] 没有成功生成的样本，无法运行 RAGAS。", file=sys.stderr)
        if first_pipeline_err:
            print(
                "[ERROR] ERR 表示 retrieve_docs / summarize 抛错，不一定只是「没召回到」。"
                " 首条异常（通常 30 题同源）：",
                file=sys.stderr,
            )
            print(f"       {first_pipeline_err}", file=sys.stderr)
        return 4

    has_full_reference = all(bool(_row_reference(raw_rows[i])) for i in evaluable_indices)
    metrics = [faithfulness, answer_relevancy]
    if has_full_reference:
        metrics = [faithfulness, answer_relevancy, context_recall]

    eval_samples = [samples[i] for i in evaluable_indices]
    dataset = EvaluationDataset.from_list(eval_samples)

    run_config = build_ragas_run_config(args)
    print(
        f"运行 RAGAS（{len(eval_samples)} 条，指标: {[m.name for m in metrics]}）；"
        f" RunConfig(timeout={run_config.timeout}s, max_workers={run_config.max_workers})…"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=chat_model,
            embeddings=embedding_model,
            show_progress=not args.no_progress,
            raise_exceptions=False,
            run_config=run_config,
        )

    score_keys: list[str] = []
    if result.scores:
        score_keys = sorted(result.scores[0].keys())

    score_by_index: dict[int, dict[str, Any]] = {}
    for j, src_i in enumerate(evaluable_indices):
        score_by_index[src_i] = dict(result.scores[j])

    per_row_scores: list[dict[str, Any]] = [
        score_by_index.get(i, {k: "" for k in score_keys}) for i in range(len(row_meta))
    ]

    row_fieldnames = [
        "id",
        "category",
        "question",
        "reference",
        "n_contexts",
        "rag_llm_ms",
        "retrieve_to_answer_total_ms",
        "error",
        "response",
        *score_keys,
    ]

    def write_rows_fh(fh: Any) -> None:
        w = csv.DictWriter(fh, fieldnames=row_fieldnames, extrasaction="ignore")
        w.writeheader()
        for meta, scores_row, sample in zip(row_meta, per_row_scores, samples, strict=True):
            rec = {
                "id": meta["id"],
                "category": meta["category"],
                "question": meta["question"],
                "reference": meta["reference"],
                "n_contexts": meta["n_contexts"],
                "rag_llm_ms": meta["rag_llm_ms"],
                "retrieve_to_answer_total_ms": meta["retrieve_to_answer_total_ms"],
                "error": meta["error"],
                "response": sample.get("response", ""),
            }
            for k in score_keys:
                v = scores_row.get(k, "")
                if _is_nan(v):
                    rec[k] = ""
                else:
                    rec[k] = f"{float(v):.6f}" if isinstance(v, (int, float)) else str(v)
            w.writerow(rec)

    _write_csv_atomic(rows_out, write_rows_fh)

    metrics_used = ";".join(m.name for m in metrics)

    def write_summary_fh(fh: Any) -> None:
        sw = csv.DictWriter(
            fh,
            fieldnames=[
                "metric",
                "mean",
                "std",
                "n_valid",
                "dataset",
                "n_total",
                "n_evaluated",
                "metrics_used",
            ],
        )
        sw.writeheader()
        for key in score_keys:
            vals: list[float] = []
            for i in evaluable_indices:
                v = per_row_scores[i].get(key)
                if v == "" or v is None:
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            mean_v, std_v, n_v = _nanmean_std(vals)
            sw.writerow({
                "metric": key,
                "mean": f"{mean_v:.6f}" if not math.isnan(mean_v) else "",
                "std": f"{std_v:.6f}" if not math.isnan(std_v) else "",
                "n_valid": str(n_v),
                "dataset": str(ds_path),
                "n_total": str(len(raw_rows)),
                "n_evaluated": str(len(evaluable_indices)),
                "metrics_used": metrics_used,
            })

    try:
        _write_csv_atomic(summary_out, write_summary_fh)
    except (PermissionError, OSError) as e:
        fb = summary_out.with_name(
            f"{summary_out.stem}_fallback_{int(time.time())}{summary_out.suffix}"
        )
        _write_csv_atomic(fb, write_summary_fh)
        print(
            f"[WARN] 无法写入 {summary_out}（{e}）。汇总已写入: {fb}",
            file=sys.stderr,
        )

    print(f"\n[OK] 明细 CSV: {rows_out}")
    print(f"[OK] 汇总 CSV: {summary_out}")
    if score_keys:
        print(f"RAGAS: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
