"""RAG 检索前 Query 扩展：多查询改写（广度）与子问题分解（深度）。

合并多路向量粗排命中后交由现有父块展开与 Rerank；Rerank 仍使用原始用户问题字符串。
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from utils.log import logger

# ---------------------------------------------------------------------------
# Streamlit 工作台：用户可临时关闭扩展（不修改 rag.yml），仅当前请求上下文生效
# ---------------------------------------------------------------------------

_FORCE_UI_EXPAND_OFF: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_FORCE_UI_EXPAND_OFF",
    default=False,
)


def push_ui_query_expand_force_off() -> contextvars.Token:
    """工作台关闭扩展时调用；返回 reset 用 token（须在 finally 中 reset）。"""
    return _FORCE_UI_EXPAND_OFF.set(True)


def reset_ui_query_expand_force_off(token: contextvars.Token) -> None:
    _FORCE_UI_EXPAND_OFF.reset(token)


def _ui_forced_expand_off() -> bool:
    return _FORCE_UI_EXPAND_OFF.get() is True


# ---------------------------------------------------------------------------
# 供 Streamlit 等前端展示：本轮检索生成的 query 列表（多工具调用可多条）
# ---------------------------------------------------------------------------

@dataclass
class QueryExpandUiRecord:
    """单次 build_search_queries 的摘要，供界面展示。"""

    path_key: str
    path_label: str
    input_preview: str
    search_queries: list[str]
    remark: str


_UI_RECORDS: list[QueryExpandUiRecord] = []


def clear_query_expand_ui_records() -> None:
    """新一轮用户提问前清空（避免串条）。"""
    _UI_RECORDS.clear()


def take_query_expand_ui_records() -> list[QueryExpandUiRecord]:
    """取出并清空队列，供助手回复完成后展示。"""
    out = list(_UI_RECORDS)
    _UI_RECORDS.clear()
    return out


def _preview_text(text: str, limit: int = 160) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= limit:
        return t
    return t[: limit - 1] + "…"


def _append_ui_record(
    *,
    path_key: str,
    path_label: str,
    retrieval_input: str,
    search_queries: list[str],
    remark: str,
) -> None:
    _UI_RECORDS.append(
        QueryExpandUiRecord(
            path_key=path_key,
            path_label=path_label,
            input_preview=_preview_text(retrieval_input),
            search_queries=list(search_queries),
            remark=remark,
        )
    )


_PATH_LABELS = {
    "decompose": "深度 · 子问题分解",
    "decompose_multi": "深度 · 分解 + 每子问多查询",
    "multi_query": "广度 · 多查询改写",
    "single": "未扩展（单条检索）",
}

_VARIANT_PROMPT = """你是投研语料检索助手。用户问题将用于向量检索（研报/财报/公告/政策等）。
请基于下方「检索输入」，生成 {n:d} 条**互不重复**的检索短语或短句，用于提高召回覆盖面。
要求：
- 可包含同一标的的不同称谓（中文名、英文名、股票代码）；
- 使用公告/研报里可能出现的术语变体；
- 不要输出解释、序号或 Markdown，只输出 JSON 数组字符串。

检索输入：
{query}
"""

_DECOMPOSE_PROMPT = """你是投研检索规划助手。将用户问题拆解为至多 {m:d} 个**可独立向量检索**的中文短句。
每个短句聚焦一个可查主题；保留公司名称、代码、时间与政策名等锚点。
只输出 JSON 字符串数组，不要其它文字。

用户问题：
{query}
"""

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")

_DECOMPOSE_TRIGGER = re.compile(
    r"(对比|相比较|差别|差异|区别|优缺点|哪些方面|影响因素|分别从|分别从哪|多角度|分项|几个方面)"
)


def should_decompose_for_depth(user_query: str) -> bool:
    q = (user_query or "").strip()
    if not q:
        return False
    return bool(_DECOMPOSE_TRIGGER.search(q))


def _dedupe_queries_preserve(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        text = (q or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _message_text(resp: Any) -> str:
    if resp is None:
        return ""
    content = getattr(resp, "content", resp)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            else:
                parts.append(str(block))
        return "".join(parts).strip()
    return str(content).strip()


def parse_json_string_list(raw: str) -> list[str]:
    """从模型回复中抽取 JSON 字符串数组。"""
    text = (raw or "").strip()
    if not text:
        return []

    cand = text
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        cand = fence.group(1).strip()

    m = _JSON_ARRAY_RE.search(cand)
    if m:
        cand = m.group(0)

    try:
        data = json.loads(cand)
    except json.JSONDecodeError:
        logger.warning("query_expand: JSON 解析失败，截取内容: %s", cand[:200])
        return []

    if not isinstance(data, list):
        return []

    out: list[str] = []
    for item in data:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
        elif isinstance(item, (int, float)):
            out.append(str(item))
    return out


def llm_generate_query_variants(
    llm: Any,
    user_query: str,
    *,
    n_variants: int,
) -> list[str]:
    """生成至多 n_variants 条额外检索短语（不包含原问）。"""
    n = max(0, min(int(n_variants), 12))
    if n == 0 or llm is None:
        return []

    prompt = _VARIANT_PROMPT.format(n=n, query=user_query.strip())
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        items = parse_json_string_list(_message_text(resp))
    except Exception:
        logger.exception("query_expand: 多查询 LLM 调用失败")
        return []

    cleaned: list[str] = []
    for s in items:
        one = str(s).strip()
        if not one:
            continue
        if len(one) > 800:
            one = one[:800]
        cleaned.append(one)
        if len(cleaned) >= n:
            break
    return cleaned


def llm_generate_decomposed_subqueries(
    llm: Any,
    user_query: str,
    *,
    max_subqueries: int,
) -> list[str]:
    """生成子问题列表（不含用户原句）。"""
    m_cap = max(1, min(int(max_subqueries), 8))
    if llm is None:
        return []

    prompt = _DECOMPOSE_PROMPT.format(m=m_cap, query=user_query.strip())
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        items = parse_json_string_list(_message_text(resp))
    except Exception:
        logger.exception("query_expand: 子问题分解 LLM 失败")
        return []

    out: list[str] = []
    for s in items:
        one = str(s).strip()
        if not one or len(one) > 600:
            if one:
                one = one[:600]
            else:
                continue
        out.append(one)
        if len(out) >= m_cap:
            break
    return out


def document_dedupe_key(doc: Document) -> str:
    meta = doc.metadata if isinstance(doc.metadata, dict) else {}
    pid = meta.get("parent_id")
    if pid:
        return f"p:{pid}"
    src = str(meta.get("source", ""))
    digest = hashlib.sha256((doc.page_content or "").encode("utf-8", errors="ignore")).hexdigest()
    return f"c:{src}:{digest}"


def dedupe_documents_preserve_order(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for d in docs:
        key = document_dedupe_key(d)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def cap_documents(docs: list[Document], max_docs: int) -> list[Document]:
    mx = max(0, int(max_docs))
    if mx == 0 or len(docs) <= mx:
        return docs
    return docs[:mx]


def coarse_retrieve_union(
    retriever,
    search_queries: list[str],
    *,
    max_coarse_docs: int,
    max_workers: int = 8,
) -> list[Document]:
    """对多条 query 并行向量检索，子块层面去重并按上限截断。"""
    unique_q = _dedupe_queries_preserve(search_queries)
    if not unique_q:
        return []

    mx_workers = max(1, min(int(max_workers), len(unique_q)))

    def _one(q: str) -> list[Document]:
        return list(retriever.invoke(q))

    with ThreadPoolExecutor(max_workers=mx_workers) as pool:
        nested = list(pool.map(_one, unique_q))

    merged: list[Document] = []
    for bucket in nested:
        merged.extend(bucket)

    merged = dedupe_documents_preserve_order(merged)
    return cap_documents(merged, max_coarse_docs)


def build_search_queries(
    *,
    retrieval_input: str,
    cfg: dict,
    llm: Any,
) -> list[str]:
    """根据配置生成送入向量检索的 query 列表（已去重保序）。"""
    q0 = (retrieval_input or "").strip()
    if not q0:
        return []

    if _ui_forced_expand_off():
        _append_ui_record(
            path_key="single",
            path_label=_PATH_LABELS["single"],
            retrieval_input=q0,
            search_queries=[q0],
            remark=(
                "工作台 **已关闭**「使用 Query 扩展」：本轮强制单条原问检索，"
                "不调用多查询/分解 LLM（覆盖 `config/rag.yml` 中的扩展开关）。"
            ),
        )
        return [q0]

    exp_enabled = cfg.get("query_expansion_enabled", False)
    n_variants = cfg.get("query_expansion_variants", 5)
    include_orig = cfg.get("query_expansion_include_original", True)

    deco_enabled = cfg.get("query_decompose_enabled", False)
    deco_max = cfg.get("query_decompose_max_subqueries", 4)
    deco_expand = cfg.get("query_decompose_with_expansion", False)

    if deco_enabled and should_decompose_for_depth(q0) and llm is not None:
        subs = llm_generate_decomposed_subqueries(llm, q0, max_subqueries=int(deco_max))
        cores = _dedupe_queries_preserve([q0] + subs)
        logger.info(
            "query_expand: decomposition path subqueries=%s (triggered)",
            len(cores),
        )
        if deco_expand and exp_enabled:
            aggregated: list[str] = []
            for c in cores:
                extra = llm_generate_query_variants(llm, c, n_variants=int(n_variants))
                merged_one = ([] if not include_orig else [c]) + extra
                aggregated.extend(_dedupe_queries_preserve(merged_one))
            final = _dedupe_queries_preserve(aggregated)
            _append_ui_record(
                path_key="decompose_multi",
                path_label=_PATH_LABELS["decompose_multi"],
                retrieval_input=q0,
                search_queries=final,
                remark=(
                    "已命中深度触发词，且开启「分解 + 每子问多查询」："
                    "对每个子问再生成检索变体后合并；仅作用于本地 Chroma；"
                    "Hybrid 时 Web 仍用原问一条；Rerank/总结仍用原问。"
                ),
            )
            return final
        _append_ui_record(
            path_key="decompose",
            path_label=_PATH_LABELS["decompose"],
            retrieval_input=q0,
            search_queries=cores,
            remark=(
                "已命中深度触发词：LLM 拆成多条子问题并与原问合并后并行向量检索；"
                "Hybrid 时 Web 仍只使用原问；Rerank/总结仍用原问。"
            ),
        )
        return cores

    if exp_enabled and llm is not None:
        extra = llm_generate_query_variants(llm, q0, n_variants=int(n_variants))
        merged = ([] if not include_orig else [q0]) + extra
        uniq = _dedupe_queries_preserve(merged)
        logger.info(
            "query_expand: multi-query variants=%s (total_queries=%s)",
            len(extra),
            len(uniq),
        )
        out_q = uniq if uniq else [q0]
        _append_ui_record(
            path_key="multi_query",
            path_label=_PATH_LABELS["multi_query"],
            retrieval_input=q0,
            search_queries=out_q,
            remark=(
                f"广度：在检索输入基础上由 LLM 最多再生成 {n_variants} 条用语，"
                f"合并后共 {len(out_q)} 条并行查库；Rerank/总结仍针对原问。"
            ),
        )
        return out_q

    parts: list[str] = []
    if not exp_enabled:
        parts.append("「广度」未开启（`query_expansion_enabled`）")
    if llm is None:
        parts.append("对话模型不可用，无法调用扩写 LLM")
    if deco_enabled and llm is not None and not should_decompose_for_depth(q0):
        parts.append("已开「深度」但未命中触发词（对比/优缺点/哪些方面等）")
    remark = "本次仅使用单条原问做向量粗排（未走多检索用语扩写）。"
    if parts:
        remark += " " + "；".join(parts) + "。可在 `config/rag.yml` 调整。"

    _append_ui_record(
        path_key="single",
        path_label=_PATH_LABELS["single"],
        retrieval_input=q0,
        search_queries=[q0],
        remark=remark,
    )
    return [q0]
