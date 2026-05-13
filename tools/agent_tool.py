"""
FinSight 投研助理 Agent 的工具集。

工具分组：
- RAG 类：研报 / 年报 / 公告 / 政策原文等本地语料的检索与总结。
- Hybrid RAG：本地语料 + Web 召回 + 统一 Rerank，覆盖最新动态。
- 行情/时间/计算：投研常用辅助类工具（财务指标、市场时间、Web 搜索）。

历史保留：`get_weather_by_location` / `geocode_place` 在投研场景已不参与
工具路由（未列入 ``TOOLS``），代码保留以便后续业务扩展或单元测试复用。

将 ``TOOLS`` 绑定到 LangChain / LangGraph Agent 的 ``tools`` 参数即可。
"""

from __future__ import annotations

import ast
import json
import logging
import operator
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Callable, TypeVar
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from rag.ragsummarize import RAGSummarize, HybridRAG, _format_docs


_rag = RAGSummarize()
_hybrid_rag = HybridRAG()  # 多路召回 RAG 实例

_logger = logging.getLogger("agent")


# ---------------------------------------------------------------------------
# LLM/Embedding 链路的网络瞬态错误重试 + 软兜底
# ---------------------------------------------------------------------------
#
# 背景：DashScope（ChatTongyi / DashScopeEmbeddings）底层走 ``requests.Session``
# 长连接，阿里云接入层对长时间空闲的 keep-alive 连接会单方面 RST，下一次复用时
# OpenSSL 抛 ``[SSL: UNEXPECTED_EOF_WHILE_READING] _ssl.c:1006``。SDK 自身对该
# 类异常**不重试**，因此需要在工具层兜一层：
#   1. 命中网络/SSL 关键字 → 指数退避重试（0.8s / 1.6s / 3.2s）；
#   2. 重试耗尽 → 退回检索原文，让 agent 仍能继续推理，而不是只看到一句
#      "问答失败：..."。

_NET_RETRY_KEYWORDS: tuple[str, ...] = (
    "unexpected_eof",
    "sslerror",
    "ssl: ",
    "max retries exceeded",
    "connection aborted",
    "connection reset",
    "remote end closed",
    "bad handshake",
    "eof occurred",
    "timed out",
    "timeout",
    "read timed out",
)


def _is_transient_network_error(err: BaseException) -> bool:
    """判断异常是否为可重试的网络/SSL 类瞬态错误。"""
    msg = str(err).lower()
    return any(k in msg for k in _NET_RETRY_KEYWORDS)


_T = TypeVar("_T")


def _call_with_network_retry(
    fn: Callable[[], _T],
    *,
    max_attempts: int = 3,
    base_backoff: float = 0.8,
    op_name: str = "llm_call",
) -> _T:
    """对依赖外部 LLM/Embedding 的调用做有限次网络重试。

    - 仅对网络/SSL 类瞬态错误重试，其他异常立即抛出由上层处理；
    - 单次失败成本 ≈ 当次 RTT；总等待上限 ≈ ``base_backoff * (2^max_attempts - 1)``；
    - 失败 / 重试事件以 WARNING 级别写入 ``agent`` logger，便于线上巡检超时频率。
    """
    last_err: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - 这里需要捕获一切第三方栈错误
            last_err = e
            if attempt < max_attempts - 1 and _is_transient_network_error(e):
                wait = base_backoff * (2 ** attempt)
                _logger.warning(
                    "[%s] 第 %d 次失败（瞬态网络错误），%.1fs 后重试: %s",
                    op_name, attempt + 1, wait, e,
                )
                time.sleep(wait)
                continue
            raise
    # 理论不可达：上面的 raise / return 已覆盖所有出口
    raise last_err  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------


@tool(
    description=(
        "基于本地投研语料库（研报 / 年报 / 季报 / 重大公告 / 政策原文 / 行业百科 / 财经术语）回答用户问题："
        "先做向量检索，再把命中的资料拼进提示词由大模型总结。"
        "**适用场景**：基本面问答、研报观点提炼、年报章节定位、政策原文核对、财经术语解释、"
        "已入库标的的业务/财务/产能/客户结构等问题。"
        "**构造 query 的关键**："
        "  1) 同一标的的多种称呼都串进去（如「宁德时代 CATL 300750」），库内常只写其中一种，单一写法极易漏检；"
        "  2) 问题依赖多轮指代（「那家公司」「他」）时，必须把上文锚定的具体标的写进 query；"
        "  3) 用 dialogue_context 极简要列出上一轮的关键事实（如「用户上文聊的是宁德时代 2024 年报钠离子电池产能」），"
        "     供总结时对齐"
        "「这次问的还是同一只票」。"
        "**不适合**：最新股价 / 最新公告 / 最新新闻等实时性问题——请改用 web_search 或 hybrid_summarize。"
    )
)
def rag_summarize(query: str, dialogue_context: str = "") -> str:
    q = (query or "").strip()
    if not q:
        return "提问为空，请提供具体问题。"
    hint = (dialogue_context or "").strip()
    if hint:
        q = f"{q}\n\n【对话上下文（与检索查询一并交给模型理解，勿向用户逐字复述本标签）】\n{hint}"

    try:
        return _call_with_network_retry(
            lambda: _rag.summarize(q),
            op_name="rag_summarize",
        )
    except Exception as e:  # noqa: BLE001
        if _is_transient_network_error(e):
            _logger.warning(
                "[rag_summarize] LLM 总结连续失败，退回本地检索原文兜底: %s", e,
            )
            try:
                docs = _rag.retrieve_docs(q)
                if docs:
                    return (
                        "（注意：LLM 总结服务暂不可用，已退回本地检索原文，"
                        "请基于以下参考资料自行汇总作答）\n\n"
                        + _format_docs(docs)
                    )
            except Exception as fb_err:  # noqa: BLE001
                _logger.warning(
                    "[rag_summarize] 兜底检索也失败: %s", fb_err,
                )
        return f"问答失败：{str(e)}"


@tool(
    description=(
        "仅从本地投研语料库检索与问题相关的原文片段（不调用大模型生成回答）。"
        "**适用于**：需要**引用原文段落 + 出处**的场景，如研报原话、年报披露口径、公告原文、政策条款等；"
        "或 Agent 想先看材料再决定后续工具的场景。"
        "query 同样应包含标的的多种称呼（公司名 / 股票代码 / 英文名）以避免漏检。"
    )
)
def rag_retrieve(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "提问为空，请提供检索关键词或问题。"
    docs = _rag.retrieve_docs(q)
    if not docs:
        return (
            "（内部说明：本次向量检索未返回片段。请直接依据常识与对话上下文作答用户问题，"
            "勿向用户反复强调「检索失败」或「知识库无结果」。）"
        )
    return _format_docs(docs)


# ---------------------------------------------------------------------------
# 时间与简单计算
# ---------------------------------------------------------------------------


@tool(
    description=(
        "返回指定金融市场时区的当前本地日期与时间，用于判断 A 股 / 港股 / 美股的"
        "交易日窗口、开收盘时点、财报披露时点等。常用 timezone_name："
        "A 股/港股 → Asia/Shanghai 或 Asia/Hong_Kong；"
        "美股 → America/New_York；"
        "其他可填合法 IANA 时区或 UTC。"
        "适用场景：用户询问「现在 A 股开盘了吗」「今天是不是交易日」「美东时间几点」「最新一期季报披露窗口」等；"
        "以及**投资建议、板块/主题推荐、选股对比**等需要在回答中写明「截至某日某时」、将分析与当前市场锚定的场景（应先于或伴随 web_search/hybrid 调用）。"
        "本工具只返回时间字符串，不判断节假日；交易日具体规则需结合最新交易所公告。"
    )
)
def get_market_datetime(timezone_name: str = "Asia/Shanghai") -> str:
    name = (timezone_name or "UTC").strip()
    if name.upper() == "UTC":
        tz = timezone.utc
        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    try:
        tz = ZoneInfo(name)
    except Exception:
        return f'无效时区 "{timezone_name}". 示例: Asia/Shanghai, UTC'
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


def _eval_arith_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_arith_node(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _eval_arith_node(node.operand)
    if isinstance(node, ast.BinOp):
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }
        t = type(node.op)
        if t not in ops:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        left = _eval_arith_node(node.left)
        right = _eval_arith_node(node.right)
        if t is ast.Div and right == 0:
            raise ZeroDivisionError("除以零")
        return ops[t](left, right)
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


@tool(
    description=(
        "对纯算术表达式（仅含数字与 + - * / 和括号）做精确求值，专用于投研中的"
        "**财务指标 / 估值 / 同环比**等计算，避免模型心算出错。"
        "典型用法："
        "  • PE = 市值 / 净利润 → expression='1500 / 80'"
        "  • ROE = 净利润 / 净资产 → expression='80 / 500'"
        "  • 同比增速 = (本期 - 同期) / 同期 → expression='(312.6 - 264.0) / 264.0'"
        "  • 环比 / 毛利率 / 净利率 / EPS / 股息率 等同理。"
        "不支持幂、函数、变量；如需百分比请自行 *100 或在外层叙述时换算。"
        "调用前请把已通过 RAG/Web 拿到的真实数据填进表达式，**不要让本工具构造数据**。"
    )
)
def compute_financial_metric(expression: str) -> str:
    expr = (expression or "").strip()
    if not expr:
        return "表达式为空。"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return f"语法错误: {e}"

    try:
        val = _eval_arith_node(tree.body)
    except ZeroDivisionError:
        return "错误: 除以零。"
    except ValueError as e:
        return f"错误: {e}"
    # 整数尽量以整数显示
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)


# ---------------------------------------------------------------------------
# 天气（Open-Meteo，无需 API Key，需能访问公网）
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT_SECONDS = 12.0


def _http_get_json(
    url: str,
    *,
    timeout: float | None = None,
    service_hint: str = "外网服务",
) -> dict:
    """GET JSON；失败时抛出 ``ValueError``，消息为简短中文，供工具原样返回给模型。"""
    to = REQUEST_TIMEOUT_SECONDS if timeout is None else timeout
    req = urllib.request.Request(url, headers={"User-Agent": "rag-agent-tools/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=to) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        code = getattr(e, "code", None)
        reason = getattr(e, "reason", "") or ""
        if code == 429:
            raise ValueError(
                f"{service_hint}：请求过于频繁（HTTP 429），请稍后再试。"
            ) from e
        if code in (502, 503, 504):
            raise ValueError(
                f"{service_hint}：服务暂时不可用（HTTP {code}），请稍后重试。"
            ) from e
        raise ValueError(
            f"{service_hint}：HTTP {code} {reason}".strip()
        ) from e
    except urllib.error.URLError as e:
        nested = getattr(e, "reason", None)
        err_msg = " ".join(
            x for x in (str(e), str(nested) if nested is not None else "") if x
        ).lower()
        if "timed out" in err_msg or "timeout" in err_msg:
            raise ValueError(
                f"{service_hint}：请求超时，请检查网络或稍后再试。"
            ) from e
        if (
            "name or service not known" in err_msg
            or "getaddrinfo failed" in err_msg
            or "temporary failure in name resolution" in err_msg
        ):
            raise ValueError(
                f"{service_hint}：DNS 解析失败，请检查网络与 DNS 设置。"
            ) from e
        if "certificate" in err_msg or "ssl" in err_msg:
            raise ValueError(
                f"{service_hint}：TLS/证书校验失败，请检查网络或代理环境。"
            ) from e
        raise ValueError(f"{service_hint}：网络不可用（{e.reason or e}）。") from e
    except TimeoutError as e:
        raise ValueError(
            f"{service_hint}：请求超时，请检查网络或稍后再试。"
        ) from e
    except OSError as e:
        err_msg = str(e).lower()
        if "timed out" in err_msg or "timeout" in err_msg:
            raise ValueError(
                f"{service_hint}：请求超时，请检查网络或稍后再试。"
            ) from e
        raise ValueError(f"{service_hint}：{e}") from e

    try:
        text = raw.decode()
    except UnicodeDecodeError as e:
        raise ValueError(
            f"{service_hint}：响应不是合法 UTF-8，无法解析为 JSON。"
        ) from e

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{service_hint}：返回内容不是合法 JSON（{e}）。"
        ) from e


@tool(
    description=(
        "根据城市或地名（中文或英文）查询当前大致气温与天气代码；数据来自 Open-Meteo，需外网。"
        "适用于「北京今天多少度」等；若网络失败或地名无法解析会返回原因。"
    )
)
def get_weather_by_location(location_name: str) -> str:
    name = (location_name or "").strip()
    if not name:
        return "请提供城市或地点名称。"

    q = urllib.parse.urlencode({"name": name, "count": 1, "language": "zh"})
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?{q}"
    try:
        geo = _http_get_json(geo_url, service_hint="地理编码")
    except ValueError as e:
        return str(e)

    results = geo.get("results") or []
    if not results:
        return f'未找到与「{name}」匹配的城市，请尝试更具体的名称。'

    r0 = results[0]
    lat, lon = r0.get("latitude"), r0.get("longitude")
    label = r0.get("name", name)
    country = r0.get("country", "")
    admin = r0.get("admin1", "")
    place = ", ".join(x for x in (label, admin, country) if x)

    params = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code",
            "timezone": "auto",
        }
    )
    wx_url = f"https://api.open-meteo.com/v1/forecast?{params}"
    try:
        wx = _http_get_json(wx_url, service_hint="天气接口")
    except ValueError as e:
        return f"已解析地点「{place}」，但{str(e)}"

    cur = wx.get("current") or {}
    temp = cur.get("temperature_2m")
    code = cur.get("weather_code")
    if temp is None:
        return (
            f"地点「{place}」天气数据不完整（接口未返回 temperature_2m，"
            "可能为限流或响应格式变化）。"
        )
    return f"{place} | 当前约 {temp}°C（WMO weather_code={code}）"


@tool(
    description=(
        "将用户说的城市或地名解析为经纬度与行政区信息（Open-Meteo 地理编码，需外网）。"
        "适用于「上海在哪一带」或为后续地图/路线类工具准备坐标；不进行 GPS 定位。"
    )
)
def geocode_place(place_name: str) -> str:
    name = (place_name or "").strip()
    if not name:
        return "请提供地点名称。"
    q = urllib.parse.urlencode({"name": name, "count": 3, "language": "zh"})
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?{q}"
    try:
        geo = _http_get_json(geo_url, service_hint="地理编码")
    except ValueError as e:
        return str(e)

    rows = geo.get("results") or []
    if not rows:
        return f'未找到「{name}」的候选地点。'

    lines = []
    for i, r in enumerate(rows, 1):
        label = r.get("name", "?")
        admin = r.get("admin1", "")
        country = r.get("country", "")
        lat, lon = r.get("latitude"), r.get("longitude")
        lines.append(f"{i}. {label}, {admin}, {country} — lat={lat}, lon={lon}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 联网搜索（DuckDuckGo，免 API Key）
# ---------------------------------------------------------------------------


def _clean_html(text: str) -> str:
    """移除 HTML 标签并解码实体字符。"""
    # 移除 script 和 style 内容
    text = re.sub(r'<(script|style)[^>]*>[^<]*</\1>', '', text, flags=re.DOTALL)
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 解码常见 HTML 实体
    text = text.replace('&quot;', '"').replace('&amp;', '&')
    text = text.replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&nbsp;', ' ')
    # 压缩空白
    text = ' '.join(text.split())
    return text.strip()


@tool(
    description=(
        "使用 DuckDuckGo 搜索引擎查询投研相关的**实时信息**："
        "最新行情评论、最新公告、并购重组、业绩快报、监管动态、行业新闻、宏观数据等。"
        "**何时使用**：(1) 用户问题包含「最新」「今天」「最近」「Q3」「2025 上半年」等时效词，"
        "或问题涉及模型训练截止后的事件时，必须优先用本工具，禁止凭记忆作答；"
        "(2) **投资建议、板块推荐、潜力股/主题、配置与排序**，即使用户未写「最新」「今天」，也需调用本工具"
        "补充近期公开市场讨论与催化剂，再结合本地研报等语料。"
        "query 建议：标的名称（含代码）+ 事件关键词（如「宁德时代 300750 三季报」「英伟达 NVDA Q3 业绩」）；"
        "板块类可写「科技板块 A股 机构观点 近期」「恒生科技 成分股 动态」等。"
        "注意：DuckDuckGo HTML 接口为非官方，适合演示与小流量；生产场景请替换为商业 API。"
    )
)
def web_search(query: str, max_results: int = 5) -> str:
    """
    使用 DuckDuckGo HTML 接口进行搜索。
    注意：这是非官方接口，适合演示与小流量场景；生产环境建议使用官方 API。
    """
    q = (query or "").strip()
    if not q:
        return "搜索关键词为空，请提供要查询的内容。"

    max_results = max(1, min(int(max_results), 10))

    try:
        # 使用 DuckDuckGo HTML 版（免 JS 版本更稳定）
        params = urllib.parse.urlencode({"q": q, "kl": "zh-cn", "df": ""})
        url = f"https://html.duckduckgo.com/html/?{params}"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.0"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        results = []

        # 解析搜索结果 - DuckDuckGo HTML 结构
        # 每个结果在 .result 或 .web-result 容器中
        result_blocks = re.findall(
            r'<div class="result[^"]*"[^>]*>.*?</div>\s*</div>\s*</div>',
            html,
            re.DOTALL,
        )

        if not result_blocks:
            # 尝试备选解析模式
            result_blocks = re.findall(
                r'<div[^>]*class="[^"]*result[^"]*"[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                html,
                re.DOTALL,
            )

        for block in result_blocks[:max_results]:
            # 提取链接
            link_match = re.search(r'<a[^>]*href="([^"]*)"[^>]*class="[^"]*result__a[^"]*"[^>]*>', block)
            if not link_match:
                link_match = re.search(r'<a[^>]*href="([^"]*)"[^>]*>', block)
            link = link_match.group(1) if link_match else ""

            # 提取标题
            title_match = re.search(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>', block, re.DOTALL)
            if title_match:
                title = _clean_html(title_match.group(1))
            else:
                title = "无标题"

            # 提取摘要
            snippet_match = re.search(
                r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', block, re.DOTALL
            )
            if snippet_match:
                snippet = _clean_html(snippet_match.group(1))
            elif '<div class="result__snippet">' in block:
                snippet_match = re.search(
                    r'<div class="result__snippet">(.*?)</div>', block, re.DOTALL
                )
                snippet = _clean_html(snippet_match.group(1)) if snippet_match else ""
            else:
                snippet = ""

            if title and title != "无标题":
                results.append(f"{len(results) + 1}. {title}\n   链接: {link}\n   摘要: {snippet}\n")

        if not results:
            # 如果结构化解析失败，尝试简单的备选方案
            simple_results = re.findall(
                r'<a[^>]*href="([^"]*)"[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>',
                html,
                re.DOTALL,
            )
            for i, (link, title_raw) in enumerate(simple_results[:max_results], 1):
                title = _clean_html(title_raw)
                results.append(f"{i}. {title}\n   链接: {link}\n")

        if not results:
            return f"未找到与「{q}」相关的搜索结果。可能是网络限制或 DuckDuckGo 页面结构变化。"

        return f"DuckDuckGo 搜索结果（{len(results)}条）：\n\n" + "\n".join(results)

    except urllib.error.HTTPError as e:
        if e.code == 403:
            return "搜索被拒绝（HTTP 403），可能是请求频率限制或需要验证。请稍后再试。"
        return f"搜索 HTTP 错误：{e.code}"
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", str(e))
        if "timed out" in str(reason).lower() or "timeout" in str(reason).lower():
            return "搜索请求超时，请检查网络连接或稍后再试。"
        return f"搜索网络错误：{reason}"
    except Exception as e:
        return f"搜索失败：{str(e)}"


# ---------------------------------------------------------------------------
# 多路召回（Hybrid Search）
# ---------------------------------------------------------------------------


@tool(
    description=(
        "**多路召回**：同时检索本地投研语料库（研报/年报/公告/政策）与 Web（最新新闻/公告/行情评论），"
        "进入统一候选池后由 Rerank 精排，返回最相关的若干条原文片段。"
        "**适用场景**：撰写个股 / 行业速评等需要「基本面 + 最新动态」并存的综合性问题；"
        "或用户提问跨越「历史披露 + 近期变化」的场景（如「宁德时代基本面 + 最近季报有什么变化」）；"
        "以及**投资建议 / 板块或主题推荐**等需并列参考本地沉淀与公开市场动态时。"
        "query 应包含标的多种称呼与事件关键词。"
        "如果只关心「最新动态」用 web_search，只关心「历史披露」用 rag_retrieve / rag_summarize 即可。"
    )
)
def hybrid_search(query: str) -> str:
    """多路召回：本地知识库 + Web 搜索 + Rerank"""
    q = (query or "").strip()
    if not q:
        return "搜索关键词为空，请提供具体问题。"

    try:
        # 调用 HybridRAG 进行多路召回 + Rerank
        docs = _hybrid_rag.retrieve_docs(q)
        if not docs:
            return "（多路召回未返回相关结果，请尝试其他关键词或扩大搜索范围）"

        # 格式化结果，显示来源渠道
        parts = []
        for i, doc in enumerate(docs, start=1):
            score = doc.metadata.get("rerank_score", 0.0)
            channel = doc.metadata.get("source_channel", "unknown")
            source = doc.metadata.get("source", "")
            channel_label = "本地" if channel == "local" else "Web"

            part = f"参考{i} [{channel_label}] [相关性: {score:.3f}]"
            if source:
                part += f"\n来源: {source}"
            part += f"\n{doc.page_content}"
            parts.append(part)

        return f"多路召回结果（共{len(docs)}条）：\n\n" + "\n\n".join(parts)

    except Exception as e:
        return f"多路召回失败：{str(e)}"


@tool(
    description=(
        "**多路召回 + 总结**：从本地投研语料库 + Web 同时召回相关内容，经统一 Rerank 精排后由大模型总结回答。"
        "**适用场景**：撰写个股速评 / 行业速评 / 晨会纪要等需要「基本面 + 最新动态」综合判断的问题；"
        "用户问「最近怎么看 X 公司」「X 行业最近有什么变化」「帮我点评 / 速评 X」等综合性提问；"
        "**板块推荐 / 选股 / 投资观点**：需同时吃进库语料与联网信息时也可用本工具。"
        "**注意**：若问题强依赖「当前日期/交易日」语境，可先 `get_market_datetime` 再在回答中标明时点。"
        "返回的内容会自然综合两路数据，并标注来源；如果只需要原文片段（不要 LLM 加工）请改用 hybrid_search。"
    )
)
def hybrid_summarize(query: str) -> str:
    """多路召回 RAG 问答"""
    q = (query or "").strip()
    if not q:
        return "提问为空，请提供具体问题。"

    try:
        return _call_with_network_retry(
            lambda: _hybrid_rag.summarize(q),
            op_name="hybrid_summarize",
        )
    except Exception as e:  # noqa: BLE001
        # 软兜底：仅在瞬态网络错误时退回检索原文，避免吞掉真实的业务 bug
        if _is_transient_network_error(e):
            _logger.warning(
                "[hybrid_summarize] LLM 总结连续失败，退回检索原文兜底: %s", e,
            )
            try:
                docs = _hybrid_rag.retrieve_docs(q)
                if docs:
                    parts = []
                    for i, doc in enumerate(docs, start=1):
                        score = doc.metadata.get("rerank_score", 0.0)
                        channel = doc.metadata.get("source_channel", "unknown")
                        channel_label = "本地" if channel == "local" else "Web"
                        source = doc.metadata.get("source", "")
                        head = (
                            f"参考{i} [{channel_label}] [相关性: {score:.3f}]"
                        )
                        if source:
                            head += f"\n来源: {source}"
                        parts.append(f"{head}\n{doc.page_content}")
                    return (
                        "（注意：LLM 总结服务暂不可用，已退回多路召回原文，"
                        "请基于以下参考资料自行汇总作答）\n\n"
                        + "\n\n".join(parts)
                    )
            except Exception as fb_err:  # noqa: BLE001
                _logger.warning(
                    "[hybrid_summarize] 兜底检索也失败: %s", fb_err,
                )
        return f"问答失败：{str(e)}"


# ---------------------------------------------------------------------------
# 金融数据工具（行情 / 基本面 / 汇率）—— 实现见 tools/finance_tool.py
# ---------------------------------------------------------------------------

from tools.finance_tool import (  # noqa: E402  保持依赖关系自下而上清晰
    convert_currency,
    get_stock_basics,
    get_stock_kline,
    get_stock_quote,
)


# ---------------------------------------------------------------------------
# 对外导出
# ---------------------------------------------------------------------------

TOOLS = [
    rag_summarize,                # 本地投研语料：检索 + 总结
    rag_retrieve,                 # 本地投研语料：仅检索原文
    hybrid_search,                # 多路召回（本地 + Web）：仅检索原文
    hybrid_summarize,             # 多路召回（本地 + Web）：检索 + 总结
    web_search,                   # 实时新闻 / 公告 / 行情评论
    get_stock_quote,              # 行情快照（最新价 / 涨跌 / 成交）
    get_stock_kline,             # 近若干交易日日线 K 线（走势摘要）
    get_stock_basics,             # 基本面快照（市值 / PE / PB / 换手率）
    convert_currency,             # 货币换算（USD / CNY / HKD / EUR ...）
    compute_financial_metric,     # 财务指标 / 估值 / 同环比 等纯算术
    get_market_datetime,          # 市场时区当前时间（A 股 / 港股 / 美股）
]


__all__ = [
    "TOOLS",
    "REQUEST_TIMEOUT_SECONDS",
    "rag_summarize",
    "rag_retrieve",
    "hybrid_search",
    "hybrid_summarize",
    "web_search",
    "compute_financial_metric",
    "get_market_datetime",
    "get_stock_quote",
    "get_stock_kline",
    "get_stock_basics",
    "convert_currency",
    # 保留以下函数定义但不参与默认工具路由（投研场景不相关，留作扩展）：
    "get_weather_by_location",
    "geocode_place",
]
