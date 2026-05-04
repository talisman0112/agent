"""
Agent 可调用的基础工具集。

使用场景（选型说明）：
- RAG 类：企业内部知识、长文档问答，需先入库向量库。
- 时间与计算：回答「现在几点」「帮我算一下」等确定性问题，避免模型算错。
- 天气 / 城市解析：需要外网；Open-Meteo 免 Key，适合演示与小流量；生产可换商业天气 API。

将 `TOOLS` 绑定到 LangChain / LangGraph Agent 的 `tools` 参数即可。
"""

from __future__ import annotations

import ast
import json
import operator
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from rag.ragsummarize import RAGSummarize, _format_docs


_rag = RAGSummarize()


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------


@tool(
    description=(
        "基于本地向量知识库回答用户问题：先做检索，再把命中的资料拼进提示词由大模型总结。"
        "适用于：公司内部文档、手册、政策；用户身份/称呼等写入库的个人信息；"
        "以及计算机与 IT 入门材料（体系结构、OS、网络、数据库、算法、RAG/机器学习等）——"
        "只要问题可能落在已上传文档里，就应使用本工具，而不是凭模型记忆直接长答。"
        "当用户说「讲一些计算机知识」「入门」「科普」等宽泛问题时，仍应用用户原话或简要关键词调用本工具。"
        "检索参数 query 要尽量覆盖「用户原话里的实体 + 对话里已出现的同义称呼」（实名/昵称/外号/英文别名），"
        "并可追加「别名」「外号」等词；库内正文常只写其中一种称呼，仅用另一种称呼检索极易漏检。"
        "若问题依赖多轮对话指代（如只说「他」但上文出现过原名与外号的对应），必须把可核对的称呼写进 query；"
        "同时用 dialogue_context 极简要列出上一轮中与此人相关的语句（一两句即可），供总结时对齐「谁在问谁」。"
        "不适合：实时新闻、明显与库无关的纯闲聊。"
    )
)
def rag_summarize(query: str, dialogue_context: str = "") -> str:
    q = (query or "").strip()
    if not q:
        return "提问为空，请提供具体问题。"
    hint = (dialogue_context or "").strip()
    if hint:
        q = f"{q}\n\n【对话上下文（与检索查询一并交给模型理解，勿向用户逐字复述本标签）】\n{hint}"
    return _rag.summarize(q)


@tool(
    description=(
        "仅从向量库检索与问题相关的原文片段（不调用大模型生成回答）。"
        "适用于需要引用原文、核对出处、或 Agent 想先看材料再决定的场景。"
        "query 请包含对话中的实名、昵称、外号等同指称呼，避免单称与库内写法不一致导致漏检。"
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
        "返回指定 IANA 时区的当前本地日期与时间，例如 Asia/Shanghai、America/New_York、UTC。"
        "用于回答「现在几点」「今天是几号」等；时区名须合法，否则返回错误说明。"
    )
)
def get_local_datetime(timezone_name: str = "Asia/Shanghai") -> str:
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
        "对仅含数字与 + - * / 和括号的算术表达式求值，例如 (12+3)*4、100/5。"
        "用于替代模型心算，减少数值错误；不支持幂、函数、变量等非纯算术内容。"
    )
)
def calculate_arithmetic(expression: str) -> str:
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
# 对外导出
# ---------------------------------------------------------------------------

TOOLS = [
    rag_summarize,
    rag_retrieve,
    get_local_datetime,
    calculate_arithmetic,
    get_weather_by_location,
    geocode_place,
]

__all__ = [
    "TOOLS",
    "REQUEST_TIMEOUT_SECONDS",
    "rag_summarize",
    "rag_retrieve",
    "get_local_datetime",
    "calculate_arithmetic",
    "get_weather_by_location",
    "geocode_place",
]
