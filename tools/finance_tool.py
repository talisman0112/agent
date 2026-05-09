"""FinSight 投研助理 · 金融数据工具集（行情 / 基本面 / 汇率）。

工具一览
========
- :func:`get_stock_quote`     —— 行情快照（最新价 / 涨跌 / 成交 / 时间）
- :func:`get_stock_basics`    —— 基本面快照（市值 / PE-TTM / PB / 换手率 / 行业）
- :func:`convert_currency`    —— 货币换算（基于 ECB 等公开汇率，免 Key）

数据源
======
- 行情 / 基本面：东方财富 push2 接口 ``push2.eastmoney.com/api/qt/stock/get``
  - secid 编码：A 股沪市 ``1.<6 位代码>``、A 股深市 / 北交所 ``0.<6 位代码>``、
    港股 ``116.<5 位代码>``、美股 ``105.<symbol>``（NASDAQ）/ ``106.<symbol>``（NYSE）。
  - ``fltt=2`` 让接口直接返回浮点数，省去乘除 scale。
- 汇率：``open.er-api.com/v6/latest/<base>``（基于 ECB 等多源公开汇率，免 Key）。

特点
====
1. 所有工具**免 API Key**，适合简历项目 / 公开 demo；
2. 同一 ticker 接受多种写法（``600519`` / ``sh600519`` / ``1.600519``、
   ``00700`` / ``hk00700``、``NVDA`` / ``us NVDA``），由 ``_normalize_ticker`` 统一解析；
3. 美股若 NASDAQ 拿不到结果会自动回退到 NYSE，避免用户手动指定交易所。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# 内部 HTTP 工具（带浏览器 UA / Referer，针对东财 push2 等场景）
# ---------------------------------------------------------------------------

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

_DEFAULT_TIMEOUT = 15.0
_DEFAULT_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 0.6


class _RetriableError(ValueError):
    """对超时 / 5xx 等可重试错误的标记；外层会重试一次。"""


def _http_get_json_once(
    url: str,
    *,
    referer: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    service_hint: str = "外网服务",
) -> dict:
    """单次 GET JSON；失败抛 ``ValueError``（其中可重试错误抛 ``_RetriableError``）。"""
    headers = {"User-Agent": _BROWSER_UA, "Accept": "application/json,text/plain,*/*"}
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        code = getattr(e, "code", None)
        if code == 429:
            raise _RetriableError(
                f"{service_hint}：请求过于频繁（HTTP 429），请稍后再试。"
            ) from e
        if code in (502, 503, 504):
            raise _RetriableError(
                f"{service_hint}：服务暂时不可用（HTTP {code}），请稍后重试。"
            ) from e
        raise ValueError(
            f"{service_hint}：HTTP {code} {getattr(e, 'reason', '') or ''}".strip()
        ) from e
    except urllib.error.URLError as e:
        msg = (str(getattr(e, "reason", "") or e)).lower()
        if "timed out" in msg or "timeout" in msg:
            raise _RetriableError(f"{service_hint}：请求超时，请检查网络或稍后再试。") from e
        if "name or service not known" in msg or "getaddrinfo failed" in msg:
            raise ValueError(f"{service_hint}：DNS 解析失败，请检查网络与 DNS 设置。") from e
        raise ValueError(f"{service_hint}：网络不可用（{e}）。") from e
    except TimeoutError as e:
        raise _RetriableError(f"{service_hint}：请求超时，请检查网络或稍后再试。") from e
    except OSError as e:
        raise ValueError(f"{service_hint}：{e}") from e

    try:
        text = raw.decode("utf-8", errors="replace")
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{service_hint}：返回内容不是合法 JSON（{e}）。") from e


def _http_get_json(
    url: str,
    *,
    referer: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    service_hint: str = "外网服务",
    max_retries: int = _DEFAULT_RETRIES,
) -> dict:
    """带重试 + 指数退避的 GET JSON；仅对超时 / 429 / 5xx 等可重试错误重试。

    单次失败成本：超时（``timeout`` s） + 退避（``_RETRY_BACKOFF_SECONDS * 2^attempt`` s）；
    总失败上限 ≈ ``(max_retries + 1) * timeout + sum(backoff)``。
    """
    last_err: ValueError | None = None
    for attempt in range(max_retries + 1):
        try:
            return _http_get_json_once(
                url, referer=referer, timeout=timeout, service_hint=service_hint
            )
        except _RetriableError as e:
            last_err = e
            if attempt < max_retries:
                # 指数退避：0.6s → 1.2s → 2.4s …
                time.sleep(_RETRY_BACKOFF_SECONDS * (2 ** attempt))
                continue
            raise ValueError(str(e)) from e
        except ValueError:
            # 不可重试错误（DNS / TLS / HTTP 4xx / JSON 解析失败等）
            raise
    # 理论不可达
    raise ValueError(str(last_err) if last_err else f"{service_hint}：未知错误。")


# ---------------------------------------------------------------------------
# Ticker 规范化：把用户的多种写法转成东财 secid
# ---------------------------------------------------------------------------


def _normalize_ticker(ticker: str) -> tuple[str, str, str]:
    """把任意常见 ticker 写法解析为 ``(market, code, secid)``。

    market ∈ ``{"a", "hk", "us"}``，secid 直接拿去给东财 push2 接口。
    若识别失败抛 ``ValueError``。
    """
    s = (ticker or "").strip().upper()
    if not s:
        raise ValueError("ticker 为空")

    # 已是 secid 形式（如 "1.600519" / "116.00700" / "105.NVDA"）
    if "." in s and s.split(".", 1)[0].isdigit():
        prefix, code = s.split(".", 1)
        if prefix in {"0", "1"}:
            return "a", code, f"{prefix}.{code}"
        if prefix == "116":
            return "hk", code.zfill(5), f"116.{code.zfill(5)}"
        if prefix in {"105", "106"}:
            return "us", code, f"{prefix}.{code}"
        raise ValueError(f"未知 secid 前缀：{prefix}")

    # 显式市场前缀
    if s.startswith("SH"):
        c = s[2:].lstrip("_").zfill(6)
        return "a", c, f"1.{c}"
    if s.startswith("SZ") or s.startswith("BJ"):
        c = s[2:].lstrip("_").zfill(6)
        return "a", c, f"0.{c}"
    if s.startswith("HK"):
        c = s[2:].lstrip("_").zfill(5)
        return "hk", c, f"116.{c}"
    if s.startswith(("US", "GB_")):
        c = s.replace("US", "").replace("GB_", "")
        return "us", c, f"105.{c}"

    # 纯字母：默认美股
    if s.isalpha() and 1 <= len(s) <= 5:
        return "us", s, f"105.{s}"

    # 纯数字：A 股 6 位 / 港股 5 位以内
    digits = s.replace("-", "")
    if digits.isdigit():
        if len(digits) == 6:
            head = digits[0]
            if head in {"6", "9"}:        # 沪市主板（含科创板 688）/ B 股
                return "a", digits, f"1.{digits}"
            if head in {"0", "3"}:        # 深市主板（含中小板 002）/ 创业板
                return "a", digits, f"0.{digits}"
            if head in {"4", "8"}:        # 北交所
                return "a", digits, f"0.{digits}"
        if len(digits) <= 5:
            c = digits.zfill(5)
            return "hk", c, f"116.{c}"

    raise ValueError(f"无法识别的 ticker 写法：{ticker!r}")


# ---------------------------------------------------------------------------
# 东财接口请求辅助
# ---------------------------------------------------------------------------

_EM_QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"
_EM_REFERER = "https://quote.eastmoney.com/"


def _em_get_quote_data(secid: str, fields: str) -> dict:
    """请求东财 quote 接口；返回 ``data`` 字段（dict 或空 dict）。"""
    params = urllib.parse.urlencode({"secid": secid, "fltt": "2", "fields": fields})
    url = f"{_EM_QUOTE_URL}?{params}"
    resp = _http_get_json(url, referer=_EM_REFERER, service_hint="东方财富行情")
    return resp.get("data") or {}


def _em_get_with_us_fallback(secid: str, fields: str, market: str) -> dict:
    """美股先试 NASDAQ（105.），失败则回退 NYSE（106.）。"""
    data = _em_get_quote_data(secid, fields)
    if data and data.get("f57") is not None:
        return data
    if market == "us" and secid.startswith("105."):
        alt = "106." + secid.split(".", 1)[1]
        data = _em_get_quote_data(alt, fields)
    return data or {}


def _format_ts(unix_seconds: int | float | None) -> str:
    if not unix_seconds:
        return "-"
    try:
        # 东财时间戳通常按当地时区给出；为简化起见统一打 UTC，避免歧义
        dt = datetime.fromtimestamp(int(unix_seconds), tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError, OSError):
        return "-"


def _format_market_cap(value: float | None) -> str:
    """市值人性化展示：> 万亿 / 亿 / 万 / 元。"""
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    if v >= 1e12:
        return f"{v / 1e12:.2f} 万亿"
    if v >= 1e8:
        return f"{v / 1e8:.2f} 亿"
    if v >= 1e4:
        return f"{v / 1e4:.2f} 万"
    return f"{v:.0f}"


def _market_currency_label(market: str) -> str:
    return {"a": "CNY", "hk": "HKD", "us": "USD"}.get(market, "")


# ---------------------------------------------------------------------------
# Tool: 行情快照
# ---------------------------------------------------------------------------


@tool(
    description=(
        "查询单只股票的**实时行情快照**：最新价、涨跌额、涨跌幅、今开、昨收、最高、最低、"
        "成交量、成交额、数据时点。"
        "支持 A 股 / 港股 / 美股；ticker 可写 600519、sh600519、SH600519、000001、"
        "00700、HK00700、NVDA、AAPL、105.NVDA 等多种形式。"
        "**何时调用**：用户问「现在多少钱」「涨了多少」「最新价」「行情怎么样」「最新成交」等。"
        "**何时不要调用**：本工具不返回 PE / PB / 市值 / 财务数据（请改用 get_stock_basics）；"
        "不返回历史行情、K 线、技术指标。"
        "数据来源：东方财富 push2 接口（公开免 Key）；A 股币种 CNY、港股 HKD、美股 USD。"
        "返回值已格式化为可读字符串，可直接用于回答。"
    )
)
def get_stock_quote(ticker: str) -> str:
    try:
        market, code, secid = _normalize_ticker(ticker)
    except ValueError as e:
        return f"ticker 解析失败：{e}"

    fields = "f43,f44,f45,f46,f47,f48,f57,f58,f60,f86,f152,f169,f170"
    try:
        data = _em_get_with_us_fallback(secid, fields, market)
    except ValueError as e:
        return str(e)

    if not data or data.get("f57") is None:
        return f"未找到 ticker={ticker}（secid={secid}）的行情数据，请确认代码或更换写法重试。"

    name = data.get("f58") or "?"
    code_ret = data.get("f57") or code
    price = data.get("f43")
    chg_amt = data.get("f169")
    chg_pct = data.get("f170")
    open_p = data.get("f46")
    prev_close = data.get("f60")
    high = data.get("f44")
    low = data.get("f45")
    volume = data.get("f47")
    turnover = data.get("f48")
    ts = data.get("f86")
    ccy = _market_currency_label(market)

    sign = "+" if (isinstance(chg_amt, (int, float)) and chg_amt > 0) else ""
    direction = "📈" if (isinstance(chg_amt, (int, float)) and chg_amt > 0) else (
        "📉" if (isinstance(chg_amt, (int, float)) and chg_amt < 0) else "➖"
    )

    lines = [
        f"{direction} {name}（{code_ret}） · {ccy}",
        f"  最新价: {price}    涨跌: {sign}{chg_amt}（{sign}{chg_pct}%）",
        f"  今开: {open_p}    昨收: {prev_close}    最高: {high}    最低: {low}",
    ]
    if volume is not None or turnover is not None:
        vol_str = f"{volume:,}" if isinstance(volume, (int, float)) else "-"
        to_str = (
            f"{turnover:,.0f} {ccy}"
            if isinstance(turnover, (int, float))
            else "-"
        )
        lines.append(f"  成交量: {vol_str} 手    成交额: {to_str}")
    lines.append(f"  数据时点: {_format_ts(ts)}")
    lines.append("  来源: 东方财富（push2.eastmoney.com）")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: 基本面快照
# ---------------------------------------------------------------------------


@tool(
    description=(
        "查询单只股票的**基本面快照**：总市值、流通市值、PE-TTM、PE 静态（LYR）、"
        "PB（市净率）、换手率，以及当前价位的快照。"
        "支持 A 股 / 港股 / 美股；ticker 形式同 get_stock_quote。"
        "**何时调用**：用户问「市值多少」「PE 多少」「估值高不高」「PB 多少」「换手率」「股息率」「基本面」等。"
        "**何时不要调用**：仅需当下成交价/涨跌请用 get_stock_quote；要财报具体数据请走 RAG 或 web_search。"
        "数据来源：东方财富 push2 接口（公开免 Key）；A 股估值口径以东财统一计算为准。"
        "注意：港股 / 美股的市值与 PE 由东财以本地货币（HKD / USD）计算。"
    )
)
def get_stock_basics(ticker: str) -> str:
    try:
        market, code, secid = _normalize_ticker(ticker)
    except ValueError as e:
        return f"ticker 解析失败：{e}"

    fields = "f43,f57,f58,f86,f116,f117,f152,f162,f163,f167,f168,f170"
    try:
        data = _em_get_with_us_fallback(secid, fields, market)
    except ValueError as e:
        return str(e)

    if not data or data.get("f57") is None:
        return f"未找到 ticker={ticker}（secid={secid}）的基本面数据，请确认代码或更换写法重试。"

    name = data.get("f58") or "?"
    code_ret = data.get("f57") or code
    price = data.get("f43")
    chg_pct = data.get("f170")
    total_cap = data.get("f117")
    free_cap = data.get("f116")
    pe_lyr = data.get("f162")
    pe_ttm = data.get("f163")
    pb = data.get("f167")
    turnover_rate = data.get("f168")
    ts = data.get("f86")
    ccy = _market_currency_label(market)

    sign = "+" if (isinstance(chg_pct, (int, float)) and chg_pct > 0) else ""

    def _fmt_ratio(x: object) -> str:
        if x is None or x == "-":
            return "-"
        try:
            v = float(x)
        except (TypeError, ValueError):
            return "-"
        # 东财对部分市场（如美股）的某些字段会用 0 / 极小值占位；视为无意义口径展示为 "-"
        if abs(v) < 1e-6:
            return "-"
        return f"{v:.2f}"

    def _fmt_pct(x: object) -> str:
        if x is None or x == "-":
            return "-"
        try:
            v = float(x)
        except (TypeError, ValueError):
            return "-"
        if abs(v) < 1e-6:
            return "-"
        return f"{v:.2f}%"

    lines = [
        f"📊 {name}（{code_ret}） · {ccy}",
        f"  最新价: {price}    当日涨跌幅: {sign}{chg_pct}%",
        f"  总市值: {_format_market_cap(total_cap)} {ccy}    流通市值: {_format_market_cap(free_cap)} {ccy}",
        f"  PE-TTM: {_fmt_ratio(pe_ttm)}    PE-LYR (静态): {_fmt_ratio(pe_lyr)}    PB: {_fmt_ratio(pb)}",
        f"  换手率: {_fmt_pct(turnover_rate)}    数据时点: {_format_ts(ts)}",
        "  来源: 东方财富（push2.eastmoney.com）",
        "  ⚠️ 估值倍数为东财统一口径；不同数据商口径可能略有差异。",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: 货币换算
# ---------------------------------------------------------------------------

_FX_BASE_URL = "https://open.er-api.com/v6/latest"


@tool(
    description=(
        "在两种货币间换算金额，返回换算结果 + 当前汇率 + 数据更新时间。"
        "**何时调用**：港美股估值人民币换算、海外业绩人民币口径换算、跨货币比较等。"
        "参数说明："
        "  • amount: 待换算的金额（数字，可为整数或小数）；"
        "  • from_ccy: 源货币代码，3 字母 ISO 4217（如 USD / CNY / HKD / EUR / JPY / GBP / KRW）；"
        "  • to_ccy:   目标货币代码，同上。"
        "数据来源：open.er-api.com（基于 ECB 与多家央行公开汇率，免 Key，约 24 小时刷新）。"
        "**注意**：本工具用于宏观换算；做高频交易请使用专业接口（本工具非实时）。"
    )
)
def convert_currency(amount: float, from_ccy: str, to_ccy: str) -> str:
    fc = (from_ccy or "").strip().upper()
    tc = (to_ccy or "").strip().upper()
    if len(fc) != 3 or len(tc) != 3 or not (fc.isalpha() and tc.isalpha()):
        return "币种代码须为 3 字母 ISO 4217 代码（如 USD / CNY / HKD / EUR）。"

    try:
        amt = float(amount)
    except (TypeError, ValueError):
        return f"金额无法解析为数字：{amount!r}"
    if amt < 0:
        return "金额不可为负数。"

    if fc == tc:
        return f"{amt:,.4g} {fc} = {amt:,.4g} {tc}（同一币种，未调用接口）。"

    url = f"{_FX_BASE_URL}/{fc}"
    try:
        resp = _http_get_json(url, service_hint="汇率接口")
    except ValueError as e:
        return str(e)

    if resp.get("result") != "success":
        return f"汇率查询失败：{resp.get('error-type', resp.get('result', 'unknown'))}"

    rates = resp.get("rates") or {}
    if tc not in rates:
        return (
            f"目标币种 {tc} 不在汇率数据集中；可选项以 ISO 4217 标准货币为准。"
        )

    try:
        rate = float(rates[tc])
    except (TypeError, ValueError):
        return f"汇率数据异常：rates[{tc}]={rates[tc]!r}"

    converted = amt * rate
    update_time = resp.get("time_last_update_utc") or "-"
    return (
        f"💱 {amt:,.4g} {fc} ≈ {converted:,.4g} {tc}\n"
        f"  汇率: 1 {fc} = {rate:.6g} {tc}\n"
        f"  更新时间（UTC）: {update_time}\n"
        f"  来源: open.er-api.com（基于 ECB 等公开汇率，约 24h 刷新）"
    )


# ---------------------------------------------------------------------------
# 对外导出
# ---------------------------------------------------------------------------

FINANCE_TOOLS = [
    get_stock_quote,
    get_stock_basics,
    convert_currency,
]


__all__ = [
    "FINANCE_TOOLS",
    "get_stock_quote",
    "get_stock_basics",
    "convert_currency",
]
