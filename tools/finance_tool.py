"""FinSight 投研助理 · 金融数据工具集（行情 / 基本面 / K 线 / 汇率）。

工具一览
========
- :func:`get_stock_quote`     —— 行情快照（最新价 / 涨跌 / 成交 / 时间）
- :func:`get_stock_basics`    —— 基本面快照（市值 / PE-TTM / PB / 换手率 / 行业）
- :func:`get_stock_kline`     —— 近若干交易日日线 OHLC（区间涨跌摘要 + 最近几日明细）
- :func:`convert_currency`    —— 货币换算（基于 ECB 等公开汇率，免 Key）

数据源
======
- 行情 / 基本面：东方财富 push2 接口 ``push2.eastmoney.com/api/qt/stock/get``
  - secid 编码：A 股沪市 ``1.<6 位代码>``、A 股深市 / 北交所 ``0.<6 位代码>``、
    港股 ``116.<5 位代码>``、美股 ``105.<symbol>``（NASDAQ）/ ``106.<symbol>``（NYSE）、
    港股指数示例 ``124.HSTECH``（恒生科技）。
  - ``fltt=2`` 让接口直接返回浮点数，省去乘除 scale。
- K 线：``push2his.eastmoney.com/api/qt/stock/kline/get``（日线 ``klt=101``，前复权 ``fqt=1``）；失败时可自动尝试 ``82``/``72`` 镜像主机。
- 汇率：``open.er-api.com/v6/latest/<base>``（基于 ECB 等多源公开汇率，免 Key）。
- 可调超时（秒）：环境变量 ``EASTMONEY_HTTP_TIMEOUT``（行情）、``EASTMONEY_KLINE_HTTP_TIMEOUT``（K 线，优先）；``EASTMONEY_HTTP_MAX_RETRIES``（最大重试次数，0～10）；单次超时 clamp 至 5～60s。

特点
====
1. 所有工具**免 API Key**，适合简历项目 / 公开 demo；
2. 同一 ticker 接受多种写法（``600519`` / ``sh600519``、``00700`` / ``hk00700`` / ``00700.hk``、
   ``NVDA``、``HSTECH`` / ``恒生科技`` / ``124.HSTECH``），由 ``_normalize_ticker`` 统一解析；
3. 美股若 NASDAQ 拿不到结果会自动回退到 NYSE，避免用户手动指定交易所。
4. ``get_stock_quote``：**东财优先**；失败时自动依次尝试 **腾讯财经**、**Yahoo Finance** 公开接口。
"""

from __future__ import annotations

import errno
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

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
_DEFAULT_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 0.6

_EM_TIMEOUT_MIN = 5.0
_EM_TIMEOUT_MAX = 60.0


def _parse_optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def _parse_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(str(raw).strip(), 10)
    except ValueError:
        return None


def _clamp_em_timeout(value: float) -> float:
    return max(_EM_TIMEOUT_MIN, min(_EM_TIMEOUT_MAX, value))


_em_http_timeout_env = _parse_optional_float("EASTMONEY_HTTP_TIMEOUT")
_em_kline_timeout_env = _parse_optional_float("EASTMONEY_KLINE_HTTP_TIMEOUT")
_EM_QUOTE_HTTP_TIMEOUT = _clamp_em_timeout(
    _em_http_timeout_env if _em_http_timeout_env is not None else 18.0
)
if _em_kline_timeout_env is not None:
    _EM_KLINE_HTTP_TIMEOUT = _clamp_em_timeout(_em_kline_timeout_env)
elif _em_http_timeout_env is not None:
    _EM_KLINE_HTTP_TIMEOUT = _clamp_em_timeout(_em_http_timeout_env)
else:
    _EM_KLINE_HTTP_TIMEOUT = _clamp_em_timeout(28.0)

_em_retries_env = _parse_optional_int("EASTMONEY_HTTP_MAX_RETRIES")
if _em_retries_env is not None:
    _EM_HTTP_MAX_RETRIES = max(0, min(10, _em_retries_env))
else:
    _EM_HTTP_MAX_RETRIES = _DEFAULT_RETRIES

_TIMEOUT_OS_ERRNOS: set[int] = {errno.ETIMEDOUT}
if sys.platform == "win32":
    _TIMEOUT_OS_ERRNOS.add(10060)  # WSAETIMEDOUT


class _RetriableError(ValueError):
    """对超时 / 5xx / 连接中断等可重试错误的标记；外层按指数退避多次重试。"""


def _fetch_http_body_once(
    url: str,
    *,
    referer: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    service_hint: str = "外网服务",
) -> bytes:
    """单次 GET 原始响应体；异常语义与 JSON 版一致。"""
    headers = {"User-Agent": _BROWSER_UA, "Accept": "application/json,text/plain,*/*"}
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
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
        if "remote end closed connection" in msg or "connection reset" in msg:
            raise _RetriableError(
                f"{service_hint}：连接被对端关闭，请稍后重试。"
            ) from e
        raise ValueError(f"{service_hint}：网络不可用（{e}）。") from e
    except (http.client.RemoteDisconnected, http.client.IncompleteRead) as e:
        raise _RetriableError(
            f"{service_hint}：连接中断（{type(e).__name__}），请稍后重试。"
        ) from e
    except TimeoutError as e:
        raise _RetriableError(f"{service_hint}：请求超时，请检查网络或稍后再试。") from e
    except OSError as e:
        winerr = getattr(e, "winerror", None)
        os_err = getattr(e, "errno", None)
        code_n = winerr if winerr is not None else os_err
        msg_l = str(e).lower()
        if code_n in _TIMEOUT_OS_ERRNOS or "timed out" in msg_l or "timeout" in msg_l:
            raise _RetriableError(
                f"{service_hint}：请求超时，请检查网络或稍后再试。"
            ) from e
        raise ValueError(f"{service_hint}：{e}") from e


def _http_get_json_once(
    url: str,
    *,
    referer: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    service_hint: str = "外网服务",
) -> dict:
    """单次 GET JSON；失败抛 ``ValueError``（其中可重试错误抛 ``_RetriableError``）。"""
    raw = _fetch_http_body_once(
        url, referer=referer, timeout=timeout, service_hint=service_hint
    )
    try:
        text = raw.decode("utf-8", errors="replace")
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{service_hint}：返回内容不是合法 JSON（{e}）。") from e


def _http_get_text(
    url: str,
    *,
    referer: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    service_hint: str = "外网服务",
    encoding: str = "utf-8",
    max_retries: int = _DEFAULT_RETRIES,
) -> str:
    """带重试的 GET，按 ``encoding`` 解码文本（用于腾讯 GBK 接口等）。"""
    last_err: ValueError | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = _fetch_http_body_once(
                url, referer=referer, timeout=timeout, service_hint=service_hint
            )
            return raw.decode(encoding, errors="replace")
        except _RetriableError as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(_RETRY_BACKOFF_SECONDS * (2 ** attempt))
                continue
            raise ValueError(str(e)) from e
        except ValueError:
            raise
    raise ValueError(str(last_err) if last_err else f"{service_hint}：未知错误。")


def _http_get_json(
    url: str,
    *,
    referer: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    service_hint: str = "外网服务",
    max_retries: int = _DEFAULT_RETRIES,
) -> dict:
    """带重试 + 指数退避的 GET JSON；对超时 / 429 / 5xx / 连接中断等可重试错误重试。

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

# 中文 / 拉丁别名 → （market, 完整 secid）。港股指数等与港股同属 HKD 口径。
_TICKER_ALIAS_SECID: dict[str, tuple[str, str]] = {
    "恒生科技": ("hk", "124.HSTECH"),
    "恒生科技指数": ("hk", "124.HSTECH"),
    "恒指科技": ("hk", "124.HSTECH"),
    "HSTECH": ("hk", "124.HSTECH"),
    "HSITECH": ("hk", "124.HSTECH"),
}


def _squash_ws_upper_ascii(ticker: str) -> str:
    """去掉空白后整体 upper（中文不受影响），用于匹配拉丁别名。"""
    return "".join((ticker or "").split()).upper()


def _squash_ws_keep_case(ticker: str) -> str:
    """去掉所有空白，保留中英文大小写（中文不受影响）。"""
    return "".join((ticker or "").split())


def _try_resolve_alias(ticker: str) -> tuple[str, str, str] | None:
    squashed_cn = _squash_ws_keep_case(ticker)
    hit = _TICKER_ALIAS_SECID.get(squashed_cn)
    if hit:
        market, secid = hit
        code = secid.split(".", 1)[1]
        return market, code, secid
    squashed_lat = _squash_ws_upper_ascii(ticker)
    hit = _TICKER_ALIAS_SECID.get(squashed_lat)
    if hit:
        market, secid = hit
        code = secid.split(".", 1)[1]
        return market, code, secid
    return None


_RE_HK_SUFFIX = re.compile(r"^(\d{1,5})\.HK$", re.I)


def _normalize_ticker(ticker: str) -> tuple[str, str, str]:
    """把任意常见 ticker 写法解析为 ``(market, code, secid)``。

    market ∈ ``{"a", "hk", "us"}``，secid 直接拿去给东财 push2 / kline 接口。
    若识别失败抛 ``ValueError``。
    """
    raw = (ticker or "").strip()
    if not raw:
        raise ValueError("ticker 为空")

    alias_hit = _try_resolve_alias(raw)
    if alias_hit:
        return alias_hit

    s = raw.upper()

    # "00700.HK"：解析为港股代码（勿把前缀误判成 secid）
    m_hk = _RE_HK_SUFFIX.match(s)
    if m_hk:
        digits = m_hk.group(1)
        c = digits.zfill(5)
        return "hk", c, f"116.{c}"

    # 已是 secid 形式（如 "1.600519" / "116.00700" / "105.NVDA" / "124.HSTECH"）
    if "." in s and s.split(".", 1)[0].isdigit():
        prefix, code = s.split(".", 1)
        if prefix in {"0", "1"}:
            return "a", code, f"{prefix}.{code}"
        if prefix == "116":
            return "hk", code.zfill(5), f"116.{code.zfill(5)}"
        if prefix == "124":
            return "hk", code, f"124.{code}"
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
        c = s.replace("US", "").replace("GB_", "").strip()
        if not c:
            raise ValueError(f"无法识别的 ticker 写法：{ticker!r}")
        return "us", c, f"105.{c}"

    # 纯字母：默认美股（部分标的 ticker 长度可达 6～10）
    if s.isalpha() and 1 <= len(s) <= 10:
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
_EM_KLINE_PATH = "/api/qt/stock/kline/get"
_EM_KLINE_HOSTS = (
    "push2his.eastmoney.com",
    "82.push2his.eastmoney.com",
    "72.push2his.eastmoney.com",
)


def _em_get_kline_rows(secid: str, *, limit: int, klt: int = 101) -> list[str]:
    """拉取日线 K 线字符串列表（东财 ``klines``）；多域名依次回退。"""
    params = urllib.parse.urlencode(
        {
            "secid": secid,
            "klt": str(klt),
            "fqt": "1",
            "lmt": str(limit),
            "end": "20500101",
            "fields1": "f1,f2,f3,f4,f5,f6,f7,f8",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        }
    )
    last_err: ValueError | None = None
    for host in _EM_KLINE_HOSTS:
        url = f"https://{host}{_EM_KLINE_PATH}?{params}"
        try:
            resp = _http_get_json(
                url,
                referer=_EM_REFERER,
                service_hint="东方财富K线",
                timeout=_EM_KLINE_HTTP_TIMEOUT,
                max_retries=_EM_HTTP_MAX_RETRIES,
            )
        except ValueError as e:
            last_err = e
            continue
        data = resp.get("data") or {}
        raw = data.get("klines")
        if isinstance(raw, list):
            return [str(x) for x in raw]
        return []
    if last_err:
        raise last_err
    raise ValueError("东方财富K线：全部镜像域名均无响应。")


def _em_get_klines_with_us_fallback(secid: str, limit: int, market: str) -> list[str]:
    rows = _em_get_kline_rows(secid, limit=limit)
    if rows:
        return rows
    if market == "us" and secid.startswith("105."):
        alt = "106." + secid.split(".", 1)[1]
        return _em_get_kline_rows(alt, limit=limit)
    return []


def _parse_kline_ohlc(line: str) -> tuple[str, float, float, float, float, float | None] | None:
    """解析单行 kline：日期,开,收,高,低,量,...（东财 fields2 顺序）。"""
    parts = line.split(",")
    if len(parts) < 6:
        return None
    try:
        dt_s = parts[0].strip()
        o_p = float(parts[1])
        c_p = float(parts[2])
        h_p = float(parts[3])
        l_p = float(parts[4])
    except (TypeError, ValueError):
        return None
    vol: float | None
    try:
        vol = float(parts[5])
    except (TypeError, ValueError):
        vol = None
    return dt_s, o_p, c_p, h_p, l_p, vol


def _em_get_quote_data(secid: str, fields: str) -> dict:
    """请求东财 quote 接口；返回 ``data`` 字段（dict 或空 dict）。"""
    params = urllib.parse.urlencode({"secid": secid, "fltt": "2", "fields": fields})
    url = f"{_EM_QUOTE_URL}?{params}"
    resp = _http_get_json(
        url,
        referer=_EM_REFERER,
        service_hint="东方财富行情",
        timeout=_EM_QUOTE_HTTP_TIMEOUT,
        max_retries=_EM_HTTP_MAX_RETRIES,
    )
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


def _gtimg_query_symbol(market: str, code: str, secid: str) -> str | None:
    """腾讯 qt.gtimg.cn 的 ``q=`` 参数（A sh/sz、港股 hk、美股 us）。"""
    if market == "a":
        c = code.zfill(6)
        return f"sh{c}" if secid.startswith("1.") else f"sz{c}"
    if market == "hk":
        if secid.startswith("124."):
            return f"hk{code.upper()}"
        return f"hk{code.zfill(5)}"
    if market == "us":
        return f"us{code.upper()}"
    return None


def _parse_gtimg_inner(inner: str) -> dict[str, Any] | None:
    """解析腾讯 ``v_*=\"...\"`` 内部字段（字段位序依 qq 行情接口惯例）。"""
    parts = inner.split("~")
    if len(parts) < 36:
        return None
    try:
        name = (parts[1] or "").strip() or "?"
        code_ret = (parts[2] or "").strip()
        price = float(parts[3])
        prev_close = float(parts[4])
        open_p = float(parts[5])
    except (ValueError, IndexError):
        return None
    high = low = None
    try:
        high = float(parts[33])
        low = float(parts[34])
    except (ValueError, IndexError):
        pass
    chg_amt = chg_pct = None
    try:
        chg_amt = float(parts[31])
        chg_pct = float(parts[32])
    except (ValueError, IndexError):
        pass
    volume: float | None = None
    turnover: float | None = None
    slash = parts[35]
    if "/" in slash:
        bits = slash.split("/")
        try:
            volume = float(bits[1])
        except (ValueError, IndexError):
            pass
        try:
            turnover = float(bits[2])
        except (ValueError, IndexError):
            pass
    else:
        try:
            volume = float(parts[6])
        except (ValueError, IndexError):
            pass
        if len(parts) > 37:
            try:
                turnover = float(parts[37])
            except (ValueError, IndexError):
                pass
    ts_raw = (parts[30] or "").strip()
    return {
        "name": name,
        "code_ret": code_ret,
        "price": price,
        "prev_close": prev_close,
        "open_p": open_p,
        "high": high,
        "low": low,
        "chg_amt": chg_amt,
        "chg_pct": chg_pct,
        "volume": volume,
        "turnover": turnover,
        "ts_raw": ts_raw,
        "source_detail": "腾讯财经（qt.gtimg.cn）",
    }


def _fallback_quote_tencent(market: str, code: str, secid: str) -> dict[str, Any] | None:
    qsym = _gtimg_query_symbol(market, code, secid)
    if not qsym:
        return None
    url = f"https://qt.gtimg.cn/q={urllib.parse.quote(qsym)}"
    try:
        text = _http_get_text(
            url,
            referer="https://finance.qq.com/",
            timeout=_EM_QUOTE_HTTP_TIMEOUT,
            service_hint="腾讯财经行情",
            encoding="gb18030",
            max_retries=_EM_HTTP_MAX_RETRIES,
        )
    except ValueError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("v_"):
            continue
        m = re.search(r'="([^"]*)"', line)
        if not m:
            continue
        snap = _parse_gtimg_inner(m.group(1))
        if snap:
            return snap
    return None


def _yahoo_chart_symbol(market: str, code: str, secid: str) -> str | None:
    """Yahoo chart API 标的代码。"""
    if market == "us":
        return code.upper()
    if market == "hk":
        if secid.startswith("124."):
            return f"{code.upper()}.HK"
        if code.isdigit():
            return f"{int(code, 10):04d}.HK"
        return f"{code.upper()}.HK"
    if market == "a":
        c6 = code.zfill(6)
        return f"{c6}.SS" if secid.startswith("1.") else f"{c6}.SZ"
    return None


def _fallback_quote_yahoo(market: str, code: str, secid: str) -> dict[str, Any] | None:
    sym = _yahoo_chart_symbol(market, code, secid)
    if not sym:
        return None
    enc = urllib.parse.quote(sym, safe=".")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{enc}?range=1d&interval=1d"
    try:
        resp = _http_get_json(
            url,
            referer="https://finance.yahoo.com/",
            timeout=_EM_QUOTE_HTTP_TIMEOUT,
            service_hint="Yahoo Finance",
            max_retries=_EM_HTTP_MAX_RETRIES,
        )
    except ValueError:
        return None
    chart = resp.get("chart") or {}
    results = chart.get("result")
    if not isinstance(results, list) or not results:
        return None
    meta = results[0].get("meta") or {}
    price = meta.get("regularMarketPrice")
    if price is None:
        return None
    prev = meta.get("previousClose")
    if prev is None:
        prev = meta.get("chartPreviousClose")
    name = meta.get("shortName") or meta.get("longName") or sym
    sym_out = meta.get("symbol") or sym
    chg_amt = None
    chg_pct = None
    try:
        if prev is not None:
            chg_amt = float(price) - float(prev)
            if float(prev) != 0:
                chg_pct = chg_amt / float(prev) * 100
    except (TypeError, ValueError):
        pass
    ts_raw = ""
    cur_millis = meta.get("regularMarketTime")
    if isinstance(cur_millis, (int, float)) and cur_millis > 0:
        ts_raw = datetime.fromtimestamp(int(cur_millis), tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    return {
        "name": name,
        "code_ret": sym_out,
        "price": price,
        "prev_close": prev,
        "open_p": meta.get("regularMarketOpen"),
        "high": meta.get("regularMarketDayHigh"),
        "low": meta.get("regularMarketDayLow"),
        "chg_amt": chg_amt,
        "chg_pct": chg_pct,
        "volume": meta.get("regularMarketVolume"),
        "turnover": None,
        "ts_raw": ts_raw,
        "source_detail": "Yahoo Finance（chart v8）",
    }


def _format_gtimg_quote_time(raw: str) -> str:
    s = (raw or "").strip()
    if len(s) == 14 and s.isdigit():
        return (
            f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}:{s[12:14]} "
            "（行情快照时间，取自腾讯接口）"
        )
    return f"{s}（取自腾讯接口）" if s else "-"


def _lines_from_fallback_snap(
    snap: dict[str, Any], market: str, *, em_warn: str | None
) -> list[str]:
    ccy = _market_currency_label(market)
    name = snap["name"]
    code_ret = snap["code_ret"]
    price = snap["price"]
    chg_amt = snap.get("chg_amt")
    chg_pct = snap.get("chg_pct")
    open_p = snap.get("open_p")
    prev_close = snap.get("prev_close")
    high = snap.get("high")
    low = snap.get("low")
    volume = snap.get("volume")
    turnover = snap.get("turnover")
    ts_note = snap.get("ts_raw") or ""
    src = snap.get("source_detail") or "备用数据源"

    sign = "+" if (isinstance(chg_amt, (int, float)) and chg_amt > 0) else ""
    direction = "📈" if (isinstance(chg_amt, (int, float)) and chg_amt > 0) else (
        "📉" if (isinstance(chg_amt, (int, float)) and chg_amt < 0) else "➖"
    )
    lines: list[str] = []
    if em_warn:
        lines.append(f"  ⚠️ 东财不可用：{em_warn}")
    lines.append(f"{direction} {name}（{code_ret}） · {ccy}")
    lines.append(
        f"  最新价: {price}    涨跌: {sign}{chg_amt}（{sign}{chg_pct}%）"
    )
    lines.append(
        f"  今开: {open_p}    昨收: {prev_close}    最高: {high}    最低: {low}"
    )
    vol_suffix = "（A 股单位为手，港股/美股等为交易所口径）" if market == "a" else ""
    if volume is not None or turnover is not None:
        vol_str = f"{volume:,.0f}" if isinstance(volume, (int, float)) else "-"
        to_str = (
            f"{turnover:,.0f} {ccy}"
            if isinstance(turnover, (int, float))
            else "-"
        )
        lines.append(f"  成交量: {vol_str}{vol_suffix}    成交额: {to_str}")
    if "腾讯" in src:
        lines.append(f"  数据时点: {_format_gtimg_quote_time(str(ts_note))}")
    else:
        lines.append(f"  数据时点: {ts_note or '-'}")
    lines.append(f"  来源: 备用行情 · {src}")
    return lines


# ---------------------------------------------------------------------------
# Tool: 行情快照
# ---------------------------------------------------------------------------


@tool(
    description=(
        "查询标的**实时行情快照**：最新价、涨跌额、涨跌幅、今开、昨收、最高、最低、"
        "成交量、成交额、数据时点。"
        "支持 A 股 / 港股 / 美股 / 常见港股指数（如恒生科技 ``HSTECH``、``恒生科技``、``124.HSTECH``）；"
        "ticker 可写 600519、00700、HK00700、00700.hk、NVDA、105.NVDA 等。"
        "**何时调用**：用户问「现在多少钱」「涨了多少」「最新价」「行情怎么样」「最新成交」等。"
        "**何时不要调用**：本工具不返回 PE / PB / 市值 / 财务数据（请改用 get_stock_basics）；"
        "历史走势、近一月涨跌请改用 get_stock_kline。"
        "数据来源：**优先**东方财富 push2；若东财不可用则依次尝试 **腾讯财经 qt.gtimg.cn**、"
        "**Yahoo Finance chart API**（均为公开接口，免 Key；口径与延迟可能与东财略有差异）。"
        "返回值已格式化为可读字符串，可直接用于回答。"
    )
)
def get_stock_quote(ticker: str) -> str:
    try:
        market, code, secid = _normalize_ticker(ticker)
    except ValueError as e:
        return f"ticker 解析失败：{e}"

    fields = "f43,f44,f45,f46,f47,f48,f57,f58,f60,f86,f152,f169,f170"
    data: dict = {}
    em_err: str | None = None
    try:
        data = _em_get_with_us_fallback(secid, fields, market)
    except ValueError as e:
        em_err = str(e)

    if data and data.get("f57") is not None:
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

    fb = _fallback_quote_tencent(market, code, secid)
    if fb is None:
        fb = _fallback_quote_yahoo(market, code, secid)

    if fb is None:
        if em_err:
            return (
                f"{em_err}\n"
                "备用行情（腾讯财经、Yahoo Finance）也无法获取报价，请稍后重试或检查网络。"
            )
        return (
            f"未找到 ticker={ticker}（secid={secid}）的行情数据，请确认代码或更换写法重试。"
        )

    return "\n".join(_lines_from_fallback_snap(fb, market, em_warn=em_err))


@tool(
    description=(
        "查询标的近若干交易日的**日线 K 线（前复权）**：区间内首尾收盘价涨跌、区间高低点、"
        "累计成交量（若接口提供），以及最近 5 个交易日的开/收/高/低。"
        "ticker 规则与 get_stock_quote 一致（含恒生科技 ``HSTECH``、``HSI TECH``、``恒生科技``、"
        "``124.HSTECH``、``00700.hk`` 等）。"
        "**何时调用**：用户问「最近一周/一月走势」「近30个交易日表现」「日线大致涨跌」「K 线概况」等。"
        "**参数**：trading_days 为交易日数量（默认 30，范围约 5～250）。"
        "数据来源：东方财富 push2his（公开免 Key）；休市日不计入交易日，故跨度略短于日历日。"
    )
)
def get_stock_kline(ticker: str, trading_days: int = 30) -> str:
    try:
        td = int(trading_days)
    except (TypeError, ValueError):
        return f"trading_days 无法解析为整数：{trading_days!r}"
    td = max(5, min(td, 250))

    try:
        market, _code, secid = _normalize_ticker(ticker)
    except ValueError as e:
        return f"ticker 解析失败：{e}"

    try:
        raw_rows = _em_get_klines_with_us_fallback(secid, td, market)
    except ValueError as e:
        msg = str(e)
        if "超时" in msg:
            msg += (
                "\n  提示：可提高环境变量 EASTMONEY_HTTP_TIMEOUT / EASTMONEY_KLINE_HTTP_TIMEOUT（秒，"
                f"有效范围 {_EM_TIMEOUT_MIN:g}～{_EM_TIMEOUT_MAX:g}）、或 EASTMONEY_HTTP_MAX_RETRIES（0～10）；"
                "并检查网络/DNS/防火墙。"
            )
        return msg

    if not raw_rows:
        return f"未找到 ticker={ticker}（secid={secid}）的 K 线数据，请更换写法或稍后重试。"

    parsed: list[tuple[str, float, float, float, float, float | None]] = []
    for ln in raw_rows:
        row = _parse_kline_ohlc(ln)
        if row:
            parsed.append(row)

    if len(parsed) < 1:
        return f"K 线原始数据无法解析（secid={secid}），请稍后重试。"

    first_dt, _fo, first_close, _fh, _fl, _fv = parsed[0]
    last_dt, _lo, last_close, _lh, _ll, _lv = parsed[-1]
    highs = [x[3] for x in parsed]
    lows = [x[4] for x in parsed]

    span_pct = "-"
    if len(parsed) >= 2:
        try:
            span_pct = f"{(last_close - first_close) / first_close * 100:.2f}%"
        except ZeroDivisionError:
            span_pct = "-"

    vol_sum: float | None = None
    vol_parts = [x[5] for x in parsed if x[5] is not None]
    if vol_parts:
        vol_sum = sum(vol_parts)

    lines_out = [
        f"📉📈 {ticker}（secid={secid}）近 {len(parsed)} 根日线（请求交易日上限 {td}，前复权）",
        f"  区间（按交易日顺序）：{first_dt} 收盘 {first_close:g} → {last_dt} 收盘 {last_close:g}",
        f"  区间涨跌（首尾收盘）：{span_pct}    区间高 {max(highs):g}    区间低 {min(lows):g}",
    ]
    if vol_sum is not None:
        lines_out.append(f"  区间成交量合计：{vol_sum:,.0f}")
    lines_out.append("  最近 5 个交易日（日期 / 开 / 收 / 高 / 低）：")
    tail = parsed[-5:] if len(parsed) >= 5 else parsed
    for dt_s, o_p, c_p, h_p, l_p, _ in tail:
        lines_out.append(f"    {dt_s}  O:{o_p:g} C:{c_p:g} H:{h_p:g} L:{l_p:g}")
    lines_out.append("  来源: 东方财富 push2his（主域名或镜像域名自动切换）")
    return "\n".join(lines_out)


# ---------------------------------------------------------------------------
# Tool: 基本面快照
# ---------------------------------------------------------------------------


@tool(
    description=(
        "查询单只股票的**基本面快照**：总市值、流通市值、PE-TTM、PE 静态（LYR）、"
        "PB（市净率）、换手率，以及当前价位的快照。"
        "支持 A 股 / 港股 / 美股；指数估值字段可能为空（请结合标的性质解读）；ticker 形式同 get_stock_quote。"
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
    get_stock_kline,
    get_stock_basics,
    convert_currency,
]


__all__ = [
    "FINANCE_TOOLS",
    "get_stock_quote",
    "get_stock_kline",
    "get_stock_basics",
    "convert_currency",
]
