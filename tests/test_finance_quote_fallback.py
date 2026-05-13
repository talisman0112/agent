"""离线校验腾讯 GTIMG 解析与 Yahoo 代码映射（不访问东财）。"""

from __future__ import annotations

from tools import finance_tool as ft


def test_parse_gtimg_inner_maotai_shape() -> None:
    inner = (
        "1~贵州茅台~600519~1361.33~1372.99~1372.89~57135~27713~29422~1361.33~22~"
        "1361.32~284~1361.31~10~1361.30~27~1361.29~4~1361.89~1~1362.10~5~1362.20~1~"
        "1362.66~2~1362.79~2~~20260511161407~-11.66~-0.85~1372.89~1361.00~"
        "1361.33/57135/7790721392~57135~779072~0.46~20.61~~1372.89~1361.00~"
    )
    snap = ft._parse_gtimg_inner(inner)
    assert snap is not None
    assert snap["name"] == "贵州茅台"
    assert snap["code_ret"] == "600519"
    assert snap["price"] == 1361.33
    assert snap["prev_close"] == 1372.99
    assert snap["chg_amt"] == -11.66
    assert snap["chg_pct"] == -0.85
    assert snap["high"] == 1372.89
    assert snap["low"] == 1361.00
    assert snap["volume"] == 57135.0
    assert snap["turnover"] == 7790721392.0


def test_yahoo_chart_symbol_mappings() -> None:
    assert ft._yahoo_chart_symbol("a", "600519", "1.600519") == "600519.SS"
    assert ft._yahoo_chart_symbol("a", "000001", "0.000001") == "000001.SZ"
    assert ft._yahoo_chart_symbol("hk", "00700", "116.00700") == "0700.HK"
    assert ft._yahoo_chart_symbol("hk", "HSTECH", "124.HSTECH") == "HSTECH.HK"
    assert ft._yahoo_chart_symbol("us", "NVDA", "105.NVDA") == "NVDA"


def test_gtimg_query_symbol() -> None:
    assert ft._gtimg_query_symbol("a", "600519", "1.600519") == "sh600519"
    assert ft._gtimg_query_symbol("hk", "00700", "116.00700") == "hk00700"
    assert ft._gtimg_query_symbol("hk", "HSTECH", "124.HSTECH") == "hkHSTECH"


def test_format_gtimg_quote_time_compact() -> None:
    s = ft._format_gtimg_quote_time("20260511161407")
    assert "2026-05-11" in s
    assert "16:14:07" in s
