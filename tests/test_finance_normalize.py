"""离线校验 ``finance_tool._normalize_ticker``（不访问外网）。"""

from __future__ import annotations

import pytest

from tools import finance_tool as ft


@pytest.mark.parametrize(
    ("ticker", "expected_secid"),
    [
        ("HSTECH", "124.HSTECH"),
        ("hstech", "124.HSTECH"),
        ("HSI TECH", "124.HSTECH"),
        ("恒生科技", "124.HSTECH"),
        ("恒 生 科 技", "124.HSTECH"),
        ("124.HSTECH", "124.HSTECH"),
        ("00700.hk", "116.00700"),
        ("00700.HK", "116.00700"),
        ("HK00700", "116.00700"),
        ("NVDA", "105.NVDA"),
        ("GOOG", "105.GOOG"),
        ("600519", "1.600519"),
    ],
)
def test_normalize_ticker_ok(ticker: str, expected_secid: str) -> None:
    _m, _c, secid = ft._normalize_ticker(ticker)
    assert secid == expected_secid


def test_unknown_secid_prefix_rejected() -> None:
    with pytest.raises(ValueError, match="未知 secid 前缀"):
        ft._normalize_ticker("800700.HK")


def test_garbage_ticker_rejected() -> None:
    with pytest.raises(ValueError, match="无法识别"):
        ft._normalize_ticker("abc!@#")
