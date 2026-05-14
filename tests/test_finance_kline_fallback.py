"""离线验证 K 线请求在多域名上的回退顺序（不访问外网）。"""

from __future__ import annotations

from unittest.mock import patch

from tools import finance_tool as ft


def test_em_kline_rows_fallback_to_second_host_on_failure() -> None:
    hosts: list[str | None] = []

    def fake_http(url: str, **kwargs: object) -> dict:
        from urllib.parse import urlparse

        hosts.append(urlparse(url).hostname)
        if hosts[-1] == "push2his.eastmoney.com":
            raise ValueError("东方财富K线：请求超时，请检查网络或稍后再试。")
        return {"data": {"klines": ["2026-05-01,10,11,12,9,1000"]}}

    with patch.object(ft, "_http_get_json", side_effect=fake_http):
        rows = ft._em_get_kline_rows("124.HSTECH", limit=5)

    assert rows == ["2026-05-01,10,11,12,9,1000"]
    assert hosts[0] == "push2his.eastmoney.com"
    assert hosts[1] == "82.push2his.eastmoney.com"
