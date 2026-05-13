"""检测东方财富 K 线接口（push2his 各镜像）在本机网络下的可用性。

用法（在项目根目录）::

    python scripts/probe_eastmoney_kline.py
    python scripts/probe_eastmoney_kline.py --secid 1.600519 --lmt 5
    python scripts/probe_eastmoney_kline.py --timeout 45 --retries 0

说明：
- 默认对每个域名 **单独** 发起请求（便于看出哪一个镜像可达）。
- ``--retries 0`` 表示每个域名只try一次；与 ``finance_tool`` 内 ``max_retries=3`` 的多轮重试不同。
- 超时、镜像列表与 ``tools/finance_tool.py`` 中逻辑一致；亦受环境变量
  ``EASTMONEY_KLINE_HTTP_TIMEOUT`` / ``EASTMONEY_HTTP_TIMEOUT`` 影响（模块导入时已读取）。
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
import urllib.parse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.finance_tool import (  # noqa: E402
    _EM_HTTP_MAX_RETRIES,
    _EM_KLINE_HOSTS,
    _EM_KLINE_HTTP_TIMEOUT,
    _EM_KLINE_PATH,
    _EM_REFERER,
    _http_get_json,
)


def _build_url(host: str, secid: str, lmt: int) -> str:
    params = urllib.parse.urlencode(
        {
            "secid": secid,
            "klt": str(101),
            "fqt": "1",
            "lmt": str(lmt),
            "end": "20500101",
            "fields1": "f1,f2,f3,f4,f5,f6,f7,f8",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        }
    )
    return f"https://{host}{_EM_KLINE_PATH}?{params}"


def _dns_summary(host: str) -> str:
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        ips = sorted({x[4][0] for x in infos})
        return ", ".join(ips[:4]) + (" …" if len(ips) > 4 else "")
    except OSError as e:
        return f"DNS失败: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="探测东财 K 线接口可用性")
    parser.add_argument("--secid", default="124.HSTECH", help="东财 secid，如 124.HSTECH、1.600519")
    parser.add_argument("--lmt", type=int, default=5, help="请求 K 线条数（宜小以加快探测）")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=f"单次请求超时秒数（默认：与 finance_tool 一致，当前 {_EM_KLINE_HTTP_TIMEOUT:g}s）",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help=f"每个域名的 HTTP 重试次数（0=只请求一次；finance_tool 默认 {_EM_HTTP_MAX_RETRIES}）",
    )
    args = parser.parse_args()

    timeout = float(args.timeout) if args.timeout is not None else float(_EM_KLINE_HTTP_TIMEOUT)
    max_retries = max(0, min(10, int(args.retries)))

    print("=" * 72)
    print("东方财富 K 线接口探测")
    print(f"  secid={args.secid!r}  lmt={args.lmt}  timeout={timeout:g}s  max_retries={max_retries}")
    print(f"  配置参考：EASTMONEY_KLINE_HTTP_TIMEOUT / EASTMONEY_HTTP_MAX_RETRIES")
    print("=" * 72)

    any_ok = False
    for host in _EM_KLINE_HOSTS:
        url = _build_url(host, args.secid, args.lmt)
        print(f"\n[{host}]")
        print(f"  DNS: {_dns_summary(host)}")
        t0 = time.perf_counter()
        try:
            data = _http_get_json(
                url,
                referer=_EM_REFERER,
                service_hint="东方财富K线",
                timeout=timeout,
                max_retries=max_retries,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            payload = data.get("data") or {}
            klines = payload.get("klines")
            n = len(klines) if isinstance(klines, list) else 0
            rc = data.get("rc") if isinstance(data.get("rc"), int) else None
            if n > 0:
                print(f"  状态: OK  耗时: {elapsed_ms:.0f} ms  rc={rc}  klines={n}")
                print(f"  示例末根: {klines[-1]!s}"[:200])
                any_ok = True
            else:
                print(f"  状态: 连通但无K线  耗时: {elapsed_ms:.0f} ms  rc={rc}")
                print(f"  原始 keys: {list(data.keys())}")
        except ValueError as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"  状态: FAIL  耗时: {elapsed_ms:.0f} ms")
            print(f"  错误: {e}")

    print("\n" + "=" * 72)
    if any_ok:
        print("结论: 至少有一个镜像返回了 K 线数据；若 Agent 仍超时，请加大 EASTMONEY_* 超时/重试或检查 DNS。")
        return 0
    print("结论: 全部镜像均未拿到有效 K 线；请检查本机网络/VPN/防火墙或改用境外可用链路。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
