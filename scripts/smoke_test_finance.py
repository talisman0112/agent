"""FinSight · 金融工具冲烟测试。

覆盖：
- get_stock_quote：A 股 / 港股 / 美股 / 指数式短代码 / 错误代码 / NYSE 回退
- get_stock_basics：A / 港 / 美 + 关键字段非空校验
- convert_currency：常见币对、同币种、错误币种

通过率 = 实际命中预期关键词的用例 / 总用例。
失败用例打印原始返回，便于定位接口字段或网络问题。
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.finance_tool import (  # noqa: E402
    convert_currency,
    get_stock_basics,
    get_stock_quote,
)


# 每条用例：(标题, 工具, kwargs, 期望出现的关键词列表（任一命中即通过）)
QUOTE_CASES: list[tuple] = [
    ("A 股沪市 600519",        get_stock_quote, {"ticker": "600519"},      ["茅台"]),
    ("A 股深市 000001",        get_stock_quote, {"ticker": "000001"},      ["平安银行"]),
    ("港股 00700",             get_stock_quote, {"ticker": "00700"},       ["腾讯"]),
    ("港股 HK00700 别名",      get_stock_quote, {"ticker": "HK00700"},     ["腾讯"]),
    ("美股 NVDA NASDAQ",       get_stock_quote, {"ticker": "NVDA"},        ["英伟达", "NVDA"]),
    ("美股 BABA NYSE 回退",    get_stock_quote, {"ticker": "BABA"},        ["阿里"]),
    ("沪市带前缀 sh601318",     get_stock_quote, {"ticker": "sh601318"},    ["平安"]),
    ("非法 ticker abc!@#",     get_stock_quote, {"ticker": "abc!@#"},      ["解析失败", "未找到", "无法识别"]),
]

BASICS_CASES: list[tuple] = [
    ("基本面 600519",          get_stock_basics, {"ticker": "600519"},     ["PE-TTM", "市值"]),
    ("基本面 NVDA",            get_stock_basics, {"ticker": "NVDA"},       ["PE-TTM", "市值"]),
    ("基本面 00700",           get_stock_basics, {"ticker": "00700"},      ["PE-TTM", "市值"]),
]

FX_CASES: list[tuple] = [
    ("USD → CNY",             convert_currency, {"amount": 100, "from_ccy": "USD", "to_ccy": "CNY"}, ["CNY", "汇率"]),
    ("CNY → USD",             convert_currency, {"amount": 1000, "from_ccy": "CNY", "to_ccy": "USD"}, ["USD", "汇率"]),
    ("USD → HKD",             convert_currency, {"amount": 50, "from_ccy": "USD", "to_ccy": "HKD"}, ["HKD", "汇率"]),
    ("USD → USD 同币种",       convert_currency, {"amount": 50, "from_ccy": "USD", "to_ccy": "USD"}, ["同一币种"]),
    ("非法币种 XYZ",           convert_currency, {"amount": 1, "from_ccy": "XYZ", "to_ccy": "CNY"},  ["XYZ", "查询失败", "不在", "ISO"]),
    ("负数金额拒绝",           convert_currency, {"amount": -10, "from_ccy": "USD", "to_ccy": "CNY"}, ["不可为负"]),
]


def _truncate(s: str, n: int = 200) -> str:
    s = (s or "").replace("\n", " | ")
    return s if len(s) <= n else s[:n] + "..."


def run_block(title: str, cases: list[tuple]) -> tuple[int, int]:
    print("\n" + "=" * 70)
    print(f"{title}（共 {len(cases)} 个）")
    print("=" * 70)
    passed = 0
    for label, tool_fn, kwargs, expected_kw in cases:
        try:
            result = tool_fn.invoke(kwargs)
        except Exception as e:
            result = f"[EXCEPTION] {type(e).__name__}: {e}"
        ok = any(kw in result for kw in expected_kw)
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {label}")
        print(f"  期望: {expected_kw}")
        print(f"  返回: {_truncate(result, 240)}")
    return passed, len(cases)


def main() -> int:
    p1, t1 = run_block("get_stock_quote", QUOTE_CASES)
    p2, t2 = run_block("get_stock_basics", BASICS_CASES)
    p3, t3 = run_block("convert_currency", FX_CASES)

    total_passed = p1 + p2 + p3
    total_cases = t1 + t2 + t3

    print("\n" + "=" * 70)
    print(
        f"汇总：行情 {p1}/{t1}    基本面 {p2}/{t2}    汇率 {p3}/{t3}    "
        f"合计 {total_passed}/{total_cases}"
    )
    print("=" * 70)
    return 0 if total_passed == total_cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
