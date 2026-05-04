"""
测试报告模式切换功能

在项目根目录执行:
    python tests/test_report_mode.py

测试内容:
1. 验证 ReactAgent.execute 的 report_mode 参数正确传递
2. 验证不同的 report_mode 值触发不同的 prompt
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_report_mode_parameter():
    """测试 report_mode 参数传递"""
    from tools.reactagent import ReactAgent
    import inspect

    sig = inspect.signature(ReactAgent.execute)
    params = sig.parameters

    # 检查 report_mode 参数存在
    assert "report_mode" in params, "❌ report_mode 参数不存在"

    # 检查默认值为 False
    param = params["report_mode"]
    assert param.default == False, f"❌ report_mode 默认值应为 False, 实际为 {param.default}"

    print("✅ report_mode 参数存在且默认值为 False")


def test_prompts_loading():
    """测试 prompts 能正确加载"""
    from utils.prompts_hander import get_main_prompt, get_report_prompt

    try:
        main_prompt = get_main_prompt()
        assert main_prompt and isinstance(main_prompt, str), "❌ main prompt 为空或格式错误"
        print(f"✅ main prompt 加载成功 (长度: {len(main_prompt)})")
    except Exception as e:
        print(f"❌ main prompt 加载失败: {e}")
        raise

    try:
        report_prompt = get_report_prompt()
        assert report_prompt and isinstance(report_prompt, str), "❌ report prompt 为空或格式错误"
        print(f"✅ report prompt 加载成功 (长度: {len(report_prompt)})")
    except Exception as e:
        print(f"❌ report prompt 加载失败: {e}")
        raise

    # 验证两个 prompt 内容不同
    assert main_prompt != report_prompt, "❌ main prompt 和 report prompt 内容相同"
    print("✅ main prompt 和 report prompt 内容不同")


def test_stream_context_structure():
    """测试 stream 调用中 context 结构正确"""
    from tools.reactagent import ReactAgent
    import inspect

    # 获取 execute 方法源码中 context 的传递
    source = inspect.getsource(ReactAgent.execute)

    # 验证源码中包含正确的 context 传递
    assert 'context={"report": report_mode}' in source, "❌ execute 方法中没有正确传递 context"
    print("✅ execute 方法正确传递 context={'report': report_mode}")


def main():
    print("\n[报告模式切换功能测试]\n")

    tests = [
        ("参数签名测试", test_report_mode_parameter),
        ("Prompt 加载测试", test_prompts_loading),
        ("Context 结构测试", test_stream_context_structure),
    ]

    failed = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            test_func()
        except AssertionError as e:
            print(f"❌ 失败: {e}")
            failed.append((name, str(e)))
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            failed.append((name, str(e)))

    print("\n" + "=" * 50)
    if not failed:
        print("🎉 所有测试通过！报告模式切换功能工作正常")
        return 0
    else:
        print(f"❌ {len(failed)} 个测试失败:")
        for name, error in failed:
            print(f"   - {name}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
