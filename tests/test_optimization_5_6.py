"""
测试第5、6点优化

在项目根目录执行:
    python tests/test_optimization_5_6.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_chroma_persist_path():
    """测试 Chroma 持久化路径是否为绝对路径"""
    from utils.path_pool import get_abs_path
    from utils.config_hander import chroma_config

    # 检查配置中的路径
    persist_dir = chroma_config.get("persist_directory", "")
    print(f"配置中的 persist_directory: {persist_dir}")

    # 检查转换为绝对路径后的结果
    abs_persist_dir = get_abs_path(persist_dir)
    print(f"转换后的绝对路径: {abs_persist_dir}")

    # 验证是绝对路径
    assert Path(abs_persist_dir).is_absolute(), "❌ persist_directory 不是绝对路径"
    print("✅ persist_directory 已转换为绝对路径")

    # 检查 vector_store.py 中的代码
    vector_store_path = Path(__file__).resolve().parents[1] / "rag" / "vector_store.py"
    source = vector_store_path.read_text(encoding="utf-8")

    assert "get_abs_path(chroma_config[\"persist_directory\"])" in source, \
        "❌ vector_store.py 未使用 get_abs_path 转换 persist_directory"
    print("✅ vector_store.py 正确使用了 get_abs_path 转换路径")

    assert "logger.info(\"Chroma 持久化路径:" in source, \
        "❌ vector_store.py 未添加路径日志"
    print("✅ vector_store.py 已添加持久化路径日志")

    return True


def test_app_error_handling():
    """测试 app.py 错误处理"""
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")

    checks = [
        ("try:", "try 块"),
        ("except Exception as e:", "except 捕获"),
        ("timeout", "超时错误处理"),
        ("401", "401 认证错误处理"),
        ("429", "429 限流错误处理"),
        ("connection", "网络连接错误处理"),
        ("st.error", "st.error 错误展示"),
    ]

    all_passed = True
    for pattern, desc in checks:
        if pattern in source:
            print(f"✅ 包含 {desc}: {pattern}")
        else:
            print(f"❌ 缺少 {desc}: {pattern}")
            all_passed = False

    return all_passed


def test_imports_after_changes():
    """测试修改后的代码可以正常导入"""
    try:
        from rag.vector_store import VectorStoreService
        print("✅ VectorStoreService 导入成功")

        from app import ReactAgent
        print("✅ app.py 导入成功")

        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n[第5、6点优化测试]\n")
    print("=" * 50)

    tests = [
        ("Chroma 持久化路径测试", test_chroma_persist_path),
        ("app.py 错误处理测试", test_app_error_handling),
        ("导入测试", test_imports_after_changes),
    ]

    failed = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            if not test_func():
                failed.append(name)
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)

    print("\n" + "=" * 50)
    if not failed:
        print("🎉 第5、6点优化测试全部通过！")
        print("\n优化内容：")
        print("  第5点 ✅ - Chroma 持久化路径已转换为绝对路径")
        print("  第6点 ✅ - app.py 已添加全面的错误处理")
        return 0
    else:
        print(f"❌ {len(failed)} 个测试失败: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
