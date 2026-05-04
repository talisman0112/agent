"""
测试文件上传功能

在项目根目录执行:
    python tests/test_file_upload.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_imports():
    """测试必要的导入是否可用"""
    try:
        from utils.config_hander import chroma_config
        from utils.path_pool import get_abs_path
        from rag.vector_store import VectorStoreService
        print("✅ 所有必要导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_chroma_config():
    """测试 chroma 配置包含必要字段"""
    from utils.config_hander import chroma_config

    required_keys = [
        "database_path",
        "allow_knowledge_file_types",
        "persist_directory",
        "collection_name",
    ]

    missing = [k for k in required_keys if k not in chroma_config]
    if missing:
        print(f"❌ chroma_config 缺少字段: {missing}")
        return False

    print(f"✅ chroma_config 包含所有必要字段")
    print(f"   允许的文件类型: {chroma_config['allow_knowledge_file_types']}")
    print(f"   数据库路径: {chroma_config['database_path']}")
    return True


def test_app_structure():
    """测试 app.py 包含文件上传相关代码"""
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")

    checks = [
        ("st.file_uploader", "文件上传器"),
        ("allow_knowledge_file_types", "允许的文件类型配置"),
        ("VectorStoreService", "向量存储服务导入"),
        ("load_data", "入库方法调用"),
        ("执行入库", "入库按钮"),
        ("正在入库", "入库状态提示"),
        ("入库完成", "成功提示"),
    ]

    all_passed = True
    for pattern, desc in checks:
        if pattern in source:
            print(f"✅ 包含 {desc}: {pattern}")
        else:
            print(f"❌ 缺少 {desc}: {pattern}")
            all_passed = False

    return all_passed


def test_data_directory():
    """测试 data 目录存在且可写"""
    from utils.config_hander import chroma_config
    from utils.path_pool import get_abs_path

    data_dir = Path(get_abs_path(chroma_config["database_path"]))

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        test_file = data_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        print(f"✅ data 目录存在且可写: {data_dir}")
        return True
    except Exception as e:
        print(f"❌ data 目录测试失败: {e}")
        return False


def main():
    print("\n[文件上传功能测试]\n")

    tests = [
        ("导入测试", test_imports),
        ("配置测试", test_chroma_config),
        ("应用结构测试", test_app_structure),
        ("数据目录测试", test_data_directory),
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
        print("🎉 所有测试通过！文件上传功能已就绪")
        return 0
    else:
        print(f"❌ {len(failed)} 个测试失败: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
