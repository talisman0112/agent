"""一键重建本地 RAG 索引。

清理 ``db/chroma/``、``db/parent_store.sqlite``、``md5.txt``，然后重新扫描
``data/`` 下所有允许后缀的文件并重新向量化入库。

适用场景：
- ``data/`` 目录结构发生重大变化（重命名、删除、批量增减）；
- 切换 embedding 模型、修改 chunk 配置后需要重建；
- 切换业务主题（例如本项目从通用百科切换到 FinSight 投研语料）。

使用：
    python scripts/rebuild_index.py            # 询问后执行
    python scripts/rebuild_index.py --yes      # 跳过确认
    python scripts/rebuild_index.py --dry-run  # 仅打印将要清理 / 入库的内容

环境前置：``DASHSCOPE_API_KEY`` 已配置（`model/model.py` 依赖）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# 确保以仓库根目录为 sys.path 起点，使脚本既能从根目录运行也能从 scripts/ 子目录运行。
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config_hander import chroma_config  # noqa: E402
from utils.path_pool import get_abs_path  # noqa: E402


def _human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _list_data_corpus() -> list[Path]:
    """列出将被实际入库的语料文件（与 ``vector_store.load_data`` 的过滤口径保持一致）。"""
    # 在函数内部 import，避免脚本启动期就触发 Chroma 客户端实例化。
    from rag.vector_store import _is_non_corpus_file  # noqa: WPS433

    data_dir = Path(get_abs_path(chroma_config["database_path"]))
    if not data_dir.exists():
        return []
    allowed = tuple(s.lower() for s in chroma_config["allow_knowledge_file_types"])
    files: list[Path] = []
    for p in data_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in allowed:
            continue
        if _is_non_corpus_file(str(p)):
            continue
        files.append(p)
    return sorted(files)


def _confirm(prompt: str) -> bool:
    try:
        ans = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans in {"y", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser(description="一键重建 FinSight 本地 RAG 索引")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过交互确认")
    parser.add_argument("--dry-run", action="store_true", help="仅打印计划，不执行")
    args = parser.parse_args()

    chroma_dir = Path(get_abs_path(chroma_config["persist_directory"]))
    parent_store_rel = chroma_config.get("parent_store_sqlite", "db/parent_store.sqlite")
    parent_store = Path(get_abs_path(parent_store_rel))
    md5_path = Path(get_abs_path(chroma_config["md5_path"]))
    data_dir = Path(get_abs_path(chroma_config["database_path"]))

    print("=" * 60)
    print("FinSight · 本地 RAG 索引重建计划")
    print("=" * 60)
    print(f"项目根     : {_PROJECT_ROOT}")
    print(f"语料目录    : {data_dir}")
    print(f"Chroma 持久化: {chroma_dir}  ({_human_size(_dir_size(chroma_dir))})")
    print(f"父块 SQLite : {parent_store}  ({_human_size(_dir_size(parent_store))})")
    print(f"MD5 ledger : {md5_path}  ({_human_size(_dir_size(md5_path))})")
    print()

    corpus = _list_data_corpus()
    print(f"将被入库的语料文件：{len(corpus)} 个")
    for p in corpus:
        rel = p.relative_to(_PROJECT_ROOT)
        print(f"  - {rel}  ({_human_size(p.stat().st_size)})")
    if not corpus:
        print("  （无文件，请先把语料放入 data/ 各子目录后重试）")
    print()

    if args.dry_run:
        print("[DRY-RUN] 已打印计划，未执行任何写操作。")
        return 0

    if not args.yes and not _confirm("确认重建索引？这将清空 chroma / parent_store / md5"):
        print("已取消。")
        return 0

    print("\n[1/4] 清空 Chroma 持久化目录 ...")
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        print(f"  - removed: {chroma_dir}")
    else:
        print("  - 不存在，跳过。")

    print("[2/4] 删除父块 SQLite ...")
    if parent_store.exists():
        parent_store.unlink()
        print(f"  - removed: {parent_store}")
    else:
        print("  - 不存在，跳过。")

    print("[3/4] 清空 MD5 ledger ...")
    if md5_path.exists():
        md5_path.unlink()
        print(f"  - removed: {md5_path}")
    else:
        print("  - 不存在，跳过。")

    print("[4/4] 重新入库 ...")
    # 延迟到此处再 import，避免清理阶段就触发 Chroma 客户端持有旧文件句柄。
    from rag.vector_store import VectorStoreService  # noqa: E402

    service = VectorStoreService()
    service.load_data()
    print("\n[OK] 索引重建完成。")
    print(f"     Chroma 大小      : {_human_size(_dir_size(chroma_dir))}")
    print(f"     父块 SQLite 大小 : {_human_size(_dir_size(parent_store))}")
    print(f"     MD5 ledger 大小  : {_human_size(_dir_size(md5_path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
