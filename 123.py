"""从 Kaggle 拉取语料到项目 ``data/``，并可选触发向量入库。

默认数据集：``peopletech/peopledaily``（人民日报分类新闻，体积较大）。

用法（项目根目录）::

    python 123.py

仅下载、不入库::

    set SKIP_INGEST=1
    python 123.py

环境变量 ``KAGGLE_DATASET`` 可替换为其它 handle（如 ``username/dataset-name``）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parent
DATA_SUBDIR = ROOT / "data" / "kaggle_peopledaily"


def main() -> None:
    handle = os.environ.get("KAGGLE_DATASET", "peopletech/peopledaily").strip()
    DATA_SUBDIR.mkdir(parents=True, exist_ok=True)

    path = kagglehub.dataset_download(
        handle,
        output_dir=str(DATA_SUBDIR),
        force_download=os.environ.get("KAGGLE_FORCE", "").strip() in ("1", "true", "yes"),
    )
    print("Path to dataset files:", path)

    if os.environ.get("SKIP_INGEST", "").strip().lower() in ("1", "true", "yes"):
        print("SKIP_INGEST set — 跳过向量入库")
        return

    _root = str(ROOT)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from rag.vector_store import VectorStoreService

    print("开始入库（scan data/，增量 MD5 跳过未变更文件）…")
    VectorStoreService().load_data()
    print("入库流程已结束")


if __name__ == "__main__":
    main()
