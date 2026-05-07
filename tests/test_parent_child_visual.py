"""父子文档检索 (Parent-Child) 可视化对比测试。

对比效果：
- 普通检索：返回小块文本（可能截断上下文）
- 父子检索：返回完整父文档（通过子块命中召回完整内容）

运行：
    python tests/test_parent_child_visual.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.vector_store import VectorStoreService
from utils.config_hander import chroma_config


def _preview(text: str, limit: int = 200) -> str:
    """生成文本预览。"""
    one_line = text.replace("\n", " ").strip()
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit] + "..."


def _count_lines(text: str) -> int:
    """统计文本行数。"""
    return text.count("\n") + 1 if text else 0


def _print_doc_table(title: str, docs: list[Document], show_parent_info: bool = True) -> None:
    """打印文档表格。"""
    print(f"\n{title}")
    print("=" * 100)
    print(f"{'Rank':>4} | {'Chars':>6} | {'Lines':>5} | {'Source':<20} | {'Preview (first 200 chars)'}")
    print("-" * 100)

    for idx, doc in enumerate(docs, 1):
        content = doc.page_content or ""
        chars = len(content)
        lines = _count_lines(content)
        source = (doc.metadata.get("source", "") or "unknown")[:18]
        preview = _preview(content, 70)

        extra = ""
        if show_parent_info:
            parent_id = doc.metadata.get("parent_id")
            section = doc.metadata.get("section")
            if parent_id:
                extra = f" [parent_id={parent_id[:8]}...]"
            if section:
                extra += f" [章节:{section[:10]}]"

        print(f"{idx:>4} | {chars:>6} | {lines:>5} | {source:<20} | {preview}{extra}")

    print("=" * 100)
    print(f"Total: {len(docs)} documents, {sum(len(d.page_content or '') for d in docs)} chars")


def main() -> int:
    parser = argparse.ArgumentParser(description="Parent-Child 检索效果对比")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="RAG 系统中如何实现多路召回？",
        help="测试查询语句",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="检索返回数量",
    )
    args = parser.parse_args()

    query = args.query.strip()
    k = args.k

    print("\n" + "=" * 100)
    print("Parent-Child (父子文档) 检索效果对比")
    print("=" * 100)
    print(f"Query: {query}")
    print(f"Config: parent_child_enabled={chroma_config.get('parent_child_enabled', False)}")

    # 初始化服务
    service = VectorStoreService()
    retriever = service.get_retriever()

    # 1. 普通检索（不展开）
    print("\n" + "=" * 100)
    print("【1. 普通检索 - 返回子块 (Child Chunks)】")
    print("=" * 100)

    child_docs = retriever.invoke(query)
    child_docs = child_docs[:k]

    _print_doc_table("子块检索结果", child_docs, show_parent_info=True)

    # 2. 父子文档检索（展开为父块）
    print("\n" + "=" * 100)
    print("【2. 父子文档检索 - 子块命中后展开为父块 (Parent Documents)】")
    print("=" * 100)

    if service.parent_store is None:
        print("⚠️  警告: parent_store 未启用，请在 config/chrome.yml 中设置 parent_child_enabled: true")
        return 1

    parent_docs = service.expand_retrieval_to_parents(child_docs)
    parent_docs = parent_docs[:k]

    _print_doc_table("父块展开结果", parent_docs, show_parent_info=True)

    # 3. 对比分析
    print("\n" + "=" * 100)
    print("【对比分析】")
    print("=" * 100)

    child_chars = sum(len(d.page_content or "") for d in child_docs)
    parent_chars = sum(len(d.page_content or "") for d in parent_docs)

    print(f"普通检索 (子块):  {len(child_docs)} 条, 平均 {child_chars // max(len(child_docs), 1)} 字/条")
    print(f"父子检索 (父块):  {len(parent_docs)} 条, 平均 {parent_chars // max(len(parent_docs), 1)} 字/条")
    print(f"内容膨胀率:       {parent_chars / max(child_chars, 1):.1f}x")

    # 检查去重效果
    if len(child_docs) > len(parent_docs):
        print(f"\n✓ 去重效果: {len(child_docs)} 个子块来自 {len(parent_docs)} 个不同父块")

    # 显示第一个文档的内容对比
    if child_docs and parent_docs:
        print("\n" + "=" * 100)
        print("【Top-1 内容对比】")
        print("=" * 100)

        print(f"\n--- 子块 (Child) #{1} ---")
        print(f"长度: {len(child_docs[0].page_content or '')} 字符")
        print(f"内容: {_preview(child_docs[0].page_content, 300)}")

        print(f"\n--- 父块 (Parent) #{1} ---")
        print(f"长度: {len(parent_docs[0].page_content or '')} 字符")
        print(f"内容: {_preview(parent_docs[0].page_content, 300)}")

        # 检查是否包含更多上下文
        child_text = child_docs[0].page_content or ""
        parent_text = parent_docs[0].page_content or ""
        if len(parent_text) > len(child_text) * 1.5:
            print(f"\n✓ 父块提供了 {(len(parent_text) / len(child_text)):.1f}x 的额外上下文")

    print("\n" + "=" * 100)
    print("结论: 父子文档检索通过小块命中召回完整上下文，更适合长文档问答场景")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
