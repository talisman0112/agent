"""
在项目根目录执行（会看到检索片段；若配置了 Chat 模型，还会打印完整回答）:

    cd c:\\Users\\lenovo\\Desktop\\rag\\agent
    python tests/test_ragsummarize_terminal.py

换一个问题（PowerShell）:

    $env:RAG_TEST_QUERY='简述RAG流程'; python tests/test_ragsummarize_terminal.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    query = os.environ.get("RAG_TEST_QUERY", "我是谁")

    print("\n[RAG CLI 测试]")
    print(f"提问: {query!r}\n")

    print("--- 向量检索（Chroma + embedding）---")
    try:
        from rag.vector_store import VectorStoreService
        from rag.ragsummarize import _format_docs

        svc = VectorStoreService()
        retriever = svc.get_retriever()
        docs = retriever.invoke(query)
    except Exception as e:
        print(f"[失败] 检索不可用（路径、embedding 配置、向量库等）: {e}")
        sys.exit(1)

    print(f"召回条数: {len(docs)}")
    for i, doc in enumerate(docs, start=1):
        preview = (doc.page_content or "").replace("\n", " ").strip()
        if len(preview) > 200:
            preview = preview[:200] + "…"
        print(f"  [{i}] {preview}")

    prompt_context = _format_docs(docs)
    print("\n--- 拼进提示词的 context 预览 ---")
    print(prompt_context[:800] + ("…\n" if len(prompt_context) > 800 else "\n"))

    print("--- summarize（通义 Qwen：dashscope + langchain-community 的 ChatTongyi）---")
    try:
        from model.model import chat_model as llm
    except Exception as e:
        print(f"[跳过] 无法加载 model: {e}")
        sys.exit(0)

    if llm is None:
        print(
            "[跳过] chat_model 为 None（未安装或未正确导入 ChatTongyi）。请执行：\n"
            "  pip install dashscope langchain-community\n"
            "然后设置密钥（与向量嵌入相同，任选其一即可）：\n"
            "  $env:DASHSCOPE_API_KEY='你的 DashScope Key'\n"
            "  # 或：$env:TONGYI_API_KEY='同一 Key' （仍通过代码传给 ChatTongyi）\n"
        )
        sys.exit(0)

    try:
        from rag.ragsummarize import RAGSummarize

        rag = RAGSummarize()
        answer = rag.summarize(query)
        print(answer)
    except Exception as e:
        print(f"[失败] summarize: {e}")
        sys.exit(1)

    print("\n[完成]\n")


if __name__ == "__main__":
    main()
