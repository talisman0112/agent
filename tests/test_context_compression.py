"""
上下文压缩模块测试套件

运行方式:
    cd tests
    python test_context_compression.py
    pytest test_context_compression.py -v
"""

import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.context_compressor import (
    ContextCompressor,
    CompressionStrategy,
    CompressionStats,
    CompressionResult,
    format_compressed_docs,
)
from langchain_core.documents import Document


class TestContextCompression:
    """上下文压缩单元测试"""

    def test_no_compression_needed(self):
        """测试: Token 未超限时不应压缩"""
        compressor = ContextCompressor()

        docs = [
            Document(page_content="短文档内容", metadata={"rerank_score": 0.9})
        ]

        result = compressor.compress("查询", docs, max_tokens=5000)

        assert result.stats.compression_ratio == 0.0
        assert result.stats.method_used == "none"
        assert len(result.documents) == 1
        assert result.quality_score == 1.0

    def test_extractive_compression(self):
        """测试: 提取式压缩功能"""
        compressor = ContextCompressor()

        # 构造长文档
        long_content = "\n\n".join([
            f"第{i}段内容：这是关于人工智能的讨论，包含机器学习和深度学习的基本概念以及各种应用场景。"
            for i in range(20)
        ])

        docs = [
            Document(page_content=long_content, metadata={"rerank_score": 0.9})
        ]

        result = compressor.compress(
            "人工智能",
            docs,
            max_tokens=800,
            strategy=CompressionStrategy.EXTRACT
        )

        # 验证压缩发生了
        assert result.stats.compression_ratio > 0.2
        assert result.stats.compressed_tokens < result.stats.original_tokens
        assert result.stats.method_used == "extractive"
        assert all(d.metadata.get("compressed") for d in result.documents)
        assert result.quality_score > 0.5

    def test_compression_quality(self):
        """测试: 压缩质量评估"""
        compressor = ContextCompressor()

        docs = [
            Document(
                page_content="Python 是一种高级编程语言，支持面向对象编程。",
                metadata={"rerank_score": 0.9}
            ),
            Document(
                page_content="Java 也是一种编程语言，广泛应用于企业开发。",
                metadata={"rerank_score": 0.8}
            )
        ]

        result = compressor.compress("编程语言", docs, max_tokens=100)

        # 质量分数应在合理范围
        assert 0 <= result.quality_score <= 1.0
        assert result.quality_score > 0.3

    def test_code_preservation(self):
        """测试: 代码块完整性保留"""
        compressor = ContextCompressor()

        code_doc = Document(
            page_content="""
这里是一些说明文字。

```python
def hello():
    print("Hello World")
    return True
```

更多说明文字。
            """.strip(),
            metadata={"rerank_score": 0.9}
        )

        result = compressor.compress(
            "代码示例",
            [code_doc],
            max_tokens=200,
            strategy=CompressionStrategy.EXTRACT
        )

        # 验证代码块被保留
        compressed = result.documents[0].page_content
        assert "def hello" in compressed or "print" in compressed

    def test_auto_strategy_selection_code(self):
        """测试: 自动策略选择（代码查询应选 EXTRACT）"""
        compressor = ContextCompressor()

        docs = [
            Document(
                page_content="Python 代码示例 def hello(): pass",
                metadata={"rerank_score": 0.9}
            )
        ]

        strategy = compressor._select_strategy("Python 函数怎么写？", docs, 1000)
        assert strategy == CompressionStrategy.EXTRACT

    def test_auto_strategy_selection_fact(self):
        """测试: 自动策略选择（事实查询应选 EXTRACT）"""
        compressor = ContextCompressor()

        docs = [
            Document(
                page_content="TCP三次握手包括SYN、SYN-ACK、ACK",
                metadata={"rerank_score": 0.9}
            )
        ]

        strategy = compressor._select_strategy("什么是TCP三次握手？", docs, 1000)
        assert strategy == CompressionStrategy.EXTRACT

    def test_multi_document_compression(self):
        """测试: 多文档压缩"""
        compressor = ContextCompressor()

        docs = [
            Document(
                page_content="文档1：" + "内容" * 200,
                metadata={"rerank_score": 0.95}
            ),
            Document(
                page_content="文档2：" + "内容" * 200,
                metadata={"rerank_score": 0.85}
            ),
            Document(
                page_content="文档3：" + "内容" * 200,
                metadata={"rerank_score": 0.75}
            ),
        ]

        result = compressor.compress("查询", docs, max_tokens=500)

        # 应该保留了部分文档
        assert len(result.documents) <= len(docs)
        assert result.stats.docs_after <= result.stats.docs_before

    def test_format_compressed_docs(self):
        """测试: 格式化输出"""
        docs = [
            Document(
                page_content="压缩后的内容",
                metadata={
                    "rerank_score": 0.9,
                    "compressed": True,
                    "original_length": 1000,
                    "compressed_length": 300,
                }
            ),
            Document(
                page_content="原文内容",
                metadata={"rerank_score": 0.8}
            )
        ]

        formatted = format_compressed_docs(docs)

        # 验证格式
        assert "参考资料1" in formatted
        assert "[相关性: 0.900]" in formatted
        assert "[已压缩 1000→300字符]" in formatted
        assert "参考资料2" in formatted
        assert "[原文]" in formatted


class TestCompressionEdgeCases:
    """边界情况测试"""

    def test_empty_documents(self):
        """测试: 空文档列表"""
        compressor = ContextCompressor()
        result = compressor.compress("查询", [], max_tokens=1000)

        assert result.documents == []
        assert result.stats.compression_ratio == 0.0
        assert result.quality_score == 0.0

    def test_single_short_document(self):
        """测试: 单个短文档"""
        compressor = ContextCompressor()

        docs = [Document(page_content="短", metadata={"rerank_score": 0.9})]
        result = compressor.compress("查询", docs, max_tokens=1000)

        assert result.stats.method_used == "none"
        assert result.documents[0].page_content == "短"

    def test_very_long_document(self):
        """测试: 超长文档"""
        compressor = ContextCompressor()

        # 构造 10k+ 字符的文档
        docs = [Document(
            page_content="这是重复内容。" * 2000,
            metadata={"rerank_score": 0.9}
        )]

        result = compressor.compress(
            "查询",
            docs,
            max_tokens=500,
            strategy=CompressionStrategy.EXTRACT
        )

        # 应该被压缩
        assert result.stats.compression_ratio > 0.5
        assert result.quality_score > 0.3


class TestCompressionPerformance:
    """性能测试"""

    def test_compression_speed(self):
        """测试: 压缩处理时间"""
        import time

        compressor = ContextCompressor()

        # 构造批量文档
        docs = [
            Document(
                page_content=f"文档{i}内容" * 50 + "。" * 10,
                metadata={"rerank_score": 0.9 - i * 0.05}
            )
            for i in range(10)
        ]

        start = time.time()
        result = compressor.compress("查询", docs, max_tokens=1000)
        elapsed = (time.time() - start) * 1000

        # 纯提取式应在 1000ms 内完成
        assert elapsed < 1000, f"压缩耗时过长: {elapsed:.1f}ms"
        print(f"\n压缩耗时: {elapsed:.1f}ms")


def run_manual_test():
    """手动运行测试，显示详细输出"""
    print("=" * 70)
    print("上下文压缩模块测试")
    print("=" * 70)

    compressor = ContextCompressor()

    # 测试 1: 长文档压缩
    print("\n【测试 1】长文档提取式压缩")
    long_doc = Document(
        page_content="\n\n".join([
            f"第{i}段：TCP协议是一种面向连接的、可靠的传输层协议，广泛应用于互联网通信。"
            for i in range(30)
        ]),
        metadata={"rerank_score": 0.95, "source": "network.txt"}
    )

    result = compressor.compress(
        "TCP协议特点",
        [long_doc],
        max_tokens=800,
        strategy=CompressionStrategy.EXTRACT
    )

    print(f"  原始 Token: {result.stats.original_tokens}")
    print(f"  压缩后 Token: {result.stats.compressed_tokens}")
    print(f"  压缩率: {result.stats.compression_ratio:.1%}")
    print(f"  质量分数: {result.quality_score:.2f}")
    print(f"  处理方法: {result.stats.method_used}")
    print(f"  处理时间: {result.stats.processing_time_ms:.1f}ms")

    # 测试 2: 多文档压缩
    print("\n【测试 2】多文档压缩")
    multi_docs = [
        Document(
            page_content=f"文档{i}：" + "这是一个关于机器学习的文档内容。" * 50,
            metadata={"rerank_score": 0.9 - i * 0.1, "source": f"doc{i}.txt"}
        )
        for i in range(5)
    ]

    result2 = compressor.compress(
        "机器学习应用",
        multi_docs,
        max_tokens=600
    )

    print(f"  文档数: {result2.stats.docs_before} → {result2.stats.docs_after}")
    print(f"  压缩率: {result2.stats.compression_ratio:.1%}")
    print(f"  自动选择策略: {result2.stats.method_used}")

    # 测试 3: 格式化输出
    print("\n【测试 3】格式化输出示例")
    formatted = format_compressed_docs(result2.documents[:2])
    print(formatted[:500] + "..." if len(formatted) > 500 else formatted)

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    run_manual_test()
