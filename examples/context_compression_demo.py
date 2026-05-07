"""
上下文压缩功能使用示例

展示如何在 RAG 系统中使用上下文压缩功能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from rag.context_compressor import (
    ContextCompressor,
    CompressionStrategy,
    format_compressed_docs,
)


def demo_basic_compression():
    """基础压缩演示"""
    print("=" * 70)
    print("示例 1: 基础上下文压缩")
    print("=" * 70)

    # 创建压缩器
    compressor = ContextCompressor()

    # 模拟检索到的长文档
    documents = [
        Document(
            page_content="""TCP协议是一种面向连接的、可靠的、基于字节流的传输层通信协议。

TCP三次握手是建立TCP连接的过程。第一次握手：客户端发送SYN包到服务器，进入SYN_SENT状态。第二次握手：服务器收到SYN包，确认客户的SYN（ack=j+1），同时自己也发送一个SYN包（syn=k），即SYN+ACK包，进入SYN_RECV状态。第三次握手：客户端收到服务器的SYN+ACK包，向服务器发送确认包ACK（ack=k+1），此包发送完毕，客户端和服务器进入ESTABLISHED状态，完成三次握手。

TCP协议的特点包括：
1. 面向连接：通信双方必须先建立连接
2. 可靠传输：通过确认和重传机制保证数据可靠
3. 流量控制：使用滑动窗口机制
4. 拥塞控制：防止网络过载

TCP与UDP的区别：
- TCP是面向连接的，UDP是无连接的
- TCP提供可靠传输，UDP不保证可靠
- TCP有流量控制和拥塞控制，UDP没有
- TCP首部开销大（20字节），UDP首部开销小（8字节）

TCP应用场景：
- HTTP/HTTPS 网页浏览
- FTP 文件传输
- SMTP/POP3 邮件传输
- 需要可靠传输的场景

更多细节...
TCP的滑动窗口机制包括发送窗口和接收窗口...
拥塞控制包括慢启动、拥塞避免、快重传、快恢复...
""",
            metadata={"rerank_score": 0.95, "source": "network/tcp_guide.txt"}
        ),
        Document(
            page_content="""HTTP协议是基于TCP的应用层协议，用于Web数据传输。

HTTP发展历史：
- HTTP/0.9（1991年）：只有GET方法，无首部
- HTTP/1.0（1996年）：增加POST、HEAD，引入首部
- HTTP/1.1（1997年）：持久连接、管道化、分块传输
- HTTP/2（2015年）：二进制分帧、多路复用、头部压缩
- HTTP/3（2022年）：基于QUIC，解决队头阻塞

HTTP/1.1 特性：
1. 持久连接：Connection: keep-alive
2. 管道化：允许发送多个请求
3. 分块传输：Transfer-Encoding: chunked
4. 缓存控制：Cache-Control
5. 范围请求：Range 头部

HTTP方法：
GET - 获取资源
POST - 提交数据
PUT - 更新资源
DELETE - 删除资源
HEAD - 获取首部
OPTIONS - 查询支持的方法
PATCH - 部分更新

状态码分类：
1xx - 信息响应
2xx - 成功
3xx - 重定向
4xx - 客户端错误
5xx - 服务器错误

常用状态码：
200 OK - 请求成功
301 Moved Permanently - 永久重定向
400 Bad Request - 请求语法错误
401 Unauthorized - 未授权
403 Forbidden - 禁止访问
404 Not Found - 资源不存在
500 Internal Server Error - 服务器内部错误
502 Bad Gateway - 网关错误
503 Service Unavailable - 服务不可用
""",
            metadata={"rerank_score": 0.82, "source": "network/http_guide.txt"}
        ),
    ]

    query = "什么是TCP三次握手？"

    print(f"\n查询: {query}")
    print(f"原始文档数: {len(documents)}")

    # 执行压缩
    result = compressor.compress(
        query=query,
        documents=documents,
        max_tokens=1000,  # 压缩到 1000 tokens
        strategy=CompressionStrategy.AUTO
    )

    print(f"\n压缩结果:")
    print(f"  原始 Token: {result.stats.original_tokens}")
    print(f"  压缩后 Token: {result.stats.compressed_tokens}")
    print(f"  压缩率: {result.stats.compression_ratio:.1%}")
    print(f"  使用策略: {result.stats.method_used}")
    print(f"  质量评分: {result.quality_score:.2f}")
    print(f"  处理时间: {result.stats.processing_time_ms:.1f}ms")

    print(f"\n压缩后文档片段:")
    for i, doc in enumerate(result.documents, 1):
        meta = doc.metadata
        print(f"\n  文档{i} [{meta.get('source', 'unknown')}]")
        if meta.get('compressed'):
            print(f"    原长度: {meta.get('original_length')}字符")
            print(f"    压缩后: {meta.get('compressed_length')}字符")
            print(f"    保留段落: {meta.get('segments_kept')}/{meta.get('segments_total')}")
        print(f"    内容预览: {doc.page_content[:150]}...")


def demo_code_preservation():
    """代码保留演示"""
    print("\n" + "=" * 70)
    print("示例 2: 代码查询时的压缩（保留代码完整性）")
    print("=" * 70)

    compressor = ContextCompressor()

    code_doc = Document(
        page_content="""Python 列表推导式是创建列表的简洁方式。

基本语法：
```python
# 基本列表推导式
squares = [x**2 for x in range(10)]
# 结果: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 带条件的列表推导式
evens = [x for x in range(10) if x % 2 == 0]
# 结果: [0, 2, 4, 6, 8]

# 嵌套列表推导式
matrix = [[i*j for j in range(3)] for i in range(3)]
# 结果: [[0, 0, 0], [0, 1, 2], [0, 2, 4]]
```

更多说明...

字典推导式：
```python
# 基本字典推导式
squares_dict = {x: x**2 for x in range(5)}
# 结果: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# 交换键值
original = {'a': 1, 'b': 2}
swapped = {v: k for k, v in original.items()}
```

集合推导式...
生成器表达式...

性能对比：
- 列表推导式比普通 for 循环快约 1.5-2 倍
- 因为列表推导式在 C 层执行
- 生成器表达式更省内存

使用建议：
1. 简单转换用列表推导式
2. 大数据集用生成器表达式
3. 复杂逻辑用普通 for 循环（可读性更好）
""",
        metadata={"rerank_score": 0.9, "source": "python/list_comprehension.md"}
    )

    query = "Python 列表推导式怎么写？"

    print(f"\n查询: {query}")

    # 自动策略会选择 EXTRACT（因为查询包含"代码"关键词）
    result = compressor.compress(
        query=query,
        documents=[code_doc],
        max_tokens=800,
        strategy=CompressionStrategy.AUTO
    )

    print(f"\n自动选择策略: {result.stats.method_used}")
    print(f"压缩率: {result.stats.compression_ratio:.1%}")

    # 检查代码块是否被保留
    has_code = any("```python" in doc.page_content for doc in result.documents)
    print(f"代码块保留: {has_code}")

    print(f"\n压缩后内容预览:")
    print(result.documents[0].page_content[:500])


def demo_different_strategies():
    """不同压缩策略对比"""
    print("\n" + "=" * 70)
    print("示例 3: 不同压缩策略对比")
    print("=" * 70)

    long_doc = Document(
        page_content="\n\n".join([
            f"段落{i}：这是关于深度学习神经网络的详细解释。" +
            "深度学习是机器学习的一个子领域，基于人工神经网络。" * 5
            for i in range(20)
        ]),
        metadata={"rerank_score": 0.92, "source": "ai/deep_learning.txt"}
    )

    query = "深度学习的核心概念是什么？"

    strategies = [
        (CompressionStrategy.EXTRACT, "提取式压缩"),
        (CompressionStrategy.HYBRID, "混合压缩"),
    ]

    print(f"\n查询: {query}")
    print(f"原始文档 Token: ~{len(long_doc.page_content) // 2}")

    for strategy, name in strategies:
        compressor = ContextCompressor()
        result = compressor.compress(
            query=query,
            documents=[long_doc],
            max_tokens=600,
            strategy=strategy
        )

        print(f"\n{name} ({strategy.value}):")
        print(f"  压缩后 Token: {result.stats.compressed_tokens}")
        print(f"  压缩率: {result.stats.compression_ratio:.1%}")
        print(f"  质量分: {result.quality_score:.2f}")
        print(f"  耗时: {result.stats.processing_time_ms:.1f}ms")


def demo_integration_with_rag():
    """与 RAG 系统集成演示"""
    print("\n" + "=" * 70)
    print("示例 4: 在 RAG 系统中使用上下文压缩")
    print("=" * 70)

    # 方式 1: 通过配置启用（推荐）
    print("\n方式 1: 通过配置文件启用")
    print("""
# config/rag.yml
compression_enabled: true
compression_max_tokens: 3500
compression_strategy: auto
    """)

    # 方式 2: 代码中显式启用
    print("\n方式 2: 代码中显式启用")
    print("""
from rag.ragsummarize import RAGSummarize

# 启用上下文压缩
rag = RAGSummarize(compression_enabled=True)

# 查询会自动经过压缩处理
answer = rag.summarize("什么是TCP三次握手？")
    """)

    # 方式 3: 获取压缩统计
    print("\n方式 3: 获取压缩统计信息")
    print("""
# 查看日志输出
# 会看到类似:
# 上下文压缩: 4200 → 2800 tokens (33.3%压缩率, 策略:extractive, 质量:0.85, 耗时:45.2ms)
    """)


def demo_formatting():
    """格式化输出演示"""
    print("\n" + "=" * 70)
    print("示例 5: 格式化输出（用于 Prompt）")
    print("=" * 70)

    compressed_docs = [
        Document(
            page_content="TCP三次握手包括：SYN、SYN-ACK、ACK三个步骤。",
            metadata={
                "rerank_score": 0.95,
                "compressed": True,
                "original_length": 1500,
                "compressed_length": 300,
                "source": "network/tcp.txt"
            }
        ),
        Document(
            page_content="HTTP协议基于TCP，用于Web数据传输。",
            metadata={
                "rerank_score": 0.82,
                "source": "network/http.txt"
            }
        ),
    ]

    formatted = format_compressed_docs(compressed_docs)

    print("\n格式化后的 Prompt 片段:")
    print("-" * 50)
    print(formatted)
    print("-" * 50)

    print("\n说明:")
    print("  - [相关性: x.xxx]: Rerank 分数")
    print("  - [已压缩 A→B字符]: 压缩信息")
    print("  - [原文]: 未压缩的原始文档")


def demo_performance_tips():
    """性能优化建议"""
    print("\n" + "=" * 70)
    print("性能优化建议")
    print("=" * 70)

    tips = """
1. Token 预算设置
   - 设置为 LLM 上下文限制的 70-80%
   - 预留空间给 Prompt 模板和用户问题

2. 策略选择
   - 代码查询: 使用 extract（保留完整性）
   - 事实查询: 使用 extract（保留精确信息）
   - 超长文档: 使用 hybrid（提取+摘要）
   - 一般情况: 使用 auto（自动选择）

3. 关闭 LLM 摘要（默认）
   - LLM 摘要质量好但增加延迟和成本
   - 提取式压缩纯本地计算，速度快

4. 阈值调优
   - 高质量文档库: score_threshold 0.5+
   - 一般文档库: score_threshold 0.3-0.4

5. 监控指标
   - 压缩率: 30-60% 为正常范围
   - 质量分数: < 0.6 时需要调整参数
   - 处理时间: > 500ms 考虑优化
    """

    print(tips)


def main():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("上下文压缩功能演示")
    print("=" * 70)

    demo_basic_compression()
    demo_code_preservation()
    demo_different_strategies()
    demo_integration_with_rag()
    demo_formatting()
    demo_performance_tips()

    print("\n" + "=" * 70)
    print("演示完成！查看 docs/context_compression_guide.md 获取完整文档")
    print("=" * 70)


if __name__ == "__main__":
    main()
