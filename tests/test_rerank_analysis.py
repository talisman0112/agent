"""Rerank 评分分布分析工具 - 诊断精度问题"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.documents import Document
from rag.vector_store import VectorStoreService
from rag.reranker import get_reranker
from rag.reranker_enhanced import get_enhanced_reranker


def analyze_rerank_distribution(query: str, docs: list[Document]):
    """分析 Rerank 评分分布，诊断区分度问题。"""
    print(f"\n{'='*70}")
    print(f"查询: {query}")
    print(f"{'='*70}")
    
    # 1. 普通 Rerank
    print("\n【1. 普通版 Rerank (qwen3-rerank)】")
    normal = get_reranker(enabled=True, model="qwen3-rerank", top_n=10)
    result_normal = normal.rerank(query, docs)
    
    scores_normal = []
    for i, doc in enumerate(result_normal, 1):
        score = doc.metadata.get("rerank_score", 0.0)
        scores_normal.append(score)
        marker = "✓" if i <= 3 else " "
        print(f"  {marker} [{score:.4f}] {doc.page_content[:50]}...")
    
    _print_score_stats(scores_normal, "普通版")
    
    # 2. 增强版 Rerank（带指令）
    print("\n【2. 增强版 Rerank (带 Q&A 指令)】")
    enhanced_qa = get_enhanced_reranker(
        model="qwen3-rerank",
        top_n=10,
        score_threshold=0.0,  # 先不设阈值，看全部分数
        instruct="Given a web search query, retrieve relevant passages that answer the query.",
    )
    result_enhanced = enhanced_qa.rerank(query, docs)
    
    scores_enhanced = []
    for i, doc in enumerate(result_enhanced, 1):
        r_score = doc.metadata.get("rerank_score", 0.0)
        c_score = doc.metadata.get("combined_score", r_score)
        scores_enhanced.append(c_score)
        marker = "✓" if i <= 3 else " "
        print(f"  {marker} [R:{r_score:.4f}, C:{c_score:.4f}] {doc.page_content[:50]}...")
    
    _print_score_stats(scores_enhanced, "增强版(Q&A)")
    
    # 3. 增强版（带阈值过滤效果展示）
    print("\n【3. 增强版 Rerank (阈值=0.4 过滤效果)】")
    enhanced_filter = get_enhanced_reranker(
        model="qwen3-rerank",
        top_n=5,
        score_threshold=0.4,
    )
    result_filtered = enhanced_filter.rerank(query, docs)
    
    for i, doc in enumerate(result_filtered, 1):
        c_score = doc.metadata.get("combined_score", 0.0)
        print(f"  [{c_score:.4f}] {doc.page_content[:50]}...")
    print(f"  过滤后保留: {len(result_filtered)}/{len(docs)} 个文档")

    # 4. 建议
    print("\n【诊断建议】")
    _provide_suggestions(scores_normal, scores_enhanced)
    
    print("="*70)


def _print_score_stats(scores: list[float], label: str):
    """打印分数统计信息。"""
    if len(scores) < 2:
        return
    
    import statistics
    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0
    max_gap = max(scores) - min(scores)
    top3_gap = scores[0] - scores[2] if len(scores) >= 3 else 0
    
    print(f"\n  {label} 统计:")
    print(f"    - 平均分: {mean:.4f}, 标准差: {std:.4f}")
    print(f"    - 最大差距: {max_gap:.4f}")
    print(f"    - 头部差距(Top1-Top3): {top3_gap:.4f}")
    
    if std < 0.1:
        print(f"    ⚠️ 警告: 标准差 < 0.1，分数过于集中，区分度不足！")
    if top3_gap < 0.05:
        print(f"    ⚠️ 警告: 头部差距 < 0.05，前3名难以区分！")


def _provide_suggestions(scores_normal: list[float], scores_enhanced: list[float]):
    """根据分析结果给出建议。"""
    import statistics
    
    std_normal = statistics.stdev(scores_normal) if len(scores_normal) > 1 else 0
    std_enhanced = statistics.stdev(scores_enhanced) if len(scores_enhanced) > 1 else 0
    
    if std_enhanced > std_normal:
        print("  ✓ 增强版（带指令）的区分度更好，建议使用")
    else:
        print("  • 增强版与普通版区分度相近")
    
    min_score = min(scores_enhanced)
    if min_score > 0.3:
        print(f"  ⚠️ 最低分数为 {min_score:.4f}，整体偏高，建议：")
        print("    1. 提高 score_threshold 到 0.5-0.6")
        print("    2. 增加 vector_search_k 召回更多候选")
    
    if max(scores_enhanced) - min(scores_enhanced) < 0.2:
        print("  ⚠️ 整体分数差距 < 0.2，建议：")
        print("    1. 检查文档切分质量（是否切得太碎）")
        print("    2. 检查向量召回质量（粗排是否已偏离）")
        print("    3. 考虑使用混合检索 (BM25 + 向量)")


def test_with_real_data(query: str):
    """使用真实向量库测试。"""
    print(f"\n{'='*70}")
    print("使用真实向量库测试")
    print(f"{'='*70}")
    
    try:
        store = VectorStoreService()
        retriever = store.chroma.as_retriever(search_kwargs={"k": 20})
        docs = retriever.invoke(query)
        
        if not docs:
            print("向量库为空或未加载数据，请先执行数据入库")
            return
        
        print(f"向量检索召回: {len(docs)} 个文档")
        analyze_rerank_distribution(query, docs)
        
    except Exception as e:
        print(f"错误: {e}")


def test_with_mock_data():
    """使用模拟数据测试。"""
    # 模拟不同质量的文档
    test_docs = [
        # 高度相关（直接回答问题）
        Document(page_content="TCP三次握手是建立TCP连接的过程，包括：1.SYN 2.SYN-ACK 3.ACK。客户端发送SYN，服务器回复SYN-ACK，客户端再发送ACK确认。"),
        Document(page_content="TCP三次握手的目的是同步双方序列号，确认双方收发能力正常，防止历史重复连接请求。"),
        # 中等相关（相关主题但不直接回答）
        Document(page_content="TCP协议是面向连接的传输层协议，提供可靠的字节流传输服务，包含流量控制和拥塞控制机制。"),
        Document(page_content="TCP四次挥手是连接终止过程，比三次握手更复杂，因为需要确保双方数据都传输完毕。"),
        # 弱相关
        Document(page_content="HTTP协议基于TCP实现，是应用层协议，用于Web浏览器和服务器之间的通信。"),
        Document(page_content="UDP协议与TCP不同，是无连接协议，不保证可靠传输，常用于视频流媒体。"),
        # 无关
        Document(page_content="Python是一种解释型高级编程语言，由Guido van Rossum于1991年创建。"),
        Document(page_content="机器学习是人工智能的一个分支，包括监督学习、无监督学习和强化学习。"),
        Document(page_content="深度学习使用神经网络，在图像识别、自然语言处理等领域取得突破。"),
    ]
    
    analyze_rerank_distribution("什么是TCP三次握手？", test_docs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rerank 精度分析工具")
    parser.add_argument("--query", "-q", type=str, help="测试查询")
    parser.add_argument("--real", "-r", action="store_true", help="使用真实向量库")
    args = parser.parse_args()
    
    if args.real and args.query:
        test_with_real_data(args.query)
    else:
        # 默认使用模拟数据
        test_with_mock_data()
        
        # 再测一个例子
        print("\n" + "="*70)
        print("附加测试：不同查询类型")
        print("="*70)
        
        docs2 = [
            Document(page_content="秦始皇（前259年—前210年），嬴姓，赵氏，名政，是中国历史上第一个称帝的君主。"),
            Document(page_content="秦始皇统一六国，建立秦朝，实行郡县制，统一文字、货币、度量衡。"),
            Document(page_content="兵马俑是秦始皇陵的陪葬坑，位于西安，是世界文化遗产。"),
            Document(page_content="汉武帝刘彻是西汉第七位皇帝，开创汉武盛世，北击匈奴。"),
            Document(page_content="唐朝是中国历史上的强盛朝代，唐太宗李世民开创贞观之治。"),
            Document(page_content="Python的Flask是一个轻量级Web框架，适合构建小型应用。"),
        ]
        analyze_rerank_distribution("秦始皇有哪些历史功绩？", docs2)
