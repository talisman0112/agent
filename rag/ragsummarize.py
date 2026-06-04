import re
from typing import List, Tuple

from utils.prompts_hander import get_rag_prompt
from utils.config_hander import rerank_config
from utils.log import logger
from model.model import chat_model
from rag.vector_store import VectorStoreService
from rag.reranker import get_reranker
from rag.reranker_enhanced import get_enhanced_reranker, InstructMode
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate

# 上下文压缩模块
from rag.context_compressor import ContextCompressor, CompressionStrategy, format_compressed_docs
from rag.query_expand import build_search_queries, coarse_retrieve_union

def _format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        score = doc.metadata.get("rerank_score", 0.0)
        parts.append(f"参考资料{i} [相关性: {score:.3f}]:\n{doc.page_content}")
    return "\n\n".join(parts)

def print_prompt(prompt):
    print("=" * 20 + " PROMPT sent to LLM " + "=" * 20)
    print(prompt)
    print("=" * 60)
    return prompt
class RAGSummarize:
    """支持 Rerank 和上下文压缩的 RAG 问答链。

    配置说明（可在 config/rag.yml 添加）：
        # Rerank 配置
        rerank_enabled: true          # 是否启用 Rerank
        rerank_model: qwen3-rerank    # 可选: qwen3-rerank(推荐), gte-rerank-v2
        rerank_top_n: 5               # 精排后取 Top-N
        rerank_search_k: 20           # 向量检索召回数量（需 > rerank_top_n）
        rerank_enhanced: true         # 是否使用增强版 Rerank
        rerank_score_threshold: 0.4   # 分数阈值

        # 上下文压缩配置（新增）
        compression_enabled: true     # 是否启用上下文压缩
        compression_max_tokens: 3500  # 压缩后目标 Token 上限
        compression_min_tokens: 200   # 单个文档最小保留 Token
        compression_extract_ratio: 0.6 # 提取式压缩保留比例
    """

    def __init__(
        self,
        rerank_enabled: bool = None,
        rerank_top_n: int = None,
        vector_search_k: int = None,
        rerank_model: str = None,
        use_enhanced: bool = None,
        score_threshold: float = None,
        compression_enabled: bool = None,
    ):
        # 从配置读取默认值（参数可覆盖）
        cfg = rerank_config
        self.rag_prompt = get_rag_prompt()
        self.rag_model = chat_model
        self.vector_store = VectorStoreService()

        # 向量检索器：召回更多，给 Rerank 提供候选池
        self.vector_k = vector_search_k or cfg.get("vector_search_k", 20)
        self.rag_retriever = self.vector_store.chroma.as_retriever(
            search_kwargs={"k": self.vector_k}
        )

        # 选择使用增强版或普通版 Reranker
        is_enhanced = use_enhanced if use_enhanced is not None else cfg.get("enhanced", True)
        
        if is_enhanced:
            # 增强版：带阈值过滤、指令引导、自适应 Top-N
            # 解析 instruct_mode
            mode_str = cfg.get("instruct_mode", "qa")
            try:
                instruct_mode = InstructMode(mode_str)
            except ValueError:
                instruct_mode = InstructMode.QA
            
            self.reranker = get_enhanced_reranker(
                model=rerank_model or cfg["model"],
                top_n=rerank_top_n or cfg["top_n"],
                score_threshold=score_threshold or cfg.get("score_threshold", 0.4),
                instruct_mode=instruct_mode,
                custom_instruct=cfg.get("custom_instruct"),
                auto_instruct=cfg.get("auto_instruct", True),
                dedup_threshold=cfg.get("dedup_threshold", 0.80),
            )
            logger = __import__("utils.log", fromlist=["logger"]).logger
            logger.info("使用增强版 Reranker（阈值: %.2f, 去重: %.2f, 模式: %s, 自动: %s）",
                       score_threshold or cfg.get("score_threshold", 0.4),
                       cfg.get("dedup_threshold", 0.80),
                       mode_str,
                       cfg.get("auto_instruct", True))
        else:
            # 普通版
            self.reranker = get_reranker(
                enabled=rerank_enabled if rerank_enabled is not None else cfg["enabled"],
                model=rerank_model or cfg["model"],
                top_n=rerank_top_n or cfg["top_n"],
            )

        self.prompt_template = PromptTemplate.from_template(self.rag_prompt)

        # 初始化上下文压缩器
        self.compression_enabled = compression_enabled if compression_enabled is not None \
                                   else cfg.get("compression_enabled", True)

        if self.compression_enabled:
            self.compressor = ContextCompressor(
                llm_client=chat_model if cfg.get("compression_use_llm", False) else None,
                embedding_client=None,
                config={
                    "compression_max_tokens": cfg.get("compression_max_tokens", 3500),
                    "compression_min_tokens": cfg.get("compression_min_tokens", 200),
                    "compression_extract_ratio": cfg.get("compression_extract_ratio", 0.6),
                    "compression_quality_threshold": cfg.get("compression_quality_threshold", 0.7),
                }
            )
            logger.info("上下文压缩已启用（目标 Token: %d, 策略: %s）",
                       cfg.get("compression_max_tokens", 3500),
                       cfg.get("compression_strategy", "auto"))
        else:
            self.compressor = None

        self.chain = self._create_chain()

    def _rerank_docs(self, query: str) -> list[Document]:
        """增强版检索流程：向量检索 → Rerank 精排 → 上下文压缩（可选）"""
        # 1. 向量检索（粗排）；可选多 query 合并后再父块展开
        cfg_q = rerank_config
        search_queries = build_search_queries(
            retrieval_input=query,
            cfg=cfg_q,
            llm=getattr(self, "rag_model", None),
        )
        coarse_docs = coarse_retrieve_union(
            self.rag_retriever,
            search_queries,
            max_coarse_docs=cfg_q.get("query_expansion_max_coarse_docs", 100),
            max_workers=cfg_q.get("query_expansion_max_workers", 8),
        )
        coarse_docs = self.vector_store.expand_retrieval_to_parents(coarse_docs)
        if not coarse_docs:
            return []

        # 2. Rerank 精排
        reranked_docs = self.reranker.rerank(query, coarse_docs)

        # 3. 上下文压缩（新增）
        if self.compression_enabled and self.compressor:
            strategy_str = rerank_config.get("compression_strategy", "auto")
            try:
                strategy = CompressionStrategy(strategy_str)
            except ValueError:
                strategy = CompressionStrategy.AUTO

            result = self.compressor.compress(
                query=query,
                documents=reranked_docs,
                max_tokens=self.compressor.max_tokens,
                strategy=strategy
            )

            # 记录压缩统计
            if result.stats.compression_ratio > 0:
                logger.info(
                    "上下文压缩: %d→%d tokens (%.1f%%压缩率, 策略:%s, 质量:%.2f, 耗时:%.1fms)",
                    result.stats.original_tokens,
                    result.stats.compressed_tokens,
                    result.stats.compression_ratio * 100,
                    result.stats.method_used,
                    result.quality_score,
                    result.stats.processing_time_ms
                )

                # 质量警告
                if result.quality_score < self.compressor.quality_threshold:
                    logger.warning(
                        "压缩质量较低(%.2f)，可能影响回答效果",
                        result.quality_score
                    )

            return result.documents

        return reranked_docs

    def _create_chain(self):
        return (
            {
                "context": RunnableLambda(self._rerank_docs) | _format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt_template
            | print_prompt
            | self.rag_model
            | StrOutputParser()
        )

    def retrieve_docs(self, query: str) -> list[Document]:
        """对外暴露：获取 Rerank 后的文档。"""
        return self._rerank_docs(query)

    def retrieve_docs_with_scores(self, query: str) -> list[tuple[Document, float]]:
        """返回带 Rerank 分数的文档。"""
        docs = self._rerank_docs(query)
        return [(doc, doc.metadata.get("rerank_score", 0.0)) for doc in docs]

    def summarize(self, query: str) -> str:
        return self.chain.invoke(query)

    def summarize_with_docs(self, query: str, docs: list[Document]) -> str:
        """??????? docs ?????????????"""
        formatted = _format_docs(docs)
        inputs = {"context": formatted, "question": query}
        return (self.prompt_template | self.rag_model | StrOutputParser()).invoke(inputs)


class HybridRAG:
    """多路召回 RAG：Web 搜索 + 本地知识库 + 统一 Rerank + 上下文压缩

    同时从多个数据源召回文档，经 Rerank 精排和上下文压缩后返回最相关的内容。

    配置说明（可在 config/rag.yml 添加）：
        hybrid_enabled: true          # 是否启用多路召回
        hybrid_web_max_results: 5     # Web 搜索返回结果数
        hybrid_local_k: 15            # 本地检索召回数
        hybrid_rerank_top_n: 5        # 精排后取 Top-N
        compression_enabled: true    # 是否启用上下文压缩
    """

    def __init__(
        self,
        web_max_results: int = None,
        local_search_k: int = None,
        rerank_top_n: int = None,
        rerank_model: str = None,
        score_threshold: float = None,
        compression_enabled: bool = None,
    ):
        # 从配置读取默认值
        cfg = rerank_config
        self.rag_prompt = get_rag_prompt()
        self.rag_model = chat_model
        self.vector_store = VectorStoreService()

        # 参数配置
        self.web_max_results = web_max_results or cfg.get("hybrid_web_max_results", 5)
        self.local_k = local_search_k or cfg.get("hybrid_local_k", 15)
        self.rerank_top_n = rerank_top_n or cfg.get("hybrid_rerank_top_n", 5)

        # 本地检索器
        self.rag_retriever = self.vector_store.chroma.as_retriever(
            search_kwargs={"k": self.local_k}
        )

        # 初始化 Reranker（增强版）
        mode_str = cfg.get("instruct_mode", "qa")
        try:
            instruct_mode = InstructMode(mode_str)
        except ValueError:
            instruct_mode = InstructMode.QA

        self.reranker = get_enhanced_reranker(
            model=rerank_model or cfg.get("model", "qwen3-rerank"),
            top_n=self.rerank_top_n,
            score_threshold=score_threshold or cfg.get("score_threshold", 0.25),
            instruct_mode=instruct_mode,
            custom_instruct=cfg.get("hybrid_custom_instruct"),  # 多路专用 instruct
            auto_instruct=cfg.get("auto_instruct", True),
            dedup_threshold=cfg.get("dedup_threshold", 0.80),
        )

        # 初始化上下文压缩器
        self.compression_enabled = compression_enabled if compression_enabled is not None \
                                   else cfg.get("compression_enabled", True)

        if self.compression_enabled:
            self.compressor = ContextCompressor(
                llm_client=chat_model if cfg.get("compression_use_llm", False) else None,
                embedding_client=None,
                config={
                    "compression_max_tokens": cfg.get("compression_max_tokens", 3500),
                    "compression_min_tokens": cfg.get("compression_min_tokens", 200),
                    "compression_extract_ratio": cfg.get("compression_extract_ratio", 0.6),
                    "compression_quality_threshold": cfg.get("compression_quality_threshold", 0.7),
                }
            )
            logger.info(
                "初始化 HybridRAG（Web: %d条, 本地: %d条, Rerank: Top-%d, 阈值: %.2f, 压缩: 启用）",
                self.web_max_results, self.local_k, self.rerank_top_n,
                score_threshold or cfg.get("score_threshold", 0.25)
            )
        else:
            self.compressor = None
            logger.info(
                "初始化 HybridRAG（Web: %d条, 本地: %d条, Rerank: Top-%d, 阈值: %.2f, 压缩: 禁用）",
                self.web_max_results, self.local_k, self.rerank_top_n,
                score_threshold or cfg.get("score_threshold", 0.25)
            )

        self.prompt_template = PromptTemplate.from_template(self.rag_prompt)
        self.chain = self._create_chain()

    def _parse_web_results(self, results_text: str) -> List[Document]:
        """将 Web 搜索结果解析为 Document 列表

        解析 web_search 返回的文本格式：
        1. 标题
           链接: xxx
           摘要: yyy
        """
        docs = []
        if not results_text or "未找到" in results_text or "失败" in results_text:
            logger.warning("Web 搜索无结果: %s", results_text[:100] if results_text else "空")
            return docs

        # 匹配搜索结果格式
        pattern = r'\d+\.\s*(.*?)\n\s*链接:\s*(.*?)\n\s*摘要:\s*(.*?)(?=\n\s*\d+\.|$)'
        matches = re.findall(pattern, results_text, re.DOTALL)

        for idx, (title, link, snippet) in enumerate(matches, 1):
            title = title.strip()
            link = link.strip()
            snippet = snippet.strip()

            if not title or title == "无标题":
                continue

            # 整合标题和摘要作为文档内容
            content = f"标题: {title}\n内容: {snippet}"

            doc = Document(
                page_content=content,
                metadata={
                    "source": link,
                    "title": title,
                    "web_rank": idx,  # 搜索引擎排名
                    "source_channel": "web",
                }
            )
            docs.append(doc)

        logger.info("Web 搜索解析: 原始 %d 条, 有效 %d 条", len(matches), len(docs))
        return docs

    def _multi_retrieve(self, query: str) -> List[Document]:
        """多路召回：Web 搜索 + 本地向量检索"""
        all_docs = []
        local_docs: List[Document] = []
        web_docs: List[Document] = []

        # 1. 本地向量库检索（可选 Query 扩展，仅扩展本地召回；Web 仍用原始 query）
        try:
            cfg_q = rerank_config
            search_queries = build_search_queries(
                retrieval_input=query,
                cfg=cfg_q,
                llm=getattr(self, "rag_model", None),
            )
            local_docs = coarse_retrieve_union(
                self.rag_retriever,
                search_queries,
                max_coarse_docs=cfg_q.get("query_expansion_max_coarse_docs", 100),
                max_workers=cfg_q.get("query_expansion_max_workers", 8),
            )
            local_docs = self.vector_store.expand_retrieval_to_parents(local_docs)
            for doc in local_docs:
                md = dict(doc.metadata) if doc.metadata else {}
                md["source_channel"] = "local"
                doc.metadata = md
            all_docs.extend(local_docs)
            logger.info("本地检索: %d 条", len(local_docs))
        except Exception as e:
            logger.error("本地检索失败: %s", e)

        # 2. Web 搜索
        try:
            # 延迟导入，避免 rag.ragsummarize 与 tools.agent_tool 的循环导入
            from tools.agent_tool import web_search

            # web_search 在 tools 中被 @tool 装饰后是 StructuredTool，需用 invoke 调用
            if hasattr(web_search, "invoke"):
                web_text = web_search.invoke(
                    {"query": query, "max_results": self.web_max_results}
                )
            else:
                web_text = web_search(query, max_results=self.web_max_results)
            web_docs = self._parse_web_results(web_text)
            all_docs.extend(web_docs)
            logger.info("Web 搜索: %d 条", len(web_docs))
        except Exception as e:
            logger.error("Web 搜索失败: %s", e)

        if not all_docs:
            logger.warning("多路召回无结果: %s", query[:50])
            return []

        logger.info("多路召回总计: %d 条 (本地 %d + Web %d)",
                   len(all_docs), len(local_docs), len(web_docs))

        return all_docs

    def _rerank_docs(self, query: str) -> List[Document]:
        """多路召回 → 统一 Rerank → 上下文压缩"""
        # 1. 多路召回
        docs = self._multi_retrieve(query)
        if not docs:
            return []

        # 2. 统一 Rerank 精排
        reranked = self.reranker.rerank(query, docs)

        # 记录各源的入选情况
        local_count = sum(1 for d in reranked if d.metadata.get("source_channel") == "local")
        web_count = sum(1 for d in reranked if d.metadata.get("source_channel") == "web")
        logger.info("Rerank 结果: %d 条 (本地 %d + Web %d)",
                   len(reranked), local_count, web_count)

        # 3. 上下文压缩
        if self.compression_enabled and self.compressor:
            strategy_str = rerank_config.get("compression_strategy", "auto")
            try:
                strategy = CompressionStrategy(strategy_str)
            except ValueError:
                strategy = CompressionStrategy.AUTO

            result = self.compressor.compress(
                query=query,
                documents=reranked,
                max_tokens=self.compressor.max_tokens,
                strategy=strategy
            )

            # 记录压缩统计
            if result.stats.compression_ratio > 0:
                logger.info(
                    "HybridRAG 上下文压缩: %d→%d tokens (%.1f%%压缩率, 质量:%.2f)",
                    result.stats.original_tokens,
                    result.stats.compressed_tokens,
                    result.stats.compression_ratio * 100,
                    result.quality_score
                )

            return result.documents

        return reranked

    def _create_chain(self):
        """创建 RAG 调用链"""
        return (
            {
                "context": RunnableLambda(self._rerank_docs) | _format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt_template
            | print_prompt
            | self.rag_model
            | StrOutputParser()
        )

    def retrieve_docs(self, query: str) -> List[Document]:
        """对外暴露：获取多路召回 + Rerank 后的文档"""
        return self._rerank_docs(query)

    def retrieve_docs_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """返回带 Rerank 分数的文档"""
        docs = self._rerank_docs(query)
        return [(doc, doc.metadata.get("rerank_score", 0.0)) for doc in docs]

    def summarize(self, query: str) -> str:
        """多路召回 RAG 问答"""
        return self.chain.invoke(query)

    def summarize_with_docs(self, query: str, docs: list[Document]) -> str:
        """??????? docs ?????????????"""
        formatted = _format_docs(docs)
        inputs = {"context": formatted, "question": query}
        return (self.prompt_template | self.rag_model | StrOutputParser()).invoke(inputs)

