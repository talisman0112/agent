import re
from typing import List, Tuple, Optional

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

# 导入 Web 搜索工具
from tools.agent_tool import web_search


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
    """支持 Rerank 的 RAG 问答链。

    配置说明（可在 config/rag.yml 添加）：
        rerank_enabled: true          # 是否启用 Rerank
        rerank_model: qwen3-rerank    # 可选: qwen3-rerank(推荐), gte-rerank-v2
        rerank_top_n: 5               # 精排后取 Top-N
        rerank_search_k: 20           # 向量检索召回数量（需 > rerank_top_n）
        rerank_enhanced: true         # 是否使用增强版 Rerank
        rerank_score_threshold: 0.4   # 分数阈值
    """

    def __init__(
        self,
        rerank_enabled: bool = None,
        rerank_top_n: int = None,
        vector_search_k: int = None,
        rerank_model: str = None,
        use_enhanced: bool = None,
        score_threshold: float = None,
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
        self.chain = self._create_chain()

    def _rerank_docs(self, query: str) -> list[Document]:
        """先向量检索，再 Rerank 精排。"""
        # 1. 向量检索（粗排）
        coarse_docs = self.rag_retriever.invoke(query)
        if not coarse_docs:
            return []
        # 2. Rerank 精排
        return self.reranker.rerank(query, coarse_docs)

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


class HybridRAG:
    """多路召回 RAG：Web 搜索 + 本地知识库 + 统一 Rerank

    同时从多个数据源召回文档，经 Rerank 精排后返回最相关的内容。

    配置说明（可在 config/rag.yml 添加）：
        hybrid_enabled: true          # 是否启用多路召回
        hybrid_web_max_results: 5     # Web 搜索返回结果数
        hybrid_local_k: 15            # 本地检索召回数
        hybrid_rerank_top_n: 5        # 精排后取 Top-N
    """

    def __init__(
        self,
        web_max_results: int = None,
        local_search_k: int = None,
        rerank_top_n: int = None,
        rerank_model: str = None,
        score_threshold: float = None,
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

        logger.info(
            "初始化 HybridRAG（Web: %d条, 本地: %d条, Rerank: Top-%d, 阈值: %.2f）",
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

        # 1. 本地向量库检索
        try:
            local_docs = self.rag_retriever.invoke(query)
            for doc in local_docs:
                doc.metadata["source_channel"] = "local"
            all_docs.extend(local_docs)
            logger.info("本地检索: %d 条", len(local_docs))
        except Exception as e:
            logger.error("本地检索失败: %s", e)

        # 2. Web 搜索
        try:
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
                   len(all_docs), len(local_docs) if 'local_docs' in dir() else 0,
                   len(web_docs) if 'web_docs' in dir() else 0)

        return all_docs

    def _rerank_docs(self, query: str) -> List[Document]:
        """多路召回后统一 Rerank"""
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

