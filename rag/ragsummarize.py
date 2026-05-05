from utils.prompts_hander import get_rag_prompt
from utils.config_hander import rerank_config
from model.model import chat_model
from rag.vector_store import VectorStoreService
from rag.reranker import get_reranker
from rag.reranker_enhanced import get_enhanced_reranker
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate


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
            self.reranker = get_enhanced_reranker(
                model=rerank_model or cfg["model"],
                top_n=rerank_top_n or cfg["top_n"],
                score_threshold=score_threshold or cfg.get("score_threshold", 0.4),
                instruct=cfg.get("instruct"),
            )
            logger = __import__("utils.log", fromlist=["logger"]).logger
            logger.info("使用增强版 Reranker（阈值: %.2f）", 
                       score_threshold or cfg.get("score_threshold", 0.4))
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

