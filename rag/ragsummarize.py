from utils.prompts_hander import get_rag_prompt
from model.model import chat_model
from rag.vector_store import VectorStoreService
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate


def _format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"参考资料{i}:\n{doc.page_content}")
    return "\n\n".join(parts)

def print_prompt(prompt):
    print("=" * 20 + " PROMPT sent to LLM " + "=" * 20)
    print(prompt)
    print("=" * 60)
    return prompt
class RAGSummarize:
    def __init__(self):
        self.rag_prompt = get_rag_prompt()
        self.rag_model = chat_model
        self.vector_store = VectorStoreService()
        self.rag_retriever = self.vector_store.get_retriever()
        self.prompt_template = PromptTemplate.from_template(self.rag_prompt)
        self.chain = self._create_chain()

    def _create_chain(self):
        return (
            {
                "context": self.rag_retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt_template
            | print_prompt
            | self.rag_model
            | StrOutputParser()
        )

    def retrieve_docs(self, query: str) -> list[Document]:
        return self.rag_retriever.invoke(query)

    def summarize(self, query: str) -> str:
        return self.chain.invoke(query)

