from abc import ABC, abstractmethod
from typing import Any, Optional
from langchain_core.embeddings import Embeddings
from utils.config_hander import rag_config
import os
try:
    from langchain_community.chat_models.tongyi import ChatTongyi
except ImportError:
    ChatTongyi = None

try:
    from langchain_community.embeddings import DashScopeEmbeddings
except ImportError:
    DashScopeEmbeddings = None


class base_model(ABC):
    @abstractmethod
    def generator(self) -> Optional[Any]:
        pass


class chat_model(base_model):
    def generator(self) -> Optional[Any]:
        if ChatTongyi is None:
            return None
        # Tongyi 集成走 DashScope；未传 api_key 时由 ChatTongyi 读环境变量 DASHSCOPE_API_KEY
        api_key = os.getenv("TONGYI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        kwargs: dict[str, Any] = {"model_name": rag_config["chat_model"]}
        if api_key:
            kwargs["api_key"] = api_key

        top_p = rag_config.get("top_p")
        if top_p is not None:
            kwargs["top_p"] = float(top_p)

        model_kwargs: dict[str, Any] = {}
        t = rag_config.get("temperature")
        if t is not None:
            model_kwargs["temperature"] = float(t)
        mt = rag_config.get("max_tokens")
        if mt is not None:
            model_kwargs["max_tokens"] = int(mt)
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs

        return ChatTongyi(**kwargs)


class embedding_model(base_model):
    def generator(self) -> Optional[Embeddings]:
        if DashScopeEmbeddings is None:
            return None
        return DashScopeEmbeddings(model=rag_config["embedding_model"])


chat_model = chat_model().generator()
embedding_model = embedding_model().generator()