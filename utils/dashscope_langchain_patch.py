"""
修补 LangChain Community 的 Tongyi check_response。

原实现：对非 200/400/401 使用 ``HTTPError(..., response=resp)``，其中 ``resp`` 为
DashScope 的响应对象，不是 ``requests.Response``。``requests.HTTPError`` 初始化时会
访问 ``response.request``，从而触发 DashScope 字典式 API 的 ``KeyError: 'request'``，
掩盖真实的 ``status_code`` / ``message``。

修补后：仍对可重试场景抛出 ``HTTPError``，但**不传入** ``response=``，既避免 KeyError，
又保留 ``chat_models/tongyi.py`` 里 tenacity 对 ``HTTPError`` 的重试逻辑。
"""

from __future__ import annotations

from typing import Any


def _resp_get(resp: Any, key: str, default: Any = None) -> Any:
    if isinstance(resp, dict):
        return resp.get(key, default)
    try:
        return resp[key]
    except Exception:
        return getattr(resp, key, default)


def apply_dashscope_langchain_patch() -> None:
    import langchain_community.chat_models.tongyi as chat_tongyi
    import langchain_community.llms.tongyi as llm_tongyi
    from requests import HTTPError

    def check_response(resp: Any) -> Any:
        code = _resp_get(resp, "status_code")
        if code == 200:
            return resp
        if code in (400, 401):
            raise ValueError(
                f"request_id: {_resp_get(resp, 'request_id')} \n "
                f"status_code: {code} \n "
                f"code: {_resp_get(resp, 'code')} \n message: {_resp_get(resp, 'message')}"
            )
        msg = (
            "DashScope 请求失败: "
            f"status_code={_resp_get(resp, 'status_code')!r} "
            f"code={_resp_get(resp, 'code')!r} "
            f"message={_resp_get(resp, 'message')!r} "
            f"request_id={_resp_get(resp, 'request_id')!r}"
        )
        # 403 等客户端拒绝对重试无意义，直接抛出以免 tenacity 反复打 API
        if code == 403:
            raise ValueError(msg)
        raise HTTPError(msg)

    llm_tongyi.check_response = check_response
    chat_tongyi.check_response = check_response
