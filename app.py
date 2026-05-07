import hashlib
import os
import shutil
from pathlib import Path

import streamlit as st

from memory.conversation_memory import ConversationMemoryManager
from model.model import chat_model
from tools.reactagent import ReactAgent
from utils.config_hander import agent_config, chroma_config
from utils.file_hander import get_file_md5
from utils.path_pool import get_abs_path


def _get_bytes_md5(file_bytes: bytes) -> str:
    """计算字节流的 MD5 哈希值。"""
    return hashlib.md5(file_bytes).hexdigest()

st.set_page_config(page_title="Agent Demo", page_icon="🤖")
st.title("Agent demo")
st.divider()

# 侧边栏：设置与文件上传
with st.sidebar:
    st.header("设置")
    report_mode = st.toggle(
        "📋 报告模式",
        value=False,
        help="开启后使用报告模式提示词（结构化、可沉淀的回答），关闭则使用常规对话模式"
    )
    if report_mode:
        st.caption("当前：报告模式 - 输出结构化、可沉淀的回答")
    else:
        st.caption("当前：对话模式 - 常规对话交互")
    st.divider()

    # 文件上传区域
    st.header("📁 知识库上传")

    # 获取允许的文件类型
    allowed_types = chroma_config.get("allow_knowledge_file_types", [".txt", ".pdf"])
    # 移除点号，因为 st.file_uploader 使用不带点的扩展名
    allowed_exts = [ext.lstrip(".") for ext in allowed_types]

    uploaded_files = st.file_uploader(
        "上传文件到知识库",
        type=allowed_exts,
        accept_multiple_files=True,
        help=f"支持格式：{', '.join(allowed_types)}。上传后需点击「入库」按钮。"
    )

    # 保存上传的文件到 data 目录（带 MD5 去重检测）
    if uploaded_files:
        data_dir = Path(get_abs_path(chroma_config["database_path"]))
        data_dir.mkdir(parents=True, exist_ok=True)

        # 扫描现有文件，建立 MD5 -> 文件路径 的映射
        existing_files_md5: dict[str, str] = {}
        allowed_exts = tuple(chroma_config.get("allow_knowledge_file_types", [".txt", ".pdf"]))
        for file_path in data_dir.iterdir():
            if file_path.is_file() and str(file_path).lower().endswith(allowed_exts):
                try:
                    file_md5 = get_file_md5(str(file_path))
                    existing_files_md5[file_md5] = file_path.name
                except Exception:
                    pass  # 忽略无法计算 MD5 的文件

        saved_files = []
        skipped_files = []
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            file_md5 = _get_bytes_md5(file_bytes)

            # 检查是否已存在相同内容的文件
            if file_md5 in existing_files_md5:
                existing_name = existing_files_md5[file_md5]
                if existing_name == uploaded_file.name:
                    skipped_files.append((uploaded_file.name, "文件内容已存在（与现有文件相同）"))
                else:
                    skipped_files.append((uploaded_file.name, f"内容已存在于 {existing_name}"))
                continue

            file_path = data_dir / uploaded_file.name
            try:
                with open(file_path, "wb") as f:
                    f.write(file_bytes)
                saved_files.append(uploaded_file.name)
                # 更新映射，避免同一批上传中重复保存相同内容
                existing_files_md5[file_md5] = uploaded_file.name
            except Exception as e:
                st.error(f"保存 {uploaded_file.name} 失败: {e}")

        if saved_files:
            st.success(f"✅ 已保存 {len(saved_files)} 个新文件到 data/ 目录")
            st.caption("文件列表: " + ", ".join(saved_files))
        if skipped_files:
            for name, reason in skipped_files:
                st.info(f"ℹ️ 跳过 {name}：{reason}")

    # 入库按钮
    if st.button("🚀 执行入库（向量化）", type="primary"):
        with st.spinner("正在入库，请稍候...大文件可能需要较长时间"):
            try:
                from rag.vector_store import VectorStoreService

                service = VectorStoreService()
                service.load_data()
                st.success("✅ 入库完成！新知识已加入向量库")
            except Exception as e:
                st.error(f"❌ 入库失败: {e}")
                st.info("提示：请确保已设置 DASHSCOPE_API_KEY 环境变量")

    st.divider()

if "agent" not in st.session_state:
    st.session_state.agent = ReactAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []

memory_manager = ConversationMemoryManager(
    llm=chat_model,
    config={
        "recent_turns": agent_config.get("conversation_recent_turns", 6),
        "summary_trigger_turns": agent_config.get("conversation_summary_trigger_turns", 12),
        "summary_increment_turns": agent_config.get("conversation_summary_increment_turns", 4),
        "max_history_tokens_before_summary": agent_config.get(
            "conversation_max_history_tokens_before_summary", 3000
        ),
        "summary_max_chars": agent_config.get("conversation_summary_max_chars", 1200),
    },
)
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = memory_manager.init_summary_state()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("输入你的问题")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        assistant_reply = ""
        with st.spinner("Thinking..."):
            try:
                history = st.session_state.messages
                summary_state = st.session_state.conversation_summary
                if memory_manager.should_compact(history, summary_state):
                    st.session_state.conversation_summary = memory_manager.update_summary(
                        history, summary_state
                    )
                stream = st.session_state.agent.execute(
                    prompt,
                    conversation_history=history,
                    short_term_turns=memory_manager.recent_turns,
                    memory_summary=st.session_state.conversation_summary.get("summary_text", ""),
                    report_mode=report_mode,
                )
                # Streamlit 1.28+：逐 token/片段写入
                full = st.write_stream(stream)
                calls = getattr(st.session_state.agent, "last_tool_calls", None) or []
                if calls:
                    with st.expander("本次调用的工具（可判断是否走了 RAG / 天气等）", expanded=False):
                        for i, c in enumerate(calls, start=1):
                            st.markdown(f"**{i}. `{c['name']}`**")
                            st.caption("工具返回（摘录）")
                            st.code(c.get("content_preview", ""), language=None)
                assistant_reply = full if isinstance(full, str) else "".join(full)
            except Exception as e:
                # 捕获常见异常类型，提供友好的中文错误提示
                error_msg = str(e).lower()
                if "timeout" in error_msg or "timed out" in error_msg:
                    assistant_reply = "⏱️ 请求超时：模型响应时间过长，请稍后重试或简化问题。"
                elif "401" in error_msg or "authentication" in error_msg or "api key" in error_msg:
                    assistant_reply = "🔑 认证失败：请检查 DASHSCOPE_API_KEY 环境变量是否正确设置。"
                elif "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                    assistant_reply = "🚦 请求过于频繁：已触发限流，请稍等片刻再试。"
                elif "connection" in error_msg or "network" in error_msg or "urlopen" in error_msg:
                    assistant_reply = "🌐 网络连接问题：请检查网络连接或稍后再试。"
                elif "embedding" in error_msg or "embed" in error_msg:
                    assistant_reply = "🔧 向量模型未就绪：请确保已配置 DASHSCOPE_API_KEY 并安装必要依赖。"
                else:
                    # 通用错误提示
                    assistant_reply = f"😅 服务暂时不可用，请稍后重试。\n\n（错误信息：{str(e)[:200]}）"
                st.error(assistant_reply)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
