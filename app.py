import streamlit as st

from tools.reactagent import ReactAgent

st.set_page_config(page_title="Agent Demo", page_icon="🤖")
st.title("Agent demo")
st.divider()

# 侧边栏：模式切换
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

if "agent" not in st.session_state:
    st.session_state.agent = ReactAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("输入你的问题")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream = st.session_state.agent.execute(
                prompt,
                conversation_history=st.session_state.messages,
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
    text = full if isinstance(full, str) else "".join(full)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": text})
