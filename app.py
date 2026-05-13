"""FinSight · 投研助理 Streamlit 入口。

财经主题 UI：深蓝 + 金的克制配色 + A 股涨跌习惯（红涨绿跌）。
仅在样式与文案上做投研化改造，对话与工具编排逻辑保持不变。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from memory.conversation_memory import ConversationMemoryManager
from model.model import chat_model
from rag.query_expand import (
    push_ui_query_expand_force_off,
    reset_ui_query_expand_force_off,
    take_query_expand_ui_records,
)
from tools.agent_tool import TOOLS
from tools.reactagent import ReactAgent
from utils.config_hander import agent_config, chroma_config, rerank_config
from utils.file_hander import get_file_md5
from utils.path_pool import get_abs_path


# ---------------------------------------------------------------------------
# 页面基本配置 + 主题 CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FinSight · 投研助理",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 财经主题配色：深蓝（信任）+ 金（强调）+ A 股红绿（涨跌）。
# 设计上克制使用色彩，避免与 Streamlit 默认风格冲突。
_THEME_CSS = """
<style>
:root {
    --finsight-primary: #0B3B6F;
    --finsight-primary-soft: #1B5EA0;
    --finsight-accent: #C8A464;
    --finsight-up: #D62F2F;
    --finsight-down: #0E8E50;
    --finsight-bg-soft: #F4F6FA;
    --finsight-border: #E1E5EC;
    --finsight-muted: #6B7280;
}

/* 主区留白：padding-top 需大于 Streamlit 顶栏高度，否则首屏品牌标题会被工具栏遮挡 */
.block-container {
    padding-top: 4.25rem;
    padding-bottom: 6rem;
    max-width: 1180px;
}

/* 顶部品牌区 */
.fs-brand {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 0.25rem;
}
.fs-brand-title {
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1.35;
    color: var(--finsight-primary);
    letter-spacing: 0.5px;
}
.fs-brand-en {
    font-size: 0.95rem;
    color: var(--finsight-muted);
    font-weight: 500;
    letter-spacing: 0.4px;
}
.fs-brand-tag {
    display: inline-block;
    padding: 0.1rem 0.55rem;
    margin-left: 0.4rem;
    font-size: 0.72rem;
    color: var(--finsight-accent);
    border: 1px solid var(--finsight-accent);
    border-radius: 999px;
    letter-spacing: 0.5px;
}
.fs-brand-sub {
    color: var(--finsight-muted);
    font-size: 0.92rem;
    margin-bottom: 1.0rem;
}

/* 模式徽章 */
.fs-mode-badge {
    display: inline-block;
    padding: 0.18rem 0.7rem;
    font-size: 0.78rem;
    border-radius: 999px;
    font-weight: 600;
    letter-spacing: 0.4px;
}
.fs-mode-chat   { background: #E8F1FB; color: var(--finsight-primary); border: 1px solid #BBD4ED; }
.fs-mode-report { background: #FFF6E0; color: #8C5A00;                 border: 1px solid #E6C56A; }

/* Hero 提问卡片（无消息时显示） */
.fs-hero {
    border: 1px solid var(--finsight-border);
    background: linear-gradient(180deg, #FAFCFF 0%, #F4F6FA 100%);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: 0.4rem 0 1.2rem 0;
}
.fs-hero h3 {
    color: var(--finsight-primary);
    margin: 0 0 0.4rem 0;
    font-size: 1.05rem;
}
.fs-hero p {
    color: var(--finsight-muted);
    font-size: 0.9rem;
    margin: 0;
}

/* 侧边栏统计指标卡 */
.fs-stat-card {
    display: flex;
    flex-direction: column;
    padding: 0.5rem 0.7rem;
    background: var(--finsight-bg-soft);
    border: 1px solid var(--finsight-border);
    border-radius: 8px;
}
.fs-stat-label {
    font-size: 0.72rem;
    color: var(--finsight-muted);
    letter-spacing: 0.3px;
}
.fs-stat-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--finsight-primary);
    font-feature-settings: "tnum";
}

/* 侧边栏「可用工具」单条 */
.fs-tool-item {
    padding: 0.45rem 0.6rem;
    margin-bottom: 0.35rem;
    background: #F8FAFD;
    border-left: 3px solid var(--finsight-primary-soft);
    border-radius: 4px;
}
.fs-tool-name {
    font-family: "JetBrains Mono", "Consolas", monospace;
    font-size: 0.82rem;
    color: var(--finsight-primary);
    font-weight: 600;
}
.fs-tool-desc {
    font-size: 0.75rem;
    color: var(--finsight-muted);
    margin-top: 0.15rem;
    line-height: 1.4;
}

/* 涨跌色工具类（后续展示行情时复用） */
.fs-up   { color: var(--finsight-up); font-weight: 600; }
.fs-down { color: var(--finsight-down); font-weight: 600; }

/* 底部免责声明 */
.fs-disclaimer {
    margin-top: 1.5rem;
    padding: 0.6rem 0.9rem;
    background: #FFF8E1;
    border-left: 3px solid var(--finsight-accent);
    border-radius: 4px;
    color: #6B5B2E;
    font-size: 0.78rem;
    line-height: 1.55;
}

/* 聊天气泡微调（避免气泡背景与卡片冲突） */
[data-testid="stChatMessage"] {
    background: transparent;
}
</style>
"""
st.markdown(_THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _get_bytes_md5(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def _human_size(num_bytes: int) -> str:
    f = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024:
            return f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}TB"


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


@st.cache_data(ttl=30, show_spinner=False)
def _kb_stats() -> dict:
    """统计本地知识库规模：文档数 / chunk 数 / 持久化大小。"""
    data_dir = Path(get_abs_path(chroma_config["database_path"]))
    chroma_dir = Path(get_abs_path(chroma_config["persist_directory"]))
    parent_store = Path(get_abs_path(
        chroma_config.get("parent_store_sqlite", "db/parent_store.sqlite")
    ))

    allowed = tuple(s.lower() for s in chroma_config.get(
        "allow_knowledge_file_types", [".txt", ".md", ".pdf"]
    ))
    doc_count = 0
    if data_dir.exists():
        non_corpus = {"readme.md", "readme.txt", ".gitkeep"}
        for p in data_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in allowed:
                continue
            if p.name.lower() in non_corpus:
                continue
            doc_count += 1

    chunk_count: int | None = None
    try:
        # 直接查 collection 的条数；失败时返回 None，由 UI 显示「-」
        from langchain_chroma import Chroma  # noqa: WPS433
        from model.model import embedding_model  # noqa: WPS433
        if embedding_model is not None and chroma_dir.exists():
            cli = Chroma(
                collection_name=chroma_config["collection_name"],
                embedding_function=embedding_model,
                persist_directory=str(chroma_dir),
            )
            chunk_count = cli._collection.count()  # noqa: WPS437
    except Exception:
        chunk_count = None

    return {
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "chroma_size": _path_size(chroma_dir),
        "parent_size": _path_size(parent_store),
    }


def _tool_brief(desc: str, max_len: int = 70) -> str:
    desc = (desc or "").replace("\n", " ").strip()
    return desc if len(desc) <= max_len else desc[:max_len].rstrip() + "…"


# ---------------------------------------------------------------------------
# 顶部品牌区
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="fs-brand">
        <span class="fs-brand-title">📈 FinSight · 投研助理</span>
        <span class="fs-brand-en">Equity Research Copilot</span>
        <span class="fs-brand-tag">DEMO</span>
    </div>
    <div class="fs-brand-sub">
        基于本地研报语料 + 实时网络的财经问答助手 ·
        多路召回 RAG · 结构化报告输出
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="font-size:1.05rem; font-weight:700; color:#0B3B6F;
                    letter-spacing:0.4px; margin-bottom:0.6rem;">
            ⚙️ 工作台
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- 模式切换 -----
    report_mode = st.toggle(
        "📋 报告模式（个股 / 行业 / 晨会）",
        value=False,
        help="开启后输出结构化投研报告（个股速评 / 行业速评 / 晨会纪要）；关闭则普通对话。",
    )
    if report_mode:
        st.markdown(
            '<span class="fs-mode-badge fs-mode-report">📋 报告模式 · 结构化输出</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="fs-mode-badge fs-mode-chat">💬 对话模式 · 自由问答</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ----- 知识库统计 -----
    st.markdown("**📊 投研语料库**")
    stats = _kb_stats()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="fs-stat-card">
                <div class="fs-stat-label">已入库文档</div>
                <div class="fs-stat-value">{stats['doc_count']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        chunk_str = (
            str(stats["chunk_count"]) if stats["chunk_count"] is not None else "-"
        )
        st.markdown(
            f"""
            <div class="fs-stat-card">
                <div class="fs-stat-label">向量切片</div>
                <div class="fs-stat-value">{chunk_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        f"Chroma {_human_size(stats['chroma_size'])} · "
        f"父块 {_human_size(stats['parent_size'])}"
    )

    st.divider()

    # ----- Query 扩展（配置说明，与 rag.yml 同步） -----
    qe_on = bool(rerank_config.get("query_expansion_enabled", False))
    qd_on = bool(rerank_config.get("query_decompose_enabled", False))
    qm_on = bool(rerank_config.get("query_decompose_with_expansion", False))
    with st.expander("🔍 Query 扩展（本地检索）", expanded=False):
        st.toggle(
            "使用 Query 扩展（本地检索）",
            key="query_expand_user_enabled",
            help=(
                "关闭后：即使 `config/rag.yml` 已开启扩写，本轮对话触发的本地检索也**只用单条原问**，"
                "不调用多查询/分解 LLM。开启时以配置文件为准。"
            ),
        )
        st.markdown(
            f"| 项 | 状态 |\n| --- | --- |\n"
            f"| 广度 · 多查询改写（`query_expansion_enabled`） | {'✅ 开启' if qe_on else '❌ 关闭'} |\n"
            f"| 深度 · 子问题分解（`query_decompose_enabled`） | {'✅ 开启' if qd_on else '❌ 关闭'} |\n"
            f"| 分解后再多查询（`query_decompose_with_expansion`） | {'✅ 开启' if qm_on else '❌ 关闭'} |\n"
        )
        st.caption(
            "仅作用于 **本地 Chroma 粗排**：可多条检索用语并行召回后合并去重；"
            "**Rerank 与最终总结仍使用用户原问**（含可选对话上下文）。"
            " **Hybrid 工具**下 Web 搜索仍只发 **一条原问**，不随扩展倍增。"
            " 上方开关可临时覆盖配置文件；修改 `config/rag.yml` 后请**重启** Streamlit。"
        )

    st.divider()

    # ----- 可用工具 -----
    with st.expander(f"🛠️ 可用工具（{len(TOOLS)} 个）", expanded=False):
        for t in TOOLS:
            name = getattr(t, "name", "?")
            desc = getattr(t, "description", "")
            st.markdown(
                f"""
                <div class="fs-tool-item">
                    <div class="fs-tool-name">{name}</div>
                    <div class="fs-tool-desc">{_tool_brief(desc, 90)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ----- 投研语料上传 -----
    st.markdown("**📁 上传投研语料**")
    st.caption("研报 / 年报 / 季报 / 公告 / 政策原文等。建议放至子目录：研报、公告、政策。")

    allowed_types = chroma_config.get("allow_knowledge_file_types", [".txt", ".pdf"])
    allowed_exts = [ext.lstrip(".") for ext in allowed_types]

    uploaded_files = st.file_uploader(
        "选择文件",
        type=allowed_exts,
        accept_multiple_files=True,
        help=f"支持：{', '.join(allowed_types)}。上传后请点击下方「执行入库」。",
        label_visibility="collapsed",
    )

    if uploaded_files:
        data_dir = Path(get_abs_path(chroma_config["database_path"]))
        data_dir.mkdir(parents=True, exist_ok=True)

        existing_files_md5: dict[str, str] = {}
        allowed_exts_tuple = tuple(allowed_types)
        for file_path in data_dir.rglob("*"):
            if file_path.is_file() and str(file_path).lower().endswith(allowed_exts_tuple):
                try:
                    existing_files_md5[get_file_md5(str(file_path))] = file_path.name
                except Exception:
                    pass

        saved_files: list[str] = []
        skipped_files: list[tuple[str, str]] = []
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            file_md5 = _get_bytes_md5(file_bytes)
            if file_md5 in existing_files_md5:
                existing_name = existing_files_md5[file_md5]
                if existing_name == uploaded_file.name:
                    skipped_files.append((uploaded_file.name, "内容已存在"))
                else:
                    skipped_files.append(
                        (uploaded_file.name, f"内容已存在于 {existing_name}")
                    )
                continue

            file_path = data_dir / uploaded_file.name
            try:
                with open(file_path, "wb") as f:
                    f.write(file_bytes)
                saved_files.append(uploaded_file.name)
                existing_files_md5[file_md5] = uploaded_file.name
            except Exception as e:
                st.error(f"保存 {uploaded_file.name} 失败: {e}")

        if saved_files:
            st.success(f"✅ 已保存 {len(saved_files)} 个新文件")
            st.caption("· " + " · ".join(saved_files))
        for name, reason in skipped_files:
            st.info(f"ℹ️ {name}：{reason}")

    if st.button("🚀 执行入库（向量化）", type="primary", use_container_width=True):
        with st.spinner("正在入库，研报较大可能需要数十秒..."):
            try:
                from rag.vector_store import VectorStoreService

                service = VectorStoreService()
                service.load_data()
                st.success("✅ 入库完成，新语料已加入向量库")
                _kb_stats.clear()  # 立即刷新指标卡
            except Exception as e:
                st.error(f"❌ 入库失败：{e}")
                st.info("提示：请确保已设置 DASHSCOPE_API_KEY 环境变量。")

    st.divider()

    # ----- 重置对话 -----
    if st.button("🔄 重置对话", use_container_width=True, help="清空当前对话历史与摘要记忆"):
        st.session_state.messages = []
        if "conversation_summary" in st.session_state:
            del st.session_state["conversation_summary"]
        if "memory_facts" in st.session_state:
            del st.session_state["memory_facts"]
        st.success("✅ 已重置对话")
        st.rerun()

    st.caption("数据来源：研报 / 年报 / 公告 / 政策语料及工具实时拉取。")


# ---------------------------------------------------------------------------
# 会话状态初始化
# ---------------------------------------------------------------------------

if "agent" not in st.session_state:
    st.session_state.agent = ReactAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None

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
        "memory_facts_enabled": agent_config.get("conversation_memory_facts_enabled", True),
        "memory_facts_max_chars": agent_config.get("conversation_memory_facts_max_chars", 800),
    },
)
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = memory_manager.init_summary_state()
if "memory_facts" not in st.session_state:
    st.session_state.memory_facts = memory_manager.init_memory_facts()
if "query_expand_user_enabled" not in st.session_state:
    st.session_state.query_expand_user_enabled = True


# ---------------------------------------------------------------------------
# Hero 区（无消息时显示）+ 三连击示例提问
# ---------------------------------------------------------------------------

EXAMPLE_PROMPTS_CHAT = [
    ("📈 实时行情", "查一下贵州茅台（600519）现在的股价 + 基本面"),
    ("📚 财经术语", "什么是 ROE？请用杜邦分析拆解一下"),
    ("🔬 行业研究", "HBM 在 AI 算力里起什么作用？谁是主要供应商？"),
]
EXAMPLE_PROMPTS_REPORT = [
    ("📋 个股速评", "帮我写一份英伟达（NVDA）的个股速评，带上最新行情和估值"),
    ("📋 行业速评", "帮我做一份 AI 算力产业链 2025 年的行业速评"),
    ("📋 晨会纪要", "帮我写一份新能源车行业的晨会纪要要点"),
]

if not st.session_state.messages:
    examples = EXAMPLE_PROMPTS_REPORT if report_mode else EXAMPLE_PROMPTS_CHAT
    mode_label = "报告模式" if report_mode else "对话模式"
    hero_hint = (
        "选择下方一个问题快速体验，或在底部输入框自由提问。"
        if not report_mode
        else "选择一种报告类型快速体验结构化输出。"
    )
    st.markdown(
        f"""
        <div class="fs-hero">
            <h3>👋 欢迎使用 FinSight 投研助理（当前：{mode_label}）</h3>
            <p>{hero_hint}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for i, (label, prompt_text) in enumerate(examples):
        with cols[i]:
            if st.button(
                f"{label}\n\n{prompt_text}",
                key=f"example_{i}",
                use_container_width=True,
            ):
                st.session_state.queued_prompt = prompt_text
                st.rerun()


# ---------------------------------------------------------------------------
# 历史消息回放
# ---------------------------------------------------------------------------

USER_AVATAR = "🧑‍💼"
ASSISTANT_AVATAR = "📈"

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# ---------------------------------------------------------------------------
# 输入框 / 一键示例提问处理
# ---------------------------------------------------------------------------

chat_input_placeholder = (
    "请输入投研问题，例：宁德时代 2024 年报关于钠离子电池产能的规划"
    if not report_mode
    else "请描述你需要的报告：如「写一份新能源车行业速评」"
)

prompt = st.chat_input(chat_input_placeholder)

# 处理一键示例：将示例提问按钮塞入输入路径
if prompt is None and st.session_state.queued_prompt:
    prompt = st.session_state.queued_prompt
    st.session_state.queued_prompt = None

if prompt:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        assistant_reply = ""
        with st.spinner("📊 正在检索语料、调用工具..."):
            try:
                history = st.session_state.messages
                summary_state = st.session_state.conversation_summary
                if memory_manager.should_compact(history, summary_state):
                    st.session_state.conversation_summary = memory_manager.update_summary(
                        history, summary_state
                    )
                    st.session_state.memory_facts = memory_manager.extract_facts(
                        st.session_state.memory_facts,
                        summary_text=st.session_state.conversation_summary.get("summary_text", ""),
                        old_messages_excerpt=memory_manager.get_messages_for_summary(history),
                    )
                memory_facts_text = memory_manager.format_memory_facts_text(
                    st.session_state.memory_facts
                )
                expand_off_token = None
                if not st.session_state.get("query_expand_user_enabled", True):
                    expand_off_token = push_ui_query_expand_force_off()
                try:
                    stream = st.session_state.agent.execute(
                        prompt,
                        conversation_history=history,
                        short_term_turns=memory_manager.recent_turns,
                        memory_summary=st.session_state.conversation_summary.get("summary_text", ""),
                        memory_facts_text=memory_facts_text or None,
                        report_mode=report_mode,
                    )
                    full = st.write_stream(stream)
                finally:
                    if expand_off_token is not None:
                        reset_ui_query_expand_force_off(expand_off_token)
                calls = getattr(st.session_state.agent, "last_tool_calls", None) or []
                if calls:
                    with st.expander(
                        f"🛠️ 本次调用了 {len(calls)} 个工具",
                        expanded=False,
                    ):
                        for i, c in enumerate(calls, start=1):
                            st.markdown(
                                f"**{i}.** `{c['name']}`",
                            )
                            st.caption("工具返回（摘录）")
                            st.code(c.get("content_preview", ""), language=None)
                qx_records = take_query_expand_ui_records()
                if qx_records:
                    with st.expander(
                        f"🔍 Query 扩展详情（本地检索 {len(qx_records)} 次）",
                        expanded=False,
                    ):
                        for i, rec in enumerate(qx_records, start=1):
                            st.markdown(f"**{i}. {rec.path_label}** · `{rec.path_key}`")
                            st.caption(rec.remark)
                            st.markdown("**检索输入（摘要）**")
                            st.code(rec.input_preview, language=None)
                            st.markdown(
                                f"**参与向量检索的用语（{len(rec.search_queries)} 条）**"
                            )
                            for j, sq in enumerate(rec.search_queries, 1):
                                disp = sq if len(sq) <= 400 else sq[:399] + "…"
                                st.text(f"{j}. {disp}")
                assistant_reply = full if isinstance(full, str) else "".join(full)
            except Exception as e:
                error_msg = str(e).lower()
                if "timeout" in error_msg or "timed out" in error_msg:
                    assistant_reply = "⏱️ 请求超时：模型响应时间过长，请稍后重试或简化问题。"
                elif "401" in error_msg or "authentication" in error_msg or "api key" in error_msg:
                    assistant_reply = "🔑 认证失败：请检查 DASHSCOPE_API_KEY 环境变量。"
                elif "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                    assistant_reply = "🚦 请求过于频繁：已触发限流，请稍等片刻再试。"
                elif "connection" in error_msg or "network" in error_msg or "urlopen" in error_msg:
                    assistant_reply = "🌐 网络连接问题：请检查网络或稍后再试。"
                elif "embedding" in error_msg or "embed" in error_msg:
                    assistant_reply = "🔧 向量模型未就绪：请确保已配置 DASHSCOPE_API_KEY 与依赖。"
                else:
                    assistant_reply = (
                        f"😅 服务暂时不可用，请稍后重试。\n\n（错误信息：{str(e)[:200]}）"
                    )
                st.error(assistant_reply)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})


# ---------------------------------------------------------------------------
# 底部免责声明（始终显示）
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="fs-disclaimer">
        ⚠️ <b>说明</b>：回答基于公开信息（研报 / 年报 / 公告 / 政策原文 / 实时网络）与本地语料；
        请自行核验关键数据与投资结论。本项目为个人作品集 demo，不用于商业用途。
    </div>
    """,
    unsafe_allow_html=True,
)
