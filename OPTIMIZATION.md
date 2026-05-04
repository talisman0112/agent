# Agent 项目：可优化项与落地流程

本文汇总当前原型之后建议补齐的工程与体验项，并给出可按步骤执行的流程，便于迭代与多人协作。

---

## 一、依赖与可复制部署

**状态：已落地** — 仓库根目录已提供 [`requirements.txt`](requirements.txt)、[`requirements-optional.txt`](requirements-optional.txt)（Excel/PPT 等非结构化 Loader 可选用）、[`ENV.example`](ENV.example)。

### 优化目标

新环境能用同一套依赖跑通 Streamlit + Agent + Chroma + DashScope，减少「我这能跑你那报错」。

### 建议流程

1. **锁定依赖**：使用根目录 **`requirements.txt`**（已按当前项目 import 对齐并固定主版本）；若需 **`xlsx/xls/ppt/pptx`** 入库，叠加 **`requirements-optional.txt`**（`unstructured` 较重，按需安装）。
2. **标注版本**：当前为「可工作的固定主版本」；大版本升级时请在本机回归 `streamlit run app.py` 与 `python rag/vector_store.py load`。
3. **环境与密钥**：见 **`ENV.example`**（复制为 `.env` 或在终端设置变量）：
   - `DASHSCOPE_API_KEY`（或可选 `TONGYI_API_KEY`，与 `model/model.py` 读取逻辑一致）
   - 可选：模型名以 `config/rag.yml` 为准
4. **冒烟验证**（新机器 / 新 clone）：

   ```text
   cd <项目根目录>
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   REM 按需: pip install -r requirements-optional.txt
   set DASHSCOPE_API_KEY=***
   python tests/test_ragsummarize_terminal.py
   streamlit run app.py
   ```

---

## 二、配置驱动对话模型超参

**状态：已落地** — `config/rag.yml` 中可配置 `temperature`、`max_tokens`、`top_p`；`model/model.py` 在构造 `ChatTongyi` 时读取（`top_p` 为模型字段；`temperature` / `max_tokens` 经 `model_kwargs` 传入 DashScope `Generation`。某键不写或值为 `null` 则不传入，沿用集成默认值）。

### 优化目标

`config/rag.yml` 中的 `temperature`、`max_tokens`、`top_p` 等不仅写在注释里，而是真实传入 `ChatTongyi`。

### 建议流程

1. 在 `config/rag.yml` 中维护上述字段（当前仓库已为对话模型填入示例默认值）。
2. 已实现：`model/model.py` 从 `rag_config` 读取并传入 `ChatTongyi`。
3. 改参后**重启** Streamlit / 任一加载 `chat_model` 的进程（模块级单例在本进程启动时固化），再用同一问题对比风格与长度。


---

## 三、报告模式与主对话模式的切换

### 优化目标

`ReactAgent` 中 `context={"report": True}` 应与产品需求一致：`report_prompt` 与 `main_prompt` 由场景决定，而不是写死。

### 建议流程

1. 明确两种模式差异：主对话（`get_main_prompt()`）与报告（`get_report_prompt()`）分别对应哪些页面或按钮。
2. 在 `tools/reactagent.py` 的 `execute` 签名中增加可选参数，例如：`report_mode: bool = False`，在 `stream(..., context={"report": report_mode})` 中传入。
3. 在 `app.py` 用 `st.toggle` 或侧边栏选项设置 `report_mode`，调用 `agent.execute(..., report_mode=...)`。
4. 自测：切换开关后，抽查 `log/agent.log` 中 middleware 是否仍能正确切换（日志可辅助确认 prompt 策略）。

---

## 四、知识库运维（上传与重建索引）

### 优化目标

减少「手工拷贝到 `data/` 后忘记跑入库」导致的空检索或旧向量。

### 建议流程（当前以代码入口为主）

1. 将待入库文件放入 `config/chrome.yml`（或实际使用的 chroma 配置文件）中 `database_path` 所指目录（默认相对项目根的 `data`）。
2. 执行向量库入库。在项目根目录下（需已配置 `DASHSCOPE_API_KEY` 且依赖齐全）：

   ```text
   python rag/vector_store.py load
   ```

   传入 `ingest` 与 `load` 等价，均会执行 `VectorStoreService().load_data()`。

   若以模块方式调用，需注意 `rag/vector_store.py` 开头的 `sys.path` 注入；建议在仓库根目录用上述脚本入口。
3. 确认：`chroma` 持久化目录中存在数据，检索测试能返回非零条数。
4. （可选演进）在 Streamlit 增加「上传到 `data` + 一键触发 `load_data`」，并提示「大文件入库耗时」，避免重复点击。

---

## 五、Chroma 持久化路径与启动目录

### 优化目标

无论从哪个工作目录启动 `streamlit run app.py`，都指向同一向量库路径。

### 建议流程

1. 审查 `persist_directory` 是否为相对路径；若是，在项目根解析为绝对路径（例如通过 `utils/path_pool.get_abs_path`）。
2. 全局搜索 `persist_directory`、`Chroma(` 初始化处，保证只有一种解析规则。
3. 在日志或启动脚本中打印一次「实际持久化路径」，便于排障。

---

## 六、Streamlit 侧错误与用户提示

### 优化目标

模型超时、密钥无效、工具网络失败时，用户看到可读说明而非空白。

### 建议流程

1. 在 `app.py` 的 `execute`/`write_stream` 外层包 `try/except`，捕获常见异常类型（超时、401、429 等），将简短中文说明写入会话消息。
2. 对工具类错误：`ReactAgent.execute` 内可在 `finally` 或except 分支写一条 `logger.exception`，前端展示降级文案「服务暂时不可用，请稍后重试」。
3. 自测：临时填错 API Key、断网调用天气接口，验证 UI 与 `agent.log`。

---

## 七、日志与生产的平衡

### 优化目标

开发期有足够上下文；上线后避免整份 `state` 刷屏与敏感信息泄漏。

### 建议流程

1. 审视 `tools/mid_ware.py` 中 `before_model` / `after_model` 是否打印完整 `state`；生产可改为：只打消息条数、最后一条摘要、耗时。
2. 增加环境变量切换日志级别：`LOGLEVEL=DEBUG`（开发）与 `INFO`（默认）。
3. 定期轮转或限制 `log/agent.log` 体积（简单做法：运维侧 logrotate；代码侧可用 `logging.handlers.RotatingFileHandler`）。

---

## 八、外网工具（天气 / 地理）健壮性

### 优化目标

Open-Meteo 类接口偶发超时或限流时不拖垮整轮对话。

### 建议流程

1. 已为 URL 调用设置超时则保持；可统一常量 `REQUEST_TIMEOUT_SECONDS`。
2. 失败时返回短错误串（工具内已实现时可审查是否覆盖 `JSONDecodeError`、超时、DNS）。
3. 如需更高 SLA：替换为付费天气 API，工具层保持「输入地名 → 结构化结果」接口不变。

---

## 九、测试与简易评估

### 优化目标

改 prompt 或工具后快速回归。

### 建议流程

1. **`pytest`**：为 `calculate_arithmetic`、`get_local_datetime`（非法时区等）编写纯函数测试。
2. **Mock Agent**：对 `ReactAgent.execute` 在 CI 中用 mock LLM 或离线 stub，断言「给定用户句是否触发某工具」（解析 `updates` 或记录 `last_tool_calls`）。
3. **手动评估集**：维护 `tests/fixtures/sample_questions.md`（10～30 条），记录期望行为（必选工具、禁用幻觉等），发版前跑一轮人工勾选。

---

## 十、安全与算术工具（审计备忘）

### 优化目标

在面向不完全可信用户前，收紧工具边界。

### 建议流程

1. 复核 `calculate_arithmetic`：仅允许的 AST 节点类型，禁止 `eval` 裸执行。
2. 外网调用：服务端出口白名单或代理策略按单位安全规范执行。
3. RAG：`data/` 内容权限与租户隔离若在多用户场景需要，再在存储与检索侧分层（本阶段单用户 demo 可不实现）。

---

## 文档维护

随实现推进，建议在对应小节末尾增加「✅ 已完成」与生效日期（或指向 PR/issue），避免文档与代码长期漂移。
