# MCP 落地实施文档

## 1. 文档目标

本文档用于指导当前项目落地接入 MCP（Model Context Protocol），目标不是只说明“可以接”，而是给出一套可以直接实施的方案。

本方案默认优先实现：

1. 将当前项目改造成一个 **MCP Server**
2. 对外暴露现有 RAG、混合检索、时间、计算、天气、联网搜索等能力
3. 保留当前 `Streamlit + ReactAgent` 的运行方式，不破坏原有功能

如果后续还需要让本项目去调用外部 MCP Server，可在本方案完成后继续扩展 MCP Client 能力。

## 2. 为什么本项目适合接入 MCP

当前项目已经具备典型的工具型 Agent 架构，和 MCP 的抽象非常接近：

- `tools/agent_tool.py` 中已经定义了一组清晰的工具函数，并统一收敛在 `TOOLS` 列表中
- `tools/reactagent.py` 中的 `ReactAgent` 已经把模型和工具组织成可执行 Agent
- `rag/ragsummarize.py`、`rag/context_compressor.py`、`rag/vector_store.py` 等模块已经提供了核心知识检索能力
- 项目本身已经有清晰的配置分层，例如 `config/rag.yml`

也就是说，你现在缺的不是“工具能力”，而是一个 **MCP 协议适配层**。

## 3. 推荐落地方向

### 3.1 第一阶段：先做 MCP Server

推荐优先把本项目实现为 MCP Server，而不是先做 MCP Client。

原因：

- 当前项目已经拥有本地能力，最自然的第一步是把这些能力标准化暴露出去
- MCP Server 对当前架构侵入更小
- 做完后可直接被 Cursor、Claude Desktop、其他支持 MCP 的 Agent/IDE 消费
- 后续如果要做 MCP Client，可以在不推翻 Server 方案的前提下继续增加

### 3.2 服务边界

建议把 MCP Server 的职责限定为：

- 暴露工具
- 负责参数校验
- 调用现有业务模块
- 将结果以文本或结构化文本返回

不建议在第一版 MCP Server 中做的事情：

- 把整个 `ReactAgent` 直接封装成一个“大一统聊天工具”
- 在 MCP 层重写 RAG 逻辑
- 在 MCP 层重复实现配置系统
- 一上来就同时支持 Stdio、SSE、HTTP 三种传输

第一版建议只做 **Stdio Transport**，这是本地 IDE / Agent 集成最常见、最稳定的方式。

## 4. 当前项目与 MCP 的映射关系

### 4.1 现有工具映射

当前 `tools/agent_tool.py` 中的能力可以直接映射为 MCP tools：

| 当前能力 | 对应函数 | 是否建议暴露为 MCP Tool | 备注 |
|---------|---------|------------------------|------|
| RAG 问答 | `rag_summarize` | 是 | 核心能力 |
| RAG 检索 | `rag_retrieve` | 是 | 适合需要原文片段的场景 |
| 多路召回检索 | `hybrid_search` | 是 | 本地 + Web |
| 多路召回答复 | `hybrid_summarize` | 是 | 高价值能力 |
| 当前时间 | `get_local_datetime` | 是 | 低风险通用工具 |
| 算术计算 | `calculate_arithmetic` | 是 | 低风险通用工具 |
| 天气查询 | `get_weather_by_location` | 可选 | 依赖外网 |
| 地理编码 | `geocode_place` | 可选 | 依赖外网 |
| 联网搜索 | `web_search` | 是 | 演示价值高，但要提示外网依赖 |

### 4.2 不建议直接暴露的对象

以下对象不建议在第一版直接暴露为 MCP 资源或工具：

- `ReactAgent`
- Streamlit UI 状态
- `st.session_state.messages`
- Chroma 底层数据库文件
- 日志文件内容

原因是它们更偏“内部运行机制”，不适合作为稳定的 MCP 协议接口。

## 5. 建议的目录与文件改动

建议新增以下文件：

```text
agent/
├─ mcp_server.py                    # MCP Server 启动入口
├─ mcp/
│  ├─ __init__.py
│  ├─ server.py                     # MCP Server 构建逻辑
│  ├─ tool_adapter.py               # LangChain 工具 -> MCP tool 适配
│  ├─ schemas.py                    # 输入参数 schema（可选）
│  └─ handlers.py                   # 具体调用封装（可选）
└─ docs/
   └─ mcp_implementation_guide.md   # 本文档
```

如果你希望第一版尽量简单，也可以压缩成：

```text
agent/
├─ mcp_server.py
└─ mcp/
   ├─ __init__.py
   └─ server.py
```

## 6. 依赖方案

### 6.1 新增依赖

建议在 `requirements.txt` 中增加 MCP Python SDK。

推荐思路：

```txt
mcp
```

如果需要固定版本，建议在实际安装验证后再锁定，而不是先拍脑袋写死版本。

### 6.2 安装方式

```bash
pip install -r requirements.txt
pip install mcp
```

如果后续确认稳定，再把 `mcp` 回写进 `requirements.txt`。

## 7. 实施方案总览

### 7.1 总体设计

整体结构如下：

```text
MCP Client
   |
   |  MCP 协议调用
   v
mcp_server.py
   |
   v
mcp/server.py
   |
   +--> 将 MCP tool 调用映射到现有业务函数
           |
           +--> tools/agent_tool.py
                   |
                   +--> rag.ragsummarize / HybridRAG / web_search / weather ...
```

### 7.2 核心原则

1. **复用现有业务逻辑**，不要在 MCP 层复制一份 RAG 实现
2. **保持接口稳定**，MCP tool 名称尽量简洁明确
3. **入参显式化**，不要把所有参数都做成一个自由文本字符串
4. **输出尽量可读**，第一版以文本返回为主即可
5. **失败信息友好**，把网络失败、配置缺失、空查询等转成可理解的错误说明

## 8. 第一版建议暴露的 MCP Tools

建议第一版先暴露以下 6 个工具：

### 8.1 `rag_summarize`

- 用途：基于本地知识库生成答案
- 入参：
  - `query: str`
  - `dialogue_context: str = ""`

### 8.2 `rag_retrieve`

- 用途：仅返回检索片段，不做总结
- 入参：
  - `query: str`

### 8.3 `hybrid_search`

- 用途：本地知识库 + Web 多路召回检索
- 入参：
  - `query: str`

### 8.4 `hybrid_summarize`

- 用途：多路召回后生成答案
- 入参：
  - `query: str`

### 8.5 `web_search`

- 用途：联网搜索时效性信息
- 入参：
  - `query: str`
  - `max_results: int = 5`

### 8.6 `get_local_datetime`

- 用途：返回指定时区当前时间
- 入参：
  - `timezone_name: str = "Asia/Shanghai"`

其余如天气、地理编码、算术计算可以作为第二批工具补充。

## 9. 具体实现步骤

## 9.1 第一步：新增 MCP Server 启动入口

新增 `mcp_server.py`，只负责启动，不承载业务细节。

建议职责：

- 初始化 MCP Server
- 注册 tools
- 通过 stdio 启动

示例骨架：

```python
from mcp.server.stdio import stdio_server
from mcp.server import Server
from mcp.types import Tool

from mcp.server.models import InitializationOptions
from mcp.server.lowlevel import NotificationOptions

from mcp.server import Server

from mcp.server.stdio import stdio_server
```

上面只是说明依赖方向，实际代码建议收敛到 `mcp/server.py` 中，`mcp_server.py` 仅保留几行启动逻辑。

## 9.2 第二步：建立工具适配层

不要把 `tools/agent_tool.py` 中的 `@tool` 对象直接硬塞给 MCP。更稳妥的方式是建立一个适配层，把：

- MCP 的 `name`
- MCP 的 `description`
- MCP 的 `inputSchema`
- 真实执行函数

统一组织起来。

建议定义一个简单的数据结构：

```python
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class MCPToolSpec:
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., Any]
```

然后为每个工具提供显式 schema。

## 9.3 第三步：为每个工具定义 JSON Schema

MCP 工具的入参需要明确 schema。建议不要依赖自动推断，第一版直接手写，最稳。

示例：

```python
RAG_SUMMARIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "用户问题或检索查询"
        },
        "dialogue_context": {
            "type": "string",
            "description": "与本轮问题相关的极简对话上下文",
            "default": ""
        }
    },
    "required": ["query"]
}
```

`web_search` 则可以这样设计：

```python
WEB_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "搜索关键词"
        },
        "max_results": {
            "type": "integer",
            "description": "返回结果条数，建议 1-10",
            "default": 5,
            "minimum": 1,
            "maximum": 10
        }
    },
    "required": ["query"]
}
```

## 9.4 第四步：封装工具调用处理器

建议新增一层轻量 handler，而不是让 MCP Server 直接 import 并裸调业务函数。

这样做的好处：

- 可以统一做异常处理
- 可以统一做日志记录
- 可以对返回值做格式整理
- 后续可插入鉴权、监控、超时控制

建议风格：

```python
from tools.agent_tool import (
    rag_summarize,
    rag_retrieve,
    hybrid_search,
    hybrid_summarize,
    web_search,
    get_local_datetime,
)


def call_rag_summarize(query: str, dialogue_context: str = "") -> str:
    return rag_summarize.invoke(
        {
            "query": query,
            "dialogue_context": dialogue_context,
        }
    )
```

注意点：

- 如果你使用的是 LangChain `StructuredTool` / `Tool` 对象，通常应通过 `invoke()` 调用
- 不要假设 MCP 层一定知道 LangChain 工具对象内部结构
- 如果测试后发现直接调底层业务函数更稳，也可以改为直接 import 原始函数

## 9.5 第五步：注册 MCP Tools

在 `mcp/server.py` 中完成工具注册。

建议职责分离：

- `list_tools()`：返回可用工具定义
- `call_tool(name, arguments)`：按名称路由到具体 handler

示例骨架：

```python
from mcp.server import Server

server = Server("rag-agent")


@server.list_tools()
async def list_tools():
    return [
        {
            "name": "rag_summarize",
            "description": "基于本地向量知识库回答问题",
            "inputSchema": RAG_SUMMARIZE_SCHEMA,
        },
        {
            "name": "rag_retrieve",
            "description": "仅检索原文片段，不生成总结",
            "inputSchema": RAG_RETRIEVE_SCHEMA,
        },
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "rag_summarize":
        result = call_rag_summarize(
            query=arguments["query"],
            dialogue_context=arguments.get("dialogue_context", ""),
        )
        return [{"type": "text", "text": result}]

    raise ValueError(f"未知工具: {name}")
```

### 返回格式建议

第一版建议统一返回：

```python
[{"type": "text", "text": "..."}]
```

原因：

- 与现有工具返回的字符串天然兼容
- 成本最低
- 容易调试

如果以后要做更强的结构化输出，再扩展为多段 content 或 resource 引用。

## 9.6 第六步：增加启动命令

建议在项目说明或脚本中加入 MCP 启动方式。

例如：

```bash
python mcp_server.py
```

如果你后续要给 Cursor 配置本地 MCP Server，这个命令就是最直接的启动入口。

## 10. 与现有 Streamlit 方案如何共存

这部分很重要。MCP 落地后，不需要替换当前 `app.py`。

推荐关系如下：

- `app.py`：保留现有 Streamlit 交互页面
- `tools/agent_tool.py`：保留现有工具定义
- `tools/reactagent.py`：继续作为站内 Agent 执行器
- `mcp_server.py`：新增对外标准化接口

也就是：

- **UI 层** 继续跑当前网页
- **协议层** 新增 MCP
- **业务层** 继续复用原有工具与 RAG 逻辑

这是最稳、最少破坏的方案。

## 11. 推荐的最小代码组织方式

下面是一套更接近落地实现的最小结构示意：

```python
# mcp/server.py
from tools.agent_tool import (
    rag_summarize,
    rag_retrieve,
    hybrid_search,
    hybrid_summarize,
    web_search,
    get_local_datetime,
)


TOOL_SPECS = {
    "rag_summarize": {
        "description": "基于本地向量知识库回答问题",
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "dialogue_context": {"type": "string", "default": ""},
            },
            "required": ["query"],
        },
        "handler": lambda args: rag_summarize.invoke(
            {
                "query": args["query"],
                "dialogue_context": args.get("dialogue_context", ""),
            }
        ),
    },
    "rag_retrieve": {
        "description": "仅检索原文片段，不生成总结",
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
        "handler": lambda args: rag_retrieve.invoke({"query": args["query"]}),
    },
}
```

这类实现不够“优雅”，但足够稳定，适合第一版快速落地。

## 12. 配置与环境变量要求

MCP Server 运行时仍然依赖当前项目原有配置。

至少要保证以下条件成立：

### 12.1 模型环境变量

需要沿用现有模型配置逻辑，确保：

- `DASHSCOPE_API_KEY` 已配置
  或
- `TONGYI_API_KEY` 已配置

### 12.2 RAG 配置可读

确保以下配置文件可正常读取：

- `config/rag.yml`
- 与 Chroma / 数据库路径相关的配置

### 12.3 向量库准备完成

如果暴露 `rag_summarize`、`rag_retrieve`、`hybrid_search` 等工具，需要确保知识库已经完成入库，否则工具虽然能注册成功，但效果会很差甚至为空。

## 13. 异常处理建议

MCP 接入后，最容易影响体验的不是“工具不能调”，而是错误信息太生硬。

建议统一处理以下几类错误：

### 13.1 配置缺失

例如：

- API Key 未设置
- `config/rag.yml` 缺失
- Chroma 路径不存在

建议返回类似：

```text
工具执行失败：未检测到 DASHSCOPE_API_KEY，请先完成模型环境变量配置。
```

### 13.2 外网依赖失败

例如：

- DuckDuckGo 超时
- Open-Meteo 网络失败

建议返回类似：

```text
工具执行失败：联网搜索当前不可用，请检查网络连接或稍后再试。
```

### 13.3 业务无结果

例如：

- 向量检索无结果
- 地点无法解析

建议返回业务友好的正常文本，而不是抛异常。

## 14. 日志建议

建议 MCP 层也复用当前日志体系，至少记录：

- 工具名
- 入参摘要
- 耗时
- 成功 / 失败
- 错误原因

建议日志粒度：

```text
MCP 调用工具: rag_summarize args={"query": "..."}
MCP 工具完成: rag_summarize elapsed_ms=5321
MCP 工具失败: web_search error="搜索请求超时"
```

这样后续排查 Cursor / Claude Desktop 调用问题会容易很多。

## 15. Cursor 中的接入方式

当本项目具备 MCP Server 能力后，可以在支持 MCP 的客户端中注册它。

典型方式是配置一个本地命令型服务，例如：

```json
{
  "mcpServers": {
    "rag-agent": {
      "command": "python",
      "args": ["c:/Users/lenovo/Desktop/rag/agent/mcp_server.py"]
    }
  }
}
```

注意：

- Windows 下路径建议使用绝对路径
- 如果项目依赖虚拟环境，最好把 `python` 换成虚拟环境解释器绝对路径
- 如果工作目录敏感，可能还需要在启动脚本中显式切换到项目根目录

## 16. 验收清单

落地完成后，至少验证以下项目：

### 16.1 基础可用性

1. `python mcp_server.py` 可以正常启动
2. MCP Client 能成功列出 tools
3. `rag_summarize` 能被调用并返回文本
4. `rag_retrieve` 能返回检索片段
5. `web_search` 在有网时能返回结果

### 16.2 错误场景

1. 未配置 API Key 时错误信息清晰
2. 网络不可用时 `web_search` / 天气类工具错误可读
3. 空 query 时返回友好提示
4. 未知 tool 名称时返回明确错误

### 16.3 兼容性

1. `app.py` 原功能不受影响
2. Streamlit 页面仍可正常调用现有工具
3. 原有测试不因 MCP 新增而被破坏

## 17. 分阶段实施计划

建议按以下顺序推进：

### 阶段 A：最小可用版

- 增加 `mcp` 依赖
- 新增 `mcp_server.py`
- 暴露 2 个工具：
  - `rag_summarize`
  - `rag_retrieve`
- 用 Stdio 跑通

目标：先证明本项目已经能作为 MCP Server 被调用。

### 阶段 B：补齐高价值工具

- 增加：
  - `hybrid_search`
  - `hybrid_summarize`
  - `web_search`
  - `get_local_datetime`
- 完善错误处理和日志

目标：形成可用的 MCP 工具集。

### 阶段 C：增强可维护性

- 拆分 `schemas.py`
- 拆分 `tool_adapter.py`
- 增加更清晰的 handler 层
- 视情况补充测试

目标：让实现从“能跑”变成“可维护”。

## 18. 常见实现误区

### 18.1 误区一：直接把 `ReactAgent` 暴露成一个 MCP tool

不推荐。

原因：

- `ReactAgent` 自己还会调工具，容易形成嵌套复杂性
- 不利于 MCP Client 精细选择工具
- 难以做稳定 schema

第一版应该优先暴露“原子工具”，而不是暴露“完整 Agent”。

### 18.2 误区二：在 MCP 层重写一套业务逻辑

不推荐。

MCP 层应该只做协议转换和轻量封装，业务能力仍应来自现有模块。

### 18.3 误区三：第一版就做资源、提示词、采样、流式输出全家桶

不推荐。

第一版先把 tool 跑通，后续再扩展：

- resources
- prompts
- streaming
- client mode

## 19. 推荐的落地结论

对当前项目，最适合的 MCP 落地方案是：

1. **保留现有 Streamlit 和 LangChain Agent 架构**
2. **新增一个独立的 MCP Server 入口**
3. **将 `tools/agent_tool.py` 中的高价值工具做 MCP 映射**
4. **第一版先走 Stdio，本地可调通即可**
5. **后续再逐步扩展到更多工具和更强的结构化能力**

这条路径风险最低、复用最高、交付最快。

## 20. 建议的下一步

如果要从文档进入实现，建议直接按下面顺序动手：

1. 在 `requirements.txt` 中加入 `mcp`
2. 新增 `mcp_server.py`
3. 新增 `mcp/server.py`
4. 先注册 `rag_summarize` 和 `rag_retrieve`
5. 用本地 MCP Client 验证能否列出并调用工具
6. 再补 `hybrid_search`、`hybrid_summarize`、`web_search`

---

如果你希望，我下一步可以继续直接帮你把这份文档对应的 **MCP 最小可用实现代码** 一并落到项目里。
