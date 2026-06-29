# LangChain 基础学习指南

> 基于 docs.langchain.com 官方文档 v1.x 整理
> 生成日期：2026-06-27

---

## 学习路线总览

```
模块 1：三层架构         → 理解生态系统全貌
模块 2：Quickstart       → 第一个 Agent
模块 3：Model            → 模型的使用
模块 4：Message          → Agent 的通信语言
模块 5：Tool             → 给 Agent 的能力
模块 6：Agent            → 核心循环详解
模块 7：Streaming        → 流式输出
模块 8：Memory & Context → 记忆与上下文
模块 9：Structured Output → 结构化输出
```

> **建议顺序**：1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9
>
> 其中 2（Quickstart）可以跳过，你的 demo1.py 已经完成了同样的功能。

---

## 模块 1：三层架构

**文档地址**：https://docs.langchain.com/oss/python/concepts/products

### 核心概念

LangChain 生态分三个层次，层层叠加：

| 层次 | 名称 | 角色 | 典型场景 |
|------|------|------|---------|
| **Framework** | **LangChain** | 提供 Model、Tool、Message、Agent 等抽象 | 快速搭建、标准化开发 |
| **Runtime** | **LangGraph** | 持久化执行、流式输出、人在回路、状态管理 | 生产级、长时间运行 |
| **Harness** | **Deep Agents** | 内置文件系统、子代理、技能、上下文压缩 | 复杂多步任务 |

### 关键理解

```
Deep Agent = LangChain Agent + LangGraph Runtime + 内置中间件
```

- LangChain 提供 `create_agent`—— 轻量、可配置的 Agent 框架
- Deep Agents 提供 `create_deep_agent` —— 开箱即用的全功能 Agent
- LangGraph 是底层运行时，两者都构建在它之上
- LangSmith 用于追踪、调试和评估

### 什么时候用什么

| 工具 | 用在哪 |
|------|--------|
| LangChain (`create_agent`) | 简单场景，需要精细化控制 |
| Deep Agents (`create_deep_agent`) | 需要文件系统、子代理、上下文管理的复杂场景 |
| LangGraph | 需要底层控制、自定义编排、生产级部署 |

---

## 模块 2：Quickstart（快速开始）

**文档地址**：https://docs.langchain.com/oss/python/langchain/quickstart

### 最小 Agent

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="openai:gpt-5.5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)
```

### 对比：create_agent vs create_deep_agent

| 能力 | `create_agent` | `create_deep_agent` |
|------|---------------|-------------------|
| 工具调用 | ✅ | ✅ |
| 文件系统 | ❌ 需手动添加 | ✅ 内置 |
| 子代理 | ❌ 需手动添加 | ✅ 内置 |
| 上下文压缩 | ❌ 需手动添加 | ✅ 内置 |
| 技能系统 | ❌ 需手动添加 | ✅ 内置 |

---

## 模块 3：Model（模型）

**文档地址**：https://docs.langchain.com/oss/python/langchain/models

### 核心概念

Model 是 LLM 在 LangChain 中的封装。所有 Provider 使用统一的接口。

### 初始化模型

两种方式：

```python
# 方式一：init_chat_model（推荐，方便切换 Provider）
from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-5.5")

# 方式二：直接实例化
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-5.5")
```

模型命名格式：`{provider}:{model_name}`，如 `openai:gpt-5.5`、`anthropic:claude-sonnet-4-6`、`google_genai:gemini-3.5-flash`。

### 三个核心方法

```python
# 1. invoke — 单次调用，返回完整结果
response = model.invoke("Why do parrots talk?")

# 2. stream — 流式输出，逐 token 返回
for chunk in model.stream("Why do parrots talk?"):
    print(chunk.text, end="", flush=True)

# 3. batch — 批量并行处理
responses = model.batch([
    "Question 1",
    "Question 2",
    "Question 3",
])
```

### 调用配置（Invocation Config）

模型调用时可通过 `config` 参数传入 `RunnableConfig` 字典，运行时控制执行行为、回调和元数据追踪：

```python
response = model.invoke(
    "Tell me a joke",
    config={
        "run_name": "joke_generation",      # LangSmith 追踪标识
        "tags": ["humor", "demo"],          # 标签（继承到子调用）
        "metadata": {"user_id": "123"},     # 自定义元数据（继承）
        "callbacks": [my_callback_handler], # 回调处理器
        "max_concurrency": 5,               # batch 最大并行数
        "recursion_limit": 25,              # 链式递归深度上限
    }
)
```

对比 `config` 与 `context`：
| | `config` (`RunnableConfig`) | `context`（自定义） |
|--|----------------------------|-------------------|
| 用途 | LangChain 运行时配置 | 你的业务数据 |
| 内容 | tags, metadata, callbacks 等 | user_id, api_key 等 |
| 类型 | 固定结构 | 通过 `context_schema` 自定义 |
| 访问方式 | LangGraph 内部自动传递 | `runtime.context` |

> 完整参数见 [`RunnableConfig`](https://reference.langchain.com/python/langchain-core/runnables/config/RunnableConfig) 参考文档。

### 关键参数

| 参数 | 作用 | 默认值 |
|------|------|--------|
| `temperature` | 控制随机性（0=确定，1=随机） | 取决于 Provider |
| `max_tokens` | 最大输出长度 | 取决于 Provider |
| `max_retries` | 失败重试次数（指数退避） | 6 |
| `timeout` | 请求超时（秒） | 取决于 Provider |

### Tool Calling（绑定工具到模型）

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."

model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What's the weather like in Boston?")
# response.tool_calls → [{"name": "get_weather", "args": {"location": "Boston"}, "id": "call_..."}]
```

- `tool_choice="any"` 强制使用任意工具
- `tool_choice="tool_name"` 强制使用指定工具
- `parallel_tool_calls=False` 禁用并行工具调用

### 模型参数（进阶）

```python
model = init_chat_model(
    "claude-sonnet-4-6",
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
    max_retries=6,
)
```

### 连接弹性（Connection Resilience）

LangChain 自动对失败的 API 请求进行指数退避重试。默认重试 **6 次**，覆盖网络错误、限流(429)和服务端错误(5xx)。

401（未授权）、404 等客户端错误**不会**重试。

不稳定的网络环境下推荐提高 `max_retries`，并结合 checkpointer 在失败时保留进度：

```python
model = init_chat_model(
    "google_genai:gemini-3.5-flash",
    max_retries=10,   # 不稳定的网络环境（默认 6）
    timeout=120,      # 慢连接增加超时
)
```

> 长时间运行的 Agent 在不稳定网络上建议 `max_retries=10~15`。

### 高级流式（Advanced Streaming）

```python
# astream_events — 按事件类型过滤
async for event in model.astream_events("Hello"):
    if event["event"] == "on_chat_model_start":
        print(f"Input: {event['data']['input']}")
    elif event["event"] == "on_chat_model_stream":
        print(f"Token: {event['data']['chunk'].text}")
    elif event["event"] == "on_chat_model_end":
        print(f"Full: {event['data']['output'].text}")

# 自动流式（Auto-streaming）
# 在 Agent 中使用 model.invoke() 时，若环境处于流式模式，
# LangChain 自动切换到内部流式，通过回调系统触发 on_llm_new_token，
# 使 LangGraph 的 stream() / astream_events() 可以实时输出。
```

### Token 用量统计（Token Usage）

Provider 返回的 token 用量信息保存在 `AIMessage.usage_metadata` 中：

```python
response = model.invoke("Hello!")
print(response.usage_metadata)
# {'input_tokens': 8, 'output_tokens': 304, 'total_tokens': 312, ...}
```

跨多个模型追踪总量：

```python
from langchain_core.callbacks import UsageMetadataCallbackHandler

model_1 = init_chat_model("gpt-5.4-mini")
model_2 = init_chat_model("claude-haiku-4-5-20251001")

callback = UsageMetadataCallbackHandler()
model_1.invoke("Hello", config={"callbacks": [callback]})
model_2.invoke("Hello", config={"callbacks": [callback]})
print(callback.usage_metadata)
# {'gpt-5.4-mini': {'input_tokens': 8, ...}, 'claude-haiku-4-5-20251001': {...}}
```

或用上下文管理器：

```python
from langchain_core.callbacks import get_usage_metadata_callback

with get_usage_metadata_callback() as cb:
    model_1.invoke("Hello")
    model_2.invoke("Hello")
    print(cb.usage_metadata)
```

> 注意：OpenAI / Azure OpenAI 的流式场景需要主动 opt-in，详见 Provider 集成文档。

### Model Profiles（模型能力档案）

`model.profile` 返回字典，暴露模型支持的能力：

```python
model.profile
# {
#     "max_input_tokens": 400000,
#     "image_inputs": True,
#     "reasoning_output": True,
#     "tool_calling": True,
#     ...
# }
```

应用场景：
1. **Summarization Middleware** 根据上下文窗口大小决定是否触发摘要
2. **Structured Output** 自动推断原生支持（check `structured_output` 字段）
3. **输入门控** 根据 `image_inputs` / `max_input_tokens` 过滤输入
4. **模型切换器** Deep Agents Code 根据 `tool_calling` 和 text I/O 过滤可用模型

```python
# 覆盖 Profile 数据
custom_profile = {
    "max_input_tokens": 100_000,
    "tool_calling": True,
    "structured_output": True,
}
model = init_chat_model("...", profile=custom_profile)
```

> Model Profiles 是 Beta 特性（`langchain>=1.1`），格式可能变化。

---

## 模块 4：Message（消息）

**文档地址**：https://docs.langchain.com/oss/python/langchain/messages

### 核心概念

Message 是 Agent 通信的基本单位。每个消息包含 Role（角色）、Content（内容）、Metadata（元数据）。

### 四种消息类型

| 类 | Role | 用途 |
|------|------|------|
| `SystemMessage` | system | 系统指令，设置模型行为 |
| `HumanMessage` | user | 用户输入 |
| `AIMessage` | assistant | 模型回复（含工具调用） |
| `ToolMessage` | tool | 工具执行结果 |

### 基本用法

```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("Hello!"),
    AIMessage("Hi there! How can I help?"),
    HumanMessage("What's the weather?"),
]
response = model.invoke(messages)
```

或者用字典格式：

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]
```

### Content Blocks（新版标准化内容格式）

Message 的 `.content_blocks` 属性提供了跨 Provider 统一的格式：

```python
# 文本块
{"type": "text", "text": "Hello world"}

# 推理/思考链块
{"type": "reasoning", "reasoning": "用户想知道天气..."}

# 工具调用块
{"type": "tool_call", "name": "get_weather", "args": {"location": "SF"}, "id": "call_123"}

# 图片块
{"type": "image", "url": "https://..."}

# 文件块
{"type": "file", "url": "https://...", "mime_type": "application/pdf"}
```

### AIMessage 关键属性

```python
response = model.invoke("Hello")
response.text              # 文本内容
response.content           # 原始内容
response.content_blocks    # 标准化内容块列表
response.tool_calls        # 工具调用列表
response.usage_metadata    # Token 用量
response.response_metadata # 响应元数据
```

### ToolMessage

```python
from langchain.messages import ToolMessage

tool_message = ToolMessage(
    content="Sunny, 72°F",
    tool_call_id="call_123",  # 必须匹配 AIMessage 中的 tool_call id
    name="get_weather",
)
```

---

## 模块 5：Tool（工具）

**文档地址**：https://docs.langchain.com/oss/python/langchain/tools

### 核心概念

Tool 是给 Agent 的能力。Agent 通过 Tool 与外部世界交互（查数据库、搜网页、执行代码等）。

### 定义工具

```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
```

> **规则**：
> 1. 必须有类型注解
> 2. 必须有 docstring（给模型看的）
> 3. 工具名用 `snake_case`
> 4. 参数名不能是 `config` 或 `runtime`（保留字）

### 自定义工具属性

```python
@tool("web_search")  # 自定义名称
@tool(return_direct=True)  # 直接返回给用户，不再经过模型处理
```

### 高级 Schema

```python
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    location: str = Field(description="City name")
    units: str = Field(default="celsius", description="Temperature unit")

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius") -> str:
    """Get current weather."""
    ...
```

### 运行时上下文（ToolRuntime）

新版统一入口，替代旧的 `InjectedState`、`InjectedStore` 等：

```python
from langchain.tools import tool, ToolRuntime

@tool
def my_tool(runtime: ToolRuntime) -> str:
    # runtime.state           → 当前对话状态（短时记忆）
    # runtime.context         → 调用时传入的上下文（用户 ID 等，不可变）
    # runtime.store           → 持久化存储（长时记忆，跨对话）
    # runtime.stream_writer   → 实时流式更新
    # runtime.execution_info  → 线程 ID、运行 ID、重试次数
    # runtime.server_info     → 服务器信息（LangGraph Server 上运行时）
    # runtime.tool_call_id    → 当前工具调用的 ID
    ...
```

#### 访问 State（短时记忆）

```python
@tool
def get_last_user_message(runtime: ToolRuntime) -> str:
    """Get the most recent message from the user."""
    messages = runtime.state["messages"]
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return "No user messages found"
```

#### 更新 State

```python
from langgraph.types import Command
from langchain.messages import ToolMessage

@tool
def set_user_name(new_name: str, runtime: ToolRuntime) -> Command:
    """Set the user's name in the conversation state."""
    return Command(
        update={
            "user_name": new_name,
            "messages": [
                ToolMessage(
                    content=f"User name set to {new_name}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
```

#### 访问 Context（运行时上下文）

```python
from dataclasses import dataclass

@dataclass
class UserContext:
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id
    return f"User ID: {user_id}"
```

#### 访问 Store（长时记忆）

```python
@tool
def save_user_info(user_id: str, info: dict, runtime: ToolRuntime) -> str:
    """Save user info across conversations."""
    store = runtime.store
    store.put(("users",), user_id, info)
    return "Saved."
```

### 工具返回值类型

| 返回值 | 用途 |
|--------|------|
| `str` | 纯文本结果 |
| `dict` | 结构化数据 |
| `list[dict]` | 多模态内容（文本+图片等） |
| `Command` | 需要同时更新 State |

---

## 模块 6：Agent（代理）

**文档地址**：https://docs.langchain.com/oss/python/langchain/agents

### 核心概念

```
Agent = Model + Harness（工具 + 提示词 + 中间件）
```

**核心循环**：
```
用户输入 → 模型决定调用哪个工具 → 执行工具 → 结果返回模型 → 模型给出最终回答
```

### 创建 Agent

```python
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search, get_weather],
    system_prompt="You are a helpful assistant.",
)
```

### Agent 核心组件

| 参数 | 作用 | 文档 |
|------|------|------|
| `model` | 使用的模型 | Models 模块 |
| `tools` | 工具列表 | Tools 模块 |
| `system_prompt` | 系统提示词 | - |
| `response_format` | 结构化输出 | Structured Output 模块 |
| `checkpointer` | 对话历史持久化 | Memory 模块 |
| `context_schema` | 运行时上下文类型 | Context 模块 |
| `middleware` | 中间件列表 | - |
| `state_schema` | 自定义 State 类型 | - |

### 带记忆的对话

```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7

agent = create_agent(
    model="...",
    tools=[...],
    checkpointer=InMemorySaver(),  # 启用对话记忆
)

# 第一轮
config = {"configurable": {"thread_id": str(uuid7())}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather?"}]},
    config=config,
)

# 第二轮：复用 thread_id 保持上下文
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What about tomorrow?"}]},
    config=config,
)
```

### 中间件（Middleware）

中间件是扩展 Agent 能力的核心机制。`create_agent` 默认没有中间件，需要手动添加。

`create_deep_agent` 默认包含：
1. `FilesystemMiddleware` — 文件系统
2. `SummarizationMiddleware` — 上下文压缩
3. `SubAgentMiddleware` — 子代理
4. `TodoListMiddleware` — 任务管理
5. `MemoryMiddleware` — 记忆

### 常用中间件

```python
from deepagents.middleware import FilesystemMiddleware, SummarizationMiddleware
from deepagents.backends import StateBackend

agent = create_agent(
    model="...",
    tools=[...],
    middleware=[
        FilesystemMiddleware(backend=StateBackend()),
        SummarizationMiddleware(model=model, backend=backend),
    ],
)
```

### invoke vs stream_events

```python
# invoke — 简单调用，返回最终结果
result = agent.invoke({"messages": [{"role": "user", "content": "Hi"}]})

# stream_events — 流式调用，推荐方式（v3）
stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "What's the weather?"}]},
    version="v3",
)
for message in stream.messages:
    print(message.text)
final = stream.output
```

---

## 模块 7：Streaming（流式输出）

**文档地址**：https://docs.langchain.com/oss/python/langchain/streaming

### 核心概念

Streaming 让 Agent 在执行过程中实时输出进度，提升用户体验。

### 推荐方式：stream_events（v3）

```python
stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "What's the weather?"}]},
    version="v3",
)

# 遍历消息
for message in stream.messages:
    for token in message.text:
        print(token, end="", flush=True)

# 获取最终状态
final_state = stream.output
```

### 交错多种事件

```python
for kind, item in stream.interleave("messages", "tool_calls"):
    if kind == "messages":
        for token in item.text:
            print(token, end="", flush=True)
    elif kind == "tool_calls":
        print(f"\nTool: {item.tool_name}({item.input})")
        for delta in item.output_deltas:
            print(delta, end="", flush=True)
        print(f"\nResult: {item.output}")
```

### 三种流模式（v2）

| 模式 | 内容 | 用途 |
|------|------|------|
| `messages` | LLM token + 元数据 | 逐 token 显示 |
| `updates` | 每步状态更新 | 显示 Agent 执行进度 |
| `custom` | 自定义数据 | 工具中实时进度 |

```python
# v2 方式
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Hi"}]},
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        print(token.text, end="")
    elif chunk["type"] == "updates":
        print(f"\n[Step]: {chunk['data']}")
```

### 工具中自定义流

```python
from langgraph.config import get_stream_writer

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"
```

### 流式推理/思考链

```python
stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "What's the weather?"}]},
    version="v3",
)
for message in stream.messages:
    for token in message.reasoning:  # 思考链
        print(f"[thinking] {token}")
    for token in message.text:       # 最终回答
        print(token, end="", flush=True)
```

---

## 模块 8：Memory & Context（记忆与上下文）

**文档地址**：
- Memory：https://docs.langchain.com/oss/python/concepts/memory
- Context：https://docs.langchain.com/oss/python/concepts/context

### 核心概念

| 维度 | 短时记忆（Short-term） | 长时记忆（Long-term） |
|------|----------------------|---------------------|
| 作用域 | 单次对话（thread） | 跨对话 |
| 存储位置 | State（checkpointer 持久化） | Store（数据库） |
| 存什么 | 消息历史、中间结果 | 用户偏好、知识库 |
| 怎么访问 | `runtime.state` | `runtime.store` |

### 短时记忆（State）

通过 `checkpointer` 启用：

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="...",
    tools=[...],
    checkpointer=InMemorySaver(),
)

# thread_id 相同 = 同一对话
config = {"configurable": {"thread_id": "my-thread-1"}}
agent.invoke({"messages": [{"role": "user", "content": "Hi"}]}, config=config)
agent.invoke({"messages": [{"role": "user", "content": "还记得刚才吗？"}]}, config=config)
```

### 长时记忆（Store）

通过 `store` 参数启用：

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
agent = create_agent(
    model="...",
    tools=[...],
    store=store,  # 跨对话持久化
)

# 在工具中读写
@tool
def get_preference(runtime: ToolRuntime) -> str:
    store = runtime.store
    data = store.get(("preferences",), "user-123")
    return str(data.value) if data else "No preferences"
```

### 三种 Context

```python
from dataclasses import dataclass

@dataclass
class MyContext:
    user_id: str
    is_admin: bool

agent = create_agent(
    model="...",
    tools=[...],
    context_schema=MyContext,
)

# 传入运行时上下文（每次调用传入）
agent.invoke(
    {"messages": [{"role": "user", "content": "Hi"}]},
    context=MyContext(user_id="u123", is_admin=True),
)
```

| Context 类型 | 可变性 | 生命周期 | 访问方式 |
|-------------|--------|---------|---------|
| 静态运行时上下文 | ❌ 不可变 | 单次运行 | `context` 参数传入 |
| 动态运行时上下文（State） | ✅ 可变 | 单次运行 | `runtime.state` |
| 跨对话上下文（Store） | ✅ 可变 | 持久化 | `runtime.store` |

---

## 模块 9：Structured Output（结构化输出）

**文档地址**：https://docs.langchain.com/oss/python/langchain/structured-output

### 核心概念

让 Agent 返回格式化的结构化数据（Pydantic、TypedDict、JSON Schema），而不是自然语言。

### 基本用法（Pydantic）

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent

class Movie(BaseModel):
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    rating: float = Field(description="Rating out of 10")

agent = create_agent(
    model="gpt-5.5",
    response_format=Movie,  # 自动选择最佳策略
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Details about Inception"}]
})
print(result["structured_response"])
# Movie(title="Inception", year=2010, rating=8.8)
```

### 支持的 Schema 类型

| 类型 | 返回值 | 适用场景 |
|------|--------|---------|
| Pydantic `BaseModel` | Pydantic 实例 | 需要验证、嵌套结构 |
| `TypedDict` | dict | 简单场景 |
| `dataclass` | dict | 简单场景 |
| JSON Schema `dict` | dict | 跨语言兼容 |

### ProviderStrategy vs ToolStrategy

两种实现策略：

| 策略 | 原理 | 适用模型 |
|------|------|---------|
| `ProviderStrategy` | Provider 原生支持（最可靠） | OpenAI、Anthropic、Gemini 等 |
| `ToolStrategy` | 通过 Tool Calling 实现（通用） | 所有支持 Tool Calling 的模型 |

```python
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

# Provider 原生（自动选择，传 Schema 即可）
agent = create_agent(..., response_format=Movie)

# 手动指定 Provider 策略
agent = create_agent(..., response_format=ProviderStrategy(Movie))

# 手动指定 Tool 策略
agent = create_agent(..., response_format=ToolStrategy(Movie))
```

### ToolStrategy 错误处理

```python
ToolStrategy(
    schema=Movie,
    handle_errors=True,         # 默认：自动重试
    # handle_errors=False,      # 不重试，直接抛异常
    # handle_errors="自定义错误消息",  # 自定义重试消息
    # handle_errors=ValueError,      # 只对特定异常重试
    tool_message_content="数据已提取",  # 自定义工具消息
)
```

### Union 类型（多选一）

```python
from typing import Union

class ProductReview(BaseModel):
    rating: int
    sentiment: str

class Complaint(BaseModel):
    issue_type: str
    severity: str

agent = create_agent(
    ...,
    response_format=ToolStrategy(Union[ProductReview, Complaint]),
)
```

---

## 各模块与官方文档对照表

| 模块 | 页面 | URL |
|------|------|-----|
| 0. 学习路径 | Learn | https://docs.langchain.com/oss/python/learn |
| 1. 三层架构 | Frameworks, runtimes, and harnesses | https://docs.langchain.com/oss/python/concepts/products |
| 2. Quickstart | Quickstart | https://docs.langchain.com/oss/python/langchain/quickstart |
| 3. Model | Models | https://docs.langchain.com/oss/python/langchain/models |
| 4. Message | Messages | https://docs.langchain.com/oss/python/langchain/messages |
| 5. Tool | Tools | https://docs.langchain.com/oss/python/langchain/tools |
| 6. Agent | Agents | https://docs.langchain.com/oss/python/langchain/agents |
| 7. Streaming | Streaming | https://docs.langchain.com/oss/python/langchain/streaming |
| 8. Memory | Memory overview | https://docs.langchain.com/oss/python/concepts/memory |
| 8. Context | Context overview | https://docs.langchain.com/oss/python/concepts/context |
| 9. Structured Output | Structured output | https://docs.langchain.com/oss/python/langchain/structured-output |
| - | LangChain 总览 | https://docs.langchain.com/oss/python/langchain/overview |
| - | Deep Agents 总览 | https://docs.langchain.com/oss/python/deepagents/overview |

---

## 你的项目对应情况

| 你的文件 | 对应模块 | 状态 |
|---------|---------|------|
| `app/deepseek_model.py` | 模块 3：Model | ✅ |
| `app/qwen_model.py` | 模块 3：Model | ✅ |
| `app/demo1.py`, `app/agent_demo.py` | 模块 2/6：Agent | ✅ |
| `app/agent_message.py` | 模块 4：Message（content_blocks） | ✅ |
| `app/agent_web_tool.py` | 模块 5：Tool | ✅ |
| `app/agent_tool_runtime.py` | 模块 5：Tool（ToolRuntime） | ✅ |
| `app/agent_stream.py` | 模块 7：Streaming | ✅ |
| `app/agent_context.py` | 模块 8：Context | ✅ |
| `app/agent_mermory.py` | 模块 8：Memory（State） | ✅ |
| `app/agent_store_memory.py` | 模块 8：Memory（Store） | ✅ |
| `app/agent_structured_output.py` | 模块 9：Structured Output | ✅ |
| `app/agent_token_usage.py` | 模块 4：Message Token | ✅ |
| `app/agent_batch.py` | 模块 3：Model.batch() | ✅ |

现在 **9/9 个模块**的 demo 已全部覆盖，可以开始学习 Deep Agent 了。
