# LangChain Tool 系统性学习指南

> 基于 docs.langchain.com Tools 章节 + 项目 demo 实践
> 目标：用体系化的思维理解 Tool 在 LangChain 中的完整面貌

---

## 目录

- [一、Tool 全景图](#一tool-全景图)
- [二、基础 Tool 定义](#二基础-tool-定义)
- [三、ToolRuntime 运行时上下文](#三toolruntime-运行时上下文)
- [四、三种返回值类型](#四三种返回值类型)
- [五、工具执行与错误处理](#五工具执行与错误处理)
- [六、动态 Tool 选择](#六动态-tool-选择)
- [七、流式输出（Stream Writer）](#七流式输出stream-writer)
- [八、Tool Execution Info（执行信息）](#八tool-execution-info执行信息)
- [九、长时记忆（Store）](#九长时记忆store)
- [十、设计思想总览](#十设计思想总览)
- [附录：项目文件映射表](#附录项目文件映射表)

---

## 一、Tool 全景图

### 1.1 Tool 在 LangChain 生态中的位置

```
LangChain Tool 架构
┌─────────────────────────────────────────────────────┐
│                   Agent（协调者）                      │
│  create_agent / create_deep_agent                   │
│  - 接收用户输入                                      │
│  - 决定何时调用 Tool                                  │
│  - 将 Tool 结果返回给 LLM                             │
├─────────────────────────────────────────────────────┤
│                   Tool（工具层）                       │
│  @tool 装饰器 / StructuredTool / BaseTool            │
│  - 将函数变为 LLM 可调用的工具                        │
│  - Type Hints + Docstring 定义 Schema                │
│  - ToolRuntime 访问运行时上下文                       │
├─────────────────────────────────────────────────────┤
│             ToolRuntime（运行时上下文）                 │
│  state / context / store / stream_writer /           │
│  execution_info / tool_call_id                       │
├─────────────────────────────────────────────────────┤
│               Model（推理引擎）                        │
│  LLM 生成 tool_calls → Agent 执行 → LLM 消费结果     │
└─────────────────────────────────────────────────────┘
```

### 1.2 Tool 的核心职责

```
┌─────────────────────────────────────────────────────────┐
│                    Tool                                  │
│                                                          │
│  本质: 带 Schema 的可调用函数                              │
│  功能: 让 LLM 能调用外部系统                               │
│                                                          │
│  核心特性:                                                 │
│  ├─ @tool 装饰器 — 函数 → 工具                            │
│  ├─ Type Hints — 参数 Schema（给 LLM 看）                 │
│  ├─ Docstring — 工具描述（给 LLM 看）                     │
│  ├─ ToolRuntime — 运行时注入（对 LLM 透明）               │
│  └─ Command — 修改 Agent 状态                             │
│                                                          │
│  Tool 不负责决定"何时调用"，只负责"如何执行"                │
│  决策是 LLM 的事，执行是 Tool 的事，协调是 Agent 的事      │
└─────────────────────────────────────────────────────────┘
```

### 1.3 学习路径

```text
基础层 → @tool 装饰器 → Type Hints + Docstring → args_schema
   │
运行时层 → ToolRuntime
   ├─ state（短时记忆，读写）
   ├─ context（调用上下文，只读）
   ├─ store（长时记忆，跨对话）
   ├─ stream_writer（流式进度推送）
   ├─ execution_info（执行元数据）
   └─ tool_call_id（工具调用标识）
   │
返回值层 → str → dict → multimodal → Command → return_direct
   │
进阶层 → 错误处理（wrap_tool_call）
       → 动态 Tool 选择（Filtering / Runtime Registration）
```

---

## 二、基础 Tool 定义

### 2.1 @tool 装饰器

Tool 的核心是 `@tool` 装饰器：将普通函数变为 LLM 可调用的工具。

**三大铁律（缺一不可）：**

```
① 必须有类型注解（Type Hints）
   └─ 定义 Tool 的输入 Schema，LLM 据此生成参数

② 必须有 Docstring
   └─ 定义 Tool 的描述，LLM 据此判断"什么时候用这个工具"

③ 函数体是 Python 执行的，不给 LLM 看
   └─ LLM 只看 Type Hints + Docstring + 函数名
```

```python
from langchain.tools import tool

@tool
def get_weather(city: str, limit: int = 10) -> str:
    """
    city of weather query tool,
    params:
    city: city name;
    limit: day count;
    """
    # 函数体不会发送给 LLM
    return f"{city}最近{limit}天的天气:..."
```

**📁 相关文件：** `app/tool/base_tool.py`

### 2.2 自定义 Tool Name & Description

```python
# 自定义 name + description（覆盖函数名和 docstring）
@tool(name_or_callable="weather_tool",
      description="city of weather query tool, params: city: city name; limit: day count;")
def get_weather2(city: str, limit: int = 10) -> str:
    """天气查询函数（这个 docstring 被 description 覆盖了）"""
    ...

print(get_weather2.name)  # weather_tool
```

| 参数 | 作用 | 说明 |
|------|------|------|
| `name_or_callable` | 自定义工具名 | 默认是函数名，建议 `snake_case` |
| `description` | 工具描述 | 默认是 docstring，显式传入会覆盖 |

> ⚠️ 工具名必须 `snake_case`，避免空格和特殊字符。某些 Provider 拒绝带空格的工具名。

### 2.3 args_schema：高级参数定义

当参数逻辑复杂时，用 Pydantic `BaseModel` 定义 Schema，替代 docstring 中的参数描述：

```python
from pydantic import BaseModel, Field
from typing import Literal

class Weather(BaseModel):
    location: str = Field(description="city name")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=Weather, name_or_callable="weather_plus",
      description="query temp in location and feature weather")
def get_weather_plus(location: str, units: str = "celsius",
                     include_forecast: bool = False) -> str:
    ...
```

**两种参数定义方式的对比：**

| 方式 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| Docstring 描述参数 | 简单直接，一行搞定 | 参数多时混乱 | 2-3 个简单参数 |
| Pydantic `args_schema` | 结构清晰，支持复杂类型（Literal、嵌套） | 多一个类定义 | 参数多的复杂工具 |

### 2.4 保留参数名

以下参数名被 LangChain **预留**，不能作为 Tool 的参数名：

| 保留名 | 用途 |
|--------|------|
| `config` | `RunnableConfig` 内部传递 |
| `runtime` | `ToolRuntime` 运行时注入 |

如果需要在 Tool 中访问运行时信息，使用 `runtime: ToolRuntime` 参数（详见第三章）。

### 2.5 Tool 的整体调用流程

```
Model 视角：
  HumanMessage → LLM → AIMessage(tool_calls) → ToolMessage → LLM → AIMessage

Agent 视角：
  agent.invoke(messages)
    → create_agent 自动绑定 tools
    → LLM 生成 tool_calls（决定"调哪个工具，传什么参数"）
    → Agent 执行对应工具函数
    → 工具返回 ToolMessage
    → 把 ToolMessage 放回消息列表
    → 再次调用 LLM（基于工具结果生成最终回答）
    → 重复直到 LLM 返回纯文本（无 tool_calls）
```

---

## 三、ToolRuntime 运行时上下文

### 3.1 ToolRuntime 全景

`ToolRuntime` 是 LangChain v1 引入的统一运行时访问接口。取代了旧版的 `InjectedState`、`InjectedStore`、`InjectedToolCallId` 等分散注入方式。

在 tool 函数中声明 `runtime: ToolRuntime` 参数，框架自动注入，**对模型完全透明**（模型看不到这个参数）。

```
ToolRuntime 的 6 个核心组件：

┌─────────────────────────────────────────────────────────────────────┐
│  runtime.state           短时记忆（state）    可读写，同 thread 共享   │
│  runtime.context         调用上下文           只读，每次 invoke 传入  │
│  runtime.store           长时记忆（store）    跨 thread 持久化        │
│  runtime.stream_writer   流式进度推送         只对 stream() 生效      │
│  runtime.execution_info  执行元数据           thread_id / run_id     │
│  runtime.tool_call_id    本次工具调用的 ID    用于追踪                │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 State — 短时记忆（可读写）

State 是 Agent 的**短时记忆**，同 `thread_id` 的多轮对话共享，由 LangGraph 自动维护。

```python
class UseState(AgentState):
    user_id: str
    user_preferences: dict

@tool(description="find user preferences")
def get_user_preference(runtime: ToolRuntime) -> str:
    messages = runtime.state["messages"]          # 对话历史
    user_prefs = runtime.state.get("user_preferences", {})  # 自定义字段
    return str(user_prefs)

agent = create_agent(
    model=qwen_model,
    state_schema=UseState,
    tools=[get_user_preference]
)

# invoke 时传入自定义 state 字段
response = agent.invoke({
    "messages": [HumanMessage("...")],
    "user_id": "abc",
    "user_preferences": {"lol": "League of Legends"}
})
```

**State 跨多轮会话的两个前提条件（缺一不可）：**

```
┌─ checkpointer ─────────────────────────────────────────────────┐
│  InMemorySaver() 或 PostgresSaver 等，负责保存 state 快照      │
│  无 checkpointer → 每次 invoke 都是全新的 state，之前全丢      │
└────────────────────────────────────────────────────────────────┘
┌─ thread_id ────────────────────────────────────────────────────┐
│  标识"哪场对话"，checkpointer 按 thread_id 存/取 state          │
│  无 thread_id → 每次都是新 thread，checkpointer 也找不到旧数据  │
└────────────────────────────────────────────────────────────────┘

总结: checkpointer 决定"能不能存"，thread_id 决定"存到哪/从哪取"
```

#### 更新 State 的方式

Tool 可以通过返回 `Command` 来更新 state（详见第四章）：

```python
from langgraph.types import Command
from langchain.messages import ToolMessage

class CounterState(AgentState):
    total: int
    last_action: str

@tool(description="处理订单")
def process_order(item: str, runtime: ToolRuntime) -> Command:
    return Command(
        update={
            "total": 1,
            "last_action": f"order:{item}",
            "messages": [
                ToolMessage(content=f"订单 {item} 已处理",
                            tool_call_id=runtime.tool_call_id)
            ],
        }
    )
```

#### Reducer（合并策略）

并发 Tool 调用时，多个 Tool 可能同时更新同一个 state 字段。Reducer 定义如何合并冲突：

```python
from typing import Annotated
from operator import add

def merge_strings(current: str, update: str) -> str:
    if not current:
        return update
    return f"{current}; {update}"

class CounterState(AgentState):
    total: Annotated[int, add]              # 累加
    last_action: Annotated[str, merge_strings]  # 拼接
```

| Reducer | 行为 | 适用场景 |
|---------|------|---------|
| `operator.add` | 数字累加、列表合并 | 计数、累加器 |
| 自定义函数 | 自定义合并逻辑 | 字符串拼接、优先级合并 |
| 无 Reducer | 并发更新报错 | 不允许并发操作的字段 |

**📁 相关文件：** `app/tool/tool_runtime.py`

### 3.3 Context — 调用上下文（只读）

Context 是每次 `invoke` 时传入的**不可变配置数据**，tool 只能读不能写。

```python
from dataclasses import dataclass

@dataclass
class UserInfo:
    user_id: str

@tool(description="find user info")
def find_user_info(runtime: ToolRuntime[UserInfo]) -> str:
    user_id = runtime.context.user_id  # 读取 context
    return f"user info: ..."

agent = create_agent(
    model=qwen_model,
    tools=[find_user_info],
    context_schema=UserInfo  # 声明 context 类型
)

response = agent.invoke(
    {"messages": HumanMessage("How user balance")},
    context=UserInfo(user_id="user123")  # 传入 context
)
```

#### Context vs State 对比

```
                    State（短时记忆）                      Context（上下文）
  ──────────────────────────────────────────────────────────────────────
  生命周期    整个 thread，跨多轮对话                      单次 invoke，每次都要传
  可变性      tool 可通过 Command(update=...) 修改          不可变，tool 只能读不能写
  用途        对话历史、中间计算结果、累加器                user_id、用户角色、API key、功能开关
  传参方式    invoke({"user_id": "abc", ...})               invoke(..., context=UserContext(user_id="abc"))
  类型声明    state_schema=UseState                         context_schema=UserContext
  tool 内访问 runtime.state["field"]                        runtime.context.field

  关键区别:
  ─────────
  1. state = 对话过程的数据（可读写），需要 reducer 处理并发更新
  2. context = 调用时就知道的配置（只读），每次 invoke 传入

  重要特性对比:
  ─────────────
  state 受 checkpointer + thread_id 管理:
    checkpointer 决定"能不能存"，thread_id 决定"存到哪/从哪取"
    同 thread 多次 invoke → state 跨轮次保持

  context 不受 checkpointer 管理，也不受 thread_id 影响:
    每次 invoke 都必须显式传入 context=...
    同 thread 的两次 invoke 之间，context 不会自动延续
    不传就取不到（runtime.context.xxx → 报错）
```

```python
# Context + State + Checkpointer 完整演示
from langgraph.checkpoint.memory import InMemorySaver

@dataclass
class UserContext:
    user_id: str
    role: str = "guest"

class ChatState(AgentState):
    user_name: str

checkpointer = InMemorySaver()

agent2 = create_agent(
    model=qwen_model,
    tools=[get_context, set_name],
    context_schema=UserContext,
    state_schema=ChatState,
    checkpointer=checkpointer,
)

# 第一轮: thread_a, 设置 user_name
r1 = agent2.invoke(
    {"messages": [HumanMessage("设置用户名为 小明")], "user_name": ""},
    config={"configurable": {"thread_id": thread_a}},
    context=UserContext(user_id="user_123", role="vip"),
)

# 第二轮: 新 thread_b, state 是全新的
r2 = agent2.invoke(
    {"messages": [HumanMessage("查我的 context")]},
    config={"configurable": {"thread_id": thread_b}},
    context=UserContext(user_id="user_456", role="admin"),
)

# 第三轮: 回到 thread_a, 上一轮的 state 还在
r3 = agent2.invoke(
    {"messages": [HumanMessage("查我的 context")]},
    config={"configurable": {"thread_id": thread_a}},
    context=UserContext(user_id="user_789", role="normal"),  # ← context 变了
)
```

**📁 相关文件：** `app/tool/tool_context.py`

### 3.4 Store — 长时记忆（跨对话）

Store 是 LangChain 的**长时记忆**系统，数据跨 thread、跨对话持久化（详见第九章）。

### 3.5 Stream Writer — 流式进度推送

详见第七章。

### 3.6 Tool Call ID — 每次调用的唯一标识

`runtime.tool_call_id` 是当前工具调用的唯一 ID，用于构建 `ToolMessage`：

```python
@tool
def my_tool(runtime: ToolRuntime) -> Command:
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="done",
                    tool_call_id=runtime.tool_call_id  # 必须匹配
                )
            ],
        }
    )
```

### 3.7 ToolRuntime 在 deepagents 中的完整演练

`agent_tool_runtime.py` 使用 `create_deep_agent` 完整演示了 ToolRuntime 的 5 个核心能力：

```python
# 1. runtime.state — 读取对话状态（短时记忆）
@tool
def check_history(runtime: ToolRuntime) -> str:
    messages = runtime.state["messages"]
    return f"对话中共有 {len(messages)} 条消息"

# 2. runtime.context — 读取运行时上下文
@tool
def get_user_info(runtime: ToolRuntime[UserContext]) -> str:
    ctx = runtime.context
    return f"用户 {ctx.user_id}（{'VIP' if ctx.is_vip else '普通'}用户）"

# 3. runtime.store — 长时记忆持久化
@tool
def save_note(key: str, content: str, runtime: ToolRuntime) -> str:
    runtime.store.put(("notes",), key, {"text": content})
    return f"笔记 '{key}' 已保存。"

# 4. runtime.stream_writer — 工具内实时流式输出
@tool
def step_by_step(task: str, runtime: ToolRuntime) -> str:
    writer = runtime.stream_writer
    writer(f"开始执行任务: {task}")
    writer(f"  → 步骤 1/3: 分析任务...")
    writer(f"  → 步骤 2/3: 处理数据...")
    writer(f"  → 步骤 3/3: 生成结果...")
    return f"任务 '{task}' 已完成。"

# 5. runtime.tool_call_id + Command — 更新 State
@tool
def increment_counter(runtime: ToolRuntime) -> Command:
    old_val = runtime.state.get("counter", 0)
    new_val = old_val + 1
    return Command(
        update={
            "counter": new_val,
            "messages": [ToolMessage(
                content=f"计数器: {old_val} → {new_val}",
                tool_call_id=runtime.tool_call_id,
            )],
        }
    )
```

**📁 相关文件：** `app/tool/agent_tool_runtime.py`

---

## 四、三种返回值类型

### 4.1 对比总览

```
Tool 返回值类型:
┌─────────────────────────────────────────────────────────────────────┐
│  1. 返回 str             — 最简单，最常见，给 LLM 提供事实          │
│  2. 返回对象 (dict/list)  — 可进一步解析的数据，或多模态内容         │
│  3. 返回 Command          — 修改 Agent 状态                         │
│  4. return_direct=True    — 跳过 LLM 后处理，直接返回给用户          │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 返回字符串

最常用方式，适合向模型提供事实或文本信息：

```python
@tool(description="query weather by cityname")
def get_weather(city: str) -> str:
    return f"weather is rain in {city}"
```

### 4.3 返回对象（dict / list）

当工具需要提供可进一步解析的数据时，返回 dict 更合适。也支持返回多模态内容：

```python
# 返回对象
@tool(description="screenshot tool")
def get_capture_screenshot() -> list[dict]:
    import base64
    with open("screenshot.jpg", "rb") as f:
        encode = base64.b64encode(f.read()).decode("utf-8")
    return [
        {"type": "text", "text": "this is screenshot of the current page"},
        {"type": "image", "base64": encode, "mime_type": "image/jpeg"}
    ]
```

### 4.4 返回 Command（修改 State）

当 Tool 不只是返回数据，还要修改 Agent 状态时，返回 `Command`：

```python
from langgraph.types import Command
from langchain.messages import ToolMessage

@tool(description="change user language")
def set_user_language(language: str, runtime: ToolRuntime) -> Command:
    return Command(
        update={
            "language": language,
            "messages": [
                ToolMessage(
                    content=f"please language {language} response user",
                    tool_call_id=runtime.tool_call_id
                )
            ],
        }
    )
```

### 4.5 return_direct=True（直接返回）

当 Tool 的输出已经是最终答案，不需要模型再加工时，使用 `return_direct=True`：

```python
@tool(return_direct=True, description="query order state")
def fetch_order_status(order_no: str) -> str:
    return f"order: {order_no} is finished"
```

| 特性 | 说明 |
|------|------|
| 性能优化 | 减少一次 LLM 调用（Tool 返回后不再调 LLM 加工） |
| Token 成本 | 降低，少一轮 LLM 交互 |
| 确定性 | 保证输出按原样返回，不被 LLM 重新组织语言 |
| 适用场景 | 查订单状态、查天气等"答案已确定"的场景 |

**📁 相关文件：** `app/tool/tool_return_values.py`

---

## 五、工具执行与错误处理

### 5.1 wrap_tool_call 中间件

LangChain 并不要求每个 Tool 自己处理异常，而是提供了统一的中间件机制，将异常处理、日志、重试、监控等横切关注点集中管理。

```python
from langchain.tools.tool_node import ToolCallRequest
from langchain.agents.middleware import wrap_tool_call
from collections.abc import Callable

@wrap_tool_call
def handle_tool(request: ToolCallRequest,
                handler: Callable[[ToolCallRequest], ToolMessage]) -> ToolMessage:
    """
    wrap_tool_call 中间件:
      - request:  包含 tool_call 信息的请求对象
      - handler:  链中下一个处理环节（最终执行工具函数）
    """
    try:
        return handler(request)
    except Exception as e:
        # 统一捕获异常，返回友好的 ToolMessage
        return ToolMessage(
            content=f"Tool error: cause by {e}",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model=qwen_model,
    tools=[chufajisuanqi],
    middleware=[handle_tool]
)

# 即使计算 10/0，也不会让 Agent 崩溃
response = agent.invoke({
    "messages": HumanMessage("使用除法计算器，计算10/0")
})
```

```
wrap_tool_call vs tool 内 try/except:

  wrap_tool_call 中间件                      Tool 内 try/except
  ──────────────────────                    ─────────────────────
  全局统一处理                                每个 Tool 各自处理
  横切关注点集中管理                            代码重复
  可叠加多个中间件                              改动影响单个 Tool
  AOP 思想（面向切面）                          侵入式处理

  推荐：开发阶段用 wrap_tool_call 全局兜底
       关键 Tool 内部再加一层精细化处理
```

**📁 相关文件：** `app/tool/tool_error_handling.py`

---

## 六、动态 Tool 选择

### 6.1 概念对比

LangChain 支持两种动态 Tool 选择方式：

```
动态 Tool 选择
├─ 方式一: Filtering pre-registered tools（过滤已注册工具）
│   ├─ 基于 State — 对话状态过滤（是否 VIP）
│   ├─ 基于 Store — 长时记忆过滤（用户权限） ← 最常用
│   └─ 基于 Context — 调用上下文过滤（用户角色）
│
└─ 方式二: Runtime tool registration（运行时注册）
    └─ 工具在运行时才被发现或创建（如从 MCP 服务器加载）
```

对比：

| 方式 | 工具是否预知 | 需要的 Hook | 典型场景 |
|------|------------|------------|---------|
| Filtering | 预注册所有工具 | `wrap_model_call` | 基于用户权限控制功能 |
| Runtime Registration | 运行时才知道 | `wrap_model_call` + `wrap_tool_call` | 从 MCP 加载、动态生成工具 |

### 6.2 基于 Store + Runtime Context 过滤（推荐方案）

```
核心思路:
  wrap_model_call 中间件在 LLM 被调用前拦截请求,
  从 Store（跨对话持久化存储）中读取用户权限/功能开关,
  用 request.override(tools=...) 替换 LLM 可见的 tool 列表。
  这样 LLM 根本不知道被过滤掉的 tool 存在。

完整流程:
  1. Store 预存用户权限数据
     store.put(('user',), 'abc123', ['vip_query'])

  2. Agent 创建时注册所有工具
     tools=[normal_query, vip_query]

  3. 中间件拦截 ModelRequest
     a. 从 request.runtime.context 读取 user_id
     b. 从 request.runtime.store 查询该用户的 tool 权限列表
     c. 筛选有权限的 tool
     d. 用 request.override(tools=filtered) 替换

  4. LLM 只能看到被授权的工具
```

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langgraph.store.memory import InMemoryStore

# 1. 定义工具
@tool(description="query vip user level")
def vip_query(name: str) -> str:
    return f"{name}, you is vip"

@tool(description="query normal user level")
def normal_query(name: str) -> str:
    return f"{name}, you is normal"

# 2. 初始化 Store，存储用户权限
store = InMemoryStore()
store.put(('user',), 'abc123', ['vip_query'])  # abc123 用户有 vip_query 权限

# 3. 定义 Context Schema
@dataclass
class UserInfo:
    user_id: str

# 4. 中间件：根据 Store 中的权限过滤 tool
@wrap_model_call
def model_handler(request: ModelRequest,
                  handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    store = request.runtime.store
    user_id = request.runtime.context.user_id
    tools = request.tools

    user_tool_names = store.get(('user',), user_id).value
    filtered_tools = [t for t in tools if t.name in user_tool_names]

    return handler(request.override(tools=filtered_tools))

# 5. 创建 agent
agent = create_agent(
    model=qwen_model,
    tools=[normal_query, vip_query],
    middleware=[model_handler],
    context_schema=UserInfo,
    store=store,
)

# 6. 调用
response = agent.invoke(
    {"messages": HumanMessage("query my level")},
    context=UserInfo(user_id="abc123"),
)
```

### 6.3 基于 State 过滤（简单方案）

从 `request.state` 读取对话状态决定工具可见性：

```python
@wrap_model_call
def model_handler(request: ModelRequest,
                  handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    state = request.state
    tools = request.tools

    if state["is_vip"]:
        filtered = [t for t in tools if t.name.startswith("vip")]
    else:
        filtered = [t for t in tools if not t.name.startswith("vip")]

    return handler(request.override(tools=filtered))
```

**📁 相关文件：** `app/tool/tool_dynamic_tool_selection_filter.py`

### 6.4 Runtime Tool Registration（运行时注册）

当工具在运行时才被发现或创建时（例如从 MCP 服务器加载、基于用户数据生成），需要同时处理工具的注册和执行。这需要两个 hooks：

```
wrap_model_call — 在 LLM 调用前将动态工具添加到 tool 列表
wrap_tool_call  — 拦截该工具的执行请求，分发给正确的实现

为什么需要 wrap_tool_call？
  Agent 需要知道如何执行不在原始工具列表中的工具。
  如果没有它，Agent 不知道如何调用动态添加的工具。
```

```python
from langchain.agents.middleware import AgentMiddleware

class DynamicToolMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        # 在 LLM 调用前，将动态工具注入
        update = request.override(tools=[*request.tools, get_weather])
        return handler(update)

    def wrap_tool_call(self, request, handler):
        # 拦截动态工具的执行请求
        tool_name = request.tool_call["name"]
        if tool_name == "get_weather":
            return handler(request.override(tool=get_weather))
        return handler(request)

agent = create_agent(
    model=qwen_model,
    tools=[jisuanqi],  # 只注册静态工具
    middleware=[DynamicToolMiddleware()],
)
```

**📁 相关文件：** `app/tool/tool_dynamic_tool_selection_runtime.py`

---

## 七、流式输出（Stream Writer）

### 7.1 stream_writer 的作用

`runtime.stream_writer` 提供在 Tool 执行过程中**向客户端推送实时进度**的能力。

```
stream_writer 不是普通 print()，它不会输出到控制台。
它的设计目标：在 tool 执行过程中，向流式客户端推送实时进度。

典型场景（Web 服务）:
  用户发起一个耗时的工具调用（如查数据库、调 API），
  stream_writer 可以分阶段推送：
    "正在查询..." → "解析数据中..." → "完成"
  前端（如 websocket / SSE）收到这些中间状态并展示给用户，
  而不是让用户干等直到最终结果返回。
```

### 7.2 使用方式

```python
@tool(args_schema=Weather, name_or_callable="weather_plus",
      description="query temp in location and feature weather")
def get_weather_plus(runtime: ToolRuntime, location: str,
                     units: str = "celsius",
                     include_forecast: bool = False) -> str:
    writer = runtime.stream_writer

    # stream_writer 写入流管道，非控制台
    writer(f"tool start, Looking up data for {location}")

    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"

    writer("tool end")
    return result
```

### 7.3 消费方式

```
agent.stream() / astream() / astream_events()
  └─ stream_writer 写入的信号出现在流事件管道中
  └─ agent.invoke() 时 stream_writer 静默丢弃（因为没有活跃的流）

关键理解：
  1. stream_writer ≠ print()
  2. 只在流式调用期间生效
  3. 写入内容可通过 astream_events 的 custom event 捕获
```

```python
# agent.stream() 能看到 stream_writer 输出
for chunk in agent.stream({
    "messages": HumanMessage("what it is temp today in dezhou and feature 5 day weather")
}):
    if "__end__" not in chunk:
        for node, output in chunk.items():
            print(f"[{node}]: {output}")
```

**📁 相关文件：** `app/tool/tool_stream.py`、`app/tool/agent_tool_runtime.py`（第 4 个示例）

---

## 八、Tool Execution Info（执行信息）

### 8.1 execution_info 提供的信息

`runtime.execution_info` 提供当前工具调用的执行元数据：

```python
@tool(args_schema=Weather, name_or_callable="weather_plus",
      description="query temp in location and feature weather")
def get_weather_plus(runtime: ToolRuntime, location: str,
                     units: str = "celsius",
                     include_forecast: bool = False) -> str:
    info = runtime.execution_info
    print(f"Thread: {info.thread_id}, Run: {info.run_id}")
    print(f"Attempt: {info.node_attempt}")
    # Thread: uuid7 — 对话线程唯一 ID，区分不同对话
    # Run: uuid7   — 单次 agent invoke 的唯一 ID，追踪完整调用链路
    # Attempt: 0   — 当前节点重试次数，调试重试/失败场景
    ...
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `thread_id` | `str` | 对话线程唯一 ID，区分不同对话 |
| `run_id` | `str` | 单次 agent invoke 的唯一 ID，追踪完整调用链路 |
| `node_attempt` | `int` | 当前节点重试次数，调试重试/失败场景 |

```python
# 在 invoke 时设置 run_id 和 thread_id
config = {
    "run_id": str(uuid7()),
    "configurable": {"thread_id": str(uuid7())}
}

response = agent.invoke({...}, config=config)
```

> 需要 `deepagents>=0.5.0`（或 `langgraph>=1.1.5`）

**📁 相关文件：** `app/tool/tool_execution_info.py`

---

## 九、长时记忆（Store）

### 9.1 为什么需要 Store？

```
短时记忆（State + checkpointer）只在同一个 thread_id 内有效
  └─ 换一个 thread_id，之前的 state 就找不到了
  └─ checkpointer 重启，内存数据全丢

长时记忆（Store）跨对话、跨 thread 持久化数据
  └─ 用户偏好、知识库、历史行为记录
  └─ 不受 thread_id 影响，不受 checkpointer 影响
```

### 9.2 Store 核心概念

```
Store 的数据结构：namespace + key 两级
───────────────────────────────────

namespace = ("user",)        → 类似文件夹
key       = "abc123"         → 类似文件名
value     = {"name": "jack"} → 文件内容

store.put(("user",), "abc123", {"name": "jack"})
           ↑namespace  ↑key      ↑value

store.get(("user",), "abc123")
           ↑namespace  ↑key      → Item 对象，.value 取值
```

### 9.3 Tool 中使用 Store

```python
from langgraph.store.memory import InMemoryStore

long_save = InMemoryStore()

@tool(description="query user info by user_id")
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    user_info = runtime.store.get(("user",), user_id)
    if user_info:
        return str(user_info)
    return "unknown user"

@tool(args_schema=Userinfo, description="save user info")
def save_user_info(user_id: str, user_info: dict, runtime: ToolRuntime) -> str:
    runtime.store.put(("user",), user_id, user_info)
    return "successed save user info"

agent = create_agent(
    model=qwen_model,
    store=long_save,
    tools=[get_user_info, save_user_info]
)
```

### 9.4 Store 跨 Thread 验证

```python
import uuid

thread_a = str(uuid.uuid7())
thread_b = str(uuid.uuid7())

# thread_a 中保存数据
agent.invoke(
    {"messages": HumanMessage("save my userinfo: my name is jack, ...")},
    config={"configurable": {"thread_id": thread_a}}
)

# thread_b 中读取 — Store 不受 thread_id 影响，依然能读到
response = agent.invoke(
    {"messages": HumanMessage("get user info for user with id 'abc123'")},
    config={"configurable": {"thread_id": thread_b}}
)
print(response["messages"][-1].content)  # 成功读取
```

### 9.5 InMemoryStore vs 生产级 Store

| 存储 | 适用场景 | 特点 |
|------|---------|------|
| `InMemoryStore` | 开发测试/实验 | 内存存储，重启丢失 |
| `PostgresStore` | 生产环境 | 数据持久化到数据库 |

**📁 相关文件：** `app/tool/tool_store.py`

---

## 十、设计思想总览

### 10.1 LangChain Tool 的核心原则

```
① 函数是代码，Schema 是给 LLM 的说明书
   ─────────────────────────
   Type Hints → 参数类型和结构（LLM 据此生成参数）
   Docstring  → 工具的用途说明（LLM 据此判断何时调用）
   函数体     → 实际执行逻辑（LLM 看不到，也不需要看）
   @tool 装饰器 → 把函数变成 LLM 可理解的工具

② 模型不负责执行，只负责决策
   ─────────────────────────
   LLM 生成 tool_calls → 决定"调哪个工具，传什么参数"
   Agent 执行工具函数 → 处理实际业务逻辑
   LLM 是大脑，Tools 是手脚，Agent 是协调者

③ 运行时注入对 LLM 透明
   ─────────────────────────
   runtime: ToolRuntime 参数在 @tool 中被自动隐藏
   LLM 不需要知道 state、context、store 的存在
   它只关心业务参数（city、limit、location...）

④ Tool 的三种形态
   ─────────────────────────
   返回字符串  → 给 LLM 提供信息
   返回 Command → 修改 Agent 状态
   return_direct → 直接给用户最终答案（跳过 LLM 加工）

⑤ 横切关注点集中管理
   ─────────────────────────
   wrap_tool_call → 异常处理、日志、重试统一处理
   wrap_model_call → 动态 Tool 选择、权限控制
   而不是在每个 Tool 里重复写 try/except
```

### 10.2 Tool 各组件之间的关系

```
                       Tool（@tool 装饰器）
                            │
            ┌───────────────┼───────────────┐
            │               │               │
       Type Hints       Docstring      args_schema
       （参数 Schema）   （工具描述）    （高级参数定义）
            │
            ▼
       ToolRuntime（运行时注入，对 LLM 透明）
            │
    ┌───────┼───────┬───────┬──────┬──────┐
    │       │       │       │      │      │
  state  context store  stream  exec  tool_call
  (可读写) (只读) (持久) (writer) (info)  (id)
            │
            ▼
       返回值类型
    ┌───────┼───────┬──────┐
    │       │       │      │
   str    dict/list Command return_direct
```

### 10.3 本项目各 demo 的定位

```
基础定义:   base_tool.py                    ← @tool 装饰器入门
            __init__.py                     ← 空包文件

Runtime:    tool_runtime.py                 ← state 读写 + reducer
            tool_context.py                 ← context + state + checkpointer
            agent_tool_runtime.py           ← ToolRuntime 6 组件完整演练（deepagents）

返回值:     tool_return_values.py           ← str / dict / Command / return_direct

流式:       tool_stream.py                  ← stream_writer 用法

错误处理:   tool_error_handling.py          ← wrap_tool_call 统一异常处理

动态选择:   tool_dynamic_tool_selection_filter.py   ← Store + Context 过滤
            tool_dynamic_tool_selection_runtime.py  ← 运行时注册

执行信息:   tool_execution_info.py          ← execution_info 元数据

长时记忆:   tool_store.py                   ← Store 跨 thread 持久化
```

### 10.4 常见陷阱与注意事项

```
❶ 参数名不能用 runtime / config（保留字）
   → 运行时注入 ToolRuntime 用 runtime: ToolRuntime 参数
   → config 是 RunnableConfig 内部传递的保留参数名

❷ args_schema 的字段名要和函数参数名一致
   → Pydantic 模型字段名 ≠ 函数参数名 → 运行时匹配失败

❸ Docstring 要简洁、抓住重点
   → LLM 上下文窗口有限，太长的 docstring 浪费 token
   → 但太短又不够模型理解工具用途

❹ state 跨 thread 需要 checkpointer + thread_id
   → checkpointer 决定"能不能存"
   → thread_id 决定"存到哪/从哪取"
   → 缺一不可

❺ context 不受 checkpointer 管理
   → 每次 invoke 都必须显式传入
   → 同 thread 的两次调用之间 context 不会延续
   → 不传就取不到

❻ return_direct 需谨慎使用
   → Tool 返回直接 → 跳过 LLM 后处理
   → 适合"答案已确定"的场景（查订单、查天气）
   → 不适合需要 LLM 重新组织语言的场景

❼ stream_writer 只在 stream() 下生效
   → invoke() 时写入的内容静默丢弃
   → 需要客户端用 stream_events 消费
```

---

## 附录：项目文件映射表

| 你的文件 | 对应知识点 | 推荐阅读顺序 |
|---------|-----------|------------|
| `app/tool/base_tool.py` | @tool 装饰器 + name/description + args_schema | ① |
| `app/tool/tool_runtime.py` | state 读写 + Command 更新 + reducer | ② |
| `app/tool/tool_context.py` | context 只读 + context+state+checkpointer 对比 | ② |
| `app/tool/agent_tool_runtime.py` | ToolRuntime 6 组件完整演练（deepagents） | ② |
| `app/tool/tool_return_values.py` | str / dict / multimodal / Command / return_direct | ③ |
| `app/tool/tool_stream.py` | stream_writer 流式进度推送 | ④ |
| `app/tool/tool_execution_info.py` | execution_info 执行元数据 | ④ |
| `app/tool/tool_store.py` | Store 长时记忆 + 跨 thread 持久化 | ⑤ |
| `app/tool/tool_error_handling.py` | wrap_tool_call 统一异常处理 | ⑥ |
| `app/tool/tool_dynamic_tool_selection_filter.py` | 动态 Tool 过滤（Store+Context） | ⑦ |
| `app/tool/tool_dynamic_tool_selection_runtime.py` | 动态 Tool 注册（运行时） | ⑦ |

---

> 本文档是 `left_tool.md`，与 `left_cycle.md` 属于同一系列文档。
>
> 核心原则：
> - 函数体是写给人看的，Type Hints + Docstring 是写给 LLM 看的
> - LLM 只做决策，不做执行
> - Tool 的运行时注入对 LLM 完全透明
> - 横切关注点（异常、权限、日志）通过中间件集中管理
