# 短期记忆管理：Trim / Delete / Summarize

> 官方文档：https://docs.langchain.com/oss/python/langchain/short-term-memory  
> 参考代码：`trim_message.py` · `delete_message.py` · `summarize_message.py`

---

## 一、背景

Agent 的短期记忆存储在 `state["messages"]` 中，由 checkpointer 持久化。随着对话进行，消息列表不断增长，会面临 LLM 上下文窗口超限、Token 费用增加、响应变慢等问题。

LangChain 提供了三种内置策略来解决：

```
短期记忆管理
    │
    ├── Trim（裁剪）     → 保留开头 + 末尾若干条
    ├── Delete（删除）   → 按 ID 删除指定消息
    └── Summarize（摘要） → 用 LLM 生成摘要替换旧消息
```

---

## 二、Trim Messages（批量裁剪）

在 `@before_model` 中，保留第一条消息和末尾若干条，删掉中间过期的内容。

### 官方示例

```python
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state["messages"]
    if len(messages) <= 3:
        return None
    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }
```

| 要素 | 说明 |
|------|------|
| 时机 | `@before_model`（LLM 调用前） |
| 方式 | `REMOVE_ALL_MESSAGES` 清空 + 重新 Add |
| 效果 | System + 末尾 N 条保留，中间丢弃 |

### 项目代码对照（`trim_message.py`）

按内容过滤而非按长度裁剪，但 Replace 模式相同：

```python
@before_model
def trim_message(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state['messages']
    if len(messages) > 0:
        new_messages = [m for m in messages
                        if d_name not in m.content and s_name not in m.content]
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages
            ]
        }
    return None
```

核心模式：**Remove All + Add New = Replace**。

---

## 三、Delete Messages（指定删除）

通过 `RemoveMessage(id=具体ID)` 精确删除某几条消息，删除后其余顺序不变。

### 官方示例

```python
@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    messages = state["messages"]
    if len(messages) > 2:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None
```

### RemoveMessage 的工作方式

- **按 ID 删除**，不是按下标：`RemoveMessage(id="msg_abc")`
- Message ID 来源：自动 UUID、显式指定 `HumanMessage(id="x")`、或 `get_state()` 查询
- **`REMOVE_ALL_MESSAGES`** 是 sentinel 常量 `'__remove_all__'`，reducer 看到它直接清空
- 依赖 `add_messages` reducer：默认 `AgentState` 继承 `MessagesState` 自带此配置
- 外部删除：`graph.update_state(config, {"messages": [RemoveMessage(id=...)]})`

### 合法消息约束

```
❌ 首条是 AI         → AI("Hi") → Human(...)
❌ tool_call 无结果   → AI(tool_call) → AI(...)      缺 ToolMessage
❌ ToolMessage 无调用 → ToolMessage → AI(...)        缺 AI(tool_call)
```

### 项目代码对照（`delete_message.py`）

```python
@after_model
def delete_message(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state['messages']
    if len(messages) > 2:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None
```

效果：**永远只保留最近的 1-2 条消息**。

---

## 四、Summarize Messages（摘要）

用 LLM 生成旧消息的摘要，替换原始消息，既压缩 Token 又保留关键信息。

### 官方示例

```python
SummarizationMiddleware(
    model="gpt-5.4-mini",
    trigger=("tokens", 4000),
    keep=("messages", 20)
)
```

| 参数 | 类型 | 含义 |
|------|------|------|
| `trigger` | `("tokens", N)` 或 `("messages", N)` | 触发摘要的阈值 |
| `keep` | `("messages", N)` | 摘要后保留的最新消息条数 |

### 项目代码对照（`summarize_message.py`）

```python
agent = create_agent(
    model=qwen_model,
    middleware=[
        SummarizationMiddleware(
            model=qwen_model,
            trigger=('tokens', 10),
            keep=('messages', 2)
        ),
        trim_message,
    ],
    checkpointer=pg_save,
)
```

---

## 五、三种策略对比

| 特性 | Trim | Delete | Summarize |
|------|------|--------|-----------|
| 信息保留 | 部分保留（头尾） | 保留其余所有 | 通过摘要保留语义 |
| Token 压缩 | 高 | 高 | 中 |
| 精度 | 丢失中间内容 | 精确控制 | 依赖 LLM 摘要质量 |
| 复杂度 | 低 | 低 | 高（需 LLM 调用） |

---

## 六、Access Memory：从 Tool 访问和修改短期记忆

除了 middleware 之外，Tool 也可以通过 `runtime` 参数**读取和修改**短期记忆（state）。

### 6.1 在 Tool 中读取 State

`runtime` 参数对模型隐藏（不出现在 tool 签名中），但 tool 内部可以访问 `runtime.state`。

```python
from langchain.tools import tool, ToolRuntime

class CustomState(AgentState):
    user_id: str

@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """Look up user info."""
    user_id = runtime.state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_user_info],
    state_schema=CustomState,
)
result = agent.invoke({
    "messages": "look up user information",
    "user_id": "user_123",
})
```

### 6.2 在 Tool 中写入 State

通过 `Command(update={...})` 返回状态更新，从而修改 state：

```python
from langgraph.types import Command
from pydantic import BaseModel

class CustomState(AgentState):
    user_name: str

class CustomContext(BaseModel):
    user_id: str

@tool
def update_user_info(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """Look up and update user info."""
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name,
        "messages": [
            ToolMessage("Successfully looked up user information",
                        tool_call_id=runtime.tool_call_id)
        ]
    })

@tool
def greet(
    runtime: ToolRuntime[CustomContext, CustomState]
) -> str | Command:
    """Use this to greet the user once you found their info."""
    user_name = runtime.state.get("user_name", None)
    if user_name is None:
        return Command(update={
            "messages": [
                ToolMessage("Please call 'update_user_info' first.",
                            tool_call_id=runtime.tool_call_id)
            ]
        })
    return f"Hello {user_name}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[update_user_info, greet],
    state_schema=CustomState,
    context_schema=CustomContext,
)
agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    context=CustomContext(user_id="user_123"),
)
```

**关键点**：

| 要素 | 说明 |
|------|------|
| 读取 state | `runtime.state["field_name"]` |
| 写入 state | `Command(update={"field_name": value})` |
| 写入 messages | `Command(update={"messages": [ToolMessage(...)]})` |
| 类型标注 | `ToolRuntime[ContextSchema, StateSchema]` 分别标注 context 和 state 类型 |

### 6.3 与 Middleware 更新的对比

Tool 和 Middleware 都能更新 State，但能力不同：

| | Middleware | Tool |
|--|-----------|------|
| 返回值 | `dict` | `Command` |
| 修改 State | ✅ `return {"key": val}` | ✅ `Command(update={...})` |
| 跳转节点 | ❌ | ✅ `Command(goto=...)` |
| 中断执行 | ❌ | ✅ `Command(interrupt=...)` |

本质：`dict` = State Patch，`Command` = State Patch + Control Flow。

---

## 七、Access Memory：从 Prompt 访问

通过 `@dynamic_prompt` 可以在 system prompt 中动态注入 state 或 context 数据。

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

class CustomContext(TypedDict):
    user_name: str

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    return f"You are a helpful assistant. Address the user as {user_name}."

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    context=CustomContext(user_name="John Smith"),
)
```

---

## 八、完整执行流程

```
invoke()
    │
    ▼
读取 State（从 checkpointer）
    │
    ▼
@before_model          ← Trim / Summarize 在这里执行
    │  └─ SummarizationMiddleware 检查触发条件
    ▼  return dict | None → Reducer 合并 → 新 State
LLM 调用
    │
    ▼  Tool 执行时可通过 runtime.state 读写 State
@after_model           ← Delete 在这里执行
    │
    ▼  return dict | None → Reducer 合并 → 新 State
保存到 checkpointer（checkpoints + checkpoint_writes）
    │
    ▼
返回结果
```

---

## 九、参考代码索引

| 文件 | 策略 | 触发时机 |
|------|------|----------|
| `trim_message.py` | Trim（按内容 Replace） | `@before_model` |
| `delete_message.py` | Delete（按数量删旧消息） | `@after_model` |
| `summarize_message.py` | Summarize + 调试打印 | `SummarizationMiddleware` + `@before_model` |
