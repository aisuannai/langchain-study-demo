# Streaming with Human-in-the-Loop

> 官方文档章节：[Streaming with human-in-the-loop](https://docs.langchain.com/oss/python/langchain/streaming#streaming-with-human-in-the-loop)
>
> Demo 代码：`app/stream/stream_human_in_the_loop.py`

---

## 一、要解决什么问题？

Agent 调用工具是**自动**的——LLM 决定调用什么工具、传什么参数，然后就执行了。但在生产环境中，有些操作不能直接放行：

- **删除数据** `delete_user(user_id=123)`
- **写入操作** `send_email(to="...", content="...")`
- **高风险参数** `transfer_money(amount=10000, to="...")`
- **需要人工确认参数是否准确**

**Human-in-the-Loop** 模式就是：Agent 自动推理 → 工具执行前先**打断** → 等人审批 → 根据决策继续执行。

---

## 二、核心概念

### 2.1 HumanInTheLoopMiddleware

LangChain 提供的一个中间件，可以在指定工具执行前插入**审批点**。

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model=...,
    tools=[get_weather, delete_user],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "get_weather": True,   # get_weather 需要审批
                "delete_user": True,   # delete_user 需要审批
            }
        ),
    ],
    checkpointer=InMemorySaver(),  # ⚠️ 必须！中断需要持久化状态
)
```

关键参数：

| 参数 | 含义 |
|------|------|
| `interrupt_on` | 字典：`{工具名: True/False}`，指定哪些工具需要审批 |
| `checkpointer` | **必须提供**，不然中断后状态没法恢复 |

### 2.2 Interrupt（中断）

当 Agent 调用到需要审批的工具时，不会直接执行，而是发出一个 **Interrupt**。这个中断会出现在流式输出的 `updates` 中，`source == "__interrupt__"`。

每个 Interrupt 包含：

```python
interrupt.value = {
    "action_requests": [
        {
            "description": "Tool: get_weather\nArgs: {'city': 'Boston'}",
            "action": {...},   # 原始 tool call
        },
        {
            "description": "Tool: get_weather\nArgs: {'city': 'San Francisco'}",
            "action": {...},
        },
    ]
}
```

### 2.3 Decision（决策）

对每个 action_request，用户可以做三种决策：

| 决策类型 | 含义 |
|---------|------|
| `{"type": "approve"}` | 批准，原样执行 |
| `{"type": "edit", "edited_action": {"name": "...", "args": {...}}}` | 修改参数后再执行 |
| `{"type": "reject", "rejected_action": {...}}` | 拒绝执行 |

**决策顺序必须与 action_requests 的顺序一致。**

### 2.4 Command(resume=...)

收集完所有决策后，通过 `Command(resume=decisions)` 恢复 agent 执行：

```python
from langgraph.types import Command

decisions = {
    interrupt.id: {
        "decisions": [{"type": "approve"}, {"type": "edit", "edited_action": {...}}]
    }
}

for chunk in agent.stream(
    Command(resume=decisions),
    config=config,
    stream_mode=["messages", "updates"],
    version="v2",
):
    # 继续消费流式输出...
```

---

## 三、完整流程拆解

整个模式分为 **三个阶段**：

```
Phase 1: 首次流式调用
─────────────────────
  agent.stream(input, config, ...)
       │
       ├─ messages  → LLM token chunks（实时显示模型输出）
       ├─ updates/model → 完整 AIMessage（含 tool_calls）
       └─ updates/__interrupt__ → 收集中断 ⏸️
       │
       ▼
  流结束，但 agent 状态被 checkpointer 保存

Phase 2: 用户决策
─────────────────────
  for each action_request:
      - 展示给用户看（description）
      - 用户选择 approve / edit / reject
  → 组装 decisions 字典

Phase 3: 恢复执行
─────────────────────
  agent.stream(Command(resume=decisions), config, ...)
       │
       ├─ updates/tools → 工具被执行（或修改后执行）
       ├─ messages → LLM 总结输出
       └─ 正常结束
```

---

## 四、细节点解读

### 4.1 为什么需要 checkpointer？

中断的本质是：**Agent 执行到一半停下来了**。状态必须持久化，否则重新恢复时不知道之前干了什么。

```python
checkpointer = InMemorySaver()
agent = create_agent(..., checkpointer=checkpointer)

# 第一次调用时传入 config，thread_id 标识会话
config = {"configurable": {"thread_id": "session-123"}}
```

### 4.2 为什么 `source == "__interrupt__"` 是特殊的？

在 `stream_mode="updates"` 下，通常的 source 是节点名（`model`、`tools`）。但中断不是某个节点的输出，而是框架层的打断信号，所以 source 是 `__interrupt__`（双下划线是 LangGraph 内部约定的前缀）。

### 4.3 decisions 为什么要按 interrupt.id 分组？

一个 stream 里可能产生多个中断（比如多个工具都设置了 `interrupt_on=True`），每个中断有自己的 id。按 id 分组可以正确匹配决策到对应的中断。

```python
decisions = {
    interrupt.id: {
        "decisions": [decision1, decision2, ...]
    }
}
```

### 4.4 为什么决策顺序必须和 action_requests 一致？

因为框架按**索引**匹配决策——第一个决策对应第一个 action_request，以此类推。顺序错了会导致张冠李戴。

### 4.5 恢复流和首次流有什么区别？

恢复流使用 `Command(resume=decisions)` 而不是原始的输入。其余完全一样——同一个 `config`（同一个 `thread_id`），同样的 `stream_mode`，同样的消费循环。

---

## 五、streaming.md 例子详解

官方例子是一个天气查询场景，用户同时问 Boston 和 San Francisco 的天气。`get_weather` 被设为需要审批。

### 场景流程

```
用户输入: "Can you look up the weather in Boston and San Francisco?"

Step 1: Agent 流式输出
  └─ LLM 同时生成了两个 tool_call:
       get_weather(city="Boston")
       get_weather(city="San Francisco")

Step 2: HumanInTheLoopMiddleware 拦截
  └─ 因为 interrupt_on={"get_weather": True}
  └─ 发出 __interrupt__，包含 2 个 action_requests
  └─ 流结束（工具尚未执行）

Step 3: 用户决策
  └─ Boston: 用户编辑参数 → {"city": "Boston, U.K."}
  └─ San Francisco: 用户批准

Step 4: 恢复流
  └─ get_weather("Boston, U.K.") 执行 → "Boston, U.K. weather is sunn"
  └─ get_weather("San Francisco") 执行 → "San Francisco weather is sunn"
  └─ LLM 总结输出
```

### 关键代码片段

**1. 收集中断：**

```python
interrupts = []
for chunk in agent.stream(input, config=config, stream_mode=["messages", "updates"], version="v2"):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        # ... 渲染 token ...
    elif chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source in ("model", "tools"):
                # ... 渲染完整消息 ...
            if source == "__interrupt__":
                interrupts.extend(update)  # 收集中断
                # 显示给用户：
                for request in update[0].value["action_requests"]:
                    print(request["description"])
```

**2. 生成决策（以程序模拟人为例）：**

```python
def _get_interrupt_decisions(interrupt):
    return [
        {
            "type": "edit",
            "edited_action": {
                "name": "get_weather",
                "args": {"city": "Boston, U.K."},
            },
        }
        if "boston" in request["description"].lower()
        else {"type": "approve"}
        for request in interrupt.value["action_requests"]
    ]

decisions = {}
for interrupt in interrupts:
    decisions[interrupt.id] = {
        "decisions": _get_interrupt_decisions(interrupt)
    }
```

**3. 恢复执行：**

```python
for chunk in agent.stream(
    Command(resume=decisions),
    config=config,
    stream_mode=["messages", "updates"],
    version="v2",
):
    # 同样的消费逻辑
```

---

## 六、和 Event Streaming (v3) 的关系

官方例子用的是**低层 v2 API**（`agent.stream(stream_mode=...)`），而不是你学过的 `agent.stream_events(version="v3")`。

原因是 `stream_events` v3 是高层封装，把 messages、tool_calls、values 分成了独立的 Projection。但 **`__interrupt__` 信号在 v3 中没有对应的 Projection**——它只在低层 `updates` 中出现。

所以 Human-in-the-loop 场景目前必须用低层 v2 API 来实现。两种 API 可以理解为：

```
stream_events v3       →  日常聊天/展示（推荐）
stream v2 + interrupt  →  需要人工审批的场景（唯一选择）
```

---

## 七、总结

| 概念 | 说明 |
|------|------|
| `HumanInTheLoopMiddleware` | 在指定工具执行前插入审批点 |
| `interrupt_on` | 声明哪些工具需要审批 |
| `checkpointer` | 中断必须配合 checkpointer 使用 |
| `__interrupt__` | 中断信号，出现在 `updates` 流中 |
| `action_requests` | 待审批的动作列表 |
| `approve / edit / reject` | 三种决策类型 |
| `Command(resume=decisions)` | 恢复 agent 执行 |

**一句话总结**：Agent 正常流式输出 → 遇到敏感工具时**暂停** → 等人审批/修改参数 → 恢复执行——全程流式输出不中断。
