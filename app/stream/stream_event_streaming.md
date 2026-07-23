# Event Streaming — 事件流

> 对应文档章节：[Event Streaming](https://docs.langchain.com/oss/python/langchain/event-streaming)
>
> Demo 代码目录：`app/stream/`

---

## 一、概述

LangChain Agent 基于 LangGraph 构建，因此它继承了 LangGraph 的流式（streaming）机制，并在此基础上为 Agent 场景提供了**按用途拆分的投影视图（Projection）**。

核心 API 是 `agent.stream_events(..., version="v3")`，返回一个事件流对象，该对象提供了多个类型化的投影（Projection），开发者无需解析底层原始事件，直接按需消费即可。

---

## 二、所有 Projection 一览

| 投影 | 用途 |
|------|------|
| `for event in stream` | 原始协议事件（完整信封，可访问所有频道） |
| `stream.messages` | LLM 消息流，每次模型调用一条 |
| `message.text` | 文本增量（delta）和最终文本 |
| `message.reasoning` | 推理内容增量（需模型支持） |
| `message.tool_calls` | 模型正在构造工具调用参数时的参数块 |
| `message.output` | 模型调用完成后的最终消息对象 |
| `stream.values` | Agent 状态快照 |
| `stream.output` | Agent 最终状态 |
| `stream.subagents` | 具名 `create_agent` 子代理的事件 |
| `stream.subgraphs` | 所有嵌套图（含普通 StateGraph） |
| `stream.extensions` | 自定义 Transformer 投影 |
| `stream.tool_calls` | 工具执行生命周期（输入、输出流、最终结果、错误） |

---

## 三、核心概念：Projection（投影）

### 3.1 定义

Projection 是对**同一条底层事件流的不同观察视角**。

官方内部实际上只有**一条 Event Stream**，所有事件（LLM 输出、Tool 调用、状态更新、子 Agent……）都进入这条流。Event Streaming 会将这些事件转换成多个 Projection，开发者不需要自己解析底层事件。

```text
               一条 Event Stream
                     │
     ┌───────────────┼─────────────────┐
     │               │                 │
     ▼               ▼                 ▼
 stream.messages  stream.values  stream.tool_calls
     │               │                 │
     │               │                 │
   看消息         看状态变化       看工具执行
```

### 3.2 Projection 是"过滤器"，不是"消费者"

**最重要的概念**：多个 Projection 共享底层数据，各有各的游标，互不抢占。

```python
# 这种做法完全 OK，不会因为先读了 messages 就丢失 values
for message in stream.messages:
    ...

for value in stream.values:    # 数据依然存在
    ...
```

底层事件只记录一次，但每个 Projection 可以独立读取整条流，只筛选自己关心的事件类型。

```
  Event Stream (完整数据)
       │
       ├── stream.messages    ← 只过滤出 message 类型
       ├── stream.tool_calls  ← 只过滤出 tool_call 类型
       └── stream.values      ← 只过滤出 state 类型
```

---

## 四、各 Projection 详解

### 4.1 stream.messages — 模型消息流

每次 LLM 调用产生一条 `ChatModelStream` 对象，暴露 `.text`、`.reasoning`、`.tool_calls`、`.output` 四个子投影。

```python
stream = agent.stream_events(input, version="v3")

for message in stream.messages:
    print(f"[{message.node}] ", end="")
    for delta in message.text:
        print(delta, end="", flush=True)

    full_message = message.output
    usage = full_message.usage_metadata
    if usage:
        print(usage)
```

**Demo**: `app/stream/stream_message_reasoning.py`

### 4.2 message.reasoning — 推理内容

只有当模型支持并输出推理内容（reasoning blocks）时才可用。

```python
for message in stream.messages:
    for delta in message.reasoning:
        print(f"[thinking] {delta}", end="", flush=True)

    for delta in message.text:
        print(delta, end="", flush=True)
```

### 4.3 message.tool_calls — 模型端工具调用参数

LLM 决定调用工具并构造参数时的实时增量片段（chunks）。可通过 `.get()` 获取最终确定的 tool call。

```python
for message in stream.messages:
    for chunk in message.tool_calls:
        print(f"tool call chunk: {chunk}")

    finalized = message.tool_calls.get()
    if finalized:
        print(f"finalized tool calls: {finalized}")
```

### 4.4 stream.tool_calls — 工具执行生命周期

工具实际执行全过程的投影。与 `message.tool_calls` 互补——前者看"模型在构造参数"，后者看"工具在跑"。

```python
for call in stream.tool_calls:
    print(f"{call.tool_name}({call.input})")
    for delta in call.output_deltas:
        print(delta, end="", flush=True)
    print(call.output, call.error)
```

**核心区别**：

| | `message.tool_calls` | `stream.tool_calls` |
|---|---|---|
| 谁产生的 | LLM 模型 | 工具执行器（ToolNode） |
| 阶段 | 模型决定调用工具，构造参数时 | 工具被实际执行时 |
| 内容 | 参数块增量片段（chunks） | 输入、输出流、最终结果/错误 |
| `.get()` | 有，返回最终 tool call | 无（直接遍历） |
| 典型用途 | 实时显示"模型正在调用 XX 工具" | 实时显示工具执行进度和结果 |

**特别注意**：`stream.messages` 和 `stream.tool_calls` 共享同一个底层事件流，是**单次迭代器（single-pass iterator）**。顺序遍历会耗尽底层流，导致第二个投影无数据。有两种解决方式：

1. **同步**：用 `stream.interleave()` 交错消费
2. **异步**：用 `asyncio.gather` + `async for` 并发消费

**Demo**: `app/stream/stream_tool.py`

### 4.5 stream.values — 状态快照

Agent 每执行一个节点（node）后发出的状态快照，可实时追踪 agent 内部状态的变更。

```python
for snapshot in stream.values:
    print(snapshot)
```

**Demo**: `app/stream/stream_values.py`

### 4.6 stream.output — 最终状态

流结束后一次性获取的最终 agent 状态。

```python
final_state = stream.output
```

### 4.7 stream.subagents — 子代理事件流

当一个 `create_agent` 通过 tool 调用另一个具名 `create_agent` 时，被调用的子代理事件以嵌套命名空间暴露。`stream.subagents` 是"具名 create_agent 子代理"的专用视图，每个子代理暴露：

- `.name` — `create_agent(name=...)` 传入的名称
- `.cause` — 触发该子代理的 tool call
- `.messages` — 子代理内的 LLM 消息流
- `.values` — 子代理状态快照
- `.tool_calls` — 子代理内的工具执行生命周期
- `.output` — 子代理最终状态

```python
for subagent in stream.subagents:
    print(f"{subagent.name}: ", end="")
    for message in subagent.messages:
        for token in message.text:
            print(token, end="", flush=True)
    print()
```

**`stream.subagents` vs `stream.subgraphs`**：
- `stream.subagents`：只包含具名 `create_agent` 子代理，无需过滤
- `stream.subgraphs`：覆盖所有嵌套图（包括普通 StateGraph）

**Demo**: `app/stream/stream_subagent.py`

### 4.8 stream.extensions — 自定义投影

通过注册自定义 `StreamTransformer` 来添加内置投影之外的视图。例如追踪工具调用状态、检索进度、领域特定事件等。

```python
stream = agent.stream_events(
    input,
    version="v3",
    transformers=[ToolActivityTransformer],
)

for activity in stream.extensions["tool_activity"]:
    print(activity)
```

**Demo**: `app/stream/custom_transformer.py`

---

## 五、Multiple Projections — 并发消费多个投影

### 5.1 为什么需要并发消费

虽然多个 Projection **逻辑上**独立共享底层数据，但**物理上**事件是实时到达的。如果串行遍历：

```python
for msg in stream.messages: ...    # 等所有消息结束
for tool in stream.tool_calls: ... # agent 可能已经跑完了
```

这是**顺序问题，不是数据丢失问题**。并发是为了**实时性**——多个事件同时到达时，能同时展示给用户。

### 5.2 同步：stream.interleave()

保持事件的**原始到达顺序（Arrival Order）**输出，而不是按类型分组。

```python
for name, item in stream.interleave("messages", "tool_calls", "values"):
    if name == "messages":
        print(item.text)
    elif name == "tool_calls":
        print(item.tool_name, item.input)
    elif name == "values":
        print(item)
```

```text
真实到达顺序：
  10:00:01  Message
  10:00:02  State
  10:00:03  Message
  10:00:04  Tool

interleave 输出：
  messages 10:00:01
  values   10:00:02
  messages 10:00:03
  tool     10:00:04
```

**Demo**: `app/stream/multiple_projections.py`

### 5.3 异步：asyncio.gather + async for

```python
stream = await agent.astream_events(input, version="v3")

async def consume_messages():
    async for message in stream.messages:
        print(await message.text)

async def consume_tool_calls():
    async for call in stream.tool_calls:
        print(call.tool_name, call.input)

await asyncio.gather(consume_messages(), consume_tool_calls())
```

### 5.4 场景选择

| 场景 | 做法 |
|------|------|
| 聊天页面（只有消息） | `stream.messages` |
| 多面板 UI（消息 + 工具 + 状态） | `asyncio.gather` 或 `interleave()` |
| 调试工具 / 完整时间线 | `interleave()` |
| 获取最终结果 | `stream.output` |

**辅助文档**: `app/stream/multiple_projections.md`
**练习题 (asyncio)**: `app/stream/asyncio_exercise.md`, `app/stream/exercise.py`

---

## 六、Custom Transformer — 自定义投影

### 6.1 核心接口

```python
from langgraph.stream import StreamTransformer, StreamChannel, ProtocolEvent


class ToolActivityTransformer(StreamTransformer):
    required_stream_modes = ("tools",)   # 声明需要的流模式

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self.activity = StreamChannel[dict]("tool_activity")
        self._tool_names: dict[str, str] = {}

    def init(self) -> dict:
        return {"tool_activity": self.activity}

    def process(self, event: ProtocolEvent) -> bool:
        # 处理协议事件，推送到 channel
        self.activity.push({...})
        return True   # True = 保留原始事件

    def finalize(self) -> None:
        self.activity.push({"status": "流结束"})
```

- `StreamTransformer`：观察协议事件，维护内部状态，暴露派生视图
- `StreamChannel`：投影原语，`push` 的值可在 `stream.extensions` 上迭代
- `stream.extensions`：自定义投影的入口

### 6.2 注册方式

```python
# 方式一：直接传参
stream = agent.stream_events(input, version="v3",
    transformers=[ToolActivityTransformer])

# 方式二：在 Middleware 中注册（langchain >= 1.3.2）
class ToolActivityMiddleware(AgentMiddleware):
    transformers = (ToolActivityTransformer,)
```

**Demo**: `app/stream/custom_transformer.py`

---

## 七、Middleware 中的 Transformer

Middleware 可以在其类上声明 `transformers` 属性：

```python
from langchain.agents.middleware import AgentMiddleware

class ToolActivityMiddleware(AgentMiddleware):
    transformers = (ToolActivityTransformer,)
```

编译时，`create_agent` 的合并顺序为：

1. 内置 `ToolCallTransformer`
2. Middleware 注册的工厂（按 middleware 顺序）
3. 调用者 `create_agent(transformers=...)` 传入的工厂

内置的 `PIIMiddleware` 即利用此机制，在流式输出中脱敏 PII 数据。

---

## 八、系统架构全景

```text
                stream_events()
                      │
                      ▼
              Event Stream（唯一）
                      │
      ┌───────────────┼──────────────────┐
      │               │                  │
      ▼               ▼                  ▼
 stream.messages  stream.values   stream.tool_calls
      │               │                  │
      ▼               ▼                  ▼
 ChatModelStream    State           ToolExecution
      │
      ├── text
      ├── reasoning
      ├── tool_calls
      └── output

      ▼
 stream.subagents

      ▼
 stream.subgraphs

      ▼
 stream.output
```

**三层结构**：

| 层 | 说明 |
|----|------|
| **底层** | 一条原始 Event Stream |
| **中间层** | Transformer 将事件转换为不同 Projection（滤镜） |
| **应用层** | 根据业务需要选择不同 Projection，无需解析原始事件 |

---

## 九、Demo 文件索引

| 文件 | 覆盖内容 |
|------|---------|
| `stream_message_reasoning.py` | `stream.messages` + `message.reasoning` + `message.output` + token 用量 |
| `stream_tool.py` | `message.tool_calls` vs `stream.tool_calls` + `interleave()` |
| `stream_values.py` | `stream.values` |
| `stream_subagent.py` | `stream.subagents` 子代理事件流 |
| `custom_transformer.py` | 自定义 `StreamTransformer` + `stream.extensions` |
| `multiple_projections.py` | `stream.interleave()` 多投影交错消费 |
| `exercise.py` | asyncio 基础练习（await / gather / to_thread） |
| `multiple_projections.md` | Projection 概念深度讲解笔记 |
| `asyncio_exercise.md` | asyncio 练习题说明 |

---

## 十、关键注意事项

1. **单次迭代器**：`stream.messages` 和 `stream.tool_calls` 共享底层流，顺序遍历会耗尽。需要用 `interleave()` 或 `asyncio.gather`。

2. **`message.tool_calls` 的输出属性是 `chunks`**（增量片段），而 `stream.tool_calls` 的输出属性是 `output_deltas`（注意是 `deltas` 不是 `details`）。

3. **Projection 不是消费者**：多个 Projection 共享底层数据，各有各的游标，先读一个不会让另一个丢数据——但底层事件流本身是单次迭代器，这个约束依然存在。

4. **自定义 Transformer 需声明 `required_stream_modes`**：否则关心的频道事件不会到达 `process()` 方法。

5. **`tool-finished` / `tool-error` 事件不携带 `tool_name`**：需要用 `tool_call_id` 关联到 `tool-started` 时记录的名字（详见 `custom_transformer.py` 中的处理模式）。
