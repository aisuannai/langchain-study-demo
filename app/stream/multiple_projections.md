# Multiple Projections（多个投影视图）

> LangChain Event Streaming API 的核心思想。
> 
> 如果理解了这一章，就会明白为什么新版 Streaming API 比以前的 `stream()/astream()` 好用得多。
>
> 文档地址：https://docs.langchain.com/oss/python/langchain/event-streaming#multiple-projections

---

## 一、什么叫 Projection（投影）？

Projection 可以理解成：

> **同一条事件流的不同观察视角（View）。**

官方内部实际上只有**一条 Event Stream**，所有事件（LLM 输出、Tool 调用、状态更新、子 Agent……）都会进入这条事件流。Event Streaming 会把这些事件转换成多个不同的 Projection，开发者不用自己解析底层事件。

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

所以：

```python
stream = agent.stream_events(..., version="v3")
```

并不是生成了多个 Stream，实际上只有一个，只是官方帮你生成了很多不同的"窗口"。

---

## 二、为什么需要多个 Projection？

假设 Agent 工作流程：

```
用户 → LLM 思考 → 生成 Tool Call → Tool 执行 → LLM 总结 → 结束
```

如果只有一个 Stream，你就需要不停判断事件类型：

```python
if event["type"] == "message":
    ...
elif event["type"] == "tool":
    ...
elif event["type"] == "state":
    ...
```

整个代码会越来越复杂。新版 Event Streaming 的思想是：**按用途拆分。**

我要聊天内容：

```python
for msg in stream.messages:
    ...
```

我要 Tool：

```python
for tool in stream.tool_calls:
    ...
```

我要状态：

```python
for value in stream.values:
    ...
```

互不影响，各取所需。

---

## 三、关键：Projection 是"过滤器"，不是"消费者"

> **Multiple consumers can read these projections concurrently. Reading one projection does not consume events needed by another projection.**
>
> — LangChain 官方文档

这是整个章节最重要的概念。很多人的第一反应是：

```python
for message in stream.messages:
    ...

for value in stream.values:    # 消息读完了，values 还有数据吗？
    ...
```

**答案：有。不会读完。**

### 错误的直觉

你以为的架构是**竞争消费**（competing consumers）：

```
  Event Stream (队列)
       │
       ▼
  stream.messages  ← 消费掉消息
       │
       ▼
  stream.tool_calls ← 没东西可读了 ✗
```

这个模型是 Kafka / RabbitMQ / Java Event Bus 的工作方式——一条消息被一个消费者拿走就没了。

### 实际的架构

Projection 是**光学滤镜**，不是**消费者的手**：

```
  Event Stream (完整数据)
       │
       ├── stream.messages    ← 只过滤出 message 类型的数据
       ├── stream.tool_calls  ← 只过滤出 tool_call 类型的数据
       └── stream.values      ← 只过滤出 state 类型的数据
```

- `stream.messages` 迭代时，只看 event 类型是 `message` 的记录，**跳过其他类型**。
- `stream.tool_calls` 迭代时，只看 event 类型是 `tool_call` 的记录，**跳过其他类型**。
- 两条迭代器**各自持有自己的游标**，互不影响。

### 形象类比

想象一本会议记录本，每页同时包含三栏：

| 时间 | 说了什么 | 后台工具 | 系统状态 |
|------|---------|---------|---------|
| 10:00 | 今天天气如何 | — | 等待输入 |
| 10:01 | — | get_weather() | 调用中 |
| 10:02 | 德州晴天 | — | 完成 |

- `stream.messages` → 每页只看**"说了什么"**那一列 → `["今天天气如何", "德州晴天"]`
- `stream.tool_calls` → 每页只看**"后台工具"**那一列 → `["get_weather()"]`
- `stream.values` → 每页只看**"系统状态"**那一列 → `["等待输入", "调用中", "完成"]`

**看"说了什么"那一列，并不会让"后台工具"那一列凭空消失。** 它们只是同一份数据的不同视图。

> **结论：多个 projection 共享底层数据，各有各的游标，互不抢占。**

---

## 四、官方为什么说可以 Concurrently（并发消费）？

假设 Web 前端有三个区域：

```
┌──────────────────────────┐
│ Chat Window              │
│ 你好                     │
│ 北京今天晴天             │
└──────────────────────────┘

┌──────────────────────────┐
│ Tool Window              │
│ Searching...             │
│ Database Query...        │
└──────────────────────────┘

┌──────────────────────────┐
│ State                    │
│ current_step=3           │
└──────────────────────────┘
```

三个窗口需要的数据全部来自同一个 `stream`：

```python
# 协程 A：更新聊天窗口
async for message in stream.messages:
    update_chat(message.text)

# 协程 B：更新工具面板
async for call in stream.tool_calls:
    update_tool_panel(call.tool_name, call.input)

# 协程 C：更新状态栏
async for value in stream.values:
    update_status(value)
```

三者互不影响，各自消费各自关心的数据。

### 那为什么还要 `asyncio.gather`？

虽然数据**逻辑上**独立，但**物理上**事件是实时到达的。如果不并发：

```python
# 串行：必须等 agent 跑完所有 message 事件，才能开始看 tool_calls
for msg in stream.messages:
    ...

# agent 此时可能已经跑完了，tool_calls 的数据虽然还在，但你是"事后"才看
for tool in stream.tool_calls:
    ...
```

这是**顺序问题，不是数据丢失问题**。用 `gather` 并发是为了**实时性**——message 和 tool_call 同时到达时，两边能同时展示给用户，而不是先等所有消息显示完再显示工具调用。

---

## 五、interleave() 是干什么的？

有了 Projection 为什么还需要 `interleave`？

因为多个 Projection 默认是**按类型分组**的。`for m in stream.messages` 会一直循环到 message 事件结束才返回，这时候真实的时间顺序就丢了。

真实事件的到达顺序：

```
10:00:01  Message
10:00:02  State
10:00:03  Message
10:00:04  Tool
```

分开消费：

```python
# 得到：["Message", "Message"]     ← 时间顺序丢了
for m in stream.messages: ...

# 得到：["State"]                   ← 实际是发生在两条 Message 之间的
for v in stream.values: ...

# 得到：["Tool"]
for t in stream.tool_calls: ...
```

而 `interleave` 按到达顺序输出：

```python
for name, item in stream.interleave("messages", "values", "tool_calls"):
    print(name, item)

# 输出：
# messages 10:00:01
# values   10:00:02
# messages 10:00:03
# tool     10:00:04
```

**保持了 Arrival Order（到达顺序）。**

### 什么时候用，什么时候不用？

| 场景 | 做法 |
|------|------|
| 聊天页面（只有消息） | `stream.messages` 就够了 |
| 调试工具 / LangSmith | `interleave()` 按时间线展示 |
| 多面板 UI | 各面板独立用对应 projection |
| 需要完整时间顺序日志 | `interleave()` |

---

## 六、整个 Event Streaming 架构全景

```
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

总结三层结构：

- **底层**：只有一个 Event Stream。
- **中间层**：Transformer 把事件转换成不同 Projection（滤镜）。
- **应用层**：根据业务需要选择不同 Projection，无需解析原始事件。

---

## 七、企业项目选型建议

| 场景 | 推荐 Projection |
|------|----------------|
| 聊天窗口（ChatGPT 类） | `stream.messages` |
| 显示模型思考过程 | `message.reasoning` |
| 显示 Tool 参数生成 | `message.tool_calls` |
| 显示 Tool 执行进度 | `stream.tool_calls` |
| 显示 Agent 当前状态 | `stream.values` |
| 多 Agent 协作界面 | `stream.subagents` |
| 调试 LangGraph 工作流 | `stream.subgraphs` |
| 需要完整时间顺序日志 | `stream.interleave(...)` |
| 获取最终结果 | `stream.output` |

---

## 八、从 Java 工程师的角度理解

如果你熟悉 Java，可以把 **Projection** 理解成**多个订阅同一份日志文件的不同管道**，而不是**多个消费者抢同一个消息队列**：

```text
               完整日志文件
               （Event Stream）
                     │
       ┌─────────────┼─────────────┐
       │             │             │
  grep "message"  grep "tool"   grep "state"
       │             │             │
       ▼             ▼             ▼
 stream.messages  stream.tool  stream.values
```

底层事件只记录一次，但每个 `grep` 可以独立读取整份日志，只筛选自己关心的行。`tail -f` 跟 `grep` 互不干扰，各自有各自的游标——这恰好就是 Projection 的工作原理。

> **一套事件流，同时服务于聊天 UI、调试面板、工具监控、状态展示等多个功能，开发者不需要自己解析和分发底层事件。**
