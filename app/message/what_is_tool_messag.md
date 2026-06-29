这一章是 **Messages 模块的最后一个核心 Message 类型**。如果说：

* **SystemMessage**：告诉 AI 应该怎么工作
* **HumanMessage**：用户输入
* **AIMessage**：模型输出

那么：

> **ToolMessage 就是工具执行后的返回结果。**

它是 **Agent、Tool Calling、LangGraph** 的核心概念，也是很多初学者最容易理解错的地方。官方对它的定义是：

> **ToolMessage 用于将一次工具执行的结果传递回模型。** ([LangChain Docs][1])

---

# ToolMessage（工具消息）

当模型调用一个 Tool 后，Tool 执行完成，并不会直接返回给用户。

而是会把执行结果包装成：

```python
ToolMessage(...)
```

再发送给模型。

然后模型根据 Tool 的执行结果，再生成最终回答。

因此：

> **ToolMessage 本质上就是 Tool 与 LLM 之间通信的"桥梁"。** ([LangChain Docs][1])

---

# 为什么需要 ToolMessage？

很多人第一次接触 Tool Calling 都会疑惑：

为什么工具不能直接返回字符串？

例如：

用户：

```text
北京天气怎么样？
```

模型决定调用工具：

```text
get_weather("北京")
```

工具执行：

```text
晴
26℃
```

很多人认为：

```text
Tool

↓

字符串

↓

结束
```

其实真实流程不是这样。

真正流程：

```text
HumanMessage

↓

AIMessage
（我要调用天气工具）

↓

Weather Tool

↓

ToolMessage
（晴，26℃）

↓

AIMessage
（北京今天晴，26℃，适合外出。）
```

也就是说：

**工具的结果必须重新交给模型。**

因为：

模型需要：

* 整理语言
* 总结结果
* 多工具融合
* 推理下一步

所以：

Tool 不负责回答用户。

Tool 只负责：

> **提供数据。**

真正回答用户：

仍然是 AI。

---

# 官方示例

官方代码：

```python
from langchain.messages import AIMessage
from langchain.messages import ToolMessage

ai_message = AIMessage(
    content=[],
    tool_calls=[
        {
            "name":"get_weather",
            "args":{
                "location":"San Francisco"
            },
            "id":"call_123"
        }
    ]
)

weather_result = "Sunny, 72°F"

tool_message = ToolMessage(
    content=weather_result,
    tool_call_id="call_123"
)
```

这里发生了三件事情：

第一步：

AI 决定：

```text
我要调用：

get_weather()
```

于是产生：

```python
AIMessage(
    tool_calls=[...]
)
```

第二步：

程序真正执行：

```python
get_weather("San Francisco")
```

得到：

```text
Sunny, 72°F
```

第三步：

程序把结果包装成：

```python
ToolMessage(
    content="Sunny,72°F"
)
```

再发送给模型。

最后：

```python
messages = [
    HumanMessage(...),
    ai_message,
    tool_message
]

response = model.invoke(messages)
```

模型看到：

```text
用户：

旧金山天气？

↓

AI：

调用天气工具

↓

Tool：

Sunny,72°F

↓

AI：

旧金山今天晴，
72°F……
```

这就是官方示例真正表达的意思。 ([LangChain Docs][1])

---

# ToolMessage 为什么不能省略？

假设：

工具返回：

```text
72°F
```

如果不给模型：

模型永远不知道：

天气是多少。

因此：

```text
Tool

↓

ToolMessage

↓

LLM

↓

AIMessage
```

这一步：

**不能跳。**

Agent 所有工具调用：

几乎都遵循：

```text
AI

↓

Tool

↓

ToolMessage

↓

AI
```

---

# ToolMessage 常用属性

官方列出了四个最重要的属性。

---

## 1. content（工具执行结果）

这是：

**最重要的字段。**

例如：

```python
ToolMessage(
    content="Sunny,72°F"
)
```

content：

就是：

真正发送给模型的数据。

例如：

数据库查询：

```text
张三
26岁
北京
```

搜索：

```text
Redis 是一个……
```

计算器：

```text
42
```

RAG：

```text
Redis 是一种内存数据库……
```

这些：

都会放进：

```python
content
```

然后：

LLM 根据这些数据生成回答。

---

## 2. tool_call_id（必须）

这是：

Tool Calling 最关键的字段。

官方说明：

> **tool_call_id 必须与 AIMessage 中对应 tool_call 的 id 一致。** ([LangChain Docs][1])

例如：

AI：

```python
AIMessage(
    tool_calls=[
        {
            "id":"call_123"
        }
    ]
)
```

Tool：

必须：

```python
ToolMessage(
    tool_call_id="call_123"
)
```

为什么？

因为：

AI：

可能一次调用多个 Tool。

例如：

```text
今天北京天气，
再查一下美元汇率。
```

模型：

可能生成：

```text
Tool1

↓

Weather
id=call_1

Tool2

↓

Exchange
id=call_2
```

工具返回：

```text
ToolMessage(call_1)

ToolMessage(call_2)
```

模型才能知道：

哪个结果属于哪个 Tool。

所以：

> **tool_call_id 就像一次工具调用的"订单号"。**

---

## 3. name（工具名称）

官方示例：

```python
ToolMessage(
    name="search_books"
)
```

表示：

哪个 Tool 返回了结果。

例如：

```text
weather

calculator

search

rag

sql
```

虽然很多 Provider 并不会严格依赖它，但建议始终填写，便于调试和日志记录。 ([LangChain Docs][1])

---

## 4. artifact（隐藏数据）

这是官方特别强调的新属性，也是很多教程几乎不会讲的。

例如：

RAG：

真正给模型：

```text
Redis 是……
```

但是：

程序还需要：

```python
{
    "document_id":"doc001",
    "page":12
}
```

这些信息：

模型根本不需要。

但是：

程序需要。

于是：

官方设计：

```python
ToolMessage(
    content="Redis 是……",
    artifact={
        "document_id":"doc001",
        "page":12
    }
)
```

这里：

```text
content

↓

发送给LLM
```

而：

```text
artifact

↓

不会发送给LLM

↓

程序自己使用
```

官方建议：

可以保存：

* 原始 API 返回
* Debug 信息
* Document ID
* Chunk ID
* Score
* Source
* Page Number

例如：

```python
artifact={
    "source":"Redis.pdf",
    "page":6,
    "score":0.93
}
```

模型不会看到这些数据，但你的程序可以利用它们来展示引用来源、调试或后续处理。 ([LangChain Docs][1])

---

# ToolMessage 与 AIMessage 的关系

这是整个 Tool Calling 的核心。

很多人误以为：

```text
Human

↓

Tool

↓

AI
```

实际上：

真正流程：

```text
HumanMessage

↓

AIMessage
（我要调用Tool）

↓

Tool

↓

ToolMessage
（Tool执行结果）

↓

AIMessage
（最终回答）
```

所以：

**ToolMessage 永远位于两个 AIMessage 中间。**

---

# 完整流程示意图

以查询天气为例：

```text
用户：
北京天气怎么样？

        │
        ▼

HumanMessage
        │
        ▼

AIMessage
──────────────────────────────
tool_calls:
get_weather("北京")
──────────────────────────────

        │
        ▼

Weather Tool

        │
        ▼

ToolMessage
──────────────────────────────
content:
北京
晴
26℃
──────────────────────────────

        │
        ▼

AIMessage
──────────────────────────────
北京今天晴，
26℃，
适合外出。
──────────────────────────────
```

可以看到：

**ToolMessage 不是最终答案，它只是工具执行结果的载体。真正面向用户的自然语言回答，仍然由最后一个 AIMessage 生成。**

---

# ToolMessage 与 Agent

现在你应该能够理解整个 Agent 的消息流了：

```text
SystemMessage
        │
        ▼
规定 AI 行为

HumanMessage
        │
        ▼
用户提出问题

AIMessage
        │
        ▼
决定调用哪些工具（tool_calls）

ToolMessage
        │
        ▼
工具执行结果返回模型

AIMessage
        │
        ▼
整理工具结果，生成最终回复
```

因此，**Agent 本质上就是不断循环处理这四类 Message**。

---

# 总结

* **ToolMessage 用于表示一次工具执行的结果，它负责将工具输出传递回模型，而不是直接返回给用户。**
* **ToolMessage 总是对应某个 `AIMessage.tool_calls`，两者通过 `tool_call_id` 一一关联。**
* 常用字段包括：

  * **content**：发送给模型的工具执行结果。
  * **tool_call_id**：对应工具调用的唯一 ID，必须与 `AIMessage.tool_calls` 中的 ID 一致。
  * **name**：工具名称，便于日志和调试。
  * **artifact**：仅供程序使用的附加数据，不会发送给模型，适合保存文档来源、原始返回值、调试信息等。
* **完整的 Tool Calling 生命周期是：`HumanMessage → AIMessage(tool_calls) → ToolMessage → AIMessage(最终回答)`。理解这一消息流，是掌握 LangChain Agent 和 LangGraph 工作原理的关键。** ([LangChain Docs][1])

[1]: https://docs.langchain.com/oss/python/langchain/messages?utm_source=chatgpt.com "Messages - Docs by LangChain"
