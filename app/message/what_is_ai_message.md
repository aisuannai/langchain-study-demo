AIMessage 是 **Messages 这一章最重要的内容**，也是很多人第一次接触 Tool Calling、Agent 时容易理解错的地方。

很多初学者认为：

> HumanMessage 是输入，AIMessage 就是模型返回的一段文本。

**其实这是错误的理解。**

从 LangChain 0.3+ 开始，**AIMessage 已经不仅仅代表"回答"，它还是模型所有输出的统一载体（Container）**。模型返回的工具调用、推理内容、Token 使用统计、响应元数据等，都保存在 AIMessage 中。([LangChain Docs][1])

下面我按照适合做学习笔记的方式进行讲解。

---

# AIMessage（AI 消息）

`AIMessage` 表示**模型（LLM）的输出**。

每当调用一次 ChatModel：

```python
response = model.invoke(messages)
```

返回的对象就是：

```python
AIMessage(...)
```

而不是字符串。

例如：

```python
response = model.invoke("Explain AI")

print(type(response))
```

输出：

```python
<class 'langchain.messages.AIMessage'>
```

因此：

> **AIMessage 就是模型回答的一次完整封装，而不仅仅是一段文本。** ([LangChain Docs][1])

---

# AIMessage 的作用

官方说明：

AIMessage 可以包含：

* 文本（Text）
* 多模态内容（Multimodal Content）
* Tool Calls（工具调用）
* Provider Metadata（模型厂商元数据）

这说明：

AIMessage 不只是：

```text
你好
```

而更像：

```text
AIMessage
├── content
├── tool_calls
├── usage_metadata
├── response_metadata
├── id
└── ...
```

所以：

> **AIMessage 是模型一次完整响应的数据对象（Response Object）。**

---

# AIMessage 是如何产生的？

例如：

```python
messages = [
    HumanMessage("什么是Redis？")
]

response = model.invoke(messages)
```

执行流程：

```text
HumanMessage
      │
      ▼
 ChatModel
      │
      ▼
 AIMessage
```

也就是说：

> **只有模型调用 (`invoke()`、`stream()`) 才会真正生成 AIMessage。**

---

# AIMessage 也可以手动创建

很多人看到官方示例都会疑惑：

```python
ai_msg = AIMessage(
    "I'd be happy to help!"
)
```

为什么自己还能创建 AIMessage？

原因是：

LangChain 的消息历史（Conversation History）本质就是：

```python
[
    Message,
    Message,
    Message
]
```

如果希望模拟历史聊天：

```text
用户：
你好

AI：
你好，有什么可以帮助你的？

用户：
Redis 是什么？
```

可以写成：

```python
messages = [
    HumanMessage("你好"),
    AIMessage("你好，有什么可以帮助你的？"),
    HumanMessage("Redis是什么？")
]
```

然后直接继续聊天。

所以：

> **手动创建 AIMessage 的目的，不是生成回答，而是构造聊天历史（Conversation History）。** ([LangChain Docs][1])

---

# AIMessage 常用属性

官方列出了几个最重要的属性。

---

## 1. text

```python
response.text
```

表示：

AI 回复的纯文本。

例如：

```text
Redis 是一个高性能 Key-Value 数据库。
```

这是最简单的文本访问方式。

---

## 2. content

```python
response.content
```

这是最核心的字段。

它保存：

模型真正返回的数据。

可能是：

字符串：

```python
"Hello"
```

也可能是：

多模态内容：

```python
[
    {
        "type":"text",
        "text":"..."
    }
]
```

因此：

> **text 可以理解为 content 的文本快捷访问，而 content 才是真正的数据载体。**

---

## 3. tool_calls

这是 AIMessage 最重要的属性之一。

如果模型决定调用工具：

例如：

```text
今天天津天气怎么样？
```

模型不会直接回答。

而是：

```text
AI：

我要调用：

get_weather(
    location="天津"
)
```

那么：

```python
response.tool_calls
```

就是：

```python
[
    {
        "name":"get_weather",
        "args":{
            "location":"天津"
        },
        "id":"call_xxx"
    }
]
```

然后程序：

```
AIMessage

↓

tool_calls

↓

执行 Tool

↓

ToolMessage

↓

再次调用模型
```

所以：

> **Tool Calling 的入口，就是 AIMessage.tool_calls。** ([LangChain Docs][1])

---

## 4. usage_metadata

这个字段保存：

Token 使用统计。

例如：

```python
response.usage_metadata
```

返回：

```python
{
    "input_tokens":8,
    "output_tokens":304,
    "total_tokens":312
}
```

甚至还能统计：

```text
reasoning tokens

cache tokens

audio tokens
```

适合：

* Token 计费
* 成本统计
* 性能监控

---

## 5. response_metadata

保存模型厂商返回的信息。

例如：

```python
{
    "model_name":"gpt-5",
    "finish_reason":"stop"
}
```

不同 Provider：

返回内容不同。

例如：

OpenAI

Anthropic

Gemini

DeepSeek

都会放这里。

因此：

> **response_metadata 保存的是模型提供商特有的响应信息。**

---

## 6. id

Message 唯一 ID。

方便：

* Trace
* Debug
* Logging
* LangSmith

一般无需手动设置。

---

# AIMessage 与 Tool Calling 的关系

这是 LangChain Agent 最核心的流程。

普通聊天：

```text
Human

↓

AIMessage

↓

结束
```

Tool Calling：

```text
Human

↓

AIMessage
（包含 tool_calls）

↓

Tool

↓

ToolMessage

↓

AIMessage（最终答案）
```

因此：

> **AIMessage 并不一定直接回答用户，它也可能只是告诉程序："请帮我调用这个工具。"** ([LangChain Docs][1])

---

# AIMessage 与 Streaming

如果开启流式输出：

```python
for chunk in model.stream("Hi"):
    ...
```

返回的不是：

```python
AIMessage
```

而是：

```python
AIMessageChunk
```

每个 Chunk：

代表一小段 Token。

例如：

```
Hel

↓

lo

↓

!
```

最后：

```python
full_message = chunk1 + chunk2 + chunk3
```

得到：

```python
AIMessage
```

所以：

```
AIMessageChunk

↓

不断拼接

↓

AIMessage
```

这就是流式输出的工作原理。([LangChain Docs][1])

---

# AIMessage 与字符串有什么区别？

很多新人都会这样写：

```python
answer = model.invoke(messages)

print(answer)
```

感觉：

返回就是：

```text
Redis 是……
```

其实：

真正返回的是：

```python
AIMessage(
    content="Redis 是……",
    usage_metadata=...,
    tool_calls=[],
    response_metadata=...
)
```

只是：

```python
print()
```

帮我们格式化显示了。

因此：

> **AIMessage 是对象（Object），字符串只是它的 content。**

---

# 总结

* **AIMessage 表示模型的一次完整输出，是 ChatModel 调用后的返回对象。**
* **它不仅包含回答内容（content），还封装了工具调用（tool_calls）、Token 使用统计（usage_metadata）、模型响应信息（response_metadata）以及消息 ID 等元数据。**
* **AIMessage 可以由模型自动生成，也可以手动创建，用于构造多轮对话历史。**
* 在 Tool Calling 中，**AIMessage 往往不是最终答案，而是发起工具调用的起点；工具执行完成后，再结合 `ToolMessage` 生成新的 AIMessage 返回给用户。**
* 在流式输出场景中，模型首先返回多个 **AIMessageChunk**，最终再合并为一个完整的 **AIMessage**。([LangChain Docs][1])

## 学习建议

到这里，建议你建立这样一个整体模型：

```text
SystemMessage
        │
        ▼
规定 AI 如何工作

HumanMessage
        │
        ▼
用户输入

ChatModel
        │
        ▼
AIMessage
        │
        ├── content（回答内容）
        ├── tool_calls（工具调用）
        ├── usage_metadata（Token 统计）
        ├── response_metadata（模型元数据）
        └── id（消息标识）
```
