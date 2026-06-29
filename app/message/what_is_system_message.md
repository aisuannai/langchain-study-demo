

# SystemMessage（系统消息）

`SystemMessage` 用于向模型提供**初始指令（Initial Instructions）**，决定模型应该如何思考、如何回答以及扮演什么角色。

它通常位于整个对话的最前面，在用户消息（HumanMessage）之前发送。

可以理解为：

> **SystemMessage 就是给 LLM 制定"游戏规则"。**

---

# SystemMessage 的作用

官方文档提到三个核心作用：

## 1. Priming the Model（预设模型行为）

SystemMessage 会在模型开始回答之前，提前"引导（Prime）"模型。

例如：

```python
from langchain_core.messages import SystemMessage

SystemMessage(
    content="你是一名经验丰富的Java架构师。"
)
```

之后无论用户问什么：

```text
什么是Redis？
```

模型都会尽量站在 **Java 架构师** 的角度回答。

也就是说：

```
SystemMessage
        ↓
影响整个对话
        ↓
HumanMessage
        ↓
AIMessage
```

它不会回答一个问题，而是**影响后续所有问题的回答方式**。

---

## 2. Set the Tone（设定回答风格）

可以控制 AI 的表达方式。

例如：

正式风格

```python
SystemMessage(
    content="请使用正式、专业的语气回答。"
)
```

幽默风格

```python
SystemMessage(
    content="请用轻松幽默的方式回答。"
)
```

儿童模式

```python
SystemMessage(
    content="请用5岁小朋友能够理解的话解释问题。"
)
```

例如用户问：

```
什么是TCP？
```

不同的 SystemMessage 会得到完全不同的回答。

因此：

> **SystemMessage 可以控制"怎么回答"，而不是"回答什么"。**

---

## 3. Define the Model's Role（定义模型角色）

这是最常见的用途。

例如：

```python
SystemMessage(
    content="你是一名资深Python工程师。"
)
```

或者：

```python
SystemMessage(
    content="你是一位英语老师。"
)
```

或者：

```python
SystemMessage(
    content="你是一名健身教练。"
)
```

模型会尽可能保持这一身份回答问题。

例如：

用户：

```
如何减脂？
```

如果角色不同：

健身教练：

```
建议控制热量摄入……
```

营养师：

```
重点是饮食结构……
```

医生：

```
如果存在基础疾病……
```

同一个问题，答案重点完全不同。

---

## 4. Establish Guidelines（建立回答规则）

除了角色，还可以规定回答必须遵循的规则。

例如：

限制回答长度：

```python
SystemMessage(
    content="回答不要超过100字。"
)
```

固定输出格式：

```python
SystemMessage(
    content="""
所有回答必须使用 Markdown。
"""
)
```

要求一步步解释：

```python
SystemMessage(
    content="回答时请按照步骤分析。"
)
```

要求始终输出 JSON：

```python
SystemMessage(
    content="""
所有回答必须输出 JSON。
"""
)
```

很多 Agent 和自动化系统都会使用这种方式，保证模型输出符合程序预期。

---

# 一个完整示例

```python
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

messages = [
    SystemMessage(
        content="""
你是一名Java专家，
回答要专业、简洁，
示例代码使用Java。
"""
    ),
    HumanMessage(
        content="什么是线程池？"
    )
]
```

模型收到的是：

```
SystemMessage
│
├── 角色：Java专家
├── 风格：专业
├── 要求：简洁
└── 代码：Java

↓

HumanMessage

↓

AIMessage
```

可以看到，**SystemMessage 并不是一个问题，而是一组持续生效的约束条件。**

---

# SystemMessage 会一直生效吗？

一般来说，只要它仍然包含在发送给模型的 Message 列表中，它就会持续影响模型。

例如：

```python
messages = [
    SystemMessage(...),
    HumanMessage(...),
    AIMessage(...),
    HumanMessage(...)
]
```

第二次调用模型时：

```python
llm.invoke(messages)
```

由于 `SystemMessage` 仍然存在，所以它依然会影响模型回答。

如果移除它：

```python
messages = [
    HumanMessage(...),
    AIMessage(...),
    HumanMessage(...)
]
```

模型就失去了最初的角色和规则约束。

因此在多轮对话中，通常都会保留同一个 `SystemMessage`。

---

# 需要注意的地方

### 1. SystemMessage 并非绝对命令

它是模型的重要指导信息，但并不能保证 100% 被遵守。如果用户提出与系统提示冲突的要求，模型会综合系统消息、用户消息以及安全策略进行权衡，因此系统消息具有**较高优先级**，但不是不可违背的硬性规则。

### 2. 并非所有模型都原生支持 SystemMessage

像 OpenAI、Anthropic 等主流聊天模型都支持 `system` 角色；有些模型没有独立的 `system` 角色，LangChain 会自动把 `SystemMessage` 转换成模型支持的提示格式，因此开发者通常仍然可以统一使用 `SystemMessage`，无需关心底层差异。

---

# 总结

* **SystemMessage 是发送给模型的初始指令，用于定义整个对话的行为和规则。**
* 它通常位于消息列表最前面，并持续影响后续所有对话。
* SystemMessage 的主要作用包括：

  * **预设模型行为（Priming）**：引导模型按照指定方式思考和回答。
  * **设定回答风格（Tone）**：控制回答的语气、详细程度、表达方式等。
  * **定义模型角色（Role）**：让模型扮演特定身份，如 Java 架构师、英语老师或健身教练。
  * **建立回答规则（Guidelines）**：约束输出格式、语言、长度或其他要求。
* **最佳实践**：将角色、风格和输出规则统一放在 `SystemMessage` 中，把用户的具体问题放在 `HumanMessage` 中，这也是 LangChain 和大多数 LLM 应用推荐的提示设计方式。
