
---

# HumanMessage（用户消息）

`HumanMessage` 表示**用户发送给模型的输入（User Input）**。

每当用户向 LLM 提问、上传图片、发送文件或语音时，程序最终都会将这些内容封装成一个 `HumanMessage` 对象，发送给模型。

可以理解为：

> **HumanMessage 就是模型眼中的"用户说的话"。**

---

# HumanMessage 的作用

`HumanMessage` 用于表示用户与模型之间的交互，它可以承载各种类型的输入，而不仅仅是文本。

官方支持的内容包括：

* 文本（Text）
* 图片（Image）
* 音频（Audio）
* 文件（File）
* PDF
* 多模态内容（Multimodal Content）

例如：

普通文本：

```python
from langchain_core.messages import HumanMessage

HumanMessage(
    content="你好！"
)
```

图片输入：

```python
HumanMessage(
    content=[
        {
            "type": "text",
            "text": "请描述这张图片。"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }
    ]
)
```

因此：

> **HumanMessage 表示的是"用户输入"，而不是单纯的一段字符串。**

---

# HumanMessage 常用字段

官方示例：

```python
human_msg = HumanMessage(
    content="Hello!",
    name="alice",
    id="msg_123"
)
```

下面分别介绍每个字段。

---

## 1. content（消息内容）

这是最重要的字段。

表示用户真正输入的数据。

例如：

```python
HumanMessage(
    content="什么是Spring Boot？"
)
```

模型收到的就是：

```text
用户：
什么是Spring Boot？
```

除了字符串，也可以是：

* 图片
* PDF
* 多模态列表
* 音频

所以它的数据类型比普通字符串更丰富。

---

## 2. name（用户名称，可选）

```python
HumanMessage(
    content="Hello!",
    name="alice"
)
```

`name` 用来标识**消息发送者**。

例如：

```text
Alice：
你好

Bob：
你好
```

对应：

```python
[
    HumanMessage(
        content="你好",
        name="alice"
    ),
    HumanMessage(
        content="你好",
        name="bob"
    )
]
```

这样模型就知道：

这是两个不同的人说的话。

### 使用场景

多人聊天：

```text
Alice：
今天吃什么？

Bob：
我想吃火锅。
```

客服系统：

```text
Customer：
退款

Operator：
处理中
```

多人 Agent：

```text
Planner
↓

Coder
↓

Reviewer
```

都可以使用 `name` 区分身份。

---

### 注意事项

官方特别说明：

> **不同模型提供商对 `name` 字段的支持程度不同。**

有些模型会使用 `name` 来辅助理解消息来源；

有些模型则会完全忽略它。

因此：

> **`name` 是一个可选字段，是否生效取决于底层模型提供商。**

在开发跨模型应用时，不应依赖 `name` 来实现关键业务逻辑。

---

## 3. id（消息 ID，可选）

```python
HumanMessage(
    content="Hello!",
    id="msg_123"
)
```

`id` 是消息的唯一标识符（Unique Identifier）。

它主要供程序使用，而不是给模型阅读。

例如：

```text
msg_001

↓

用户发送

↓

AI 回复

↓

记录日志
```

常见用途：

* 消息追踪（Tracing）
* 调试（Debug）
* 日志记录（Logging）
* 数据库存储
* Message 去重

例如：

```python
HumanMessage(
    content="你好",
    id="conversation001_msg003"
)
```

方便后续定位：

> 第1个会话，第3条消息。

---

# 一个完整示例

```python
from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

messages = [
    SystemMessage(
        content="你是一名Python老师。"
    ),
    HumanMessage(
        content="解释一下装饰器。"
    )
]
```

模型看到的是：

```text
System：
你是一名Python老师。

Human：
解释一下装饰器。
```

随后生成：

```text
AI：
装饰器（Decorator）是一种...
```

整个流程如下：

```text
SystemMessage
        │
HumanMessage
        │
        ▼
      ChatModel
        │
        ▼
AIMessage
```

---

# HumanMessage 与普通字符串有什么区别？

很多初学者会疑惑：

```python
llm.invoke("你好")
```

和

```python
llm.invoke([
    HumanMessage("你好")
])
```

有什么区别？

实际上：

第一种只是 LangChain 提供的简写方式。

内部最终仍然会转换成：

```python
[
    HumanMessage("你好")
]
```

因此：

> **真正发送给模型的始终是 Message，而不是字符串。**

当涉及多轮对话、图片输入、工具调用或 Memory 时，应优先使用 `HumanMessage`，因为它能完整表达消息的角色、内容和元数据。

---

# 总结

* **HumanMessage 表示用户发送给模型的输入，是模型接收用户信息的主要载体。**
* 它支持文本、图片、音频、文件等多模态内容，而不仅限于字符串。
* 常用字段包括：

  * **content**：消息内容，是 HumanMessage 最核心的字段。
  * **name（可选）**：标识消息发送者，适用于多人对话或多 Agent 场景，但是否生效取决于模型提供商。
  * **id（可选）**：消息唯一标识，主要用于日志、调试、Tracing 和消息管理，不参与模型推理。
* **最佳实践**：简单示例可以直接传入字符串，但在正式项目中，尤其是涉及多轮对话、Memory、Agent 或多模态输入时，建议显式使用 `HumanMessage`，这样代码更规范，也更容易扩展。
