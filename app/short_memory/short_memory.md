# 短期记忆管理：Trim / Delete / Summarize

> 官方文档：https://docs.langchain.com/oss/python/langchain/short-term-memory  
> 参考代码：`trim_message.py` · `delete_message.py` · `summarize_message.py`

---

## 一、背景

Agent 的短期记忆存储在 `state["messages"]` 中，由 checkpointer 持久化。随着对话进行，消息列表不断增长，会面临：

- LLM 上下文窗口超限
- Token 费用增加
- 响应变慢
- 模型被过期/无关内容干扰

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

### 原理

在 `@before_model` 中，保留第一条消息（System）和末尾若干条消息，删掉中间过期的内容。

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

**关键点**：

| 要素 | 说明 |
|------|------|
| 时机 | `@before_model`（LLM 调用前） |
| 方式 | `REMOVE_ALL_MESSAGES` 清空 + 重新 Add |
| 效果 | System + 末尾 N 条保留，中间全部丢弃 |

### 项目代码对照（`trim_message.py`）

项目中实现的是按内容过滤，而非按长度裁剪，但模式相同：

```python
@before_model
def trim_message(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state['messages']
    if len(messages) > 0:
        # 保留不包含敏感词的消息
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

共同的核心模式：**Remove All + Add New = Replace**。

---

## 三、Delete Messages（指定删除）

### 原理

通过 `RemoveMessage(id=具体ID)` 精确删除某几条消息，删除后其余消息顺序不变。

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
- Message ID 来源：自动生成的 UUID、显式指定 `HumanMessage(id="x")`、或通过 `get_state()` 查询获取
- **`REMOVE_ALL_MESSAGES`** 是 sentinel 常量，值为 `'__remove_all__'`，reducer 看到它直接清空 `messages = []`

### 为什么需要 `add_messages` Reducer

`RemoveMessage` 能被识别并执行删除，依赖 `messages` 字段配置了 `add_messages` reducer。默认 `AgentState` 继承自 `MessagesState`，自带此配置。如果自定义 state 没有用 `MessagesState`，`RemoveMessage` 不会生效。

### 合法消息约束

删除后必须保证消息序列符合 LLM 要求：

```
❌ 首条是 AI         → AI("Hi") → Human(...)
❌ tool_call 无结果   → AI(tool_call) → AI(...)      （缺 ToolMessage）
❌ ToolMessage 无调用 → ToolMessage(...) → AI(...)   （缺 AI(tool_call)）
```

### 删除全部消息

```python
return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

### 外部删除

不经过 middleware，直接用 `graph.update_state()` 也能删：

```python
graph.update_state(config, {
    "messages": [RemoveMessage(id=target_id)]
})
```

### 项目代码对照（`delete_message.py`）

```python
@after_model
def delete_message(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state['messages']
    if len(messages) > 2:
        return {
            "messages": [RemoveMessage(id=m.id) for m in messages[:2]]
        }
    return None
```

**执行效果 trace**：

```
Round  消息数  触发删除？   剩余
──────────────────────────────
1       1      否          1
2       2      否          2
3       3      是（删前2）  1
4       2      否          2
5       3      是（删前2）  1
...    循环往复
```

最终效果：**永远只保留最近的 1-2 条消息**。

---

## 四、Trim vs Delete 对比

| 特性 | Trim | Delete |
|------|------|--------|
| 删除范围 | 批量删中间，保留头尾 | 按 ID 精确删除 |
| 实现方式 | `REMOVE_ALL_MESSAGES` + 重新 Add | `RemoveMessage(id=具体ID)` |
| 执行时机 | `@before_model`（LLM 前） | `@after_model`（LLM 后） |
| 灵活性 | 固定保留头尾 | 任意位置删除 |
| 典型场景 | 控制 Token 长度 | 删敏感/错误消息 |

**选择依据**：

- 需要保头尾 → **Trim**
- 需要删中间某条 → **Delete**
- 保头尾但中间内容不能丢 → **Summarize**

---

## 五、Summarize Messages（摘要）

### 原理

用 LLM 生成旧消息的摘要，替换原始消息，既压缩 Token 又保留关键信息。

### 官方示例

```python
SummarizationMiddleware(
    model="gpt-5.4-mini",
    trigger=("tokens", 4000),   # Token 超 4000 时触发
    keep=("messages", 20)        # 保留最近 20 条 + 摘要
)
```

### 参数说明

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

### 执行效果

```
触发前  [msg0, msg1, msg2, msg3, msg4, ...]
                           ↓
触发后  [SummaryMessage, msg_last_N, ...]
```

摘要以结构化文本呈现，包含对话要点：

```
Here is a summary of the conversation to date:

## SESSION INTENT
...
## SUMMARY
...
## NEXT STEPS
...
```

---

## 六、三种策略对比

| 特性 | Trim | Delete | Summarize |
|------|------|--------|-----------|
| 信息保留 | 部分保留（头尾） | 保留其余所有 | 通过摘要保留语义 |
| Token 压缩 | 高 | 高 | 中（摘要本身占 Token） |
| 精度 | 丢失中间内容 | 精确控制 | 依赖 LLM 摘要质量 |
| 实现复杂度 | 低（无额外调用） | 低 | 高（需 LLM 调用） |
| 推荐场景 | Token 临近限制时 | 精准编辑历史 | 长对话需要保留语义 |

---

## 七、完整执行流程

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
    ▼
@after_model           ← Delete 在这里执行
    │
    ▼  return dict | None → Reducer 合并 → 新 State
保存到 checkpointer（checkpoints + checkpoint_writes）
    │
    ▼
返回结果
```

---

## 八、参考代码索引

| 文件 | 策略 | 触发时机 |
|------|------|----------|
| `trim_message.py` | Trim（按内容做 Replace） | `@before_model` |
| `delete_message.py` | Delete（按数量删旧消息） | `@after_model` |
| `summarize_message.py` | Summarize + 调试打印 | `SummarizationMiddleware` + `@before_model` |
