# LangChain Model 系统性学习指南

> 基于 docs.langchain.com Models 章节 + 项目 demo 实践
> 目标：用体系化的思维理解 Model 在 LangChain 中的完整面貌

---

## 目录

- [一、Model 全景图](#一model-全景图)
- [二、模型初始化](#二模型初始化)
- [三、模型参数与连接弹性](#三模型参数与连接弹性)
- [四、三种调用方式](#四三种调用方式)
- [五、Configurable Models（动态配置）](#五configurable-models动态配置)
- [六、Agent 生命周期（核心流程图）](#六agent-生命周期核心流程图)
- [七、Tool Calling（工具调用）](#七tool-calling工具调用)
- [八、Structured Output（结构化输出）](#八structured-output结构化输出)
- [九、Token Usage & Callbacks](#九token-usage--callbacks)
- [十、进阶特性](#十进阶特性)
- [十一、设计思想总览](#十一设计思想总览)
- [附录：项目文件映射表](#附录项目文件映射表)

---

## 一、Model 全景图

### 1.1 Model 在 LangChain 生态中的位置

```
LangChain 三层架构
┌─────────────────────────────────────────────────────┐
│  Harness: Deep Agents                               │
│  (create_deep_agent：文件系统 + 子代理 + 上下文压缩) │
├─────────────────────────────────────────────────────┤
│  Runtime: LangGraph                                 │
│  (状态管理 + 流式输出 + 人在回路 + 持久化)          │
├─────────────────────────────────────────────────────┤
│  Framework: LangChain                               │
│  (Model + Tool + Message + Agent 抽象)              │
│                     ↑                                │
│              你在这里 ── Model 是 Agent 的推理引擎    │
└─────────────────────────────────────────────────────┘
```

### 1.2 Model 的核心职责

```
┌─────────────────────────────────────────────────────────┐
│                    Model (LLM)                          │
│                                                         │
│  输入: Messages         输出: AIMessage                 │
│  (System/User/Assistant)      ├─ text / content_blocks │
│                                ├─ tool_calls           │
│                                ├─ usage_metadata       │
│                                └─ response_metadata    │
│                                                         │
│  核心能力：                                              │
│  ├─ 文本生成（invoke / stream / batch）                 │
│  ├─ Tool Calling（调用外部工具）                         │
│  ├─ Structured Output（结构化输出）                      │
│  ├─ Reasoning（思考链）                                  │
│  └─ Multimodal（多模态，如图片理解）                     │
└─────────────────────────────────────────────────────────┘
```

### 1.3 学习路径

```text
基础层 → 模型初始化 → 三种调用方式 → 参数配置
  │
进阶层 → Configurable Models → Agent 生命周期
  │
能力层 → Tool Calling → Structured Output
  │
观测层 → Token Usage → Callbacks
  │
深入层 → Reasoning → LogProbs → Model Profiles
```

---

## 二、模型初始化

### 2.1 两种初始化方式

```
方式 A: init_chat_model()  ← 推荐，方便切换 Provider
─────────────────────────────────────────────────

  from langchain.chat_models import init_chat_model

  model = init_chat_model(
      model="glm-5.1",
      model_provider="openai",    ← 指定 Provider
      api_key=...,
      base_url=...,
  )

  特点：
  ！ 支持 configurable_fields（运行时动态切换）
  ！ 统一接口，更换 Provider 只需改 model_provider
  ！ 模型命名格式："{provider}:{model_name}"
      例: "openai:gpt-5"、"google_genai:gemini-3.5-flash"


方式 B: 直接实例化  ← 特定 Provider 的原生 SDK
─────────────────────────────────────────────────

  from langchain_qwq import ChatQwen

  model = ChatQwen(
      model="qwen3.7-plus",
      api_key=...,
      base_url=...,
  )

  特点：
  ！ 可以使用 Provider 特有的参数
  ！ ChatQwen 能获取 reasoning_content（思考过程）
  ！ 不支持 configurable_fields
```

**📁 相关文件：**
- `app/deepseek_model.py` — 方式 A（init_chat_model，实际调用智谱 glm-5.1）
- `app/qwen_model.py` — 方式 B（ChatQwen 直接实例化）

### 2.2 本项目双模型体系

```
┌──────────────────────────────────────────────────────┐
│                   本项目双模型                         │
│                                                       │
│  deepseek_model.py          qwen_model.py             │
│  (init_chat_model)          (ChatQwen 直接实例化)     │
│       │                           │                   │
│       ▼                           ▼                   │
│  model="glm-5.1"            model="qwen3.7-plus"     │
│  provider="openai"          (DashScope 兼容)          │
│                                                       │
│  ⚠️ 命名陷阱:                                          │
│  deepseek_model.py 实际是 glm-5.1（智谱），不是 DeepSeek│
│  thinking_model.py 实际是 qwen3.7-plus（通义千问）     │
└──────────────────────────────────────────────────────┘
```

### 2.3 初始化参数

| 参数 | 作用 | 示例 |
|------|------|------|
| `model` | 模型名称/标识符 | `"glm-5.1"` |
| `model_provider` | Provider 名称 | `"openai"`，仅 init_chat_model 需要 |
| `api_key` | API 密钥 | 通常从环境变量读取 |
| `base_url` | API 端点地址 | DashScope: `https://dashscope.aliyuncs.com/compatible-mode/v1/` |
| `temperature` | 随机性控制 | `0.7` |
| `max_tokens` | 最大输出 token 数 | `1000` |
| `max_retries` | 失败重试次数（默认 6） | `10`（不稳定网络） |
| `timeout` | 请求超时（秒） | `30` |

---

## 三、模型参数与连接弹性

### 3.1 标准参数速查

| 参数 | 作用 | 默认值 | 典型范围 |
|------|------|--------|---------|
| `temperature` | 控制随机性。越低越确定，越高越有创造力 | 因 Provider 而异 | 0.0 ~ 2.0 |
| `max_tokens` | 限制输出长度 | 因 Provider 而异 | 视场景而定 |
| `max_retries` | 网络错误/限流时的自动重试次数 | `6` | 6 ~ 15 |
| `timeout` | 等待响应的最大秒数 | 因 Provider 而异 | 30 ~ 120 |

### 3.2 Connection Resilience（连接弹性）

```
请求失败
  │
  ├─ 网络超时          ─┐
  ├─ Rate Limit (429)   ├─ 自动重试（指数退避 + jitter）
  ├─ Server Error (5xx) ─┘
  │
  ├─ 401 (未授权)  ── 不重试，直接抛异常
  └─ 404 (不存在)  ── 不重试，直接抛异常

  重试策略:
  第 1 次失败 → 等待 ~1s → 重试
  第 2 次失败 → 等待 ~2s → 重试
  第 3 次失败 → 等待 ~4s → 重试
  ...指数递增，最大 6 次（默认）

  ⚠️ 长时间运行 Agent + 不稳定网络 →
    建议 max_retries=10~15
```

### 3.3 Rate Limiter（限流器）

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,       # 每秒 1 个请求
    check_every_n_seconds=0.1,   # 每 0.1 秒检查一次是否可发新请求
    max_bucket_size=10,          # 最大同时支持 10 个请求
)

model = ChatQwen(..., rate_limiter=rate_limiter)
```

| 特性 | 说明 |
|------|------|
| 线程安全 | ✅ 单进程内多线程安全 |
| 多进程安全 | ❌ 多 Agent Server 场景需要自己实现 |
| 适用场景 | 单机版开发/测试，生产级多实例需外部存储 |

**📁 相关文件：** `app/qwen_model.py`（第 6-10 行）、`app/model/agent_rate_limiter.py`

---

## 四、三种调用方式

### 4.1 对比总览

```
                    invoke()             stream()               batch()
              ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    特点      │ 阻塞，一次返回 │   │ 逐 token 返回  │   │ 并行多请求    │
              │              │   │  实时消费     │   │ 全部完成再返回 │
    返回类型  │ AIMessage    │   │ AIMessageChunk│   │ list[AIMessage]│
    适用场景  │ 简单问答      │   │ 长文本/实时展示 │   │ 批量独立请求  │
    用户体验  │ 等待 → 看到   │   │ 边看边等      │   │ 等全部 → 看到 │
    对应 Demo │ agent_stream  │   │ agent_stream  │   │ agent_batch   │
              │ 中的 invoke   │   │ 中的 stream   │   │              │
```

### 4.2 invoke — 单次调用

```python
# 最简单的方式：传入字符串
response = model.invoke("Why do parrots talk?")

# 传入消息列表（带对话历史）
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]
response = model.invoke(conversation)

# 也支持 Message 对象
from langchain.messages import HumanMessage
response = model.invoke([HumanMessage("Hello!")])
```

### 4.3 stream — 流式输出

```python
# 基础流式：逐 token 输出
for chunk in model.stream("Why do parrots talk?"):
    print(chunk.text, end="", flush=True)

# 进阶：流式 content_blocks（含 reasoning + tool_call_chunk）
for chunk in model.stream("What color is the sky?"):
    for block in chunk.content_blocks:
        if block["type"] == "reasoning":
            print(f"思考: {block['reasoning']}")
        elif block["type"] == "tool_call_chunk":
            print(f"工具调用: {block}")
        elif block["type"] == "text":
            print(block["text"])

# 关键：chunk 可以通过累加合并为完整消息
full = None
for chunk in model.stream("Hello"):
    full = chunk if full is None else full + chunk  # ← 累加
```

**流式 vs 非流式的深层理解：**

```
invoke()     → LLM 生成完整响应 → 一次性返回
               用户全程等待，看不到中间过程

stream()     → LLM 每生成一个 token 就推送一次
               用户实时看到生成过程

astream_events() → 不仅流式输出 token，还流式输出事件
   ├─ on_chat_model_start  → 输入内容
   ├─ on_chat_model_stream → 每个 token
   └─ on_chat_model_end    → 完整消息

本质：LangChain 是"事件驱动执行系统"
  invoke   = 事件的批量收集方式
  stream   = 事件的实时消费方式
```

**📁 相关文件：** `app/model/agent_stream.py`（完整演示 + 详细注释）

### 4.4 batch — 批量并行

```python
# batch() — 所有任务完成后再返回
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?",
])

# batch_as_completed() — 按完成顺序逐个返回（结果可能乱序）
for task_id, result in model.batch_as_completed(inputs):
    print(f"[任务 {task_id}]: {result}")

# 控制并行度
model.batch(inputs, config={"max_concurrency": 5})
```

| 方法 | 返回时机 | 顺序 | 适用场景 |
|------|---------|------|---------|
| `batch()` | 全部完成 | 保序 | 需要按原始顺序处理结果 |
| `batch_as_completed()` | 逐个完成 | 乱序（带 index） | 需要尽快处理每个结果 |

> ⚠️ `model.batch()` ≠ Provider 的 Batch API（如 OpenAI Batch）
> model.batch() 是客户端并行，Provider Batch API 是服务端异步批处理。

**📁 相关文件：** `app/model/agent_batch.py`

---

## 五、Configurable Models（动态配置）

这是 LangChain Model 最强大的特性之一：**在运行时动态切换模型配置，而不需要重新创建模型实例。**

### 5.1 三种配置模式

```
模式一：无默认模型，运行时必须指定
────────────────────────────────────
model = init_chat_model(model_provider="openai")
# 调用时必须传:
model.invoke("hello", config={
    "configurable": {"model": "glm-5.1"}
})

特点：model=None → 自动开启 configurable_fields=("model", "model_provider")


模式二：config_prefix 多实例 + 全字段可配
────────────────────────────────────
model1 = init_chat_model(
    model_provider="openai",
    configurable_fields="any",    # "any" = 全部字段都可运行时覆盖
    config_prefix="fix_1",       # config 中 key 加 fix_1_ 前缀
)

model2 = init_chat_model(
    model_provider="openai",
    configurable_fields="any",
    config_prefix="fix_2",
)

# 统一 config 字典，各前缀对应各自的 key
config = {
    "configurable": {
        "fix_1_model": "glm-5.1",     # model1 消费
        "fix_1_api_key": "...",
        "fix_2_model": "qwen3.7-plus", # model2 消费
        "fix_2_api_key": "...",
    }
}
model1.invoke("hello", config=config)  # 用 glm-5.1
model2.invoke("hello", config=config)  # 用 qwen3.7-plus


模式三：model 固定，仅部分字段可配
────────────────────────────────────
model3 = init_chat_model(
    model="glm-5.1",                  # model 已固定
    configurable_fields=("temperature", "max_tokens"),  # 只能改这两个
    config_prefix="fix_3",
)

# fix_3_model 传了也没用，model3 固定用 glm-5.1
# 但 fix_3_temperature 和 fix_3_max_tokens 生效
```

### 5.2 理解 config_prefix

```
config_prefix 的作用：类似 Spring 的 application-{env}.yaml
─────────────────────────────────────────────────────────

背景：多个模型实例共享同一个 config 字典时
问题：key 会冲突（大家都是 "model"）
解决：加前缀区分

┌──────────────────────────────────────────────────┐
│ config = {                                        │
│   "configurable": {                               │
│     "fix_1_model": "glm-5.1",    ← model1 读这里  │
│     "fix_2_model": "qwen3.7",    ← model2 读这里  │
│     "fix_1_temperature": 0.5,                     │
│     "fix_2_temperature": 0.8,                     │
│   }                                               │
│ }                                                 │
│                                                   │
│ model1 = init_chat_model(..., config_prefix="fix_1")│
│ model2 = init_chat_model(..., config_prefix="fix_2")│
└──────────────────────────────────────────────────┘
```

### 5.3 Dynamic Model Selection（中间件方式）

除了 `configurable_fields`，还可以通过 Middleware 在运行时切换模型：

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state['messages'][-1].content)
    if message_count < 5:
        request = request.override(model=qwen_model)      # 短消息用 Qwen
    else:
        request = request.override(model=deepseek_model)  # 长消息用 DeepSeek
    return handler(request)

agent = create_agent(
    model=qwen_model,
    middleware=[dynamic_model_selection]
)
```

**两种动态模型选择方式的对比：**

| 方式 | 粒度 | 适用场景 |
|------|------|---------|
| `configurable_fields` | 模型参数级 | 同一 Provider 的不同模型间切换 |
| `@wrap_model_call` Middleware | 请求级 | 不同 Provider、不同实例间切换，可追加业务逻辑 |

**📁 相关文件：**
- `app/model/configurable_model.py` — 三种配置模式的完整 demo
- `app/model/dynamic_model _selection.py` — Middleware 方式动态选模型

---

## 六、Agent 生命周期（核心流程图）

> 这一节回答：`agent.invoke(messages)` 内部到底发生了什么？

### 6.1 完整生命周期图

```text
┌──────────────────────────────┐
│        User Input            │
│  agent.invoke(messages)      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│ 1. 创建 RunnableConfig                                │
│------------------------------------------------------│
│ run_name     → LangSmith 追踪标识                     │
│ tags         → 标签（继承到子调用）                    │
│ metadata     → 自定义元数据（继承）                    │
│ callbacks    → [重点] 事件回调处理器                    │
│ recursion_limit → 链式递归深度上限（防无限循环）       │
│ max_concurrency → batch 最大并行数                    │
│ configurable → 动态配置入口（见第五章）                │
│                                                       │
│ ⚠️ config 与 context 的区别：                          │
│   config: LangChain 运行时控制（tags/metadata/...）    │
│   context: 你的业务数据（通过 context_schema 自定义）  │
└──────────────┬────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│ 2. Middleware（@wrap_model_call）                     │
│------------------------------------------------------│
│ 中间件 = AOP 切面编程，在模型调用前后插入逻辑          │
│                                                       │
│ ✔ 动态选择模型（request.override(model=new_model)）   │
│ ✔ 修改 Prompt                                         │
│ ✔ 权限校验                                            │
│ ✔ 日志记录                                            │
│ ✔ 缓存                                                │
│ ✔ 熔断 / 降级                                        │
│                                                       │
│ 顺序：先注册的中间件先执行（洋葱模型）                 │
└──────────────┬────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│ 3. 确定最终 Model                                     │
│------------------------------------------------------│
│ 经过 Middleware 后，最终使用的 Model 可能是：          │
│   - 原始传入的 model                                  │
│   - Middleware override 后的 model                    │
│   - configurable 动态选择的 model                     │
│                                                       │
│ 原则：先确定模型，再附加能力                           │
└──────────────┬────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│ 4. Agent 自动绑定 Tool                                │
│------------------------------------------------------│
│ model.bind_tools(tools)                               │
│                                                       │
│ 支持两种工具来源：                                     │
│ ① 用户自定义工具（@tool 装饰器）                      │
│ ② Provider 内置工具（如 web_search）                 │
│                                                       │
│ tool_choice 控制:                                     │
│   "auto"  → 模型自主决定                              │
│   "any"   → 强制使用任意工具                           │
│   "xxx"   → 强制使用指定工具                           │
│                                                       │
│ parallel_tool_calls = True（默认支持并行调用）         │
└──────────────┬────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│ 5. Agent 自动绑定 Structured Output                   │
│------------------------------------------------------│
│ model.with_structured_output(Schema)                  │
│                                                       │
│ 或 agent 级别: response_format=Schema                 │
│                                                       │
│ 两种实现策略：                                        │
│   ProviderStrategy → 原生支持（更可靠）               │
│   ToolStrategy     → 通过 Tool Calling 模拟（更通用） │
└──────────────┬────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│ 6. 调用 LLM                                           │
│------------------------------------------------------│
│ HTTP POST → Provider API                              │
│                                                       │
│ 请求携带：                                             │
│ ├─ model: "glm-5.1"                                  │
│ ├─ messages: [...]                                    │
│ ├─ temperature: 0.7                                   │
│ ├─ max_tokens: 1000                                   │
│ ├─ max_retries: 6（指数退避 + jitter）               │
│ ├─ tools: [...]（如果绑定了工具）                     │
│ ├─ tool_choice: "auto"                                │
│ ├─ logprobs: True（可选，见第十章）                    │
│ └─ extra_body: {"enable_thinking": False}（可选）     │
│                                                       │
│ 限流控制：Rate Limiter 在 request 发出前检查           │
└──────────────┬────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│ 7. Tool Calling（如果需要）                            │
│------------------------------------------------------│
│ LLM 返回 tool_calls（请求执行工具）                    │
│  ↓                                                    │
│ Agent 执行对应工具                                    │
│  ↓                                                    │
│ 工具返回 ToolMessage                                  │
│  ↓                                                    │
│ 把 ToolMessage 放回消息列表                           │
│  ↓                                                    │
│ 再次调用 LLM（让模型基于工具结果生成回答）             │
│  ↓                                                    │
│ 重复以上步骤直到 LLM 返回纯文本回答（无 tool_calls）   │
│                                                       │
│ ⚠️ 每次 tool call → tool result → 再次 LLM 调用      │
│    都会消耗一轮的 recursion_limit                      │
└──────────────┬────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│ 8. Callbacks（事件系统）                               │
│------------------------------------------------------│
│ 通过 config 注入的 callback handler 会收到事件：       │
│                                                       │
│ on_chain_start          → Agent 开始执行              │
│ on_chat_model_start     → LLM 开始调用                │
│ on_chat_model_stream    → 每个 token（流式）           │
│ on_chat_model_end       → LLM 调用完成                │
│ on_tool_start           → 工具开始执行                │
│ on_tool_end             → 工具执行完成                │
│ on_chain_end            → Agent 执行完成              │
│                                                       │
│ 预置处理器：                                           │
│ UsageMetadataCallbackHandler → 跨模型累计 token 用量   │
│                                                       │
│ 下游系统：                                             │
│ LangSmith Trace（全链路追踪）                          │
│ Token 用量统计                                        │
│ 成本监控                                              │
└──────────────┬────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│ 9. 选择输出方式                                        │
│------------------------------------------------------│
│ invoke()                                              │
│   → 阻塞等待，全部完成后一次性返回 AIMessage           │
│   → 简单问答场景                                      │
│                                                       │
│ stream()（模型层）                                     │
│   → 逐 token 返回 AIMessageChunk                      │
│   → chunk 可累加（chunk1 + chunk2 = 完整消息）         │
│                                                       │
│ stream_events()（Agent 层）                            │
│   → 每步返回完整 state 快照                           │
│   → 能看到 tool 调用、中间结果、最终回答               │
│   → Agent 场景推荐（详见 app/model/agent_stream.py）   │
└──────────────┬────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│      Final Response                          │
│                                              │
│  AIMessage:                                  │
│  ├─ content / text                          │
│  ├─ content_blocks（标准化格式）              │
│  ├─ tool_calls                              │
│  ├─ usage_metadata（token 用量）              │
│  └─ response_metadata（含 logprobs 等）      │
└──────────────────────────────────────────────┘
```

### 6.2 设计思想

```
整个生命周期遵循一个重要原则：

  「先确定最终使用哪个模型，再给模型附加能力
   （Tool、Structured Output），最后才真正调用 LLM。」

因此：

  Raw Model
      ↓
  Middleware（选择模型 / 拦截逻辑）
      ↓
  bind_tools()
      ↓
  with_structured_output()
      ↓
  invoke() / stream() / batch()

而不是：

  ❌ bind_tools()
  ❌ with_structured_output()
  ❌ 再切换模型

这也是官方强调
"Dynamic Model Selection 不支持 Pre-bound Models" 的原因。
```

---

## 七、Tool Calling（工具调用）

### 7.1 两种工具来源

```
Tool Calling 流程：

  用户输入 "今天天气怎么样？"
      ↓
  LLM 判断需要调用 get_weather 工具
      ↓
  LLM 返回 tool_calls（请求执行）
      ↓
  Agent 执行对应工具函数
      ↓
  工具返回结果（ToolMessage）
      ↓
  LLM 基于工具结果生成最终回答


工具来源一：用户自定义工具
──────────────────────
  from langchain.tools import tool

  @tool
  def get_weather(location: str) -> str:
      """获取指定城市的天气。"""  ← docstring 必须，给模型看的
      return f"{location} 天气晴朗"

  model.bind_tools([get_weather])

  规则：
  ① 必须有类型注解
  ② 必须有 docstring
  ③ 参数名不能是 config 或 runtime（保留字）
  ④ 工具名用 snake_case


工具来源二：Provider 内置工具（server-side）
──────────────────────
  agent = create_agent(
      model=model,
      tools=[{"type": "web_search"}]   ← 大模型侧执行
  )

  特点：
  ├─ 不在本地执行，在大模型服务器端执行
  ├─ 不经过 tool_choice 控制
  └─ 不需要 @tool 装饰器
```

### 7.2 tool_choice 控制

```python
# 默认：模型自主决定是否调工具
model.bind_tools([tool_1, tool_2])

# 强制使用任意工具（模型必须选一个）
model.bind_tools([tool_1], tool_choice="any")

# 强制使用指定工具
model.bind_tools([tool_1], tool_choice="tool_1")

# 禁用并行工具调用
model.bind_tools([tool_1], parallel_tool_calls=False)
```

### 7.3 手动 Tool 执行循环

```python
# 当使用模型（而非 Agent）时，需要手动处理 tool calling 循环
# Agent 自动做了这些，但理解底层原理很重要

# 步骤 1：模型生成 tool_calls
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# 步骤 2：执行工具，收集结果
for tool_call in ai_msg.tool_calls:
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# 步骤 3：把工具结果放回消息列表，再次调用模型
final_response = model_with_tools.invoke(messages)
```

**📁 相关文件：**
- `app/model/agent_web_tool.py` — Provider 内置工具（web_search）
- `app/tool/agent_tool_runtime.py` — ToolRuntime 高级用法

---

## 八、Structured Output（结构化输出）

### 8.1 两种实现路径

```
路径一：模型级（推荐，简单场景）
────────────────────
  from pydantic import BaseModel, Field

  class Person(BaseModel):
      name: str = Field(description="person name")
      age: int = Field(description="person age")

  model_with_structure = model.with_structured_output(Person)
  result = model_with_structure.invoke("谁是李世民")
  # result 是 Person 实例，不是字符串


路径二：Agent 级（复杂场景，含工具调用）
────────────────────
  agent = create_agent(
      model=model,
      response_format=Person,   ← Agent 自动处理
  )

  result = agent.invoke({"messages": [...]})
  result["structured_response"]  ← 结构化结果从这里取
```

### 8.2 支持的 Schema 类型

| 类型 | 返回值 | 适用场景 |
|------|--------|---------|
| Pydantic `BaseModel` | Pydantic 实例 | 需要验证、嵌套结构 |
| `TypedDict` | dict | 简单场景，不需要运行时验证 |
| JSON Schema `dict` | dict | 跨语言兼容 |

### 8.3 ProviderStrategy vs ToolStrategy

```
ProviderStrategy（原生策略）
────────────────────────
  原理：利用 Provider 原生 API 的结构化输出能力
  优点：最可靠，质量最高
  适用：OpenAI、Anthropic、Gemini 等主流模型

ToolStrategy（工具策略）
────────────────────────
  原理：通过 Tool Calling 机制模拟结构化输出
  优点：所有支持 Tool Calling 的模型都可用
  缺点：依赖 tool_choice，与 enable_thinking=True 冲突

  ⚠️ 本项目中的坑：
  create_agent + response_format 依赖 tool_choice
  如果 model 用了 enable_thinking=True → 冲突报错
```

### 8.4 注意事项

```
❶ Field(deprecated=...) 不可用
   Pydantic v2 中 deprecated 不是 Field 的有效参数
   会导致 TypeError。
   你的 agent_structured_output.py:12 就是这个问题。

❷ create_deep_agent 不适合简单结构化输出
   因为 deep_agent 会注入大量内置工具
   （ls, execute, task...），token 消耗极大。
   建议：简单结构化输出用 model.with_structured_output() 替代。

❸ 错误处理（ToolStrategy）
   ToolStrategy(
       schema=Movie,
       handle_errors=True,      → 自动重试
       handle_errors=False,     → 不重试，抛异常
   )
```

**📁 相关文件：** `app/model/agent_structured_output.py`

---

## 九、Token Usage & Callbacks

### 9.1 Token 用量统计

```python
# 方式一：单次调用，从 response 获取
response = model.invoke("Hello!")
print(response.usage_metadata)
# {'input_tokens': 8, 'output_tokens': 304, 'total_tokens': 312}

# 方式二：跨模型累计（推荐）
from langchain_core.callbacks import UsageMetadataCallbackHandler

callback = UsageMetadataCallbackHandler()

model_1.invoke("Hello", config={"callbacks": [callback]})
model_2.invoke("Hello", config={"callbacks": [callback]})

print(callback.usage_metadata)
# {'glm-5.1': {'input_tokens': 8, ...}, 'qwen3.7-plus': {...}}

# 方式三：上下文管理器
from langchain_core.callbacks import get_usage_metadata_callback

with get_usage_metadata_callback() as cb:
    model_1.invoke("Hello")
    model_2.invoke("Hello")
    print(cb.usage_metadata)
```

### 9.2 Callback 事件系统

```
Callback 机制：

  注入（步骤 1）                   触发（步骤 8）
  ┌──────────────┐                ┌──────────────────┐
  │ config 传入   │    runtime      │ LLM 执行过程中    │
  │ callbacks     │ ──────────────→ │ 自动触发事件      │
  │ = [handler]   │                │ handler.on_xxx() │
  └──────────────┘                └──────────────────┘

  典型用途：
  ├─ Token 用量统计（UsageMetadataCallbackHandler）
  ├─ 日志记录
  ├─ 流式输出（on_llm_new_token）
  ├─ 监控告警
  └─ LangSmith 追踪
```

### 9.3 config 中 callbacks 的传递规则

```
config 中设置的 tags / metadata / callbacks
会沿着调用链自动继承到子调用：

agent.invoke(
    config={
        "tags": ["demo"],
        "metadata": {"user_id": "123"},
        "callbacks": [handler],
    }
)
  │
  ├─ model.invoke()       ← 继承 tags, metadata, callbacks
  ├─ tool.invoke()        ← 继承 tags, metadata
  └─ (子模型调用)         ← 继承
```

**📁 相关文件：**
- `app/model/agent_token_usage.py` — UsageMetadataCallbackHandler demo
- `app/model/agent_runable_config.py` — RunnableConfig 各字段演示

---

## 十、进阶特性

### 10.1 Reasoning / Thinking（思考链）

```
本项目中的 reasoning 困境：
────────────────────────

DashScope API 返回 reasoning_content 字段
         │
         ▼
init_chat_model(provider="openai")
  → LangChain OpenAI 适配器
  → ❌ 不解析 reasoning_content
  → content_blocks 中没有 type=reasoning
         │
         ▼
ChatQwen（langchain_qwq）
  → ✅ 能获取 reasoning_content
  → chunk.additional_kwargs["reasoning_content"]
         │
         ▼
enable_thinking=True 时
  → ❌ 不支持 tool_choice
  → create_agent 会出问题
```

```python
# 正确获取 reasoning 的方式（用 ChatQwen）
from langchain_qwq import ChatQwen

model = ChatQwen(model="qwen3.7-plus")
for chunk in model.stream([HumanMessage("Why do parrots talk?")]):
    reasoning = chunk.additional_kwargs.get("reasoning_content", "")
    if reasoning:
        print(f"思考: {reasoning}")
```

**📁 相关文件：**
- `app/model/agent_reasoning.py`（⚠️ 已废弃，但注释说明了原因）
- `app/thinking_model.py`

### 10.2 LogProbs（对数概率）

```
LogProbs = 模型对每个输出 token 的置信度
数值越接近 0 → 模型越有信心
数值负得越大 → 模型越不确定

主要用途：
  ① 判断模型信心：logprob 趋近 0 说明回答可靠
  ② Agent 决策：低置信度时触发确认流程
  ③ RAG 幻觉检测：对检索结果的引用置信度
  ④ Prompt 调优：对比不同 prompt 的输出置信度
```

```python
model.logprobs = True

response = model.invoke("明天是涨还是跌，只回复上涨或者下跌")

# 查看每个 token 的 logprob
print(response.response_metadata["logprobs"])
```

**📁 相关文件：** `app/model/model_logprobs.py`

### 10.3 Model Profiles（模型能力档案）

```
Beta 特性（langchain>=1.1），返回模型的能力描述字典：

  model.profile
  {
      "max_input_tokens": 400000,    ← 上下文窗口大小
      "image_inputs": True,          ← 是否支持图片输入
      "reasoning_output": True,      ← 是否支持思考链
      "tool_calling": True,          ← 是否支持工具调用
      "structured_output": True,     ← 是否支持结构化输出
      ...
  }

应用场景：
  ① Summarization Middleware：根据 max_input_tokens 决定是否触发摘要
  ② Structured Output：自动推断 Provider 原生支持还是需要 ToolStrategy
  ③ 输入门控：根据 image_inputs / max_input_tokens 过滤输入
  ④ 模型切换器：Deep Agents Code 根据 tool_calling 过滤可用模型

  也可以自定义覆盖 Profile：
  model = init_chat_model("...", profile=custom_profile)
```

**📁 相关文件：** `docs/langchain-basics-guide.md` 模块 3（Model Profiles 章节）

---

## 十一、设计思想总览

### 11.1 LangChain Model 的核心原则

```
① 统一接口，多 Provider 兼容
   ─────────────────────────
   不管用 OpenAI、Anthropic、Google、还是 DashScope，
   模型接口都是一样的：invoke() / stream() / batch()
   切换 Provider 只需要换 model_provider 参数。

② 先确定模型，再附加能力
   ─────────────────────────
   Raw Model → Middleware → bind_tools → with_structured_output → invoke
   不能在绑定工具之后才换模型（bind_tools 绑定的是具体模型的 schema）。

③ 模型不负责执行，只负责决策
   ─────────────────────────
   LLM 生成 tool_calls 但不会执行工具，
   执行是 Agent/Harness 的责任。
   LLM 是大脑，Tools 是手脚，Agent 是协调者。

④ 可靠性来自系统，不来自模型
   ─────────────────────────
   Retry（指数退避）+ Rate Limiter（防限流）
   + Checkpointer（状态持久化）+ Timeout（防死等）
   = 可靠的 LLM 应用

⑤ LangChain 本质是"事件驱动执行系统"
   ─────────────────────────
   invoke   = 事件的批量收集方式
   stream   = 事件的实时消费方式
   callback = 事件的监听分发机制
```

### 11.2 Model 各特性之间的关系

```
                          Model（统一接口）
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
      invoke/stream/batch    init_chat_model    model.profile
      （调用方式）            （初始化方式）       （能力档案）
           │
           ├──────────────────┐
           │                  │
      bind_tools()      with_structured_output()
      （工具能力）           （结构化能力）
           │
      ┌────┴────┐
      │         │
  用户工具    Provider 内置工具
  (@tool)    (web_search)
```

### 11.3 本项目各 demo 的定位

```
你的 demo 按功能分层：

调用层:   agent_batch.py        ← batch / batch_as_completed
          agent_stream.py       ← stream_events

配置层:   configurable_model.py      ← 三种配置模式
          dynamic_model_selection.py ← Middleware 动态选模型
          agent_runable_config.py    ← RunnableConfig 字段

能力层:   agent_structured_output.py ← with_structured_output
          agent_web_tool.py          ← Provider 内置工具

观测层:   agent_token_usage.py       ← UsageMetadataCallbackHandler
          model_logprobs.py          ← 对数概率

深入层:   agent_reasoning.py         ← reasoning（⚠️ 已废弃但记录坑）
          agent_rate_limiter.py      ← 限流器（指向 qwen_model.py）

模型层:   deepseek_model.py          ← init_chat_model 方式
          qwen_model.py              ← ChatQwen 直接实例化
          thinking_model.py          ← enable_thinking=True
```

---

## 附录：项目文件映射表

| 你的文件 | 对应知识点 | 推荐阅读顺序 |
|---------|-----------|------------|
| `app/deepseek_model.py` | init_chat_model + 模型初始化 | ① |
| `app/qwen_model.py` | ChatQwen 直接实例化 + Rate Limiter | ① |
| `app/model/agent_runable_config.py` | RunnableConfig 各字段 | ② |
| `app/model/agent_stream.py` | stream_events + invoke 对比 | ③ |
| `app/model/agent_batch.py` | batch / batch_as_completed | ③ |
| `app/model/configurable_model.py` | 三种配置模式 + config_prefix | ④ |
| `app/model/dynamic_model _selection.py` | Middleware 动态选模型 | ⑤ |
| `app/model/agent_structured_output.py` | response_format + ⚠️ 常见坑 | ⑥ |
| `app/model/agent_web_tool.py` | Provider 内置工具 | ⑥ |
| `app/model/agent_token_usage.py` | Token 用量统计 | ⑦ |
| `app/model/model_logprobs.py` | 对数概率 | ⑧ |
| `app/model/agent_reasoning.py` | Reasoning 困境记录 | ⑨ |
| `app/thinking_model.py` | enable_thinking 配置 | ⑨ |
| `app/model/agent_rate_limiter.py` | 限流器（指向 qwen_model.py） | ⑩ |
| `docs/langchain-basics-guide.md` | 完整学习指南（含 Model Profiles） | 参考 |

---

> 本文档是 `left_cycle.md` 的完整版，将原来的 Agent 生命周期图扩展为
> 覆盖 Model 整个章节的系统性学习指南。
>
> 核心流程图（第六章）保留了原版的设计思想，
> 并补充了初始化、调用方式、Configurable Models、Tool Calling、
> Structured Output、Token Usage、进阶特性等缺失的部分。
