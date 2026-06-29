"""
ToolRuntime — 工具的运行时上下文
=================================

1. 什么是 ToolRuntime？
   ToolRuntime 是 LangChain v1 引入的统一运行时访问接口。
   取代了旧版的 InjectedState、InjectedStore、InjectedToolCallId 等分散注入方式。
   在 tool 函数中声明 runtime: ToolRuntime 参数，框架自动注入，对模型透明。

2. ToolRuntime 提供的 6 个核心组件：

   | 属性                | 用途                          | 类似概念          |
   |--------------------|-------------------------------|------------------|
   | runtime.state      | 当前对话状态（短时记忆）        | 本次会话的消息历史 |
   | runtime.context    | 调用时传入的上下文（不可变）     | 用户 ID/API Key   |
   | runtime.store      | 持久化存储（长时记忆，跨对话）   | 数据库/Redis     |
   | runtime.stream_writer | 实时流式更新                 | WebSocket 推送    |
   | runtime.tool_call_id | 当前工具调用的唯一 ID          | 追踪 ID          |
   | runtime.execution_info | 线程 ID/运行 ID/重试次数     | 日志追踪          |

3. 注意事项：
   - runtime 参数在函数签名中声明，但模型看不到它（被 @tool 自动隐藏）
   - context 和 state 的区别：context 是每次 invoke 传入的，不可变；
     state 是 Agent 运行时内部维护的，可变
   - 要读写 state 中的自定义字段，需要定义 state_schema
"""

from dataclasses import dataclass
from uuid import uuid7

from app.deepseek_model import model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, ToolMessage
from deepagents import create_deep_agent
from langchain.agents import AgentState


# ============================================================
# 1. runtime.state — 读取对话状态（短时记忆）
# ============================================================
print("=" * 60)
print("1. runtime.state — 读取对话状态")
print("=" * 60)

@tool
def check_history(runtime: ToolRuntime) -> str:
    """Check how many messages have been exchanged in this conversation.
    This tool reads the current message history from state."""
    messages = runtime.state["messages"]
    msg_count = len(messages)
    # 提取最后一条用户消息
    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content[:50]
            break
    return f"对话中共有 {msg_count} 条消息，最近用户说: {last_user_msg}"

checkpointer = InMemorySaver()
agent = create_deep_agent(
    model=model,
    tools=[check_history],
    checkpointer=checkpointer,
)

thread_id = str(uuid7())
config = {"configurable": {"thread_id": thread_id}}

# 第一轮：先聊两句
agent.invoke(
    {"messages": [{"role": "user", "content": "我的名字是张三。"}]},
    config=config,
)
# 第二轮：调用 check_history，看看 Agent 是否记得历史
result = agent.invoke(
    {"messages": [{"role": "user", "content": "用 check_history 查看我们聊了多少条消息。"}]},
    config=config,
)
print(f"state 读取结果: {result['messages'][-1].text}")

# ============================================================
# 2. runtime.context — 读取运行时上下文
# ============================================================
print("\n" + "=" * 60)
print("2. runtime.context — 读取运行时上下文")
print("=" * 60)

@dataclass
class UserContext:
    """每次 invoke 传入的业务上下文。"""
    user_id: str        # 用户标识
    region: str         # 地区（用于个性化回复）
    is_vip: bool = False

@tool
def get_user_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get personalized greeting with user info.
    This tool reads runtime context (passed at invoke time)."""
    ctx = runtime.context
    return (
        f"用户 {ctx.user_id}（{'VIP' if ctx.is_vip else '普通'}用户）"
        f"来自 {ctx.region}，欢迎使用！"
    )

agent2 = create_deep_agent(
    model=model,
    tools=[get_user_info],
    context_schema=UserContext,
)

result = agent2.invoke(
    {"messages": [{"role": "user", "content": "调用 get_user_info。"}]},
    context=UserContext(user_id="u_001", region="北京", is_vip=True),
)
print(f"context 读取结果: {result['messages'][-1].text}")

# ============================================================
# 3. runtime.store — 长时记忆持久化
# ============================================================
print("\n" + "=" * 60)
print("3. runtime.store — 长时记忆（跨对话）")
print("=" * 60)

persistent_store = InMemoryStore()

@tool
def save_note(key: str, content: str, runtime: ToolRuntime) -> str:
    """Save a note to long-term memory so it persists across conversations.
    Args:
        key: Note identifier
        content: Note content to save"""
    store = runtime.store
    store.put(("notes",), key, {"text": content, "saved_at": "now"})
    return f"笔记 '{key}' 已保存。"

@tool
def read_note(key: str, runtime: ToolRuntime) -> str:
    """Read a note from long-term memory.
    Args:
        key: Note identifier to look up"""
    store = runtime.store
    item = store.get(("notes",), key)
    if item is None:
        return f"笔记 '{key}' 不存在。"
    return f"笔记 '{key}': {item.value['text']}"

agent3 = create_deep_agent(
    model=model,
    tools=[save_note, read_note],
    store=persistent_store,
)

# 第一次对话：保存笔记
thread_a = str(uuid7())
agent3.invoke(
    {"messages": [{"role": "user", "content": "帮我保存笔记: key=my_idea, content=用 LangChain 做一个自动总结工具"}]},
    config={"configurable": {"thread_id": thread_a}},
)

# 第二次对话（不同 thread_id）：读取笔记
thread_b = str(uuid7())
result = agent3.invoke(
    {"messages": [{"role": "user", "content": "帮我读取笔记 my_idea"}]},
    config={"configurable": {"thread_id": thread_b}},
)
print(f"store 跨对话读取结果: {result['messages'][-1].text}")

# ============================================================
# 4. runtime.stream_writer — 自定义流式进度
# ============================================================
print("\n" + "=" * 60)
print("4. runtime.stream_writer — 工具内实时流式输出")
print("=" * 60)

@tool
def step_by_step(task: str, runtime: ToolRuntime) -> str:
    """Execute a task step by step, streaming progress.
    Args:
        task: Description of the task to execute"""
    writer = runtime.stream_writer
    writer(f"📋 开始执行任务: {task}")
    writer(f"  → 步骤 1/3: 分析任务...")
    writer(f"  → 步骤 2/3: 处理数据...")
    writer(f"  → 步骤 3/3: 生成结果...")
    return f"任务 '{task}' 已完成。"

agent4 = create_deep_agent(
    model=model,
    tools=[step_by_step],
)

# stream_events 方式可以捕获 stream_writer 的输出
print("流式输出（stream_events 中的自定义消息）:")
stream = agent4.stream_events(
    {"messages": [{"role": "user", "content": "用 step_by_step 执行：数据分析报告"}]},
    version="v3",
)
for msg in stream.messages:
    if msg.text:
        print(f"  {msg.text}")

# ============================================================
# 5. runtime.tool_call_id + Command — 更新 State
# ============================================================
print("\n" + "=" * 60)
print("5. runtime.tool_call_id + Command — 更新 State")
print("=" * 60)

class CustomState(AgentState):
    """自定义 Agent 状态，含额外字段。"""
    counter: int = 0  # 记录工具调用次数

@tool
def increment_counter(runtime: ToolRuntime) -> Command:
    """Increment the internal counter by 1.
    This tool updates the agent's state using Command."""
    # runtime.tool_call_id 是本次工具调用的唯一标识
    old_val = runtime.state.get("counter", 0)
    new_val = old_val + 1
    return Command(
        update={
            "counter": new_val,
            "messages": [ToolMessage(
                content=f"计数器: {old_val} → {new_val}",
                tool_call_id=runtime.tool_call_id,
            )],
        }
    )

agent5 = create_deep_agent(
    model=model,
    tools=[increment_counter],
    state_schema=CustomState,
)

result = agent5.invoke(
    {"messages": [{"role": "user", "content": "调用 increment_counter 三次。"}]},
    # 此 demo 主要展示 Command 更新 state 的能力
    # Agent 可能只调一次，查看最终 state 即可
)
print(f"最终 counter 值: {result.get('counter', 'N/A')}")
