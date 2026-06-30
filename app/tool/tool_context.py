"""
Context vs State 对比总结
==========================

                    State（短期记忆）                      Context（上下文）
  ──────────────────────────────────────────────────────────────────────
  生命周期    整个 thread，跨多轮对话                      单次 invoke，每次都要传
  可变性      tool 可通过 Command(update=...) 修改          不可变，tool 只能读不能写
  用途        对话历史、中间计算结果、累加器                user_id、用户角色、API key、功能开关
  传参方式    invoke({"user_id": "abc", ...})               invoke(..., context=UserContext(user_id="abc"))
  类型声明    state_schema=UseState                         context_schema=UserContext
  tool 内访问 runtime.state["field"]                        runtime.context.field

关键区别
--------
1. state = 对话过程的数据（可读写），需要 reducer 处理并发更新
2. context = 调用时就知道的配置（只读），每次 invoke 传入

⚠️  state 跨多轮会话的两个前提条件（缺一不可）：
   ┌─ checkpointer ─────────────────────────────────────────────────┐
   │  InMemorySaver() 或 PostgresSaver 等，负责保存 state 快照      │
   │  无 checkpointer → 每次 invoke 都是全新的 state，之前全丢      │
   └────────────────────────────────────────────────────────────────┘
   ┌─ thread_id ────────────────────────────────────────────────────┐
   │  标识"哪场对话"，checkpointer 按 thread_id 存/取 state          │
   │  无 thread_id → 每次都是新 thread，checkpointer 也找不到旧数据  │
   └────────────────────────────────────────────────────────────────┘
   总结: checkpointer 决定"能不能存"，thread_id 决定"存到哪/从哪取"

⚠️  context 不受 checkpointer 管理，也不受 thread_id 影响：
   ┌─ 不被 checkpoint 保存 ─────────────────────────────────────────┐
   │  每次 invoke 都必须显式传入 context=...                         │
   │  同 thread 的两次 invoke 之间，context 不会自动延续             │
   └────────────────────────────────────────────────────────────────┘
   ┌─ 不传就取不到 ─────────────────────────────────────────────────┐
   │  runtime.context.xxx → 取不到值                                 │
   │  文档原话: "context carries per-run data"                       │
   └────────────────────────────────────────────────────────────────┘
   对比: state   = 对话数据（checkpointer + thread_id 持久化）
         context = 调用配置（每次 invoke 重新传入）

本文件包含两个示例：
    示例 1 — 基础 context 用法（你原来的代码，已保留）
    示例 2 — context + state + checkpointer 完整演示
"""

# =============================================================================
# 示例 1: context 基础用法（你原来的代码）
# =============================================================================

from uuid import uuid7

from app.qwen_model import model as qwen_model
from langchain.tools import ToolRuntime, tool
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from dataclasses import dataclass


USER_DATA = {
    "user123": {
        "name": "jack",
        "balance": "2000"
    },
    "user456": {
        "name": "joke",
        "balance": "3000"
    }
}


@dataclass
class UserInfo:
    user_id: str


@tool(description="find user info")
def find_user_info(runtime: ToolRuntime[UserInfo]) -> str:
    user_id = runtime.context.user_id
    if user_id in USER_DATA:
        user = USER_DATA[user_id]
        return f'user info, name: {user["name"]}, balance: {user["balance"]}'
    return "no user info"


agent = create_agent(
    model=qwen_model,
    tools=[find_user_info],
    context_schema=UserInfo
)

response = agent.invoke(
    {'messages': HumanMessage('How user balance')},
    config={'configurable': {'thread_id': str(uuid7())}},
    context=UserInfo(user_id='user123')
)

#print(f'结果:{response["messages"]}')

# =============================================================================
# 示例 2: context + state 同时使用（对比演示）
# =============================================================================

from langchain.agents import AgentState
from langgraph.types import Command
from langchain.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

@dataclass
class UserContext:
    user_id: str
    role: str = "guest"


class ChatState(AgentState):
    user_name: str


# checkpointer: state 持久化的前提，没有它 thread_id 传了也没用
checkpointer = InMemorySaver()


@tool
def get_context(runtime: ToolRuntime[UserContext]) -> str:
    """读取当前调用的 context 信息和 state 中的用户名"""
    user_name = runtime.state.get("user_name", "")
    return (
        f"当前用户: {runtime.context.user_id}, "
        f"角色: {runtime.context.role}, "
        f"state中的用户名: {user_name or '(未设置)'}"
    )


@tool
def set_name(name: str, runtime: ToolRuntime) -> Command:
    """设置用户名到 state（可写）"""
    return Command(
        update={
            "user_name": name,
            "messages": [
                ToolMessage(
                    content=f"用户名已设为: {name}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


agent2 = create_agent(
    model=qwen_model,
    tools=[get_context, set_name],
    context_schema=UserContext,
    state_schema=ChatState,
    checkpointer=checkpointer,
)

thread_a = str(uuid7())  # 固定 thread，模拟同一场对话
thread_b = str(uuid7())  # 另一个 thread，模拟另一场对话


def show_round(label: str, result: dict):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for m in result["messages"]:
        print(f"  [{m.type}] {m.content}")
    print(f"  → state['user_name'] = '{result.get('user_name', '')}'")


# ── 第一轮: thread_a, 设置 user_name = "小明" ─────────────
r1 = agent2.invoke(
    {"messages": [HumanMessage("设置用户名为 小明")], "user_name": ""},
    config={"configurable": {"thread_id": thread_a}},
    context=UserContext(user_id="user_123", role="vip"),
)
show_round("第一轮: thread_a, 设 user_name='小明'", r1)


# ── 第二轮: 新 thread_b, 没有 checkpoint 可恢复 ────────────
r2 = agent2.invoke(
    {"messages": [HumanMessage("查我的 context")]},
    config={"configurable": {"thread_id": thread_b}},
    context=UserContext(user_id="user_456", role="admin"),
)
show_round("第二轮: 新 thread_b, state 是全新的", r2)


# ── 第三轮: 回到 thread_a, state 从 checkpoint 恢复 ────────
r3 = agent2.invoke(
    {"messages": [HumanMessage("查我的 context")]},
    config={"configurable": {"thread_id": thread_a}},
    context=UserContext(user_id="user_789", role="normal"),
)
show_round("第三轮: 回到 thread_a, 上一轮的 state 还在", r3)