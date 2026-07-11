# =============================================================================
# 示例 1: state_schema 基本用法 — 自定义字段 + runtime.state 读取
# =============================================================================

from app.qwen_model import model as qwen_model
from langchain.tools import ToolRuntime, tool
from langchain.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent, AgentState


class UseState(AgentState):
    user_id: str
    user_preferences: dict


@tool(description="find user preferences")
def get_user_perference(runtime: ToolRuntime) -> str:
    """
    ToolRuntime.state: 访问当前 thread 的短期记忆（state）
    - state["messages"]: 对话历史，由 LangGraph 自动维护
    - state["自定义字段"]: 通过 state_schema 声明、invoke 时传入或 Command.update 写入
    - state 被 checkpointer 按 thread_id 持久化，跨轮次保持
    - 工具间通过 state 共享中间数据，无需经过 LLM
    """
    messages = runtime.state['messages']
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            print(f'human message:{message}')
    return runtime.state.get('user_preferences', {})


agent = create_agent(
    model=qwen_model,
    state_schema=UseState,
    tools=[get_user_perference]
)

#input参数就是AgentState类型
response = agent.invoke({
    "messages": [
        HumanMessage('i am jack'),
        HumanMessage('do you konw i like thing')
    ],
    "user_id": "abc",
    "user_preferences": {"lol": "League of Legends", "java": "Java Programming"}
})

print(f'结果:{response["messages"]}')

# =============================================================================
# 示例 2: reducer 演示 — 并发 tool call 对 state 字段的影响
# =============================================================================

from typing import Annotated
from operator import add
from langgraph.types import Command


def merge_strings(current: str, update: str) -> str:
    """自定义 reducer: 并发更新时用分号拼接"""
    if not current:
        return update
    return f"{current}; {update}"


class CounterState(AgentState):
    # messages: 内置 add_messages reducer，自动合并追加
    # total: 有 reducer → 累加，并发更新不冲突
    total: Annotated[int, add]
    # last_action: 有自定义 reducer → 并发时拼接而非报错
    last_action: Annotated[str, merge_strings]


@tool(description="处理订单")
def process_order(item: str, runtime: ToolRuntime) -> Command:
    return Command(
        update={
            "total": 1,
            "last_action": f"order:{item}",
            "messages": [
                ToolMessage(
                    content=f"订单 {item} 已处理",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool(description="处理退款")
def process_refund(item: str, runtime: ToolRuntime) -> Command:
    return Command(
        update={
            "total": 1,
            "last_action": f"refund:{item}",
            "messages": [
                ToolMessage(
                    content=f"退款 {item} 已处理",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


agent2 = create_agent(
    model=qwen_model,
    state_schema=CounterState,
    tools=[process_order, process_refund],
)

response2 = agent2.invoke({
    "messages": [
        HumanMessage("帮我处理 order:macbook 和 refund:iphone")
    ],
    "total": 0,
    "last_action": "",
})

print("\n=== Reducer 示例 ===")
for m in response2["messages"]:
    print(f"  [{m.type}] {m.content}")

print(f"\ntotal (有 reducer, add累加)     = {response2['total']}")
print(f"last_action (有 reducer)        = {response2['last_action']}")
