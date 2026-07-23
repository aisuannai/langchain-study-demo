"""
Streaming with Human-in-the-Loop - 流式输出 + 人工审批

官方文档: https://docs.langchain.com/oss/python/langchain/streaming#streaming-with-human-in-the-loop
解释文档: docs/stream_human_in_the_loop.md

核心流程:
  Phase 1 - 首次流: Agent 流式输出 -> 遇到需审批的工具 -> 打断 -> 收集 Interrupt
  Phase 2 - 决策:   对每个 action_request 做 approve / edit / reject
  Phase 3 - 恢复流: Command(resume=decisions) 恢复执行 -> 工具执行 -> LLM 总结

注意:
  - 必须配合 checkpointer (InMemorySaver)，否则中断状态无法持久化
  - 这个场景只能用 v2 API (agent.stream), v3 没有 __interrupt__ 投影
  - 使用非 OpenAI 模型时, tool_call_chunks 行为可能略有不同

运行:
  python app/stream/stream_human_in_the_loop.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Windows GBK 终端修正: 遇到 Unicode 字符不崩溃
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.qwen_model import model as llm
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessageChunk
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt
from langchain_core.utils.uuid import uuid7


# ============================================================
# 1. 定义工具
# ============================================================

@tool(description="query city weather, params: city=city name")
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"{city} weather is sunny"


# ============================================================
# 2. 创建 Agent - 带 HumanInTheLoopMiddleware
# ============================================================
# interrupt_on 声明哪些工具需要人工审批
# checkpointer 必须提供，否则中断无法恢复

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[get_weather],
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"get_weather": True}),
    ],
    checkpointer=checkpointer,
)

# ============================================================
# 3. Phase 1: 首次流式调用 - 收集 Interrupt
# ============================================================
# 同时问两个城市，让 LLM 产生两个 tool_call，方便展示审批流程

print("=" * 60)
print("Phase 1: 首次流式调用 - 收集 Interrupt")
print("=" * 60)

input_message = {
    "role": "user",
    "content": "What is the weather in Boston and San Francisco?",
}
config = {"configurable": {"thread_id": str(uuid7())}}

interrupts: list[Interrupt] = []

for chunk in agent.stream(
    {"messages": [input_message]},
    config=config,
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if isinstance(token, AIMessageChunk):
            if token.text:
                print(token.text, end="", flush=True)
            if token.tool_call_chunks:
                print(f"\n[tool call chunk]: {token.tool_call_chunks}")

    elif chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source in ("model", "tools"):
                messages = update["messages"]
                last = messages[-1]
                if hasattr(last, "tool_calls") and last.tool_calls:
                    print(f"\n[完整 Tool Calls]: {last.tool_calls}")
                if hasattr(last, "content") and last.content:
                    print(f"\n[完整消息]: {last.content[:100]}")

            # 关键: __interrupt__ 是中断信号
            if source == "__interrupt__":
                interrupts.extend(update)
                print(f"\n[中断] 共 {len(update[0].value['action_requests'])} 个操作待审批")
                for req in update[0].value["action_requests"]:
                    print(f"  -> {req['description']}")

print(f"\n\n收集到 {len(interrupts)} 个中断对象")
print("Agent 已暂停，等待人类决策...\n\n")


# ============================================================
# 4. Phase 2: 模拟人类决策
# ============================================================
# 演示三种决策类型：
#   - Boston   -> 编辑（城市名改为 Boston, U.K.）
#   - San Francisco -> 批准

print("=" * 60)
print("Phase 2: 人类决策")
print("=" * 60)


def make_decisions(interrupt: Interrupt) -> list[dict]:
    """模拟人类对每个 action_request 做出决策"""
    decisions = []
    for request in interrupt.value["action_requests"]:
        desc = request["description"].lower()
        if "boston" in desc:
            # 编辑：修改城市参数
            decision = {
                "type": "edit",
                "edited_action": {
                    "name": "get_weather",
                    "args": {"city": "Boston, U.K."},
                },
            }
            print(f"  [编辑] {request['description']}")
            print(f"      -> 改为: get_weather(city='Boston, U.K.')")
        else:
            # 批准
            decision = {"type": "approve"}
            print(f"  [批准] {request['description']}")
        decisions.append(decision)
    return decisions


resume_decisions = {}
for interrupt in interrupts:
    resume_decisions[interrupt.id] = {
        "decisions": make_decisions(interrupt)
    }

print(f"\n决策字典: {resume_decisions}\n\n")


# ============================================================
# 5. Phase 3: 恢复流 - Command(resume=decisions)
# ============================================================
# 同一个 config (thread_id) 告诉 checkpointer 从之前中断处继续
# 用 Command(resume=...) 传入决策

print("=" * 60)
print("Phase 3: 恢复执行 - 带决策恢复 Agent")
print("=" * 60)

for chunk in agent.stream(
    Command(resume=resume_decisions),
    config=config,
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if isinstance(token, AIMessageChunk):
            if token.text:
                print(token.text, end="", flush=True)

    elif chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source in ("model", "tools"):
                messages = update["messages"]
                last = messages[-1]
                if hasattr(last, "tool_calls") and last.tool_calls:
                    print(f"\n[恢复后 Tool Calls]: {last.tool_calls}")
                if hasattr(last, "content") and last.content:
                    print(f"\n[LLM 总结]: {last.content}")
            if source == "__interrupt__":
                print(f"\n[中断 - 继续审批]")
                for req in update[0].value["action_requests"]:
                    print(f"  -> {req['description']}")

print("\n\n[完成] Agent 执行完成")
