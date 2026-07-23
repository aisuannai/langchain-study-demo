"""
Streaming with Human-in-the-Loop — 交互版（真正的人工输入）

和 stream_human_in_the_loop.py 的区别:
  - Phase 2 不再用代码模拟决策
  - 通过 input() 让用户在终端里真的打字选择 approve / edit / reject

运行:
  python app/stream/stream_human_in_the_loop_interactive.py
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
from langchain.messages import AIMessageChunk
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


@tool(description="send email to someone, params: to=recipient, content=email content")
def send_email(to: str, content: str) -> str:
    """Send an email."""
    return f"Email sent to {to}: {content}"


# ============================================================
# 2. 创建 Agent
# ============================================================

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[get_weather, send_email],
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={
            "get_weather": True,
            "send_email": True,
        }),
    ],
    checkpointer=checkpointer,
)

# ============================================================
# 3. Phase 1: 首次流式调用 - 收集 Interrupt
# ============================================================

print("=" * 60)
print("Phase 1: 收集 Interrupt")
print("=" * 60)
print("提示: 你可以问天气，也可以让 agent 发邮件")
print()

user_question = input("请输入你的问题 > ")

config = {"configurable": {"thread_id": str(uuid7())}}
interrupts: list[Interrupt] = []

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": user_question}]},
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
                print(f"\n  [构造参数]: {token.tool_call_chunks}")

    elif chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source in ("model", "tools"):
                messages = update["messages"]
                last = messages[-1]
                if hasattr(last, "tool_calls") and last.tool_calls:
                    print(f"\n  [完整 Tool Calls]: {last.tool_calls}")
            if source == "__interrupt__":
                interrupts.extend(update)
                print(f"\n  [!] 等待审批: {len(update[0].value['action_requests'])} 个操作\n")


# ============================================================
# 4. Phase 2: 真正的人工输入决策
# ============================================================

print("\n" + "=" * 60)
print("Phase 2: 人工审批 — 请对每个操作做出决策")
print("=" * 60)


def prompt_decision(request: dict, idx: int) -> dict:
    """对单个 action_request 让用户在终端选择决策"""
    tool_name = request["name"]
    tool_args = request["args"]
    desc = request["description"]

    print(f"\n--- 操作 {idx + 1} ---")
    print(f"  工具: {tool_name}")
    print(f"  参数: {tool_args}")
    print()

    while True:
        choice = input("  决策? (a=批准, e=编辑参数, r=拒绝) > ").strip().lower()

        if choice == "a":
            return {"type": "approve"}

        elif choice == "e":
            print("  请输入新的参数 (每行一个 key=value, 空行结束):")
            new_args = {}
            while True:
                line = input("    ").strip()
                if not line:
                    break
                if "=" in line:
                    k, v = line.split("=", 1)
                    new_args[k.strip()] = v.strip()
            return {
                "type": "edit",
                "edited_action": {
                    "name": tool_name,
                    "args": new_args,
                },
            }

        elif choice == "r":
            return {"type": "reject"}

        else:
            print("  无效输入，请输入 a / e / r")


resume_decisions = {}
for interrupt in interrupts:
    action_requests = interrupt.value["action_requests"]
    decisions_list = []
    for i, req in enumerate(action_requests):
        decisions_list.append(prompt_decision(req, i))

    resume_decisions[interrupt.id] = {"decisions": decisions_list}

print(f"\n决策汇总: {resume_decisions}")


# ============================================================
# 5. Phase 3: 恢复执行
# ============================================================

print("\n" + "=" * 60)
print("Phase 3: 恢复执行")
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
                if hasattr(last, "content") and last.content:
                    print(f"\n[工具返回]: {last.content}")
            if source == "__interrupt__":
                print(f"\n[!] 仍有中断，继续审批...")

print("\n\n[完成]")
