import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "app")

from deepseek_model import model
from uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent
from langchain.tools import ToolRuntime

checkpointer = InMemorySaver()

def get_weather(city: str, runtime: ToolRuntime) -> str:
    return f"Weather in {city} is sunny"

agent = create_deep_agent(
    model=model,
    tools=[get_weather],
    checkpointer=checkpointer,
)

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "weather in beijing?"}]},
    config={"configurable": {"thread_id": str(uuid7())}},
    version="v3",
)

for i, result in enumerate(stream.values):
    print(f"=== 第{i}步 ===")
    print(f"  result 类型: {type(result)}")
    if isinstance(result, dict):
        print(f"  result keys: {list(result.keys())}")
    last = result["messages"][-1]
    print(f"  messages[-1] 类型: {type(last).__name__}")
    # 列出公开属性
    attrs = [a for a in dir(last) if not a.startswith("_")]
    print(f"  可用属性: {attrs}")
    print()
    if i >= 3:
        break
