"""
message.tool_calls vs stream.tool_calls
=======================================

stream_events 返回的事件流对象提供两种 tool-call 投影：

  message.tool_calls  — 模型端：LLM 正在构造工具调用参数时的实时片段
  stream.tool_calls   — 执行端：工具实际执行的生命周期（输入、输出流、最终结果、错误）

核心区别：

  ┌──────────────────────┬─────────────────────────────────┬──────────────────────────────┐
  │                     │ message.tool_calls             │ stream.tool_calls            │
  ├──────────────────────┼─────────────────────────────────┼──────────────────────────────┤
  │ 谁产生的            │ LLM 模型                        │ 工具执行器 (ToolNode)         │
  │ 阶段                │ 模型决定调用工具，构造参数时     │ 工具被实际执行时              │
  │ 内容                │ 参数块的增量片段 (chunks)       │ 输入、输出流、最终结果/错误   │
  │ 是否依赖流式        │ 是，需要流式事件才能捕获        │ 是，需要流式事件才能捕获      │
  │ .get() 方法         │ 有，返回最终确定的 tool call    │ 无（直接遍历即可）            │
  │ 输出属性            │ chunks                          │ .input / .output_deltas /     │
  │                     │                                 │   .output / .error            │
  │ 典型用途            │ 实时显示"模型正在调用 XX 工具"   │ 实时显示工具执行进度和结果    │
  └──────────────────────┴─────────────────────────────────┴──────────────────────────────┘

注意事项：
   1. stream.messages 和 stream.tool_calls 共享同一个底层事件流。
      该底层流是单次迭代器（single-pass iterator），而非独立的数据缓存。
      顺序遍历（先 messages 后 tool_calls）会耗尽底层流，导致第二个投影没有数据。
      有两种解决方式：
        a) 同步：用 stream.interleave("messages", "tool_calls") 交错消费，一个循环同时监听多个投影。
        b) 异步：用 asyncio.gather + async for 并发消费，两个协程同时从底层流读取各自关注的事件。
  2. message.tool_calls 返回的是一个 ToolCallDelta 对象，它既是可迭代的（遍历 chunks），
     也有 .get() 方法获取最终确定的 tool call。
  3. stream.tool_calls 遍历得到的每个 call 对象有 .output_deltas（注意是 deltas 不是 details）。
"""

from app.qwen_model import model as qwen_model
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain.agents import create_agent


@tool(description="query city weather, pamars:city=city name")
def get_weather(city: str) -> str:
    return f"{city} weather is sunn"


agent = create_agent(model=qwen_model, tools=[get_weather])

stream = agent.stream_events(
    {"messages": HumanMessage("what is dezhou weather")}, version="v3"
)

# ============================================================
# Part 1: message.tool_calls — 模型端的工具调用参数
# ============================================================
print("=" * 60)
print("1. message.tool_calls — 模型正在构造工具调用参数")
print("=" * 60)

for msg in stream.messages:
    # 工具调用参数增量片段（流式 chunks）
    if msg.tool_calls:
        print(f"[{msg.node}] tool call chunks:")
        for chunk in msg.tool_calls:
            print(f"  chunk: {chunk}")

        # .get() 获取最终确定的 tool call
        finalized = msg.tool_calls.get()
        if finalized:
            print(f"  finalized: {finalized}")
    else:
        print(f"[{msg.node}] (no tool call chunks in this message)")

# ============================================================
# Part 2: stream.tool_calls — 工具执行生命周期
# ============================================================
# 注意：必须重新创建 stream，因为上面已经消费完了事件流
stream2 = agent.stream_events(
    {"messages": HumanMessage("what is dezhou weather")}, version="v3"
)

print()
print("=" * 60)
print("2. stream.tool_calls — 工具执行生命周期")
print("=" * 60)

for call in stream2.tool_calls:
    print(f"  tool_name: {call.tool_name}")
    print(f"  input:     {call.input}")
    print(f"  output_deltas type: {type(call.output_deltas).__name__}")
    for delta in call.output_deltas:
        print(f"    delta: {delta}")
    print(f"  output: {call.output}")
    print(f"  error:  {call.error}")

# ============================================================
# Part 3: interleave 方式 — 同时获取两者
# ============================================================
print()
print("=" * 60)
print("3. stream.interleave — 同时消费 messages + tool_calls")
print("=" * 60)

stream3 = agent.stream_events(
    {"messages": HumanMessage("what is dezhou weather")}, version="v3"
)

for name, item in stream3.interleave("messages", "tool_calls"):
    if name == "messages":
        print(f"[message] node={item.node}, text='{str(item.text)[:80]}'")
        if item.tool_calls:
            finalized = item.tool_calls.get()
            print(f"          tool_calls.get()={finalized}")
    elif name == "tool_calls":
        print(f"[tool exec] {item.tool_name}({item.input})")
        for delta in item.output_deltas:
            print(f"            delta: {delta}")
        print(f"            output: {item.output}")
