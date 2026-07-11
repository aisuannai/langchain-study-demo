"""
Streaming Sub-Agents — 事件流中的子代理消费

核心概念（来自 LangChain 官方文档）:
  - create_agent + stream_events(version="v3") 支持子代理事件流。
  - 当 A 通过 tool 调用另一个具名 create_agent B 时，B 的内部事件会
    以嵌套命名空间的形式暴露在 stream.subagents 投影上。
  - stream.subagents 是 "具名 create_agent 子代理" 的专用视图，
    每个子代理暴露:
      .name       — create_agent(name=...) 传入的名称
      .cause      — 触发该子代理的 tool call
      .messages   — 子代理内的 LLM 消息流（含 .text / .reasoning / .tool_calls）
      .values     — 子代理状态快照
      .tool_calls — 子代理内的工具执行生命周期
      .output     — 子代理最终状态
  - stream.subgraphs 覆盖所有嵌套图（包括普通 StateGraph），
    而 stream.subagents 只包含具名 create_agent，无需过滤。
  - 对应代码模式见文档 "Streaming sub-agents" 章节:
    https://docs.langchain.com/oss/python/langchain/event-streaming#streaming-sub-agents
"""

from app.qwen_model import model as qwen_model
from langchain.tools import  tool
from langchain.messages import HumanMessage
from langchain.agents import create_agent


@tool(description='query city weather, pamars:city=city name')
def get_weather(city:str)->str:
    return f'{city} weather is sunn'


weather_agent = create_agent(
    model=qwen_model,
    tools=[get_weather],
    name='weahter_agent'
)

@tool(description='query weather agent, pamars:query=you weather proleam')
def call_weather_agent(query:str)->str:
    result = weather_agent.invoke({
        'messages': HumanMessage(query)
    })
    return result['messages'][-1].content

main_agent = create_agent(
    model=qwen_model,
    tools=[call_weather_agent],
    name='main_agent'
)

stream = main_agent.stream_events({
    'messages':HumanMessage('what is dezhou weather')
},version='v3')


for subagent in stream.subagents:
    print(f'{subagent.name}:', end='')
    for message in subagent.messages:
        for token in message.text:
            print(token,end='',flush=True)
    print()

