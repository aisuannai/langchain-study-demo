"""
State and Final Output — 状态快照与最终输出

核心概念（来自 LangChain 官方文档）:
  - stream.values: Agent 运行过程中的状态快照流（state snapshots）。
    每执行一个节点（node）后发出一次，可实时追踪 agent 内部状态的变更。
    可迭代: for snapshot in stream.values
  - stream.output: Agent 的最终状态（final agent state）。
    在流结束后一次性获取，相当于所有状态积累后的最终结果。
  - 两者配合可做到既实时追踪执行过程，又能拿到最终的完整结果。
  - 对应文档章节:
    https://docs.langchain.com/oss/python/langchain/event-streaming#state-and-final-output
"""

from app.qwen_model import model as qwen_model
from langchain.tools import  tool
from langchain.messages import HumanMessage
from langchain.agents import create_agent

@tool(description='query city weather, pamars:city=city name')
def get_weather(city:str)->str:
    return f'{city} weather is sunn'


agent = create_agent(
    model=qwen_model,
    tools=[get_weather]
)

stream = agent.stream_events({
    'messages':HumanMessage('what is dezhou weather')
},version='v3')

for snapshot in stream.values:
    print(snapshot)
