from pydantic import BaseModel,Field
from typing import Literal

from app.qwen_model import model as qwen_model
from langchain.tools import ToolRuntime, tool
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from dataclasses import dataclass
from langgraph.store.memory import InMemoryStore

# =============================================================================
# runtime.stream_writer 说明
# =============================================================================
# stream_writer 不是普通 print()，它不会输出到控制台。
# 它的设计目标：在 tool 执行过程中，向流式客户端推送实时进度。
#
# 典型场景（Web 服务）：
#   用户发起一个耗时的工具调用（如查数据库、调 API），
#   stream_writer 可以分阶段推送："正在查询..." → "解析数据中..." → "完成"
#   前端（如 websocket / SSE）收到这些中间状态并展示给用户，
#   而不是让用户干等直到最终结果返回。
#
# 所以 stream_writer 只在 agent.stream() / astream() / astream_events()
# 等流式调用期间生效，且写入的信号会出现在流事件管道中。
# 用 agent.invoke() 时 stream_writer 静默丢弃——因为没有活跃的流。
#
# 参考：LangGraph ToolRuntime 文档
#   → runtime.stream_writer: StreamWriter 类型
#   → 写入内容可通过 astream_events 的 custom event 捕获
# =============================================================================

# 参数
class Weather(BaseModel):
    location:str = Field(description="city name")
    #Literal类似于enum
    units: Literal["celsius","fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool =Field(
        default= False,
        description="Include 5-day forecast"
    )

@tool(args_schema=Weather,  name_or_callable= 'weather_plus', description="query temp in location and feature weather")
def get_weather_plus(runtime:ToolRuntime, location:str, units: str='celsius', include_forecast: bool = False)->str:
    writer = runtime.stream_writer
    # stream_writer 写入流管道，非控制台；仅在 agent.stream()/astream_events() 下生效
    writer(f'tool start,Looking up data, {locals};{units};{include_forecast}')
    if units == 'celsius':
        temp = 22
    else:
        temp = 72
    result = f'Current weather in {location}:{temp} degrees {units[0].upper()}'
    if include_forecast:
        result += '\n next 5 days:sunny'
    writer(f'tool end')    
    return result

agent = create_agent(
    model=qwen_model,
    tools=[get_weather_plus]
)

response = agent.stream_events({
    'messages':HumanMessage('what it is temp today in dezhou and feature 5 day weather')
},version='v3')



for result in response.values:
    last = result["messages"][-1]
    print(f'输出"{last}')

# === agent.stream() 版本：能看到 stream_writer 输出 ===
print('\n--- agent.stream() 输出（stream_writer 可见） ---')
for chunk in agent.stream({'messages': HumanMessage('what it is temp today in dezhou and feature 5 day weather')}):
    if '__end__' not in chunk:
        for node, output in chunk.items():
            print(f'[{node}]: {output}')