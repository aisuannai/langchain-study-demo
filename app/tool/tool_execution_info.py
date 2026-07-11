from uuid import uuid7

from app.qwen_model import model as qwen_model
from langchain.tools import tool,ToolRuntime
from langchain.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from typing import Literal

from pydantic import BaseModel,Field

short_save = InMemorySaver()


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
def get_weather_plus(runtime:ToolRuntime,location:str, units: str='celsius', include_forecast: bool = False)->str:
    # runtime.execution_info 提供当前工具调用的执行元数据
    #   thread_id      — 对话线程唯一 ID（uuid7），区分不同对话
    #   run_id         — 单次 agent invoke 的唯一 ID，追踪完整调用链路
    #   node_attempt   — 当前节点重试次数，调试重试/失败场景
    info = runtime.execution_info
    print(f'Thread:{info.thread_id}, Run:{info.run_id}')
    print(f'Attempt:{info.node_attempt}')
    if units == 'celsius':
        temp = 22
    else:
        temp = 72
    result = f'Current weather in {location}:{temp} degrees {units[0].upper()}'
    if include_forecast:
        result += '\n next 5 days:sunny'
    return result


agent = create_agent(
    model=qwen_model,
    tools=[get_weather_plus]
)

config = {
    "run_id":str(uuid7()), #设置run_id
    "configurable":{"thread_id":str(uuid7())} #设置thread_id
}

response = agent.invoke({
    'messages':HumanMessage('what it is temp today in dezhou and feature 5 day weather')
},config=config)

print(response['messages'][-1].content)