from typing import Literal

from pydantic import BaseModel,Field

from app.qwen_model import model as qwen_model
from langchain.tools import tool
from langchain.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent

#函数体是写给 Python 运行时看的；类型注解和 Docstring 是写给 LLM 看的。
#特别强调 Type Hints 必须有、Docstring 要准确且简洁——它们直接影响模型能否正确理解、选择并调用你的 Tool

datas=[
    {
        "city":"dezhou",
        "weather":"rain",
        "date":"2025-05-21"
    },
    {
        "city":"dezhou",
        "weather":"cloud",
        "date":"2025-05-22"
    },
    {
        "city":"dezhou",
        "weather":"sun",
        "date":"2025-05-23"
    },
    {
        "city":"hangzhou",
        "weather":"rain",
        "date":"2025-05-21"
    },
    {
        "city":"hangzhou",
        "weather":"cloud",
        "date":"2025-05-22"
    },
    {
        "city":"hangzhou",
        "weather":"sun",
        "date":"2025-05-23"
    }
]

# 这个必须有，langchain会把这返回自动包装
@tool
# city:str 参数名：类型 必须有
def get_weather(city:str, limit:int = 10)->str:
    # docstring tool和参数说明，必须有，而且要简介和抓住重点
    """
     city of weather query tool,
     params:
     city: city name;
     limit: day count;
    """
    #函数体，这个不会发送给LLM
    list = []
    for item in datas:
        if item.get('city') == city:
            list.append(item)
    return f"{city}最近{limit}天的天气:{list}"

# customer tool properties
# name自定义tool name, description 覆盖tool docstring,有时候函数体内最为函数注解使用
@tool(name_or_callable ="weather_tool", description="city of weather query tool,params: city: city name;limit: day count;")
def get_weather2(city:str, limit:int = 10)->str:
    #函数注解
    """
     天气查询函数
    """
    #函数体，这个不会发送给LLM
    list = []
    for item in datas:
        if item.get('city') == city:
            list.append(item)
    return f"{city}最近{limit}天的天气:{list}"


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

#args_schema=参数结构化，给LLM看的，不用在通过docstring描述， name_or_callable=tool name description=工具描述
@tool(args_schema=Weather,  name_or_callable= 'weather_plus', description="query temp in location and feature weather")
#这里的参数给python看的，但是LLM只能传递扁平参数，复杂参数需要自己处理
#runtime config是langchain的保留字段，不能用来做参数名
def get_weather_plus(location:str, units: str='celsius', include_forecast: bool = False)->str:
#def get_weather_plus(weather:Weather)->str:
    if units == 'celsius':
        temp = 22
    else:
        temp = 72
    result = f'Current weather in {location}:{temp} degrees {units[0].upper}'
    if include_forecast:
        result += '\n next 5 days:sunny'
    return result
agent = create_agent(
    model=qwen_model,
    tools=[get_weather_plus]
)

#整体的流程  humanmessage -> LLM->AIMessage[toolcall]->ToolMessage->LLM->AIMessage
# response = agent.invoke({
#     'messages':HumanMessage('what it is weather dezhou nearly 3 day')
# })

response = agent.invoke({
    'messages':HumanMessage('what it is temp today in dezhou and feature 5 day weather')
})

print(response['messages'])