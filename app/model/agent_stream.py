"""
stream_events 流式输出
=======================

1. stream_events 与 invoke 的区别：
   - invoke：等所有步骤跑完，一次性返回最终结果
     → 如果 agent 调了 3 次 tool，中间完全看不到，只能干等
   - stream_events：每走一步就返回一次当前状态
     → 实时展示 tool 调用、中间结果，用户体验更好

2. 适用场景：
   - agent 需要多次调 tool（查城市 → 查天气 → 写总结等）
   - 耗时较长的任务，用户需要进度反馈
   - 简单问答用 invoke 就够了

3. 返回格式：
   stream.values 每次循环返回一个"快照"（当前完整的 state）
   snapshot["messages"][-1]  → 这步新产生的消息
   messages 中可能包含多种类型：
     - HumanMessage: 用户消息
     - AIMessage: AI 回复（可能有 tool_calls 或 content）
     - ToolMessage: tool 执行结果

4. 注意 version="v3"：
   v3 是实验性流式协议，每次返回完整 state（不是增量 diff）。
   控制台会打印 LangChainBetaWarning，不影响功能，忽略即可。
"""

from dataclasses import dataclass
import time
from uuid import uuid7

from app.deepseek_model import model
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import AIMessage


checkpointer = InMemorySaver()

@dataclass
class UserData:
    """
    dataclass 字段默认值规则：
    ----------------------------------------
    1. 无默认值的字段（name）必须放在最前面
    2. 有默认值的字段必须放在无默认值的之后
    3. 构造时可以不传有默认值的字段：
       UserData(name="zhaiwy")                     ✅ 只传 name
       UserData(name="zhaiwy", city="dezhou")      ✅ 部分传
       UserData(name="zhaiwy", city="dezhou", weather="sunny")
    4. city | None = None 表示：
       - 类型是 str 或 None
       - 不传时默认是 None
    5. 访问时注意判断 None：
       if runtime.context.city is not None: ...
    """
    name: str
    city: str | None = None
    weather: str | None = None
    school: str | None = None

data = {
    "zhaiwy": {
        "city": "dezhou",
        "school": "实验小学",
    },
    "yuzhi": {
        "city": "zibo",
        "school": "实验中学",
    },
    "luan": {
        "city": "taian",
        "school": "实验大学",
    },
}

weather_data = {
    "dezhou": "下雨",
    "zibo": "晴天",
    "taian": "台风",
}


def get_user_city(user_id: str, runtime: ToolRuntime[UserData]) -> str:
    """根据用户名称，查询用户所在城市"""
    name = runtime.context.name
    time.sleep(0.5)
    return f"user:{user_id} city is {data.get(name, {}).get('city', 'unkown')}"


def get_city_weather(city_id: str, runtime: ToolRuntime[UserData]) -> str:
    """根据城市名称查询城市天气"""
    name = runtime.context.name
    time.sleep(0.5)
    return f"user:{name} city is {weather_data.get(city_id, '没有数据')}"


config = {"configurable": {"thread_id": str(uuid7())}}

agent = create_deep_agent(
    model=model,
    tools=[get_user_city, get_city_weather],
    context_schema=UserData,
    checkpointer=checkpointer,
    system_prompt="你是一个ai助手",
)

# ============================================================
# stream_events 用法
# 参数和 invoke 一样，只是调用方式不同：
#   invoke(...)     → 等全部完成，返回最终结果
#   stream_events() → 每步都推一次，在循环中实时处理
#
# 每次循环：
#   result["messages"][-1]  → 这一步新产生的消息
#   检查 tool_calls 知道 LLM 调了什么工具
#   检查 content 知道 LLM 回复了什么
# ============================================================
stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "my user_id is 5, what my city weather"}]},
    context=UserData(name="zhaiwy"),
    config=config,
    version="v3",
)

for result in stream.values:
    last = result["messages"][-1]
    if isinstance(last, AIMessage):
        if last.tool_calls:
            print(f'调工具: {[tc["name"] for tc in last.tool_calls]}')
        elif last.content:
            # content 可能是列表（content block 格式）或纯字符串
            if isinstance(last.content, list):
                text = last.content[0].get("text", str(last.content))
            else:
                text = last.content
            print(f"AI: {text}")