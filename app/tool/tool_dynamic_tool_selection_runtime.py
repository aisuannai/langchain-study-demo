"""
Runtime Tool Registration (动态工具注册)
============================================

参考: https://docs.langchain.com/oss/python/langchain/tools#dynamic-tool-selection

当工具在运行时才被发现或创建时（例如从 MCP 服务器加载、基于用户数据生成、
或从远程注册中心获取），需要同时处理工具的注册和执行。

这需要两个 middleware hooks：

1. wrap_model_call - 将动态工具添加到请求中
2. wrap_tool_call - 处理动态添加的工具的执行

为什么需要 wrap_tool_call？
  wrap_tool_call hook 对于运行时注册的工具是必需的，因为 agent 需要知道
  如何执行不在原始工具列表中的工具。如果没有它，agent 将不知道如何调用
  动态添加的工具。

适用场景：
  - 工具在运行时被发现（例如从 MCP 服务器）
  - 工具基于用户数据或配置动态生成
  - 正在集成外部工具注册中心

动态工具选择有两种方式（取决于工具是否预先已知）：
  1. Filtering pre-registered tools - 在 agent 创建时注册所有工具，
     运行时根据 state/permissions/context 过滤
  2. Runtime tool registration - 工具在运行时才被发现或创建，
     需要 wrap_model_call + wrap_tool_call 两个钩子

参考示例：
```python
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest

@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 20.0) -> str:
    \"\"\"Calculate the tip amount for a bill.\"\"\"
    tip = bill_amount * (tip_percentage / 100)
    return f"Tip: ${tip:.2f}, Total: ${bill_amount + tip:.2f}"

class DynamicToolMiddleware(AgentMiddleware):
    def wrap_model_call(self, request: ModelRequest, handler):
        updated = request.override(tools=[*request.tools, calculate_tip])
        return handler(updated)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        if request.tool_call["name"] == "calculate_tip":
            return handler(request.override(tool=calculate_tip))
        return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[get_weather],  # Only static tools registered here
    middleware=[DynamicToolMiddleware()],
)
```
"""

from app.qwen_model import model as qwen_model
from langchain.tools import ToolRuntime, tool
from langchain.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware


@tool(description='除法计算器, a:被除数 b:除数')
def jisuanqi(a:int, b:int)->float:
    return a/b

@tool(description='城市天气查询, city：城市名称')
def get_weather(city:str)->str:
    return f'{city} weather is sun'

class DynamicToolMiddleware(AgentMiddleware):

    def wrap_model_call(self, request, handler):
        print(f'模型绑定tool{request.tools}')
        update = request.override(tools=[*request.tools, get_weather])
        return handler(update)
    def wrap_tool_call(self, request, handler):
       tool_name = request.tool_call['name']
       print('tool执行')
       if tool_name == 'get_weather':
           return handler(request.override(tool=get_weather))
       return handler(request)


agent = create_agent(
    model=qwen_model,
    tools=[jisuanqi],
    middleware=[DynamicToolMiddleware()],
)

response = agent.invoke({
    'messages':HumanMessage('查询德州的天气')
})

print(f'结果:{response['messages']}')