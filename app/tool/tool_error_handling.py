from app.qwen_model import model as qwen_model
from langchain.tools import tool
from langchain.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from langchain.tools.tool_node import ToolCallRequest
from langchain.agents.middleware import wrap_tool_call
from collections.abc import Callable


@tool(description='除法计算器,传入参数返回计算结果,参数：a=被除数;b=除数')
def chufajisuanqi(a:int, b:int)->float:
    return a/b

#LangChain 并不是要求每个 Tool 自己处理异常，而是提供了统一的中间件机制，将异常处理、日志、重试、监控等横切关注点集中管理
#类似于java的函数式接口，Callable表明这是一个函数[[参数类型1...,参数类型N],返回类型]
@wrap_tool_call
def handle_tool(request:ToolCallRequest, handler:Callable[[ToolCallRequest], ToolMessage])->ToolMessage:
    """
    """
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f'Tool error: cause by {e}',
            tool_call_id = request.tool_call['id']
        )

agent = create_agent(
    model=qwen_model,
    tools=[chufajisuanqi],
    middleware=[handle_tool]
)

response = agent.invoke({
    'messages':HumanMessage('使用除法计算器，计算10/0')
})

print(f'结果:{response['messages']}')