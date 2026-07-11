"""
动态 Tool 选择 — 基于 Store（长时记忆）的过滤模式

参考: https://docs.langchain.com/oss/python/langchain/tools#dynamic-tool-selection
章节: Filtering pre-registered tools > Store

核心思路:
  wrap_model_call 中间件在 LLM 被调用前拦截请求,
  从 Store（跨对话持久化存储）中读取用户权限/功能开关,
  用 request.override(tools=...) 替换 LLM 可见的 tool 列表。
  这样 LLM 根本不知道被过滤掉的 tool 存在。

对比三种过滤方式（见官方文档）:
  1. State — 基于对话状态（短时记忆）,如是否已认证、对话轮数
  2. Store — 基于持久化存储（长时记忆）,如用户权限、功能开关  ← 本文件用此方式
  3. Runtime Context — 基于调用时上下文,如用户角色（admin/editor/viewer）
"""

from dataclasses import dataclass

from app.qwen_model import model as qwen_model
from langchain.tools import ToolRuntime, tool
from langchain.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent, AgentState
from typing import Callable
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langgraph.store.memory import InMemoryStore


# ============================================================
# Context: 每次 invoke 时传入的不可变业务参数
# ============================================================
@dataclass
class UserInfo():
    """每次调用 agent 时携带的业务上下文。"""
    user_id: str


# ============================================================
# State schema: 对话级别的短时记忆（本文件未使用,仅作演示）
# ============================================================
class MyAgentState(AgentState):
    """对话状态,同 thread_id 的多轮对话共享。"""
    is_vip: bool


# ============================================================
# 注册 tools
# ============================================================
@tool(description='query normal user level')
def normal_query(name: str) -> str:
    """普通用户查询工具。"""
    return f"{name},you is normal"


@tool(description='query vip user level')
def vip_query(name: str) -> str:
    """VIP 用户查询工具。"""
    return f"{name}, you is vip"


# ============================================================
# Store: 初始化长时记忆并写入用户权限数据
#
# store.put(namespace, key, value):
#   - namespace: 分类路径,如 ('user',)
#   - key:       标识符,如 'abc123'（用户 ID）
#   - value:     要存储的数据,这里存该用户有权限使用的 tool 名称列表
# ============================================================
store = InMemoryStore()
store.put(('user',), 'abc123', ['vip_query'])


# ============================================================
# 方式一（已注释）: 基于 State 过滤 — 通过对话状态 is_vip 决定 tool 可见性
#
# 官网文档: Dynamic tool selection > Filtering pre-registered tools > State
# 对应代码: request.state["is_vip"] → 按 tool.name.startswith("vip") 过滤
# ============================================================
# @wrap_model_call
# def model_handler(request:ModelRequest, handler:Callable[[ModelRequest], ModelResponse])->ModelResponse:
#     state = request.state
#     tools = request.tools
#     filtered_tools=[]
#     if state['is_vip']:
#         for t in tools:
#             if t.name.startswith('vip'):
#                 filtered_tools.append(t)
#     else:
#         for t in tools:
#             if not t.name.startswith('vip'):
#                 filtered_tools.append(t)
#     return handler(request.override(tools=filtered_tools))

# agent = create_agent(
#     model=qwen_model,
#     tools=[normal_query, vip_query],
#     middleware=[model_handler],
#     state_schema=MyAgentState
# )

# response = agent.invoke({
#     'messages':HumanMessage('my name jack,query my level'),
#     'is_vip':True
# })

# print(f'结果1：{response['messages']}')


# ============================================================
# 方式二（当前使用）: 基于 Store + Runtime Context 过滤
#
# 官网文档: Dynamic tool selection > Filtering pre-registered tools > Store
# https://docs.langchain.com/oss/python/langchain/tools#dynamic-tool-selection
#
# 流程:
#   1. 从 request.runtime.context 读取 user_id（调用时传入）
#   2. 从 request.runtime.store 查询该用户的 tool 权限列表
#   3. 从 request.tools（所有已注册 tool）中筛选出有权限的
#   4. 用 request.override(tools=filtered) 替换 LLM 的可见 tool 列表
#
# wrap_model_call 的作用:
#   @wrap_model_call 将普通函数转换为 AgentMiddleware,
#   使其能在 LLM 被调用前拦截 ModelRequest。
#   handler(request) 调用链中的下一个环节（可能是另一个 middleware 或真正的 LLM 调用）。
# ============================================================
@wrap_model_call
def model_handler(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """根据 Store 中存储的用户权限动态过滤 tools。

    Args:
        request:  包含 model/tools/state/runtime 等信息的请求对象
        handler:  链中下一个处理环节（最终调用 LLM）
    Returns:
        ModelResponse: LLM 的响应
    """
    store = request.runtime.store         # 长时记忆存储
    user_id = request.runtime.context.user_id  # 从上下文读取用户 ID
    tools = request.tools                  # 所有已注册的 tool

    filtered_tools = []

    # store.get(namespace, key) → 返回 Item 对象
    # .value 取出原始值,即 store.put() 时存的 list
    user_tool_names = store.get(('user',), user_id).value

    if user_tool_names:
        # 只保留用户有权限的 tool（按 tool.name 匹配）
        for t in tools:
            if t.name in user_tool_names:
                filtered_tools.append(t)
    else:
        # 没有权限配置时默认全部可见
        filtered_tools = tools

    # override(tools=...) 替换 LLM 本次调用可见的 tool 列表
    # LLM 完全不知道被过滤掉的 tool 存在
    return handler(request.override(tools=filtered_tools))


# ============================================================
# 创建 agent
# - middleware: 注册中间件,在 LLM 调用前拦截
# - context_schema: 声明 context 的类型（用于注入 user_id）
# - store: 关联长时记忆存储
# ============================================================
agent = create_agent(
    model=qwen_model,
    tools=[normal_query, vip_query],
    middleware=[model_handler],
    context_schema=UserInfo,
    store=store,
)

response = agent.invoke(
    {'messages': HumanMessage('my name jack,query my level')},
    context=UserInfo(user_id='abc123'),
)

print(f'结果1：{response["messages"]}')