"""
context 的作用与用法
====================

1. context 是什么？
   context 是每次 agent.invoke() 时传入的"业务参数"。
   类似 Java 的 ThreadLocal — 避免在函数调用链中逐层透传。
   但比 ThreadLocal 安全：它只绑定到本次 invoke，不会跨调用串数据。

2. context 与 config 的区别：
   - config.thread_id → 管"对话记忆"（同 thread_id 多次调用共享消息历史）
   - context         → 管"本次调用的业务数据"（user_id, api_key, feature_flag 等）
     每次 invoke 可以传不同的 context，与 thread_id 无关

3. context 怎么用？
   ① 定义数据结构：用 @dataclass 声明一个 Context 类
   ② 声明 schema：在 create_deep_agent() 里传 context_schema=Context
   ③ 注入参数：在 invoke() 里传 context=Context(...)
   ④ 读取参数：在 tool 函数中用 runtime.context.field_name 读取

4. 注意事项：
   - context 只在 tool/middleware 中通过 runtime.context 可读
   - context 不会自动注入到 LLM 的 prompt 中（要自己拼）
   - 同 thread_id 的多次 invoke，LLM 可能从历史记忆直接回答
     不调 tool → context 参数即使传了也不会被用到
     如果每次都需要调 tool，用不同的 thread_id 或用 system_prompt 强制
"""

from dataclasses import dataclass
from uuid import uuid7

from app.deepseek_model import model
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent
from langchain.tools import tool, ToolRuntime

checkpointer = InMemorySaver()

# ============================================================
# ① 定义 context 的数据结构
# 用 dataclass 声明每次 invoke 时可以传入哪些业务参数
# ============================================================
@dataclass
class Context:
    """每次调用 agent 时携带的业务上下文。"""
    user_id: str       # 当前用户 ID
    is_vip: bool       # 是否为 VIP 用户（可用于权限控制）

# ============================================================
# ② 在 tool 中读取 context
# tool 函数可以声明 runtime: ToolRuntime[Context] 参数，
# 通过 runtime.context 拿到本次 invoke 传入的业务数据
# ============================================================
def get_user_data(city: str, runtime: ToolRuntime[Context]) -> str:
    """获取用户所在城市的天气信息。"""
    # runtime.context 就是 invoke() 时传入的 Context 对象
    user_id = runtime.context.user_id
    is_vip = runtime.context.is_vip
    if is_vip:
        return f"User {user_id}: weather in {city} is sunny"
    return f"User {user_id}: no server"


# ============================================================
# ③ 声明 context_schema
# 在创建 agent 时通过 context_schema=Context 告诉框架：
# "我的 tool 里会用到这个类型的 context"
# ============================================================
agent = create_deep_agent(
    model=model,
    tools=[get_user_data],
    context_schema=Context,
    checkpointer=checkpointer,
    system_prompt="你是天气助手，回答用户的天气问题",
)

# ============================================================
# ④ 每次 invoke 传入不同的 context
#
# 注意：同 thread_id 下，LLM 会看到历史消息。
# 如果问题一样，LLM 可能直接从记忆中回答而不调 tool。
# 此时 context 虽然传进去了，但 tool 没跑，context 就没被用到。
#
# 需要每次调 tool 时，用不同 thread_id（或 system_prompt 强制）。
# ============================================================

# 第一次调用：传入 VIP 用户 context
config1 = {"configurable": {"thread_id": str(uuid7())}}
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "weather in beijing?"}]},
    context=Context(user_id="zhaiwy", is_vip=True),
    config=config1,
)
print(f"结果1（VIP）：{result1['messages'][-1].content}")

print()

# 第二次调用：用不同 thread_id，传入非 VIP 用户 context
config2 = {"configurable": {"thread_id": str(uuid7())}}
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "weather in beijing?"}]},
    context=Context(user_id="yuzhi", is_vip=False),
    config=config2,
)
print(f"结果2（非VIP）：{result2['messages'][-1].content}")