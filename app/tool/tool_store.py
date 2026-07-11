from typing import Any
from uuid import uuid7

from pydantic import BaseModel,Field

from app.qwen_model import model as qwen_model
from langchain.tools import ToolRuntime, tool
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from dataclasses import dataclass
from langgraph.store.memory import InMemoryStore

"""
1. 为什么需要长时记忆？
   - 短时记忆（State + checkpointer）只在同一个 thread_id 内有效
   - 长时记忆（Store）跨对话、跨 thread 持久化数据
   - 典型场景：用户偏好、知识库、历史行为记录

2. Store 核心概念：
   - namespace: 命名空间（类似文件夹路径），用于组织数据
     → 例如 ("users", "u_123") 或 ("preferences",)
   - key: 每条记录的 ID（类似文件名）
   - value: 存储的数据（dict 格式）

3. InMemoryStore vs 生产级 Store：
   - InMemoryStore: 内存存储，重启丢失，仅用于实验
   - PostgresStore: 生产环境用，数据持久化到数据库

4. tool 通过 runtime.store 访问长期记忆，这里展示的主要是tool如何使用长期记忆，不对长期记忆做深入的研究

"""

#长期记忆
long_save = InMemoryStore()

class Userinfo(BaseModel):
    user_id:str=Field(description='user id')
    user_info:dict[str,Any] = Field(description="user info dict eg: name='xx',address='xxxx'")

@tool(description="query user_info by user_id")
def get_user_info(user_id:str, runtime:ToolRuntime)->str:
    store = runtime.store
    # store 是 namespace + key 两级结构，不是 "user_{user_id}" 拼成一个 key
    # namespace = ("user",) → 类似文件夹
    # key = user_id         → 类似文件名
    user_info = store.get(("user",),  user_id)
    if user_info:
        return str(user_info)
    else:
        return "unkown user"


@tool(args_schema=Userinfo, description='save user info')
def save_user_info(user_id:str, user_info:dict[str,Any], runtime:ToolRuntime)->str:
    store = runtime.store
    # namespace = ("user",), key = user_id
    store.put(('user',),user_id, user_info)
    return 'successed save user info'


agent = create_agent(
    model=qwen_model,
    store=long_save,
    tools=[get_user_info,save_user_info]
)

# Store 不受 thread_id 影响，它是长期记忆，跨 thread、跨会话
thread_a = str(uuid7())
thread_b = str(uuid7())

response = agent.invoke({
    'messages':HumanMessage('save my userinfo: my name is jack, my home address is dezhou, my user_id is abc123')
},
config={'configurable':{'thread_id':thread_a}}
)

print(f'结果1: {response['messages'][-1].content}')

response = agent.invoke({
    'messages':HumanMessage("get user info for user with id 'abc123'")
},
config={'configurable':{'thread_id':thread_b}}
)

print(f'结果2: {response['messages'][-1].content}')