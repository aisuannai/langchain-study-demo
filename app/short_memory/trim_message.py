from typing import Any

from app.qwen_model import model as qwen_model
from langchain.agents import create_agent,AgentState
from langchain.agents.middleware import before_model
from langgraph.store.memory import InMemoryStore
from langchain.messages import HumanMessage,SystemMessage,RemoveMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.tools import tool
from langgraph.runtime import Runtime
from langgraph.graph.message import REMOVE_ALL_MESSAGES

# 每次 LLM 调用之前执行
@before_model
# 如果短期记忆太多影响上下文窗口，可以通过裁剪的方式
# 这是一个 demo：删除不想要的 messages，保留想要的 messages
def trim_message(state: AgentState, runtime:Runtime)->dict[str,Any] | None:
    # 短期记忆存放在 AgentState 中
    messages = state['messages']
    if len(messages) > 0:
        d_name = 'Zhaiweiyu'
        s_name = 'zhaiweiyu'
        a_name = 'Zhai Weiyu'
        new_messages = []
        for message in messages:
            if d_name not in message.content   and s_name not in  message.content  and a_name not in message.content:
                new_messages.append(message)
        return {
            "messages":[
                # RemoveMessage 不是消息，是一个消息更新指令，删除原来的 messages
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                # 保留过滤后的新 messages
                *new_messages
            ]
        }
    return None
DB_URI = "postgresql://postgres:Lr_1234@192.168.1.100:25432/langchain_save?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as pg_save:
    pg_save.setup()  # auto create tables in PostgreSQL

    agent = create_agent(
        model=qwen_model,
        system_prompt=SystemMessage('你是一个聊天助手'),
        middleware=[trim_message],
        # checkpointer 是短期记忆的持久化组件，负责保存和恢复 state
        # 底层用 PostgreSQL 作为存储，checkpointer 类似 DAO，负责 save / query
        checkpointer=pg_save,
    )

    config = {"configurable": {"thread_id": 'abc'}}

    response = agent.invoke({
        'messages':[
            HumanMessage('do you konw my name')
        ]
    },config=config)

    print(f"结果1:{response['messages'][-1].content}")