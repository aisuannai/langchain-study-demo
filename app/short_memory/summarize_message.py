from app.qwen_model import model as qwen_model
from langchain.agents import create_agent,AgentState
from langchain.agents.middleware import SummarizationMiddleware,before_model
from langchain.messages import HumanMessage,SystemMessage,RemoveMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.runtime import Runtime
from typing import Any


# 每次 LLM 调用之前执行
@before_model
# 如果短期记忆太多影响上下文窗口，可以通过裁剪的方式
# 这是一个 demo：删除不想要的 messages，保留想要的 messages
def trim_message(state: AgentState, runtime:Runtime)->dict[str,Any] | None:
    # 短期记忆存放在 AgentState 中
    messages = state['messages']
    if len(messages) > 0:
       for message in messages:
            print(f'类型{message.type}，内容:{message.content}')
    return None

DB_URI = "postgresql://postgres:Lr_1234@192.168.1.100:25432/langchain_save?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as pg_save:
    pg_save.setup()  # auto create tables in PostgreSQL

    agent = create_agent(
        model=qwen_model,
        middleware=[SummarizationMiddleware(
            model=qwen_model,
            trigger=('tokens',10),
            keep=('messages',2)
        ),trim_message],
        # checkpointer 是短期记忆的持久化组件，负责保存和恢复 state
        # 底层用 PostgreSQL 作为存储，checkpointer 类似 DAO，负责 save / query
        checkpointer=pg_save,
    )

    config = {"configurable": {"thread_id": '7788'}}
    response = agent.invoke({
        'messages':[
            HumanMessage('I am zhaiweiyu'),
            HumanMessage('i live in dezhou'),
            HumanMessage('i study python')
        ]
    },config=config)

    print(response['messages'][-1].content)

    response = agent.invoke({
        'messages':[
            HumanMessage('dou you kone my home?')
        ]
    },config=config)

    print(response['messages'][-1].content)

    response = agent.invoke({
        'messages':[
            HumanMessage('dou you kone my name?')
        ]
    },config=config)

    print(response['messages'][-1].content)