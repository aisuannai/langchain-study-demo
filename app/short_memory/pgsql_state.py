from uuid import uuid7

from app.qwen_model import model as qwen_model
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore
from langchain.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.tools import tool

@tool(description='find city weather, param: city=city name ')
def get_weather(city:str)->str:
    return f'{city} weahter is sun'

DB_URI = "postgresql://postgres:Lr_1234@192.168.1.100:25432/langchain_save?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as pg_save:
    pg_save.setup()  # auto create tables in PostgreSQL

    agent = create_agent(
        model=qwen_model,
        tools=[get_weather],
        checkpointer=pg_save,
    )

    config = {"configurable": {"thread_id": str(uuid7())}}
    response = agent.invoke(
        {
            "messages": HumanMessage("what is dezhou weahter"),
        },
        config=config,
    )

    print(f"结果1:{response['messages'][-1].content}")

    config = {"configurable": {"thread_id": str(uuid7())}}
    response = agent.invoke(
        {
            "messages": HumanMessage("what is jinan weahter"),
        },
        config=config,
    )

    print(f"结果2:{response['messages'][-1].content}")