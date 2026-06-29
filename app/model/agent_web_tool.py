from app.qwen_model import model
from langchain.agents import create_agent
from langchain.messages import HumanMessage,AIMessage

agent = create_agent(
    model=model,
    #调用大模型的自带的工具
    tools=[{"type":"web_search"}]
)


response = agent.invoke({"messages":[HumanMessage("what it is weather that jinan")]})

print(f'结果:{response["messages"][-1].content[-1].get('text')}')
