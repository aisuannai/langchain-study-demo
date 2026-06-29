from pydantic import BaseModel

from app.deepseek_model import model
from langchain.agents import create_agent
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

class Answer(BaseModel):
    city:str
    data:str

def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    data = {
        "Beijing": "25°C, sunny",
        "Shanghai": "28°C, cloudy",
        "San Francisco": "18°C, foggy",
        "Tokyo": "22°C, light rain",
        "London": "15°C, overcast",
    }
    return data.get(city, f"{city}: weather data not available")

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant that can check weather.",
    checkpointer=checkpointer
)

deep_agent=create_deep_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant that can check weather.",
    checkpointer=checkpointer,
    #结构化输出
    response_format=Answer
)

agent_result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather in Beijing and San Francisco?"}]
},
config={"configurable": {"thread_id": "great-gatsby-lc"}},
)
deep_agent_result = deep_agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather in Tokyo and London?"}]
},
config={"configurable": {"thread_id": "great-gatsby-da"}},
)

print(agent_result["messages"][-1].content)
print("\n")
print(deep_agent_result["messages"][-1].content)
print(deep_agent_result["structured_response"])
