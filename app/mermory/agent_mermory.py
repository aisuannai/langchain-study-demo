

from uuid import uuid7

from app.deepseek_model import model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent

checkpointer = InMemorySaver()

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

deep_agent=create_deep_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
    checkpointer=checkpointer,
    debug=True
)

config = {"configurable": {"thread_id":str(uuid7())}}

result = deep_agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    config=config,
)

print(f'天气:{result["messages"][-1].content}')
result = deep_agent.invoke(
    {"messages": [{"role": "user", "content": "你知道San Francisco的温度是多少吗？"}]},
    config=config,
)
print(f'地点:{result["messages"][-1].content}')