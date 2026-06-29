from app.config import OPENAI_API_KEY, OPENAI_BASE_URL
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}"


model = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="you are a helpful assistant",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "what's the weather in San Francisco?"}]
})

print(result["messages"][-1].content)
