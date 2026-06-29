from app.qwen_model import model
from langchain.agents import create_agent
from langchain.messages import HumanMessage,AIMessage


#对数概率，越接近0，说明模型约有信心
model.logprobs = True

response = model.invoke('今天晚上纳斯达克是上涨还是下跌，只回复上涨或者下跌')

#主要用途在1.判断模型信心  2.AGENT决策 3.RAG幻觉 4.Prompt调优
print(f'{response.response_metadata["logprobs"]}')