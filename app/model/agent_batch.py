from app.deepseek_model import model
from deepagents import create_deep_agent
from langchain.messages import HumanMessage

agent = create_deep_agent(
    model=model
)

list = [
    {"messages": [HumanMessage('Why do parrots have colorful feathers?')]},
    {"messages": [HumanMessage('How do airplanes fly?')]},
    {"messages": [HumanMessage('What is quantum computing?')]}
]


#agent并发请求三个
#batch() 是“等所有任务完成再返回”，batch_as_completed() 是“按完成顺序逐个返回结果”，本质是批处理系统的两种调度策略
# result = agent.batch(list)

# for value in result:
#     print(value)

reponse = agent.batch_as_completed(list)

for task_id, result in reponse:
    print(f'[{task_id}]: {result['messages'][-1].content}')