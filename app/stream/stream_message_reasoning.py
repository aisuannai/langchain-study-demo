from app.qwen_model import model as qwen_model
from langchain.tools import  tool
from langchain.messages import HumanMessage
from langchain.agents import create_agent


@tool(description='query city weather, pamars:city=city name')
def get_weather(city:str)->str:
    return f'{city} weather is sunn'


agent = create_agent(
    model=qwen_model,
    tools=[get_weather]
)

stream = agent.stream_events({
    'messages':HumanMessage('what is dezhou weather')
},version='v3')


#输出每一次调用的结果
for message in stream.messages:
    print(f'[{message.node}] ', end = "")
    for delta in message.text:
        print(f'{delta}',end='', flush=True)
    print() # text 流式结束换行

    if message.reasoning:
        print('[thinking] ', end='')
        for delta2 in message.reasoning:
            print(f'{delta2}',end='', flush=True)
        print() # reasoning 流式结束换行

    #输出每个步骤最终结果
    final_state = message.output
    usage = final_state.usage_metadata
    if usage:
        print(f'token使用情况:{usage}')

#输出最终结果        
final_state = stream.output
for final_message in final_state['messages']:
    print(f'最终的message:{final_message}')
