import base64

from langgraph.types import Command

from app.qwen_model import model as qwen_model
from langchain.tools import tool,ToolRuntime
from langchain.messages import HumanMessage, ToolMessage,SystemMessage
from langchain.agents import AgentState, create_agent

class UserSeeting(AgentState):
   language:str

#返回字符串是最简单、最常用的方式，适合那些只需要向模型提供事实或文本信息的工具
@tool(description="query weather by cityname; param city=city_name")
def get_weather(city:str)->str:
    return f'weather is rain in {city}'


#当工具需要提供可进一步解析的数据时，返回对象（如 dict）更合适
@tool(description='screenshot tool and return screenshot content')
def get_capture_screenshot()->list[dict]:
    with open("D:\\work_file\\ai\\3006.jpg","rb") as f:
     encode = base64.b64encode(f.read()).decode("utf-8")
     #返回多模态的内容,写入content里
    return [
        {"type":"text","text":'this is screenshot of the curent page'},
        {"type":'image','base64': encode, 'mime_type':'image/jpeg'}
    ]


#当 Tool 不只是返回数据，而是要修改 Agent 状态时，就应该返回 Command
@tool(description='change user language')
def set_user_language(language:str, runtime:ToolRuntime)->Command:
   print(f'user current language:{runtime.state.get('language')}')
   return Command(
      update={
         "language":language,
         "messages":[
            ToolMessage(
               content=f'please language {language} resonpe user',
               tool_call_id =runtime.tool_call_id
            )
         ]
      }
   )

#return_direct=True 是一种性能优化和行为控制机制。
# 当 Tool 的输出已经是最终答案，并且你希望保持原样、不需要模型再加工时，它可以减少一次 LLM 调用、降低 Token 成本，并保证输出的确定性
@tool(return_direct=True, description='query oreder state')
def fetch_order_status(order_no:str)->str:
   return f'order:{order_no} is finished'

# agent = create_agent(
#     model=qwen_model,
#     state_schema = UserSeeting,
#     tools=[set_user_language],
#     system_prompt='you are chat root, reponse user ask please use tool change usee language'
# )

agent = create_agent(
    model=qwen_model,
    tools=[fetch_order_status]
)

# response = agent.invoke({
#    'messages':HumanMessage('analyais screenshot content ')
# })

# response = agent.invoke({
#    'messages':[HumanMessage('i am chinese'),HumanMessage('i am lost in beijing')],
#    'language': 'english'
# })

# print(f'结果:{response['messages']}')

# response = agent.invoke({
#    'messages':[HumanMessage('我是韩国人'),HumanMessage('我在青岛迷路了')],
#    'language': 'zh-cn'
# })

# print(f'结果:{response['messages']}')



#human->llm(tool_call)->toolmessage-human
response = agent.invoke({
   'messages':HumanMessage('please check my order state, order_no:abc123456')
})

print(f'{response['messages']}')