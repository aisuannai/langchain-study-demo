from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from app.qwen_model import model as qwen_model
from app.deepseek_model import model as deepseek_model
from langchain.messages import HumanMessage

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
   message_count=  len(request.state['messages'][-1].content)
   if message_count < 5:
     request=request.override(model=qwen_model)
   else:
     request = request.override(model=deepseek_model)
   return handler(request)

agent = create_agent(
   model=qwen_model,
   middleware=[dynamic_model_selection]
)

response = agent.invoke({'messages':HumanMessage('你好')})
print(f'结果1:{response['messages'][-1].content}')


response = agent.invoke({'messages':HumanMessage('你是什么大模型，请告诉我')})
print(f'结果2:{response['messages'][-1].content}')