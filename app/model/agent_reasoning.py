from app.qwen_model import model
from langchain.agents import create_agent
from langchain.messages import HumanMessage,AIMessage

# ⚠️ agent 方案已废弃（stream_events 用法不对，且 create_agent
#    不支持 enable_thinking=True 的模型）
agent = create_agent(
    model=model
)
response = agent.stream_events({'messages':HumanMessage('"Why do parrots have colorful feathers?')}, version='v3')

for result in response.values:
    message = result['messages'][-1]
    if isinstance(message, AIMessage):
        for content in message.content_blocks:
           if content.get('type') == 'reasoning':
               print(f'原因:{content.get('reasoning','')}')

# ⚠️ model.stream() 也拿不到 reasoning
# 因为 LangChain 的 OpenAI 适配器不解析 DashScope 的 reasoning_content
#
# 如需正确显示思考过程:
#   pip install langchain-qwq
#   from langchain_qwq import ChatQwen
#   model = ChatQwen(model="qwen3.7-plus")
#   for chunk in model.stream([...]):
#       print(chunk.additional_kwargs)  # 里面有 reasoning
# for chunk in model.stream([HumanMessage('Why do parrots have colorful feathers?')]):
#     # chunk 是 AIMessageChunk 对象，不是 dict，要用 . 访问属性
#     reasoning = chunk.additional_kwargs.get('reasoning_content', '')
#     if reasoning:
#         print(f'原因:{reasoning}')
