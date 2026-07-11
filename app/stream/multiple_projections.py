from app.qwen_model import model as qwen_model
from langchain.tools import  tool
from langchain.messages import HumanMessage
from langchain.agents import create_agent
import asyncio

# async def 声明一个协程函数，调用时返回协程对象，函数体不立即执行
# async def main():
#     @tool(description='query city weather, pamars:city=city name')
#     def get_weather(city:str)->str:
#         return f'{city} weather is sunn'


#     agent = create_agent(
#         model=qwen_model,
#         tools=[get_weather]
#     )

#     # await：标识这个 stream 输出也是非阻塞的，有就用，没有就挂起，拿到结果后往下执行
#     stream = await agent.astream_events({
#         'messages':HumanMessage('what is dezhou weather')
#     },version='v3')

#     # async for = 异步迭代器，内部会 await 下一个元素到达
#     async def consume_messages():
#         async for message in stream.messages:
#             # await message.text：挂起当前协程，等文本就绪后恢复
#             print(f'[message]: {await message.text}')

#     async def consume_tool_calls():
#         async for call in stream.tool_calls:
#             print(f'[tool]: {call.tool_name}({call.input})')

#     # asyncio.gather 并发运行多个协程，等价于 Java CompletableFuture.allOf(...) + 收集结果
#     await asyncio.gather(consume_messages(), consume_tool_calls())

# # asyncio.run() 创建事件循环 → 运行 main() → 完成后关闭循环，相当于 new Thread(runnable).start() + join()
# asyncio.run(main())


@tool(description='query city weather, pamars:city=city name')
def get_weather(city:str)->str:
    return f'{city} weather is sunn'


agent = create_agent(
    model=qwen_model,
    tools=[get_weather]
)

stream =  agent.stream_events({
    'messages':HumanMessage('what is dezhou weather')
},version='v3')

#按照实际发生的顺序输出，而stream.messages这样的是按照类型输出
for name, item in stream.interleave("messages", "tool_calls", "values"):
    print(f'顺序:{name}, {item}')