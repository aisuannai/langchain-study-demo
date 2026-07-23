from app.qwen_model import model as llm
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessageChunk
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt
from langchain_core.utils.uuid import uuid7

@tool(description='send eamil tool, params:email_address like 143396@qq.com ')
def send_email(email_address:str)->str:
    print(f'[发生email] TO [{email_address}]')
    return "send email success"


savePointer = InMemorySaver()

config = {'configureable':{'thread_id':'some_id'}}

agent = create_agent(
    model=llm,
    tools=[send_email],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={'send_email',True})],
    checkpointer=savePointer
)

stream = agent.stream({
    'messages':HumanMessage('send email to james, he email_addres is james_home@qq.com')
}, 
config= config,
stream_mode=["messages", "updates"],
version="v2")

interrupts: list[Interrupt] = []

for chunk in stream:
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if isinstance(token, AIMessageChunk):
            if token.text:
                print(token.text, end="", flush=True)
            if token.tool_call_chunks:
                print(f"\n[tool call chunk]: {token.tool_call_chunks}")

    elif chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source in ("model", "tools"):
                messages = update["messages"]
                last = messages[-1]
                if hasattr(last, "tool_calls") and last.tool_calls:
                    print(f"\n[完整 Tool Calls]: {last.tool_calls}")
                if hasattr(last, "content") and last.content:
                    print(f"\n[完整消息]: {last.content[:100]}")

            # 关键: __interrupt__ 是中断信号
            if source == "__interrupt__":
                interrupts.extend(update)
                print(f"\n[中断] 共 {len(update[0].value['action_requests'])} 个操作待审批")
                for req in update[0].value["action_requests"]:
                    print(f"  -> {req['description']}")
                    
print(f"\n\n收集到 {len(interrupts)} 个中断对象")
print("Agent 已暂停，等待人类决策...\n\n")