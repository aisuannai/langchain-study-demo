from pydantic import BaseModel, Field

from app.deepseek_model import model
from langchain.agents import create_agent
from langchain.messages import HumanMessage

#结构定义推荐是basemodel
class Person(BaseModel):
    """
    a person info
    """
    name: str = Field(deprecated="person name")
    age: int = Field(description="person age")
    birthday:str = Field(description="persong birthday")

# ⚠️ create_agent + response_format 依赖 tool_choice
# 如果 model 用了 enable_thinking=True，会冲突报错
# 因为深思考模式下不支持 tool_choice
#
# 另外 create_deep_agent 会注入大量内置工具（ls, execute, task...）
# 不适合简单问答场景，token 消耗极大。
# 建议: 简单结构化输出用 model.with_structured_output(Person) 替代
agent = create_agent(
    model=model,
    system_prompt="You are a history help assistant",
    response_format=Person
)

respones=agent.invoke({
    "messages":HumanMessage("谁是李世民")
})

print(f'结果:{respones["messages"][-1].content}')
#结构化结果从这里获取
print(f'结构化结果:{respones["structured_response"]}')
