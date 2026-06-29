# 导入 Qwen 和 DeepSeek 模型
from app.qwen_model import model as qwen_model
from app.deepseek_model import model as deepseek_model

from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langchain_core.callbacks import UsageMetadataCallbackHandler

# 推荐使用这种，创建回调处理器，用于捕获每次调用的 token 用量元数据
callback = UsageMetadataCallbackHandler()

# 单次调用 Qwen 模型，注入回调以记录 token 消耗
result_1 = qwen_model.invoke('hello', config={"callbacks": [callback]})

# 单次调用 DeepSeek 模型，同样注入回调
result_2 = deepseek_model.invoke('hello', config={"callbacks": [callback]})

# 创建一个基于 Qwen 的 LangChain Agent，并传入消息执行
agent = create_agent(
    model=qwen_model
)

result_3 = agent.invoke(
    {'messages': HumanMessage('hello')},
    config={"callbacks": [callback]}
)

# 能统计全链路，各个模型的消息token,打印累计的 token 用量（包括 prompt_tokens、completion_tokens、total_tokens）
print(callback.usage_metadata)