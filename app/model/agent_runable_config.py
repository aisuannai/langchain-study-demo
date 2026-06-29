from app.qwen_model import model
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage
from langchain_core.callbacks import UsageMetadataCallbackHandler


my_callback = UsageMetadataCallbackHandler()

response = model.invoke("tell me a joke",
    # 模型调用时可通过 config 参数传入 RunnableConfig 字典，运行时控制执行行为、回调和元数据追踪
    config={
        "run_name": "joke_generation",  # LangSmith 追踪标识
        "tags": ["humor", "demo"],    # 标签，可以根据标签筛选出整条链路，支持继承
        "metadata": {"user_id": "123"}, # 自定义数据，可以放入一些业务的详细信息
        "callbacks": [my_callback],  # [重点]回调处理器，对各种事件进行监听，实现日志、流失输出和监控的基础
        "max_concurrency": 5,       # batch 最大并行数
        "recursion_limit": 25,      # 链式递归深度上限，防止出现一直循环，导致token成本上升
    }
)

print(f"Joke: {response.content}")
print(f"Token usage: {response.usage_metadata}")