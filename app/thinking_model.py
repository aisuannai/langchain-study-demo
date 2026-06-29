from app.config import OPENAI_API_KEY, OPENAI_BASE_URL
from langchain.chat_models import init_chat_model
import langchain

#langchain.debug =True

model = init_chat_model(
    model="qwen3.7-plus",
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    # ⚠️ 虽然这里 enable_thinking=True，但 LangChain 的 OpenAI 适配器
    # 不解析 DashScope 返回的 reasoning_content 字段。
    # 用 model.invoke() 或 model.stream() 拿不到思考内容，
    # 只能在 additional_kwargs 或 response_metadata 里碰运气。
    #
    # 如需正确显示 reasoning，方案有二:
    #   1. 安装 langchain-qwq，用 ChatQwen 替代 init_chat_model
    #   2. 直接调 DashScope 原始 API / httpx 拿 reasoning_content
    extra_body={"enable_thinking": True},
)
