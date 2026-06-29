from app.config import OPENAI_API_KEY, OPENAI_BASE_URL
from langchain.chat_models import init_chat_model
import langchain

#langchain.debug =True

model = init_chat_model(
    model="qwen3.5-omni-flash",
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    # ⚠️ 注意: enable_thinking=False 时模型不会输出思考过程
    # 如果要看 reasoning，需要改为 True
    # 但深思考模式下不支持 tool_choice，create_agent 会出问题
    #
    # ⚠️ 当前走的是 OpenAI 兼容模式 (model_provider="openai")
    # LangChain 没有官方 DashScope 集成，所以:
    #   - 原始 API 返回的 reasoning_content 字段会被丢掉
    #   - content_blocks 里不会出现 type=reasoning (那是 Anthropic 格式)
    #   - 想拿 reasoning 需走原始 API httpx 或装 langchain-qwq
    extra_body={"enable_thinking": False},
)
