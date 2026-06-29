# 参考: https://docs.langchain.com/oss/python/langchain/models#configurable-models
#
# 演示 init_chat_model 的三种配置模式：
#   model        —— 无默认模型，运行时通过 config["configurable"]["model"] 切换
#   model1/model2 —— config_prefix 多实例 + configurable_fields="any" 全字段可配
#   model3       —— 有默认模型，仅 temperature/max_tokens 可配
#
# 所有模型走 DashScope（同一 provider + base_url），运行时只切换 model 名称

# 加载 .env 到环境变量（设置 OPENAI_API_KEY 和 OPENAI_BASE_URL）
from app.config import OPENAI_API_KEY, OPENAI_BASE_URL  # noqa: F401

from langchain.chat_models import init_chat_model


# ── 模式一：无默认模型，运行时指定 ──────────────────────────────────
# model=None（默认）→ 自动开启 configurable_fields=("model", "model_provider")
# 调用时必须传 config["configurable"]["model"]
model = init_chat_model(model_provider="openai")

# ── 模式二：config_prefix 多实例 + "any" 全字段可配 ────────────────
# config_prefix="fix_1" → config 中的 key 需要加 fix_1_ 前缀
# configurable_fields="any" → model、api_key、base_url 等全部可运行时覆盖
# configurable_fields标识那些配置可以动态配置，如果是any表示任意字段都可以配置
# config_perfeix是配置前缀，当使用的很多model使用统一的config时，可以通过这个前缀来区别model用的哪个配置，类似applicaiton-{env}.yaml这么个东西
model1 = init_chat_model(model_provider="openai", configurable_fields="any", config_prefix="fix_1")

model2 = init_chat_model(model_provider="openai", configurable_fields="any", config_prefix="fix_2")

# ── 模式三：有默认模型，仅部分字段可配 ──────────────────────────────
# 这里演示configurable_fields只有配置的字段可以动态修改，如何没有配置，即使你改了，也不好使
# model 已固定为 glm-5.1，运行时只能覆盖 temperature / max_tokens
model3 = init_chat_model(model_provider="openai", model="glm-5.1", configurable_fields=("temperature", "max_tokens"), config_prefix="fix_3")


# 统一配置字典，各前缀对应的 key 分别被 model1 / model2 / model3 消费
config = {
    "configurable": {
        # model1 使用的前缀 → fix_1_model
        "fix_1_model": "glm-5.1",
        "fix_1_api_key": OPENAI_API_KEY,
        "fix_1_base_url": OPENAI_BASE_URL,
        # model2 使用的前缀 → fix_2_model
        "fix_2_model": "qwen3.7-plus",
        "fix_2_api_key": OPENAI_API_KEY,
        "fix_2_base_url": OPENAI_BASE_URL,
        # model3 的前缀 → 但 model3 的 model 已固定，config 中的 fix_3_model 无效
        "fix_3_model": "qwen3.7-plus",
        "fix_3_api_key": OPENAI_API_KEY,
        "fix_3_base_url": OPENAI_BASE_URL,
    }
}


# ── 模式一调用：无 prefix，直接在 configurable 中传 model ──────────
message = model.invoke(
    "what is your model?",
    config={"configurable": {"model": "glm-5.1"}},
)
print(f"结果1: {message.content}")

message = model.invoke(
    "what is your model?",
    config={"configurable": {"model": "qwen3.7-plus"}},
)
print(f"结果2: {message.content}")


# ── 模式二调用：config_prefix 匹配对应 key ──────────────────────────
# model1 读取 fix_1_model: "glm-5.1"
message = model1.invoke("what is your model?", config=config)
print(f"结果3: {message.content}")

# model2 读取 fix_2_model: "qwen3.7-plus"
message = model2.invoke("what is your model?", config=config)
print(f"结果4: {message.content}")


# ── 模式三调用：model 已固定，config 中的 fix_3_model 被忽略 ──────
# model3 的 model="glm-5.1" 不会变，仍用 glm-5.1
message = model3.invoke("what is your model?", config=config)
print(f"结果5: {message.content}")
