"""
Message 消息系统与 content_blocks 标准化格式
===========================================

1. 四种消息类型：
   - SystemMessage: 系统指令，设定 Agent 的角色和行为
   - HumanMessage:  用户输入（文本、图片、文件等多模态）
   - AIMessage:     模型回复（含工具调用、推理链、token 用量）
   - ToolMessage:   工具执行结果

2. content_blocks（新版标准化内容格式）：
   不同 Provider（OpenAI/Anthropic/Google）的原始格式不同，
   .content_blocks 属性将它们归一化为统一的 block 列表：
   - {"type": "text", "text": "..."}           → 文本
   - {"type": "reasoning", "reasoning": "..."}  → 推理/思考链
   - {"type": "tool_call", ...}                 → 工具调用
   - {"type": "image", "url": "..."}            → 图片
   - {"type": "file", "url": "..."}             → 文件

3. AIMessage 关键属性：
   - .text              → 纯文本内容
   - .content           → 原始内容（Provider 原生格式）
   - .content_blocks    → 标准化内容块列表
   - .tool_calls        → 工具调用列表 [{"name", "args", "id"}, ...]
   - .usage_metadata    → Token 用量统计
   - .response_metadata → 响应元数据（Provider 原始信息）
"""

from app.deepseek_model import model
from langchain.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.tools import tool

# ============================================================
# 1. SystemMessage — 设定模型行为
# ============================================================
print("=" * 50)
print("1. SystemMessage — 系统指令")
print("=" * 50)

sys_msg = SystemMessage("You are a terse assistant. Reply in 5 words or fewer.")

# 用字典格式也可以，效果等同
messages = [
    {"role": "system", "content": "You are a terse assistant. Reply in 5 words or fewer."},
    {"role": "user", "content": "What is the capital of France?"},
]
reply = model.invoke(messages)
print(f"SystemMessage 约束结果: {reply.text}")  # e.g. "Paris."

# ============================================================
# 2. HumanMessage — 用户输入（纯文本 + 多模态）
# ============================================================
print("\n" + "=" * 50)
print("2. HumanMessage — 用户输入")
print("=" * 50)

# 纯文本方式
text_msg = HumanMessage("Tell me a joke about programming.")
print(f"纯文本 HumanMessage: {text_msg.content}")

# 多模态内容块方式（图片 URL）
# 注意：DeepSeek 模型可能不支持图片，此处仅展示 content_blocks 格式
multimodal_msg = HumanMessage(
    content_blocks=[
        {"type": "text", "text": "Describe this image:"},
        {"type": "image", "url": "https://example.com/cat.jpg"},
    ]
)
print(f"多模态 HumanMessage content_blocks: {multimodal_msg.content_blocks}")

# ============================================================
# 3. AIMessage — 模型回复（含工具调用 + 推理 + token 用量）
# ============================================================
print("\n" + "=" * 50)
print("3. AIMessage — 模型回复")
print("=" * 50)

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"It's always sunny in {city}!"

model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What's the weather in Beijing and Tokyo?")

# 3a. 文本内容
print(f"text: {response.text}")

# 3b. content_blocks — 标准化格式
print(f"content_blocks: {response.content_blocks}")
# 输出类似：
# [{"type": "text", "text": ""},
#  {"type": "tool_call", "name": "get_weather", "args": {"city": "Beijing"}, "id": "call_..."},
#  {"type": "tool_call", "name": "get_weather", "args": {"city": "Tokyo"}, "id": "call_..."}]

# 3c. tool_calls — 提取工具调用（最常用）
if response.tool_calls:
    for tc in response.tool_calls:
        print(f"  → 工具: {tc['name']}, 参数: {tc['args']}, ID: {tc['id']}")

# 3d. usage_metadata — Token 用量
if response.usage_metadata:
    u = response.usage_metadata
    print(f"token 用量: 输入={u['input_tokens']}, 输出={u['output_tokens']}, 总计={u['total_tokens']}")

# 3e. 检查是否有推理/思考链（reasoning block）
for block in response.content_blocks or []:
    if block.get("type") == "reasoning":
        print(f"推理链: {block.get('reasoning')[:100]}...")

# ============================================================
# 4. ToolMessage — 工具执行结果
# ============================================================
print("\n" + "=" * 50)
print("4. ToolMessage — 工具执行结果")
print("=" * 50)

# 模拟一次完整的工具调用循环：
# ① 模型发出工具调用请求 (AIMessage.tool_calls)
# ② 执行工具，返回结果 (ToolMessage)
# ③ 把结果喂回模型，模型给出最终回答

# 第①步：模型请求调用工具
first_response = model_with_tools.invoke("What's the weather in Beijing?")
print(f"模型请求工具调用: {first_response.tool_calls}")

# 第②步：构造 ToolMessage（手动模拟工具执行）
tool_results = []
for tc in first_response.tool_calls:
    tool_result = get_weather.invoke(tc)  # 执行工具
    tool_results.append(tool_result)
    print(f"工具执行结果: {tool_result.content} (tool_call_id={tool_result.tool_call_id})")

# 第③步：将工具结果传回模型，继续对话
final_response = model_with_tools.invoke([
    {"role": "user", "content": "What's the weather in Beijing?"},
    first_response,        # AIMessage: 工具调用请求
    *tool_results,         # ToolMessage 列表: 工具执行结果
])
print(f"最终回答: {final_response.text}")

# ============================================================
# 5. 手动构造消息测试各种 content_blocks
# ============================================================
print("\n" + "=" * 50)
print("5. 手动构造 AIMessage 检查 content_blocks")
print("=" * 50)

# 手动构造一个含工具调用的 AIMessage
manual_ai = AIMessage(
    content="",  # 文本为空，工具调用通过 tool_calls 传递
    tool_calls=[{
        "name": "get_weather",
        "args": {"city": "Shanghai"},
        "id": "call_manual_001",
    }],
)
print(f"手动构造 AIMessage.tool_calls: {manual_ai.tool_calls}")

# 检查 content_blocks 是否会正确解析
print(f"手动构造 AIMessage.content_blocks: {manual_ai.content_blocks}")

# ============================================================
# 6. 消息类型检查 —— isinstance vs type
# ============================================================
print("\n" + "=" * 50)
print("6. 消息类型检查")
print("=" * 50)

msgs = [
    SystemMessage("You are a helper."),
    HumanMessage("Hello!"),
    AIMessage("Hi there!"),
]

for msg in msgs:
    print(f"  type={type(msg).__name__}, content={msg.content[:50]}")
