from app.deepseek_model import model
from langchain.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
import base64

# messages = HumanMessage(
#     content_blocks=[
#         {"type":"text", "text":"Describe the content of this image."},
#         {"type":"image", "url":"https://loremflickr.com/800/600/cat"}
#     ]
# )

# response = model.invoke([messages])
# print(response.content_blocks)

#from base64

# with open("D:\work_file\AI平台\ScreenShot_2026-06-03_151717_276.png","rb") as f:
#     encode = base64.b64encode(f.read()).decode("utf-8")

# messages = [
#     HumanMessage(
#         content_blocks=[
#             {"type":"text", "text":"Describe the content of this image."},
#             {"type":"image", 
#               "base64":encode,
#               "mime_type":"image/jpeg"
#              }
#         ]
#     )
# ]

# response = model.invoke(messages)
# print(response.content_blocks)

# with open("D:\\work_file\\AI平台\\sample-3s.wav", "rb") as f:
#     encode = base64.b64encode(f.read()).decode("utf-8")

# messages = HumanMessage(content=[
#     {"type": "text", "text": "Describe the content of this audio."},
#     {"type": "input_audio", "input_audio": {
#         "data": encode,
#         "format": "wav"
#     }},
# ])

# ⚠️ init_chat_model(openai) 不支持 modalities 参数,
# 所以 omni 模型的音频输出无法正常工作。
# 如果需要 omni 音频能力，请直接用 openai SDK 调用。
# response = model.invoke([messages])
# print(response.content_blocks)


# ============================================
# 视频输入示例（qwen3.5-omni-flash 支持）
# ============================================
messages = HumanMessage(content=[
    {"type": "text", "text": "Describe the content of this video."},
    {"type": "video_url", "video_url": {
        "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241115/cqqkru/1.mp4"
    }},
])

response = model.invoke([messages])
print(response.content_blocks)