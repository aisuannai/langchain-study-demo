"""
自定义 StreamTransformer 示例

演示如何写一个自定义投影（custom projection）来追踪
工具调用的生命周期，并通过 stream.extensions 消费。

概念:
  - StreamTransformer: 观察协议事件，维护内部状态，暴露派生视图
  - StreamChannel:     投影原语，push 的值可在 stream.extensions 上迭代
  - stream.extensions:  自定义投影的入口

运行:
  python app/stream/custom_transformer.py
"""

import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中，支持从子目录运行
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain.agents import create_agent
from langgraph.stream import ProtocolEvent, StreamChannel, StreamTransformer

from app.deepseek_model import model


# ============================================================
# 1. 定义一个工具
# ============================================================

def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    time.sleep(1)  # 模拟耗时
    return f"{city} 的天气是晴天，气温 25°C"


def search_news(topic: str) -> str:
    """搜索指定主题的新闻"""
    time.sleep(1.5)
    return f"找到关于 '{topic}' 的最新新闻 3 条"


# ============================================================
# 2. 自定义 Transformer —— 追踪工具调用状态
# ============================================================

class ToolActivityTransformer(StreamTransformer):
    """
    观察 tools 频道的事件，提取工具名和状态，
    推送到 tool_activity 投影上。

    关键: tool-finished / tool-error 事件里没有 tool_name，
    需要用 tool_call_id 关联到 tool-started 时记录的名字。
    """

    # 声明需要 tools 流模式 —— 没有这行，tools 事件根本不会到达 process()
    required_stream_modes = ("tools",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        # 命名 channel：值会同时出现在 stream.extensions 和原始事件流中
        self.activity = StreamChannel[dict]("tool_activity")
        # 按 tool_call_id 缓存工具名，因为 finish/error 事件不携带 tool_name
        self._tool_names: dict[str, str] = {}

    def init(self) -> dict:
        """返回投影的入口名 -> StreamChannel 映射"""
        return {"tool_activity": self.activity}

    def process(self, event: ProtocolEvent) -> bool:
        """每个协议事件都会经过这里"""
        if event["method"] != "tools":
            return True  # 不关心的频道直接放过

        data = event["params"]["data"]
        if not isinstance(data, dict):
            return True

        event_type = data.get("event", "?")
        tool_call_id = data.get("tool_call_id")

        # tool-started: 缓存 tool_name，供后续事件使用
        if event_type == "tool-started" and tool_call_id:
            self._tool_names[tool_call_id] = data.get("tool_name", "?")

        # 从缓存查 tool_name（所有事件类型都适用）
        tool_name = self._tool_names.get(tool_call_id, data.get("tool_name", "?"))

        # 映射事件类型为可读状态
        status_map = {
            "tool-started":      "[start]  开始",
            "tool-output-delta": "[delta]  输出中",
            "tool-finished":     "[done]   完成",
            "tool-error":        "[error]  出错",
        }
        status = status_map.get(event_type, event_type)

        # 推送到 channel —— 消费者通过 stream.extensions["tool_activity"] 收到
        self.activity.push({
            "tool": tool_name,
            "status": status,
            "at": time.strftime("%H:%M:%S"),
        })

        return True  # True = 保留原始事件不放行；False = 压制

    def finalize(self) -> None:
        """流结束时触发"""
        self.activity.push({"tool": "", "status": "[finish] 流结束", "at": time.strftime("%H:%M:%S")})


# ============================================================
# 3. 创建 agent 并消费自定义投影
# ============================================================

def main():
    agent = create_agent(
        model=model,
        tools=[get_weather, search_news],
    )

    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "上海的天气怎么样？顺便搜一下 AI 新闻"}]},
        version="v3",
        transformers=[ToolActivityTransformer],   # 注册自定义 transformer
    )

    # --- 方式 A: 通过 stream.extensions 消费自定义投影 ---
    print("=" * 50)
    print("[stream.extensions] 工具活动追踪")
    print("=" * 50)
    for activity in stream.extensions["tool_activity"]:
        if activity["tool"]:
            print(f"  [{activity['at']}] {activity['tool']}: {activity['status']}")
        else:
            print(f"  {activity['status']}")

    # --- 方式 B: 同时也可以消费标准投影 ---
    print("\n" + "=" * 50)
    print("[stream.messages] 模型输出")
    print("=" * 50)
    for message in stream.messages:
        print(f"  {message.text}", end="", flush=True)

    print("\n")
    print(f"[output] 最终状态: {stream.output['messages'][-1].content[:80]}...")


main()
