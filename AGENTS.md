# PROJECT KNOWLEDGE BASE

**Generated:** 2026-06-27
**Stack:** Python 3.14 · LangChain · LangGraph · DeepAgents · DashScope API

## OVERVIEW

LangChain Agent 学习/实验项目。通过独立可执行脚本演示 LangChain + DeepAgents + LangGraph 的核心概念（batch, stream, memory, context, structured output, token usage, web tool 等）。非生产项目。

## STRUCTURE

```
langchain-demo/
├── app/           # 全部源码（18 .py），扁平结构，无 __init__.py
├── docs/          # 架构知识笔记（中文 MD）
├── .omo/          # OpenCode 工作追踪
└── .vscode/       # 调试启动配置
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| 模型初始化 | `app/deepseek_model.py`, `app/qwen_model.py` | 模块级单例，导入即初始化 |
| Agent 基础 | `app/demo1.py`, `app/agent_demo.py` | 最简单的起点 |
| 流式输出 | `app/agent_stream.py` | stream_events + context |
| Context 注入 | `app/agent_context.py` | ToolRuntime 用法 |
| 记忆/对话历史 | `app/agent_mermory.py` | InMemorySaver + thread_id |
| 结构化输出 | `app/agent_structured_output.py` | Pydantic response_format |
| Token 用量 | `app/agent_token_usage.py` | UsageMetadataCallbackHandler |
| 批量处理 | `app/agent_batch.py` | batch / batch_as_completed |
| 思考链 | `app/agent_reasoning.py` | ⚠️ 已废弃，参考注释 |
| Web Search 工具 | `app/agent_web_tool.py` | 内置 web_search |
| 速率限制 | `app/qwen_model.py` | InMemoryRateLimiter（单线程安全） |
| LogProbs | `app/model_logprobs.py` | 对数概率调试 |
| 根级调试 | `debug_type.py` | stream_events 类型检查 |
| 架构文档 | `docs/what_is_agent_system.md` | Agent 系统知识（515 行中文） |
| 模型文档 | `docs/what_is_model.md` | 生产级 Agent 架构图解（146 行） |

## CODE MAP

| Symbol | Type | File | Role |
|--------|------|------|------|
| `OPENAI_API_KEY` | config | `app/config.py` | DashScope API key 入口 |
| `model` (DeepSeek) | singleton | `app/deepseek_model.py` | glm-5.1 via OpenAI-compat |
| `model` (Qwen) | singleton | `app/qwen_model.py` | qwen3.7-plus via ChatQwen |
| `model` (thinking) | singleton | `app/thinking_model.py` | enable_thinking=True |
| `create_agent()` | factory | deepagents | 标准 agent 创建 |
| `create_deep_agent()` | factory | deepagents | 子代理 agent 创建 |
| `InMemorySaver` | class | langgraph | 对话记忆检查点 |
| `ToolRuntime[T]` | class | deepagents | Context 类型化注入 |
| `UsageMetadataCallbackHandler` | class | langchain_core | Token 统计回调 |
| `InMemoryRateLimiter` | class | langchain_core | 请求限流器 |
| `ChatQwen` | class | langchain_qwq | Qwen 官方 LangChain 集成 |
| `init_chat_model()` | factory | langchain | OpenAI 兼容模型初始化 |

## CONVENTIONS

- **导入即初始化**: `deepseek_model.py` 和 `qwen_model.py` 在模块顶层实例化 model，其他文件直接 `from deepseek_model import model`。
- **入口即脚本**: 所有 `.py` 文件是独立可执行入口，**没有** `if __name__ == "__main__"` 守卫。
- **文档即注释**: 功能说明和已知坑写在文件顶部的中文 docstring 中，而非独立文档。
- **工具必须有 docstring**: deepagents 运行时强制要求 tool 函数有 docstring。
- **thread_id 用 uuid7**: 所有对话记忆的 thread_id 使用 `uuid.uuid7()` 生成。
- **API 端点**: 统一使用 DashScope（阿里云）OpenAI 兼容地址 `https://dashscope.aliyuncs.com/compatible-mode/v1/`。

## ANTI-PATTERNS (THIS PROJECT)

- **`Field(deprecated=...)`** — Pydantic v2 中 `deprecated` 不是 `Field` 有效参数，会导致 TypeError。已在 `agent_structured_output.py:12`。
- **变量覆盖内建函数**: `agent_batch.py:9` 用 `list` 作为变量名。
- **废弃代码保留**: `agent_reasoning.py` 整体废弃但未删除，含过期 API 用法。
- **`print()` 替代 logging**: 全项目 12 文件 23 处 `print()`，不适用于生产环境。
- **空/占位文件**: `agent_runable_config.py`（0 行）和 `agent_rate_limiter.py`（1 行注释）无实际内容。
- **模块级单例副作用**: `deepseek_model.py` / `qwen_model.py` 在 import 时即发起网络模型初始化，有失败风险。

## UNIQUE STYLES

- 代码注释 + 文件顶部文档均使用**中文**，变量/函数名用英文。
- 同时使用两个 Agent 框架：`create_agent`（标准 LangChain）+ `create_deep_agent`（deepagents 扩展）。
- 双模型体系：`init_chat_model`（OpenAI 兼容适配器）和 `ChatQwen`（官方 Qwen 集成）共存，各有不同的能力边界。
- 模型命名与实际不符：`deepseek_model.py` 实际初始化的是 `glm-5.1`（智谱），`thinking_model.py` 初始化的是 `qwen3.7-plus`。

## COMMANDS

```bash
# 运行任意 demo（工作目录为项目根）
python app/demo1.py
python app/agent_stream.py
python app/agent_context.py

# 调试（VS Code F5 或）
python debug_type.py

# 虚拟环境
.venv\Scripts\activate
```

## NOTES

- 所有脚本调用**外部 API**（DashScope），需要 `.env` 中有效的 `OPENAI_API_KEY`。
- DashScope 的 `reasoning_content` 字段被 LangChain OpenAI 适配器丢弃，如需思考链内容请使用 `langchain_qwq.ChatQwen` 或原始 HTTP 调用。
- `create_agent` + `response_format` 依赖 `tool_choice`，与 `enable_thinking=True` 冲突。
- `create_deep_agent` 注入大量内置工具（ls, execute, task 等），不适合简单问答场景。
- `InMemoryRateLimiter` 是线程安全但**非多进程安全**，多 Agent Server 场景需要自己实现。
- 无 `requirements.txt`：依赖锁定仅通过 `.venv/` 隐式管理。
- 无测试：所有验证通过手动 `python app/xxx.py` 运行。
