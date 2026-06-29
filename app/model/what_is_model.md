
**生产级AGENT架构**
                        ┌──────────────────────────────┐
                        │          User / API          │
                        │   (Web / App / MCP Client)  │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                ┌─────────────────────────────────────────┐
                │        API Gateway / Agent Server       │
                │   (Spring Boot / FastAPI / Node)        │
                └──────────────┬──────────────────────────┘
                               │
                               ▼
        ┌────────────────────────────────────────────────────┐
        │                 Guardrails Layer                  │
        │  (Policy / Security / Compliance / Validation)    │
        │                                                    │
        │  - Prompt Injection 防护                           │
        │  - 敏感数据过滤                                   │
        │  - Tool调用白名单                                 │
        └──────────────┬─────────────────────────────────────┘
                               │
                               ▼
        ┌────────────────────────────────────────────────────┐
        │              Agent Orchestration Layer            │
        │        (LangGraph / Deep Agents / AutoGen)        │
        │                                                    │
        │   ┌──────────────────────────────────────────┐    │
        │   │         Planner (主Agent)               │    │
        │   │  - 任务拆解                            │    │
        │   │  - 决策流控制                          │    │
        │   │  - SubAgent编排                        │    │
        │   └──────────────┬──────────────────────────┘    │
        │                  │                                │
        │     ┌────────────┼────────────┬────────────┐     │
        │     ▼            ▼            ▼            ▼     │
        │  DB Agent   Backend Agent  Frontend    DevOps    │
        │ (SubAgent)   (SubAgent)    (SubAgent)  (SubAgent)│
        │                                                    │
        │  👉 每个SubAgent = 独立 Context + 独立 Loop      │
        └──────────────┬─────────────────────────────────────┘
                               │
                               ▼
        ┌────────────────────────────────────────────────────┐
        │              Execution Environment                 │
        │                                                    │
        │  ┌──────────────┐  ┌──────────────┐              │
        │  │   Tools      │  │ Filesystem   │              │
        │  │ - DB query   │  │ /workspace   │              │
        │  │ - API calls  │  │ reports.md   │              │
        │  │ - Search     │  │ logs/        │              │
        │  └──────────────┘  └──────────────┘              │
        │                                                    │
        │  ┌──────────────────────────────────────────┐     │
        │  │ Code Execution (Sandbox)                 │     │
        │  │ - Python / Shell / SQL / Docker         │     │
        │  └──────────────────────────────────────────┘     │
        └──────────────┬─────────────────────────────────────┘
                               │
                               ▼
        ┌────────────────────────────────────────────────────┐
        │              LLM Reasoning Engine                 │
        │   (GPT / Claude / Gemini / Local Models)          │
        │                                                    │
        │  - Tool Selection                                 │
        │  - Planning                                       │
        │  - Reasoning                                     │
        │  - Structured Output (JSON / Function Call)      │
        └──────────────┬─────────────────────────────────────┘
                               │
                               ▼
        ┌────────────────────────────────────────────────────┐
        │              Fault Tolerance Layer                │
        │                                                    │
        │  - Retry (指数退避)                                │
        │  - Timeout                                        │
        │  - Circuit Breaker                                │
        │  - Rate Limit Handling                             │
        │  - Fallback Models                                 │
        └──────────────┬─────────────────────────────────────┘
                               │
                               ▼
        ┌────────────────────────────────────────────────────┐
        │              Context Management Layer             │
        │                                                    │
        │  - Summarization (压缩历史)                        │
        │  - Memory (跨会话记忆)                             │
        │  - Skills (按需加载知识)                           │
        │  - Context Window 管理                             │
        └──────────────┬─────────────────────────────────────┘
                               │
                               ▼
        ┌────────────────────────────────────────────────────┐
        │          Human-in-the-Loop (Steering)            │
        │                                                    │
        │  - Approval (SQL / delete / deploy)              │
        │  - Edit / Reject                                 │
        │  - Cost confirmation                             │
        └──────────────┬─────────────────────────────────────┘
                               │
                               ▼
        ┌────────────────────────────────────────────────────┐
        │            Observability Layer (LangSmith)        │
        │                                                    │
        │  - Trace every LLM call                          │
        │  - Tool execution tracking                        │
        │  - Debug reasoning chain                          │
        │  - Model comparison                               │
        │  - Performance monitoring                         │
        └────────────────────────────────────────────────────┘



LLM core think

1. LLM不负责执行，只负责决策
2. context不能无限增长，必须工程化管理
3. 可靠性来自系统，不来自模型

Streaming = 数据流，不是返回值
系统必须支持增量消费
任何步骤是阻塞式的就会破坏streaming

astream_events() 将 LLM 从“文本生成器”升级为“可观测的事件驱动系统”
把ai的每一个步骤都分类


Agent本质
事件 + 状态机
LLM 产生事件（token / tool call）
callback 收集事件
系统根据事件决定下一步
LangChain 的本质是“事件驱动执行系统”，stream 只是事件的实时消费方式，invoke 只是事件的批量收集方式

生产标准配置

LangGraph Agent
    ↓
Prompt Cache（降成本）

Rate Limiter（防限流）

Retry（自动重试）

Streaming（流式输出）