# Agent 系统架构知识地图

## 一、Agent 的本质

传统 LLM：

用户 → LLM → 回复

只能生成文本。

Agent：

用户 → LLM → Tool → 环境 → 观察结果 → 再推理

Agent不仅能思考（Reason），还能行动（Act）。

核心循环：

Think
↓
Act
↓
Observe
↓
Think

---

## 二、Execution Environment（执行环境）

Agent需要一个工作空间。

组成：

### 1. Tools

工具调用能力

例如：

* Web Search
* 数据库查询
* MCP工具
* 邮件发送
* API调用

作用：

让Agent具备执行能力。

---

### 2. Filesystem

文件系统

Agent可以：

* 读取文件
* 写入文件
* 保存分析结果
* 跨轮次共享信息

例如：

project/
├── src/
├── report.md
└── data.csv

作用：

让Agent拥有长期工作区，而不是只依赖上下文。

---

### 3. Code Execution

代码执行环境

例如：

python script.py

或者：

mvn test

npm run build

docker compose up

作用：

Agent能够验证自己的结果。

---

## 三、Context Management（上下文管理）

Agent最大的瓶颈：

Context Window。

上下文包括：

* 用户消息
* Agent历史记录
* Tool返回结果
* 推理过程

随着任务进行：

Context越来越大。

最终溢出。

---

### 1. Summarization

自动总结

作用：

压缩历史信息。

例如：

10000 Token
↓
500 Token

避免Context爆炸。

---

### 2. Memory

长期记忆

作用：

跨会话保存信息。

例如：

用户是Java工程师

正在学习：

* Python
* LangChain
* LangGraph

下次启动自动加载。

---

### 3. Skills

按需知识加载

不要：

一次加载全部知识。

而是：

需要Kafka时加载Kafka知识。

需要PostgreSQL时加载PostgreSQL知识。

作用：

降低Prompt长度。

---

## 四、Planning & Delegation（规划与委派）

复杂任务超出单个Agent能力。

解决方案：

任务拆分。

---

### Main Agent

负责：

* 规划
* 协调
* 汇总

---

### SubAgent

负责：

* 实际执行

例如：

Database Agent

Backend Agent

Frontend Agent

DevOps Agent

---

### Context Isolation

每个SubAgent拥有独立上下文。

数据库Agent：

只看数据库。

后端Agent：

只看后端代码。

作用：

降低上下文压力。

---

### Parallel Execution

多个Agent同时工作。

类似：

线程池

CompletableFuture

作用：

提高效率。

---

## 五、 Agent Identifier

Agent名称。

例如：

database_agent

backend_agent

report_writer_agent

作用：

多Agent系统中的身份标识。

类似：

Spring服务名：

user-service

order-service

---

## 六、Fault Tolerance（容错机制）

生产环境一定会出现异常。

常见问题：

### Rate Limit

429

请求过多。

---

### Timeout

模型超时。

---

### Transient Error

502

503

网络抖动。

---

### Fault Tolerance Middleware

统一处理：

* Retry
* Timeout
* Fallback
* Circuit Breaker

作用：

业务代码无需到处写try/catch。

类似：

* Spring Retry
* Resilience4j
* Sentinel

---

## 七、Guardrails（护栏）

作用：

强制执行规则。

核心特点：

确定性执行。

不是靠Prompt。

而是靠代码。

---

例如：

禁止访问：

/etc/passwd

即使模型想访问：

直接拒绝。

---

典型场景：

* 敏感信息过滤
* 合规检查
* 内容审核
* 数据脱敏

---

架构：

Tool Result
↓
Guardrail
↓
LLM

---

## 八、Human-in-the-Loop（人工审批）

并非所有事情都适合自动执行。

关键决策需要人参与。

---

典型场景

### 危险操作

DELETE

DROP

rm -rf

---

### 高成本调用

预计消耗：

500美元

等待审批。

---

### 专业判断

医疗

法律

金融

---

流程：

Agent
↓
Pause
↓
Human
↓
Approve / Edit / Reject
↓
Resume

---

## 九、生产级Agent架构总览

```
                User
                  │
                  ▼
         Human Approval
                  │
                  ▼
            Guardrails
                  │
                  ▼
             Main Agent
                  │
   ┌──────────────┼──────────────┐
   ▼              ▼              ▼
```

Database Agent  Backend Agent  DevOps Agent
│              │              │
└──────────────┼──────────────┘
▼
Execution Environment
├─ Tools
├─ Filesystem
└─ Code Execution
│
▼
Fault Tolerance
│
▼
Model

---

## 十、学习路线（适合Java工程师）

第一阶段

Python基础

第二阶段

LangChain

掌握：

* Prompt
* Tool
* Memory

第三阶段

LangGraph

掌握：

* StateGraph
* Workflow
* Durable Execution

第四阶段

Deep Agents

掌握：

* SubAgent
* Context Management
* Planning
* Delegation

第五阶段

Claude Code / OpenCode

研究：

* Coding Agent
* Shell执行
* Approval机制
* Agent工程化

最终目标：

能够构建企业级Agent系统，而不是只会调用LLM接口。
