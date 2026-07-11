# `langchain_save` 数据库表结构与关系

> 基于 LangGraph `PostgresSaver` 自动创建的表（`langgraph.checkpoint.postgres`）  
> 项目连接：`pgsql_state.py` → `postgresql://postgres:***@192.168.1.100:25432/langchain_save`

---

## 1. `checkpoint_migrations` — Schema 版本表

```sql
CREATE TABLE checkpoint_migrations (
    v INTEGER PRIMARY KEY
);
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `v` | `INTEGER` | schema 版本号 |

**作用**：记录当前数据库 schema 版本。`setup()` 启动时读取此表，按版本顺序执行增量迁移（如加列、建索引），保证表结构始终匹配 LangGraph 版本。

**关系**：独立表，不与任何表关联。

---

## 2. `checkpoints` — State 全量快照表

```sql
CREATE TABLE checkpoints (
    thread_id            TEXT NOT NULL,
    checkpoint_ns        TEXT NOT NULL DEFAULT '',
    checkpoint_id        TEXT NOT NULL,       -- ULID，按时间排序
    parent_checkpoint_id TEXT,                -- 构成版本链 → checkpoints.checkpoint_id
    type                 TEXT,                -- 序列化类型
    checkpoint           JSONB NOT NULL,      -- 当前全量 state
    metadata             JSONB NOT NULL DEFAULT '{}',

    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
```

| 字段 | 说明 |
|------|------|
| `thread_id` | 对话/线程标识符（如 `"great-gatsby-lc"`） |
| `checkpoint_ns` | 命名空间，用于子图隔离 |
| `checkpoint_id` | ULID 格式，单调递增可排序 |
| `parent_checkpoint_id` | **上一个 checkpoint 的 ID**，构成单向链表 |
| `type` | 序列化格式标识 |
| `checkpoint` | 从 step 0 到当前的**全量累积 state**（JSONB），包含全部 messages、channel 值等 |
| `metadata` | 元数据，如 `{"source": "loop", "step": 2, "writes": {...}}` |

**索引**：
```
CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoints_thread_id_idx ON checkpoints(thread_id);
```

---

## 3. `checkpoint_blobs` — 大对象存储表

```sql
CREATE TABLE checkpoint_blobs (
    thread_id       TEXT NOT NULL,
    checkpoint_ns   TEXT NOT NULL DEFAULT '',
    channel         TEXT NOT NULL,      -- state channel 名
    version         TEXT NOT NULL,      -- 版本号
    type            TEXT NOT NULL,      -- 序列化类型（msgpack / pickle）
    blob            BYTEA,             -- 二进制大对象数据

    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);
```

| 字段 | 说明 |
|------|------|
| `channel` | state 中的 channel 名称（如 `"messages"`） |
| `version` | 该 channel 的版本号，用于增量覆盖 |
| `type` | 序列化方式（如 `"json"`, `"pickle"`） |
| `blob` | 实际的二进制数据，当值太大无法内联到 `checkpoints.checkpoint` 时存放于此 |

**索引**：
```
CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id);
```

**关系**：
- `(thread_id, checkpoint_ns)` → 关联 `checkpoints.thread_id` / `checkpoints.checkpoint_ns`
- 是 `checkpoints.checkpoint` JSONB 的**冷存储扩展**——JSONB 存不下的值外联到这里

---

## 4. `checkpoint_writes` — 节点级中间写表

```sql
CREATE TABLE checkpoint_writes (
    thread_id       TEXT NOT NULL,
    checkpoint_ns   TEXT NOT NULL DEFAULT '',
    checkpoint_id   TEXT NOT NULL,       -- 所属的 checkpoint
    task_id         TEXT NOT NULL,       -- 节点/工具执行 ID
    idx             INTEGER NOT NULL,    -- 该 task 内的写顺序
    channel         TEXT NOT NULL,       -- 写入的 channel
    type            TEXT,               -- 序列化类型
    blob            BYTEA NOT NULL,     -- 节点的产出数据

    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- 后续 migration 加的列
ALTER TABLE checkpoint_writes ADD COLUMN IF NOT EXISTS task_path TEXT NOT NULL DEFAULT '';
```

| 字段 | 说明 |
|------|------|
| `checkpoint_id` | 关联到 `checkpoints.checkpoint_id` |
| `task_id` | 超级步内某个节点/工具的唯一执行标识 |
| `idx` | 该 task 内部多个写入的顺序号 |
| `channel` | 写入 state 中的哪个 channel |
| `task_path` | 节点路径（子图场景下标识完整路径） |

**索引**：
```
CREATE INDEX CONCURRENTLY IF NOT EXISTS checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id);
```

---

## 核心关系：`checkpoint_writes` 与 `checkpoints`

### 一句话

> **`checkpoint_writes` = 增量（delta）**  
> **`checkpoints` = 全量（snapshot）**

### 关系图

```
                      checkpoint_writes
                    ┌──────────────────────┐
        Super-step  │ task_id: llm_call    │
        中的每个节点 │   channel: messages  │
       各自执行完毕  │   blob: tool_call    │
        → 立即写入   ├──────────────────────┤
                    │ task_id: get_weather │
                    │   channel: messages  │
                    │   blob: "25°C"       │
                    └──────────────────────┘
                              │
                              │ 超级步内所有节点完成
                              ▼
                    ┌──────────────────────────────────────┐
                    │          checkpoints                  │
                    │  checkpoint_id: "xxx-step-2"          │
                    │  parent_checkpoint_id: "xxx-step-1"   │
                    │  checkpoint: {                        │
                    │    messages: [                        │
                    │      {role: "user", ...},             │ ← 来自 step-1
                    │      {role: "assistant", ...tool_call},│ ← 来自 step-1
                    │      {role: "tool", content: "25°C"}  │ ← 累加 step-2 的结果
                    │    ]                                  │
                    │  }                                    │
                    │  metadata.writes: {                   │
                    │    "get_weather": {                   │
                    │      "messages": [tool_result]        │ ← 标识这步新增了什么
                    │    }                                  │
                    │  }                                    │
                    └──────────────────────────────────────┘
```

### 数学关系

```
checkpoints(N) = apply(
    checkpoint_writes(N),     → 这组增量
    到 checkpoints(N-1)       → 上一个全量
)
```

其中 `checkpoints(N-1).checkpoint_id = checkpoints(N).parent_checkpoint_id`

### 类比

| 概念 | 类比 | 说明 |
|------|------|------|
| `checkpoint_writes` | Git 的 **diff**（文件变更） | 每个节点独立产出的一小块更改 |
| `checkpoints` | Git 的 **commit 后的完整文件树** | 所有 diff 累积 apply 后的全量 |
| `parent_checkpoint_id` | Git 的 `parent commit` | 指向历史中的上一个 checkpoint |
| `checkpoint_id` | Git 的 `commit hash` | 单调递增，可以按时间回溯 |

### 设计意义

| 特性 | 实现方式 |
|------|---------|
| **容错恢复** | 同一超级步内，节点 A 成功 → writes 已持久化; 节点 B 失败 → 恢复后只重跑 B，A 的结果从 writes 读取，不浪费 |
| **Time Travel** | 按 `checkpoint_id` 反查 `checkpoints` 表，拿到当时的全量 state 还原 |
| **Pending Writes** | `get_tuple()` 时将当前 checkpoint 关联的 writes 读出来，作为未提交的增量一并返回 |

---

## 完整查询示例

### 查看一个 thread 的所有 checkpoint

```sql
SELECT checkpoint_id, parent_checkpoint_id, metadata->>'step' AS step,
       metadata->>'source' AS source
FROM checkpoints
WHERE thread_id = 'great-gatsby-lc'
ORDER BY checkpoint_id;
```

### 查看某个 checkpoint 关联的 writes

```sql
SELECT task_id, channel, idx,
       CASE WHEN type = 'pickle' THEN '🔒 pickled' ELSE blob::text END AS blob_preview
FROM checkpoint_writes
WHERE thread_id = 'great-gatsby-lc'
  AND checkpoint_ns = ''
  AND checkpoint_id = 'xxx-step-2'
ORDER BY task_id, idx;
```

### 查看版本链

```sql
WITH RECURSIVE chain AS (
    SELECT checkpoint_id, parent_checkpoint_id, metadata->>'step' AS step
    FROM checkpoints
    WHERE thread_id = 'great-gatsby-lc'
      AND checkpoint_id = 'latest-checkpoint-id'
    UNION ALL
    SELECT c.checkpoint_id, c.parent_checkpoint_id, c.metadata->>'step'
    FROM checkpoints c
    JOIN chain ON c.checkpoint_id = chain.parent_checkpoint_id
)
SELECT * FROM chain;
```