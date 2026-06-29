"""
Store — 长时记忆（跨对话持久化）
=================================

1. 为什么需要长时记忆？
   - 短时记忆（State + checkpointer）只在同一个 thread_id 内有效
   - 长时记忆（Store）跨对话、跨 thread 持久化数据
   - 典型场景：用户偏好、知识库、历史行为记录

2. Store 核心概念：
   - namespace: 命名空间（类似文件夹路径），用于组织数据
     → 例如 ("users", "u_123") 或 ("preferences",)
   - key: 每条记录的 ID（类似文件名）
   - value: 存储的数据（dict 格式）

3. Store 的两种模式：
   - Profile 模式：单条记录持续更新（如用户画像）
   - Collection 模式：多条记录持续追加（如笔记集合）

4. InMemoryStore vs 生产级 Store：
   - InMemoryStore: 内存存储，重启丢失，仅用于实验
   - PostgresStore: 生产环境用，数据持久化到数据库

5. 语义搜索（向量检索）：
   InMemoryStore 支持 index 参数，可以按语义搜索存储的内容。
   需要提供 embedding 函数。
"""

from uuid import uuid7
from app.deepseek_model import model
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, ToolRuntime
from deepagents import create_deep_agent

# ============================================================
# 1. Store 基础操作：put / get / search
# ============================================================
print("=" * 60)
print("1. Store 基础操作：put / get / search")
print("=" * 60)

store = InMemoryStore()

# 1a. put — 写入数据
# namespace 用元组表示层级，例如 ("users", "profiles") 表示 users/profiles/
store.put(("users", "profiles"), "u_001", {
    "name": "张三",
    "age": 28,
    "city": "北京",
    "preferences": ["技术", "AI", "读书"],
})
store.put(("users", "profiles"), "u_002", {
    "name": "李四",
    "age": 35,
    "city": "上海",
    "preferences": ["金融", "投资", "旅游"],
})

# 1b. get — 读取单条记录
item = store.get(("users", "profiles"), "u_001")
if item:
    print(f"get('u_001'): name={item.value['name']}, city={item.value['city']}")

# 1c. search — 按命名空间搜索所有记录
items = store.search(("users", "profiles"))
print(f"search 所有用户: {[it.value['name'] for it in items]}")

# ============================================================
# 2. Profile 模式：单条记录持续更新
# ============================================================
print("\n" + "=" * 60)
print("2. Profile 模式：单条记录持续更新")
print("=" * 60)

# 通常用于用户画像，每次对话后更新用户偏好
profile_store = InMemoryStore()

@tool
def update_user_profile(key: str, field: str, value: str, runtime: ToolRuntime) -> str:
    """Update a field in the user's profile.
    Args:
        key: User profile key
        field: Field name to update (e.g. name, age, city, interest)
        value: New value for the field"""
    s = runtime.store
    namespace = ("users", "profiles")
    existing = s.get(namespace, key)
    profile = existing.value.copy() if existing else {}
    profile[field] = value
    s.put(namespace, key, profile)
    return f"用户 {key} 的 {field} 已更新为 {value}。"

@tool
def get_user_profile(key: str, runtime: ToolRuntime) -> str:
    """Get the complete user profile.
    Args:
        key: User profile key"""
    s = runtime.store
    item = s.get(("users", "profiles"), key)
    if item is None:
        return f"用户 {key} 不存在。"
    return f"用户信息: " + ", ".join(f"{k}={v}" for k, v in item.value.items())

agent_profile = create_deep_agent(
    model=model,
    tools=[update_user_profile, get_user_profile],
    store=profile_store,
)

# 第一次对话：创建并更新 profile
agent_profile.invoke(
    {"messages": [{"role": "user", "content": "更新用户 profile_" + str(uuid7())[:8] + " 的 name 为 Alice"}]},
    config={"configurable": {"thread_id": str(uuid7())}},
)

# ============================================================
# 3. Collection 模式：多条记录追加（笔记/记忆集合）
# ============================================================
print("\n" + "=" * 60)
print("3. Collection 模式：笔记集合")
print("=" * 60)

# Collection 模式通常使用带 key 的多条记录，每条独立
note_store = InMemoryStore()

# 预置几条笔记演示 search
note_store.put(("notes",), "note_1", {"title": "Python 技巧", "content": "使用 list comprehension", "tags": ["python"]})
note_store.put(("notes",), "note_2", {"title": "LangChain 笔记", "content": "create_agent 比 create_deep_agent 更轻量", "tags": ["langchain", "ai"]})
note_store.put(("notes",), "note_3", {"title": "工具函数", "content": "@tool 装饰器自动解析函数签名", "tags": ["python", "langchain"]})

# search 出所有笔记
all_notes = note_store.search(("notes",))
for n in all_notes:
    print(f"  {n.key}: {n.value['title']} — {n.value['content'][:40]}")

# ============================================================
# 4. 语义搜索（向量检索）
# ============================================================
print("\n" + "=" * 60)
print("4. 语义搜索（向量检索）")
print("=" * 60)

def dummy_embed(texts: list[str]) -> list[list[float]]:
    """简单的 dummy embedding 函数，实际项目中用 OpenAI/本地 embedding 模型。
    这里返回固定维度向量，仅演示 InMemoryStore 搜索 API 的用法。"""
    return [[0.1, 0.2, 0.3] for _ in texts]

semantic_store = InMemoryStore(index={"embed": dummy_embed, "dims": 3})

semantic_store.put(("docs",), "doc_1", {"text": "LangChain is a framework for LLM apps"})
semantic_store.put(("docs",), "doc_2", {"text": "Python is a programming language"})
semantic_store.put(("docs",), "doc_3", {"text": "Machine learning uses neural networks"})

# search 支持 filter 和 query（向量相似度排序）
results = semantic_store.search(
    ("docs",),
    query="LLM application",  # 语义查询，按向量相似度排序
)
print(f"语义搜索结果（按相关性排序）:")
for r in results:
    print(f"  {r.key}: {r.value['text'][:50]} (score={r.score:.2f})")

# ============================================================
# 5. 跨 namespace 搜索
# ============================================================
print("\n" + "=" * 60)
print("5. 跨 namespace 搜索")
print("=" * 60)

# Store 支持在不同 namespace 下组织不同种类的数据
ns_store = InMemoryStore()

ns_store.put(("用户", "北京"), "u_01", {"name": "张三"})
ns_store.put(("用户", "上海"), "u_02", {"name": "李四"})
ns_store.put(("用户", "北京"), "u_03", {"name": "王五"})
ns_store.put(("设置",), "global", {"theme": "dark"})

# 搜索特定 namespace
bj_users = ns_store.search(("用户", "北京"))
print(f"北京用户: {[it.value['name'] for it in bj_users]}")

# ============================================================
# 6. 用 Agent 读写 Store（跨对话持久化）
# ============================================================
print("\n" + "=" * 60)
print("6. Agent 中使用 Store — 跨对话读写")
print("=" * 60)

agent_store = InMemoryStore()

@tool
def save_fact(fact_key: str, fact_value: str, runtime: ToolRuntime) -> str:
    """Save a fact to long-term memory.
    Facts are remembered across conversations.
    Args:
        fact_key: Unique key for this fact (e.g. 'user_hobby')
        fact_value: The fact content to remember (e.g. 'loves hiking')"""
    s = runtime.store
    # 读取已有事实，追加新内容
    namespace = ("facts",)
    existing = s.get(namespace, fact_key)
    data = existing.value.copy() if existing else {}
    data["value"] = fact_value
    s.put(namespace, fact_key, data)
    return f"已记住: {fact_key} = {fact_value}。"

@tool
def recall_fact(fact_key: str, runtime: ToolRuntime) -> str:
    """Recall a fact previously saved to long-term memory.
    Args:
        fact_key: The key of the fact to recall"""
    s = runtime.store
    item = s.get(("facts",), fact_key)
    if item is None:
        return f"不记得 {fact_key} 了。"
    return f"我记得: {item.value['value']}。"

agent_mem = create_deep_agent(
    model=model,
    tools=[save_fact, recall_fact],
    store=agent_store,
)

# 第一次对话（thread_1）：保存事实
agent_mem.invoke(
    {"messages": [{"role": "user", "content": "帮我记住: fact_key=user_hobby, fact_value=我喜欢爬山和摄影"}]},
    config={"configurable": {"thread_id": str(uuid7())}},
)
print("第一次对话：已保存事实。")

# 第二次对话（thread_2，不同会话）：回忆
result = agent_mem.invoke(
    {"messages": [{"role": "user", "content": "我的爱好是什么？用 recall_fact 查 fact_key=user_hobby"}]},
    config={"configurable": {"thread_id": str(uuid7())}},
)
print(f"第二次对话（跨 thread）: {result['messages'][-1].text}")

# ============================================================
# 7. Store 注意事项
# ============================================================
print("\n" + "=" * 60)
print("7. 注意事项")
print("=" * 60)

print("""
- InMemoryStore 重启后数据丢失，生产环境请用 PostgresStore
- namespace 建议按 (entity_type, scope) 组织，如 ("users", "profiles")
- store.put() 是 upsert 语义：key 存在则覆盖，不存在则创建
- store.search() 当前 namespace 下搜索，不支持跨 namespace
- 语义搜索需要提供 embedding 函数，且 store 初始化时传入 index 参数
- 生产环境建议加上 filter 参数做精确筛选
""")
