# Python asyncio 与 Java 并发编程对比

> 从 Java 工程师视角理解 Python 的 async/await 事件循环

---

## 一、架构总览

### Python asyncio（协作式 + 单线程）

```
┌─────────────────────────────────────────────────────┐
│                  事件循环 (Event Loop)                 │
│                    单线程运行                          │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ 协程 A   │  │ 协程 B   │  │ 协程 C   │    ...     │
│  │ main()   │  │ consume_ │  │ consume_ │           │
│  │          │  │ messages │  │ tool_calls│           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │              │             │                  │
│       │   await 时   │  主动交出    │                  │
│       │ ──────────►  │ ──────────► │                  │
│       │  控制权      │  控制权      │                  │
└───────┴──────────────┴─────────────┴──────────────────┘
               │                ▲
               │  async def     │  asyncio.run()
               ▼                │
        user code ──────────────┘
```

### Java 线程（抢占式 + 多线程）

```
┌──────────────────────────────────────────────────────┐
│                    JVM                                │
│                                                       │
│  ┌──────────────┐   ┌──────────────┐                 │
│  │  线程池       │   │  线程池       │                 │
│  │  ThreadPool  │   │  ForkJoin    │                 │
│  │  Executor    │   │  Pool        │                 │
│  └──────┬───────┘   └──────┬───────┘                 │
│         │                  │                          │
│  ┌──────▼────────┐  ┌─────▼─────────┐                │
│  │ Thread-1      │  │ Thread-2      │                │
│  │ 栈: 1MB       │  │ 栈: 1MB       │                │
│  │ running...    │  │ waiting...    │                │
│  └───────────────┘  └───────────────┘                │
│                                                       │
│  操作系统内核 KERNEL                                   │
│  ┌──────────────────────────────────────┐            │
│  │ 线程调度器 (Preemptive Scheduler)     │            │
│  │ 随时暂停线程 → 切换上下文 → 恢复其他   │            │
│  └──────────────────────────────────────┘            │
└──────────────────────────────────────────────────────┘
```

---

## 二、调度方式对比图

### 场景：两个任务——"烧水 2s" + "切菜 1s"

#### Python asyncio（协作式调度）

```
时间(ms)     0    500   1000   1500   2000
            │     │     │     │     │
main ───────┼─────┼─────┼─────┼─────┼───►
            │
boil_water  ────► await sleep(2) ───────────►
            │         │(挂起)                  │
cut_veg     │         ────► await sleep(1)──► │
            │               │(挂起)            │
            │               │                 │
线程1:  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (一直活跃)
            ↑ 事件循环不断轮询 ready 队列
```

关键：**只有一个线程，全程活跃。** 协程主动挂起时，线程立刻去跑另一个协程。

#### Java Thread（抢占式调度）

```
时间(ms)     0    500   1000   1500   2000
            │     │     │     │     │
main ───────┼─────┼─────┼─────┼─────┼───►
            │
Thread-1    ────► sleep(2000) ──────────────►
(boil)      │    ░░░░░(阻塞)░░░░░░░░░░       │
            │
Thread-2    │     ──► sleep(1000) ──►        │
(cut_veg)   │         ░░░(阻塞)░░░            │
            │                                 │
Thread-1:  ▓▓░░░░░░░░░░░░░░░░░▓▓▓▓▓▓
Thread-2:  ░░░▓▓▓░░░░░░░▓▓▓░░░░░░░░
            ↑ JVM 的线程调度器随时可以暂停任意线程
```

关键：**两个线程都可能被 JVM 随时暂停**，即使它们没有阻塞。线程切换由内核决定，你无法控制。

---

## 三、await vs Thread.sleep 行为对比

```python
# Python：await 挂起协程，线程不阻塞
async def fetch():
    print("开始请求")           # 执行
    await asyncio.sleep(2)     # 挂起协程 → 事件循环跑其他协程
    print("请求完成")           # 2s 后恢复
    return "data"

# 此时线程：▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (一直在工作)
```

```java
// Java：Thread.sleep 阻塞整个线程
String fetch() throws InterruptedException {
    System.out.println("开始请求");     // 执行
    Thread.sleep(2000);                // 线程阻塞 → 不能干任何事
    System.out.println("请求完成");     // 2s 后继续
    return "data";
}

// 此时线程：▓▓▓░░░░░░░░░░░░░░░░░▓▓▓ (阻塞了2秒)
```

| 效果 | `await asyncio.sleep(2)` | `Thread.sleep(2000)` |
|------|--------------------------|----------------------|
| 当前执行单元 | 协程挂起 | 线程阻塞 |
| 线程状态 | **RUNNABLE**（仍在跑其他协程） | **TIMED_WAITING**（闲置） |
| 能利用这段时间做别的吗 | ✅ 事件循环分配其他协程 | ❌ 这个线程废了，必须等 |
| 谁决定何时恢复 | 事件循环（用户态） | 操作系统（内核态） |
| 10000 个并发 | ✅ 轻松支持（1 线程） | ❌ 创建 10000 个线程会 OOM |

---

## 四、并行 vs 并发

```
并发 (Concurrency)：交替执行，看起来像同时
并行 (Parallelism)：真正同时执行

Python asyncio：
  ──A────A────A────A──  （单线程，协程交替）
  ──B────B────B────B──
  ↑ 并发（concurrent），但不是并行（parallel）

Java 多线程：
  ──A──────────────────  （线程1，独立 CPU）
  ──────B──────────────  （线程2，独立 CPU）
  ↑ 可能真正并行（parallel）
```

| | Python asyncio | Java Thread |
|---|---|---|
| 利用多核？ | ❌ 单线程，用不了多核 | ✅ 多线程，可以用满多核 |
| 何时用多核？ | 加 `multiprocessing` 或 `concurrent.futures` | 默认就支持 |
| 切换开销 | 函数调用级别（~几十 ns） | 内核上下文切换（~µs 级） |

---

## 五、锁与共享变量

```
Python asyncio（单线程）：

  协程A                   协程B
    │                      │
    │ read count=0         │
    │ count += 1           │  ← 同一线程，不会同时执行
    │ write count=1        │
    │                      │ read count=1
    │    ...               │ count += 1
    │                      │ write count=2
    │                      │
  └── 不需要锁，同一时间只有一个人在操作 ──┘


Java 多线程：

  Thread-1                 Thread-2
    │                        │
    │ read count=0           │ read count=0    ← 同时读到 0！
    │ count += 1             │ count += 1
    │ write count=1          │ write count=1   ← 都写了 1，结果错误
    │                        │
  └── 必须加 synchronized / Lock ──┘
```

| | Python asyncio | Java Thread |
|---|---|---|
| 共享变量安全吗 | ✅ 天然安全（单线程） | ❌ 需要锁保护 |
| 需要 synchronized？ | ❌ 不需要 | ✅ 必须 |
| 竞态条件（race condition） | 极难出现 | 极容易出现 |

---

## 六、并发规模对比

```
Python asyncio（1 线程）
  ┌────┐
  │    │ 100,000 个协程共存
  │    │ 每个协程 ≈ 几个字节（栈在堆上）
  │    │
  └────┘ ✓ 没问题

Java 多线程
  ┌────┐
  │    │ 100,000 个线程共存
  │    │ 每个线程 ≈ 1MB 栈空间
  │    │ 总共 ≈ 100GB 内存
  │    │
  └────┘ ✗ OOM
```

---

## 七、代码模式对比

### 启动入口

```python
# Python
asyncio.run(main())    # 创建事件循环 → 运行 → 关闭
```

```java
// Java
new Thread(() -> {     // 创建线程 → 运行
    // ...
}).start();            // 启动
```

### 等待一个任务

```python
result = await fetch_data()     # 挂起协程，等结果
```

```java
Future<Result> future = executor.submit(task);
Result result = future.get();   // 阻塞线程，等结果
```

### 等待多个任务

```python
results = await asyncio.gather(
    fetch("https://a.com"),
    fetch("https://b.com"),
    fetch("https://c.com"),
)
```

```java
CompletableFuture<String> a = fetch("https://a.com");
CompletableFuture<String> b = fetch("https://b.com");
CompletableFuture<String> c = fetch("https://c.com");
List<String> results = CompletableFuture.allOf(a, b, c)
    .thenApply(v -> Stream.of(a, b, c)
        .map(CompletableFuture::join)
        .toList())
    .join();
```

### 超时控制

```python
try:
    result = await asyncio.wait_for(
        fetch_data(), timeout=5.0
    )
except asyncio.TimeoutError:
    print("超时")
```

```java
try {
    Result result = future.get(5, TimeUnit.SECONDS);
} catch (TimeoutException e) {
    System.out.println("超时");
}
```

---

## 八、总结对照表

| 维度 | Python asyncio | Java 多线程 |
|------|---------------|-------------|
| **调度方式** | 协作式（协程主动让出） | 抢占式（JVM/OS 随时暂停） |
| **执行单元** | 协程（coroutine） | 线程（Thread） |
| **底层数量** | 1 个线程 | N 个线程 |
| **切换开销** | ~几十 ns（函数调用） | ~µs 级（内核态切换） |
| **10 万并发** | ✅ 轻松 | ❌ 内存爆炸 |
| **利用多核** | ❌ 默认不行（需 +multiprocessing） | ✅ 默认支持 |
| **共享变量** | ✅ 不需要锁 | ❌ 需要锁 |
| **切换谁决定** | 程序员（await 的位置） | JVM + OS（不可控） |
| **适合场景** | I/O 密集型（网络、数据库、文件） | 两者都行，CPU 密集型更合适 |
| **阻塞 vs 挂起** | await 挂起协程，线程继续工作 | Thread.sleep 阻塞线程，线程闲置 |

---

## 九、一句话总结

> **Python asyncio 是用一个线程模拟出"多任务并行"的假象，代价是只能跑 I/O 密集型；Java 多线程是用多个真正线程干多件事，代价是内存占用和锁的复杂度。**

---

# 附录：Python 并发三大家族

> asyncio 只是 Python 并发拼图的一块。实际开发中还会遇到 threading 和 multiprocessing。

## 一、概览

```
                  Python 并发
                      │
        ┌─────────────┼─────────────┐
        │             │             │
     asyncio      threading     multiprocessing
        │             │             │
   async/await    ThreadPool    ProcessPool
                  Executor       Executor
```

| 维度 | asyncio | threading | multiprocessing |
|------|---------|-----------|-----------------|
| **执行单元** | 协程（单线程内） | 线程（共享内存） | 进程（独立内存） |
| **调度方式** | 协作式（主动 await） | 抢占式（OS 调度） | 抢占式（OS 调度） |
| **GIL 影响** | 无（单线程） | 有（计算要抢 GIL） | 无（独立解释器） |
| **真正并行** | ❌ 否 | ❌ 否（GIL 限制） | ✅ 是（多核） |
| **切换开销** | ~几十 ns | ~µs 级 | ~ms 级 |
| **内存共享** | 天然共享（同线程） | ✅ 共享（需锁） | ❌ 独立（需序列化） |
| **适用场景** | I/O 密集型 | 阻塞 I/O + 少量计算 | CPU 密集型 |

## 二、asyncio — 协程并发

```python
# 适合：网络请求、API 调用、数据库查询、文件读写
import asyncio

async def fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

async def main():
    results = await asyncio.gather(
        fetch("https://a.com"),
        fetch("https://b.com"),
        fetch("https://c.com"),
    )

asyncio.run(main())
```

| 单线程 | 协作式调度 | I/O 密集型首选 |
|--------|-----------|---------------|

当你的程序主要花时间在"等"（等网络、等数据库、等文件读取）时，asyncio 是最省资源的选择。

## 三、threading — 线程并发

```python
# 适合：阻塞 I/O、调用不支持 async 的 SDK
from concurrent.futures import ThreadPoolExecutor
import time

def blocking_io(url):
    time.sleep(1)  # 模拟阻塞调用
    return f"done {url}"

with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(blocking_io, [
        "https://a.com",
        "https://b.com",
        "https://c.com",
    ]))
```

**重要限制：CPython 有 GIL（全局解释器锁）。**

```
线程A: ▓▓▓▓▓▓▓ 计算中 ▓▓▓▓▓▓▓
                ↓ 抢到 GIL
线程B: ░░░░░░░░ 等待GIL ░░░░░░
                ↑ 没抢到，干等

结论：两个线程做纯计算，不会比一个线程快。
```

但 I/O 阻塞时会**释放 GIL**，所以线程池在 I/O 场景下仍然有效：

```
线程A: ▓▓ read ▓▓░░░░ 等磁盘 ░░░░▓▓ read ▓▓░░░░
                                           ↓ GIL 释放
线程B: ░░░░▓▓▓▓ 计算 ▓▓▓▓░░░░ 等磁盘 ░░░░
                ↑ 趁机抢到 GIL
```

## 四、multiprocessing — 进程并发

```python
# 适合：CPU 密集型、需要绕过 GIL
from multiprocessing import Pool

def cpu_intensive(n):
    """纯 Python 计算，不会释放 GIL"""
    total = 0
    for i in range(n):
        total += i * i
    return total

# 4 个进程 = 4 个独立 Python 解释器
# 没有 GIL 争抢，真正利用多核
with Pool(4) as pool:
    results = pool.map(cpu_intensive, [10**7, 10**7, 10**7, 10**7])
```

```
进程1:  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  CPU 核心 1
进程2:  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  CPU 核心 2
进程3:  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  CPU 核心 3
进程4:  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  CPU 核心 4
                              ↑ 真正同时运行
```

代价：进程间不共享内存，传递数据需要序列化（pickle），开销远大于线程切换。

## 五、实际项目怎么选

### 场景 1：Web 后端点播

```python
# FastAPI / Quart / Sanic → 框架强制 async
@app.post("/chat")
async def chat(request: Request):
    prompt = await request.json()
    response = await llm.generate(prompt)    # async
    await db.save(history)                   # async
    return {"reply": response}
```

→ **asyncio**。Web 框架本身就是异步的，整个调用链都是 async。

### 场景 2：批量处理文件

```python
# 省事写法，不用写 async/await
from concurrent.futures import ThreadPoolExecutor

def process_file(path):
    with open(path) as f:
        data = f.read()
    transformed = transform(data)
    with open(path + ".out", "w") as f:
        f.write(transformed)

with ThreadPoolExecutor(max_workers=8) as pool:
    pool.map(process_file, file_list)
```

→ **ThreadPoolExecutor**，简单直接。

### 场景 3：调用阻塞的第三方 SDK

```python
# asyncio 调阻塞代码的逃生舱
result = await asyncio.to_thread(sdk_blocking_call, arg1, arg2)
```

→ **asyncio.to_thread**，在线程池里跑阻塞代码，不阻塞事件循环。

### 场景 4：CPU 密集型计算

```python
# 多进程，真正利用多核
from multiprocessing import Pool

with Pool(os.cpu_count()) as pool:
    results = pool.map(heavy_computation, huge_dataset)
```

→ **multiprocessing.Pool**。

### 场景 5：机器学习推理服务

```python
# 推理本身在 GPU 上跑，Python 只做调度
async def predict(inputs):
    async with semaphore:            # 限流
        result = await asyncio.to_thread(
            model.predict, inputs    # GPU 计算，释放 GIL
        )
        return result
```

→ **asyncio**。主要时间花在等 GPU 返回，协程最适合。

## 六、选择流程图

```
你的任务
    │
    ├─ 主要在等网络/数据库/文件？ ──► asyncio
    │
    ├─ 调用不支持 async 的 SDK？  ──► asyncio.to_thread
    │
    ├─ 纯 CPU 计算？              ──► multiprocessing
    │
    └─ 简单并发任务不想写 async？ ──► ThreadPoolExecutor
```

## 七、你需要掌握到什么程度

### 必须掌握（日常每天都在用）

```python
# 1. async def / await
async def handler():
    data = await fetch_data()
    return process(data)

# 2. asyncio.gather
results = await asyncio.gather(t1(), t2(), t3())

# 3. asyncio.to_thread（调阻塞代码的逃生舱）
result = await asyncio.to_thread(blocking_func, arg)

# 4. asyncio.run
asyncio.run(main())
```

### 偶尔需要

```python
# 5. asyncio.wait_for（超时控制）
result = await asyncio.wait_for(fetch(), timeout=5)

# 6. asyncio.Semaphore（限流）
sem = asyncio.Semaphore(10)
async with sem:
    await fetch(url)

# 7. async for / async with（异步迭代器、上下文管理器）
async for chunk in stream:
    process(chunk)
```

**够用就行，不用全学。** 上面这些已经能覆盖 90% 的开发场景。
