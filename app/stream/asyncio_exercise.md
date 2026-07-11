# asyncio 练习题

> 用你学到的 async/await、gather、to_thread 知识完成下面三个练习。

---

## 练习一：理解 await 的阻塞感

补全下面的代码，让 `task1` 和 `task2` **串行**执行，总耗时约 3 秒。

```python
import asyncio
import time


async def task1():
    print(f"[{time.strftime('%H:%M:%S')}] task1 开始")
    await asyncio.sleep(2)
    print(f"[{time.strftime('%H:%M:%S')}] task1 结束")


async def task2():
    print(f"[{time.strftime('%H:%M:%S')}] task2 开始")
    await asyncio.sleep(1)
    print(f"[{time.strftime('%H:%M:%S')}] task2 结束")


async def main():
    # TODO: 用 await 让 task1 和 task2 串行执行，总耗时约 3 秒
    await task1()
    await task2()


asyncio.run(main())
```

**预期输出**（时间仅供参考）：

```
[10:00:00] task1 开始
[10:00:02] task1 结束
[10:00:02] task2 开始
[10:00:03] task2 结束
```

---

## 练习二：理解 gather 并发

用 `asyncio.gather` 改写练习一，让两个 task **并发**执行，总耗时约 2 秒。

```python
import asyncio
import time


async def task1():
    print(f"[{time.strftime('%H:%M:%S')}] task1 开始")
    await asyncio.sleep(2)
    print(f"[{time.strftime('%H:%M:%S')}] task1 结束")
    return "task1 结果"


async def task2():
    print(f"[{time.strftime('%H:%M:%S')}] task2 开始")
    await asyncio.sleep(1)
    print(f"[{time.strftime('%H:%M:%S')}] task2 结束")
    return "task2 结果"


async def main():
    # TODO: 用 asyncio.gather 并发执行 task1 和 task2
    result = await asyncio.gather(task1(), task2())
    # TODO: 打印 gather 返回的结果
    print(result)


asyncio.run(main())
```

**预期输出**（task2 先结束，task1 后结束，总耗时 ~2 秒）：

```
[10:00:00] task1 开始
[10:00:00] task2 开始
[10:00:01] task2 结束
[10:00:02] task1 结束
返回结果: ['task1 结果', 'task2 结果']
```

---

## 练习三：用 to_thread 解决阻塞问题

有一个**不支持 async 的同步函数** `blocking_query`，每次调用需要 2 秒。

```python
def blocking_query(query: str) -> str:
    """模拟一个阻塞的 SDK 调用（比如某个不支持 async 的数据库驱动）"""
    time.sleep(2)
    return f"查询结果: {query}"


async def main():
    queries = ["天气", "新闻", "股票"]
    # TODO: 用 asyncio.to_thread 把 blocking_query 变成异步的
    tasks = [
         asyncio.to_thread(blocking_query, query)
         for query in queries
    ]
    # TODO: 用 asyncio.gather 并发查询 3 次
    result =  await asyncio.gather(*tasks)
    # TODO: 打印每次查询的结果
    print(result)

asyncio.run(main())
```

**预期输出**（总耗时 ~2 秒，而不是 6 秒）：

```
查询结果: 天气
查询结果: 新闻
查询结果: 股票
总耗时约 2 秒
```

---

## 提交方式

1. 在 `D:\langchain-demo\app\stream\` 下新建 `exercise.py`
2. 把三个练习的完整代码写进去
3. 运行确认输出符合预期
4. 让我检查
