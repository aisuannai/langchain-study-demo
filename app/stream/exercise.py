import asyncio
import time

# ============================================================
# 共享函数
# ============================================================

async def task1():
    """sleep 2 秒后返回结果"""
    print(f"[{time.strftime('%H:%M:%S')}] task1 开始")
    await asyncio.sleep(2)
    print(f"[{time.strftime('%H:%M:%S')}] task1 结束")
    return "task1 结果"


async def task2():
    """sleep 1 秒后返回结果"""
    print(f"[{time.strftime('%H:%M:%S')}] task2 开始")
    await asyncio.sleep(1)
    print(f"[{time.strftime('%H:%M:%S')}] task2 结束")
    return "task2 结果"


def blocking_query(query: str) -> str:
    """模拟不支持 async 的阻塞 SDK 调用"""
    time.sleep(2)
    return f"查询结果: {query}"


# ============================================================
# 练习一：理解 await 的阻塞感（串行）
# ============================================================

async def exercise1():
    print("=== 练习一：串行 await ===")
    start = time.time()

    await task1()
    await task2()

    print(f"总耗时: {time.time() - start:.2f}s\n")


# ============================================================
# 练习二：理解 gather 并发
# ============================================================

async def exercise2():
    print("=== 练习二：gather 并发 ===")
    start = time.time()
    #表示等待 asyncio.gather这个协程，这个协程里面并发执行task1(),task2()
    results = await asyncio.gather(task1(), task2())
    print(f"返回结果: {results}")

    print(f"总耗时: {time.time() - start:.2f}s\n")


# ============================================================
# 练习三：to_thread 解决阻塞问题
# ============================================================

async def exercise3():
    print("=== 练习三：to_thread ===")
    queries = ["天气", "新闻", "股票"]
    start = time.time()

    # 把同步函数丢到线程池执行，避免阻塞事件循环
    tasks = [asyncio.to_thread(blocking_query, q) for q in queries]
    #等待并发执行
    results = await asyncio.gather(*tasks)

    for res in results:
        print(res)
    print(f"总耗时: {time.time() - start:.2f}s\n")


# ============================================================
# 运行全部练习
# ============================================================

async def main():
    await exercise1()
    await exercise2()
    await exercise3()


asyncio.run(main())
