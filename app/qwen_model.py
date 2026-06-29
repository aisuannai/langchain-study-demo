from app.config import OPENAI_API_KEY, OPENAI_BASE_URL
from langchain_qwq import ChatQwen
from langchain_core.rate_limiters import InMemoryRateLimiter

#InMemoryRateLimiter是langchian自带的限流器，是线程安全的，因为放在内存，只使用与单机版，多个Agetn Server需要自己实现
myRateLimiter =  InMemoryRateLimiter(
    requests_per_second=1, #每秒1个请求
    check_every_n_seconds=0.1, #如果发生限流，每0.1秒检查一次是否产生新的请求
    max_bucket_size= 10 #最大同时支持10个请求
)

model = ChatQwen(
    model='qwen3.7-plus',
    api_key= OPENAI_API_KEY,
    base_url= OPENAI_BASE_URL,
    rate_limiter=myRateLimiter
)