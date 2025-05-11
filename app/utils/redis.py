import os
import redis.asyncio as redis
# from redis import Redis
from dotenv import load_dotenv
from typing import List
import json

load_dotenv()

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    username=os.getenv("REDIS_USERNAME"),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True,
)

async def cache_append_to_list(key: str, item: dict):
    await redis_client.rpush(key, json.dumps(item))

async def cache_get_list(key: str) -> List[dict]:
    items = await redis_client.lrange(key, 0, -1)
    return [json.loads(i) for i in items]

async def cache_set(key: str, value: str, ex: int = None):
    return await redis_client.set(key, value, ex=ex)

async def cache_get(key: str):
    return await redis_client.get(key)

async def cache_delete(key: str):
    return await redis_client.delete(key)

async def cache_hset(name: str, key: str, value: str):
    return await redis_client.hset(name, key, value)

async def cache_hget(name: str, key: str):
    return await redis_client.hget(name, key)

async def cache_hgetall(name: str):
    return await redis_client.hgetall(name)

async def cache_hdel(name: str, key: str):
    return await redis_client.hdel(name, key)

async def cache_clear():
    return await redis_client.flushdb()
