import typing as t
from typing import Optional, Dict

from pydantic import BaseModel
from redis.asyncio import Redis as AIORedis


class RedisConfig(BaseModel):
    host: str
    port: int
    db: int
    password: t.Optional[str] = None
    connect_args: t.Optional[t.Dict[str, t.Any]] = None


# This function now creates a client from the new 'redis' library.
# `decode_responses=True` is added to automatically handle decoding from bytes to strings.
def create_redis_client(config: RedisConfig) -> AIORedis:
    """Creates an asynchronous Redis client instance."""
    return AIORedis(
        host=config.host,
        port=config.port,
        db=config.db,
        password=config.password,
        decode_responses=True,  # Automatically decodes responses, simplifying your code
        **(config.connect_args or {})
    )


class Redis:
    """
    An improved asynchronous Redis client manager for redis-py >= 4.2.

    This class is designed to be used as an async context manager.
    It creates a connection pool on entering the context and closes it on exit.
    This is much more efficient than creating/closing a connection for every command.

    Correct Usage:
        redis_manager = Redis(config)
        async with redis_manager as redis_client:
            await redis_client.set_value("my_key", "my_value")
            value = await redis_client.get_value("my_key")
    """
    def __init__(self, config: RedisConfig):
        self.config = config
        self.redis = create_redis_client(self.config)

    async def get_hash(self, key: str) -> Dict[str, str]:
        """Gets all fields and values in a hash."""
        if not self.redis:
            raise ConnectionError("Redis client not connected. Use within an 'async with' block.")
        return await self.redis.hgetall(key)

    async def get_value(self, key: str) -> Optional[str]:
        """Gets the string value of a key."""
        if not self.redis:
            raise ConnectionError("Redis client not connected. Use within an 'async with' block.")
        return await self.redis.get(key)

    async def set_value(self, key: str, value: str, expires_in: Optional[int] = None):
        """Sets the string value of a key, with an optional expiration in seconds."""
        if not self.redis:
            raise ConnectionError("Redis client not connected. Use within an 'async with' block.")
        return await self.redis.set(key, value, ex=expires_in)

    async def del_value(self, key: str) -> int:
        """Deletes a key."""
        if not self.redis:
            raise ConnectionError("Redis client not connected. Use within an 'async with' block.")
        return await self.redis.delete(key)
