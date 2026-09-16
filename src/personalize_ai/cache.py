from __future__ import annotations

import json
import time
from threading import Lock
from typing import Any, Protocol


class Cache(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl_seconds: int) -> None: ...


class MemoryTTLCache:
    def __init__(self) -> None:
        self._values: dict[str, tuple[float, Any]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            expires, value = self._values.get(key, (0.0, None))
            if expires <= time.monotonic():
                self._values.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        with self._lock:
            self._values[key] = (time.monotonic() + ttl_seconds, value)


class RedisCache:
    def __init__(self, redis_url: str) -> None:
        import redis
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)

    def get(self, key: str) -> Any | None:
        value = self.client.get(key)
        return json.loads(value) if value is not None else None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        self.client.setex(key, ttl_seconds, json.dumps(value))
