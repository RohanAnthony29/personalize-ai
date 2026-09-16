from __future__ import annotations

import json
from typing import Any, Protocol


class FeatureStore(Protocol):
    def active_version(self) -> str: ...
    def user(self, user_id: int) -> dict[str, Any] | None: ...
    def items(self, item_ids: list[int]) -> dict[int, dict[str, Any]]: ...
    def popular_items(self, limit: int) -> list[int]: ...


class RedisFeatureStore:
    def __init__(self, redis_url: str) -> None:
        import redis
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)

    def active_version(self) -> str:
        value = self.client.get("features:active_version")
        if not value:
            raise RuntimeError("No active feature version has been published")
        return value

    def user(self, user_id: int) -> dict[str, Any] | None:
        value = self.client.get(f"features:{self.active_version()}:user:{user_id}")
        return json.loads(value) if value else None

    def items(self, item_ids: list[int]) -> dict[int, dict[str, Any]]:
        version = self.active_version()
        values = self.client.mget([f"features:{version}:item:{item}" for item in item_ids])
        return {
            item: json.loads(value)
            for item, value in zip(item_ids, values)
            if value is not None
        }

    def popular_items(self, limit: int) -> list[int]:
        return [
            int(item) for item in self.client.zrevrange(
                f"features:{self.active_version()}:item_popularity", 0, limit - 1
            )
        ]


class DictionaryFeatureStore:
    """Small deterministic store used by tests and local smoke checks."""
    def __init__(self, version: str, users: dict[int, dict], items: dict[int, dict]) -> None:
        self.version, self.users, self.item_values = version, users, items

    def active_version(self) -> str:
        return self.version

    def user(self, user_id: int) -> dict | None:
        return self.users.get(user_id)

    def items(self, item_ids: list[int]) -> dict[int, dict]:
        return {item: self.item_values[item] for item in item_ids if item in self.item_values}

    def popular_items(self, limit: int) -> list[int]:
        return [
            item for item, _ in sorted(
                self.item_values.items(),
                key=lambda value: (-float(value[1].get("weighted_popularity", 0)), value[0]),
            )[:limit]
        ]
