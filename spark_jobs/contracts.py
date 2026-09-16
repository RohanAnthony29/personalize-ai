"""Dependency-free contracts shared by Spark jobs and tests."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VALID_EVENTS = frozenset({"view", "addtocart", "transaction"})
EVENT_WEIGHTS = {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}


@dataclass(frozen=True)
class IncrementalState:
    watermark_ms: int = -1
    latest_version: str | None = None

    @classmethod
    def load(cls, path: Path) -> "IncrementalState":
        if not path.exists():
            return cls()
        value = json.loads(path.read_text())
        return cls(int(value["watermark_ms"]), value.get("latest_version"))

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.__dict__, indent=2) + "\n")
        temporary.replace(path)


def render_sql(path: Path, source_view: str, as_of_ms: int) -> str:
    sql = path.read_text()
    if "{{source_view}}" not in sql or "{{as_of_ms}}" not in sql:
        raise ValueError(f"SQL template lacks required placeholders: {path}")
    return sql.replace("{{source_view}}", source_view).replace(
        "{{as_of_ms}}", str(as_of_ms)
    )


def validate_manifest(manifest: dict) -> list[str]:
    failures = []
    for field in ("feature_version", "watermark_ms", "row_counts", "format"):
        if field not in manifest:
            failures.append(f"missing manifest field: {field}")
    for name, count in manifest.get("row_counts", {}).items():
        if not isinstance(count, int) or count < 0:
            failures.append(f"invalid row count for {name}: {count!r}")
    if manifest.get("incremental") and not manifest.get("parent_version"):
        failures.append("incremental snapshot must identify parent_version")
    return failures
