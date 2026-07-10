"""Shared file I/O helpers for the SAE pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Read the small YAML subset used by SAE config files."""
    root: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not raw_line.startswith(" "):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                root[key] = _parse_scalar(value)
                current_section = None
            else:
                current_section = {}
                root[key] = current_section
        else:
            if current_section is None:
                raise ValueError(f"nested config value without a section: {raw_line}")
            key, value = line.split(":", 1)
            current_section[key.strip()] = _parse_scalar(value)
    return root
