from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "base":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return payload


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = _load_yaml(config_path)
    base_entry = payload.get("base")
    if not base_entry:
        return payload
    base_path = (config_path.parent / base_entry).resolve()
    base_payload = load_config(base_path)
    return _merge_dicts(base_payload, payload)
