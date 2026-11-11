"""YAML config loading with `extends` inheritance and dotted overrides."""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """Load `path`, resolving `extends:` relative to the file, then apply `a.b=c` overrides."""
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = os.path.join(os.path.dirname(os.path.abspath(path)), parent)
        cfg = _deep_merge(load_config(parent_path), cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        node = cfg
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise TypeError(f"cannot set {dotted!r}: {key!r} is not a mapping")
        node[keys[-1]] = _parse_scalar(raw)
    return cfg


def flatten(cfg: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested config to dotted keys, for logging as run parameters."""
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, name + "."))
        else:
            flat[name] = value
    return flat
