"""Read registry YAML files and render new entries as appendable YAML text."""

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from discovery.models import RegistryEntry, RejectedEntry

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = REPO_ROOT / "registry"
T = TypeVar("T", bound=BaseModel)


def _load_list(path: Path, model: type[T]) -> list[T]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a YAML list, got {type(data).__name__}")
    return [model.model_validate(item) for item in data]


def load_models(path: Path) -> list[RegistryEntry]:
    return _load_list(path, RegistryEntry)


def load_rejected(path: Path) -> list[RejectedEntry]:
    return _load_list(path, RejectedEntry)


def known_keys(models: list[RegistryEntry], rejected: list[RejectedEntry]) -> frozenset[str]:
    keys: set[str] = set()
    for m in models:
        keys |= m.keys()
    keys |= {r.key for r in rejected}
    return frozenset(keys)


def render_yaml_block(items: list[dict]) -> str:
    """YAML list items to append to an existing list file (leading newline, no document markers)."""
    text = yaml.safe_dump(items, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return "\n" + text
