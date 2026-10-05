"""Per-model manifests under ``fixtures/``, read for the SMG names they pin."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

_KEYS = (
    ("tool", "tool_parser"),
    ("reasoning", "reasoning_parser"),
    ("renderer", "renderer"),
    ("tokenizer", "tokenizer"),
)


@dataclass
class Manifest:
    slug: str
    model: str
    smg: dict[str, str]  # kind -> SMG name


def read_manifests(fixtures_dir: Path) -> list[Manifest]:
    manifests = []
    for path in sorted(fixtures_dir.glob("*/manifest.toml")):
        data = tomllib.loads(path.read_text())
        smg = data.get("smg", {})
        names = {kind: smg[key] for kind, key in _KEYS if isinstance(smg.get(key), str)}
        manifests.append(Manifest(path.parent.name, str(data.get("model", "")), names))
    return manifests
