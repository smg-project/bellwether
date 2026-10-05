"""Name normalization and the alias table that merges names across systems."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .registries import KINDS, SYSTEMS


def normalize(name: str) -> str:
    """Lowercase letters and digits only: ``minimax-m2``, ``minimax_m2`` and ``MiniMaxM2`` agree."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def default_aliases_path() -> Path:
    return Path(str(resources.files("bellwether.gaps") / "aliases.toml"))


@dataclass
class Aliases:
    """``(kind, system, name) -> canonical row id``; a name absent here is its own normalized form."""

    schema_version: int
    table: dict[tuple[str, str, str], str]
    path: str

    @classmethod
    def load(cls, path: Path) -> Aliases:
        data = tomllib.loads(path.read_text())
        version = data.get("schema_version")
        if version != 1:
            raise ValueError(f"{path}: schema_version {version!r}, expected 1")
        table: dict[tuple[str, str, str], str] = {}
        for kind, systems in data.items():
            if kind == "schema_version":
                continue
            if kind not in KINDS:
                raise ValueError(f"{path}: unknown kind {kind!r}")
            for system, names in systems.items():
                if system not in SYSTEMS:
                    raise ValueError(f"{path}: unknown system {kind}.{system}")
                for name, canonical in names.items():
                    if normalize(canonical) != canonical:
                        raise ValueError(f"{path}: {kind}.{system}.{name} maps to {canonical!r}, not a normalized id")
                    table[(kind, system, name)] = canonical
        return cls(version, table, str(path))

    def canonical(self, kind: str, system: str, name: str) -> str:
        return self.table.get((kind, system, name), normalize(name))
