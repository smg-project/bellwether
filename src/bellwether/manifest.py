"""Per-model manifests: the pinned revision, the authority order and the parser names.

One ``fixtures/<slug>/manifest.toml`` per model. The slug is the lowercase last path segment of
the Hugging Face id with anything but letters, digits, dots and dashes turned into a dash.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

KINDS = ("render", "parse", "tokenize", "detokenize")


@dataclass
class Manifest:
    slug: str
    path: Path
    model: str
    revision: str
    authority: dict[str, list[str]]
    smg: dict[str, str]
    engines: dict[str, dict[str, str]]


def slug_for(model: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", model.split("/")[-1].lower()).strip("-")


def load_manifest(path: Path) -> Manifest:
    try:
        data = tomllib.loads(path.read_text())
    except ValueError as err:  # not TOML, or not text: the message does not say which file
        raise ValueError(f"{path}: {err}") from None
    for key in ("model", "revision"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"{path}: `{key}` is required")
    authority = data.get("authority", {})
    for kind, sources in authority.items():
        if kind not in KINDS:
            raise ValueError(f"{path}: unknown kind `{kind}` under [authority]")
        if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
            raise ValueError(f"{path}: authority.{kind} must be a list of sources")
    return Manifest(
        slug=path.parent.name,
        path=path,
        model=data["model"],
        revision=data["revision"],
        authority={k: list(v) for k, v in authority.items()},
        smg={k: v for k, v in data.get("smg", {}).items() if isinstance(v, str)},
        engines={e: dict(v) for e, v in data.get("engines", {}).items() if isinstance(v, dict)},
    )


def find_manifest(fixtures_dir: Path, model: str) -> Manifest:
    """The manifest whose ``model`` is ``model``; models are matched exactly, never by slug."""
    seen = []
    for path in sorted(fixtures_dir.glob("*/manifest.toml")):
        manifest = load_manifest(path)
        if manifest.model == model:
            return manifest
        seen.append(manifest.model)
    known = ", ".join(seen) or "none"
    raise FileNotFoundError(f"no manifest under {fixtures_dir} for model {model!r}; manifests exist for: {known}")
