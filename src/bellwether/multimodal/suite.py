"""The multimodal suite's files under ``multimodal/``: the manifest of images, the tiers, the processor families, the
reference sets and the Git LFS pointers that stand for the images in a clone.

The suite is the image side of bellwether: does SMG's media pipeline turn an image into the same pixels the serving
engine's own image path gives, bit for bit? Its inputs (the images, by dataset) and its results (one reference set per
processor family) live together, because a reference belongs to a processor family, which many checkpoints share, and
not to one checkpoint as the text fixtures do. ``multimodal/README.md`` describes the layout; this module reads it.
"""

from __future__ import annotations

import csv
import hashlib
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path("multimodal")
MANIFEST = "manifest.tsv"
TIERS = ("corpus-50", "corpus-500", "all")
# The reference sources, in authority order: the engine's own image path, recorded inside the engine's image, and the
# cross-check, transformers' processor behind the same loader semantics.
SOURCES = ("vllm", "hf")
SETS_FILE = "sets.toml"
# The columns a manifest row must have; the rest of the catalogue travels with it.
COLUMNS = ("id", "path", "source", "class", "w", "h", "bytes", "sha256")
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_POINTER = re.compile(rb"\Aversion https://git-lfs\.github\.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n\Z")
_FAMILY_SLUG = re.compile(r"[a-z0-9.-]+")


@dataclass(frozen=True)
class Image:
    """One row of the manifest: an image by its path from the suite root, with what the catalogue says about it."""

    id: str
    path: str
    source: str
    cls: str
    w: int
    h: int
    bytes: int
    sha256: str
    row: dict[str, str] = field(compare=False, hash=False)

    @property
    def name(self) -> str:
        """The file's name, which every recorder and dump keys its lines by."""
        return self.path.rsplit("/", 1)[-1]

    @property
    def short_side(self) -> int:
        return min(self.w, self.h)


@dataclass(frozen=True)
class Family:
    """A processor family: the checkpoints whose image preprocessing inputs are equal, recorded once under one slug."""

    slug: str
    path: Path
    smg_processor: str
    model: str
    revision: str
    checkpoints: tuple[str, ...]
    engine: dict[str, str]
    inputs: dict[str, str]
    trust_remote_code: bool = False

    @property
    def config(self) -> Path:
        """The preprocessor config SMG's processor reads."""
        return self.path / "preprocessor_config.json"


def read_manifest(root: Path) -> list[Image]:
    """Every image of the manifest, in file order. A row missing a column, or two rows with one id or one path, stop
    the read: the manifest is the suite's index and every command keys by it."""
    path = root / MANIFEST
    images: list[Image] = []
    ids: set[str] = set()
    paths: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [column for column in COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: columns missing: {', '.join(missing)}")
        for number, row in enumerate(reader, 2):
            if row["id"] in ids:
                raise ValueError(f"{path}:{number}: id {row['id']} is already listed")
            if row["path"] in paths:
                raise ValueError(f"{path}:{number}: path {row['path']} is already listed")
            ids.add(row["id"])
            paths.add(row["path"])
            images.append(
                Image(
                    id=row["id"],
                    path=row["path"],
                    source=row["source"],
                    cls=row["class"],
                    w=int(row["w"]),
                    h=int(row["h"]),
                    bytes=int(row["bytes"]),
                    sha256=row["sha256"],
                    row=row,
                )
            )
    return images


def by_name(images: list[Image]) -> dict[str, Image]:
    """Images keyed by file name. A name in two datasets stops the read: dumps and recorders key by it."""
    found: dict[str, Image] = {}
    for image in images:
        if image.name in found:
            raise ValueError(f"two images are named {image.name}: {found[image.name].path} and {image.path}")
        found[image.name] = image
    return found


def read_tier(root: Path, tier: str) -> list[str]:
    """The image paths a tier names, one a line under ``tiers/``, in the tier's order."""
    path = root / "tiers" / f"{tier}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"no tier {tier!r}: {path} is not a file")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tier_images(root: Path, tier: str, images: list[Image] | None = None) -> list[Image]:
    """The tier's images, in the tier's order; a path the manifest does not list stops the read."""
    images = read_manifest(root) if images is None else images
    by_path = {image.path: image for image in images}
    picked = []
    for line in read_tier(root, tier):
        if line not in by_path:
            raise ValueError(f"tiers/{tier}.txt names {line}, which the manifest does not list")
        picked.append(by_path[line])
    return picked


def read_families(root: Path) -> dict[str, Family]:
    """Every family under ``families/``, by slug: ``family.toml`` names the SMG processor, the hub checkpoints the
    inputs were taken from with their revision, and each input with its sha256."""
    found: dict[str, Family] = {}
    base = root / "families"
    if not base.is_dir():
        return found
    for directory in sorted(p for p in base.iterdir() if p.is_dir()):
        found[directory.name] = read_family(directory)
    return found


def read_family(directory: Path) -> Family:
    path = directory / "family.toml"
    if not _FAMILY_SLUG.fullmatch(directory.name):
        raise ValueError(f"{directory}: a family's directory is a lowercase slug of letters, digits, dots and dashes")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    for key in ("smg_processor", "model", "revision", "inputs"):
        if key not in data:
            raise ValueError(f"{path}: `{key}` is required")
    inputs = data["inputs"]
    if not isinstance(inputs, dict) or not all(isinstance(v, str) and len(v) == 64 for v in inputs.values()):
        raise ValueError(f"{path}: [inputs] maps each file to its sha256")
    return Family(
        slug=directory.name,
        path=directory,
        smg_processor=str(data["smg_processor"]),
        model=str(data["model"]),
        revision=str(data["revision"]),
        checkpoints=tuple(str(c) for c in data.get("checkpoints", [])),
        engine={k: str(v) for k, v in data.get("engine", {}).items()},
        inputs={str(k): str(v) for k, v in inputs.items()},
        trust_remote_code=bool(data.get("trust_remote_code", False)),
    )


def pointer(path: Path) -> tuple[str, int] | None:
    """``(sha256, size)`` of the content a Git LFS pointer at ``path`` stands for; None when the file is content."""
    with path.open("rb") as handle:
        head = handle.read(len(LFS_POINTER_PREFIX))
        if head != LFS_POINTER_PREFIX:
            return None
        rest = handle.read(512)
    match = _POINTER.match(head + rest)
    if match is None:
        raise ValueError(f"{path} begins like a Git LFS pointer but is not one")
    return match.group(1).decode(), int(match.group(2))


def identity(path: Path) -> tuple[str, int]:
    """The sha256 and size of the content at ``path``: read from its pointer when Git LFS has not fetched it, so a
    clone that holds pointers alone is checked against the manifest without a download."""
    found = pointer(path)
    if found is not None:
        return found
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1 << 20), b""):
            digest.update(piece)
            size += len(piece)
    return digest.hexdigest(), size


def read_sets(path: Path) -> dict[tuple[str, str], dict]:
    """``(source, family) -> table`` from a ``references/<family>/sets.toml``; empty when there is none."""
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {(source, family): table for source, families in data.items() for family, table in families.items()}


def write_sets(path: Path, tables: dict[tuple[str, str], dict]) -> None:
    """Write the tables sorted, each field on its line, through a partial file renamed into place."""
    lines: list[str] = []
    for source, family in sorted(tables):
        lines.append(f"[{source}.{family}]")
        for key, value in tables[(source, family)].items():
            lines.append(f"{key} = {_toml(value)}")
        lines.append("")
    partial = path.with_name(path.name + ".partial")
    partial.write_text("\n".join(lines), encoding="utf-8")
    partial.replace(path)


def _toml(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {_toml(v)}" for k, v in value.items()) + " }"
    raise TypeError(f"cannot write {type(value).__name__} to sets.toml")
