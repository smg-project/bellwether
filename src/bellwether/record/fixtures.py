"""Fixture files: JSON Lines, one case per line, sorted by id, canonical, schema-checked on write.

A set is plain (``<set>.jsonl``) or zstd-compressed (``<set>.jsonl.zst``, for benchmark sets); both hold the same
lines, and readers take either.

A line is canonical in what bellwether controls: the top-level keys come in a fixed order and the
reference and witnesses carry sorted keys. The request is written exactly as the corpus gave it,
because key order inside it is part of what the oracle rendered: a template that serialises tool
definitions with ``tojson`` writes the keys in the order it received them, so a re-sorted request
would no longer be the request the reference answers.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from functools import cache
from importlib import resources
from pathlib import Path

import zstandard
from jsonschema import Draft202012Validator

from bellwether import jsonl

COMPRESSED_SUFFIX = ".jsonl.zst"
# Level 19, one thread: the compressed bytes are a function of the content for a given zstandard version, which
# uv.lock pins. sets.toml records the plain content's sha256, so nothing depends on them.
ZSTD_LEVEL = 19


def schema_path() -> Path:
    """``case.schema.json``, packaged with the module so an installed wheel finds it too."""
    return Path(str(resources.files("bellwether") / "schemas" / "case.schema.json"))


@cache
def validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(schema_path().read_text()))


TOP_LEVEL_ORDER = ("id", "kind", "model", "request", "reference", "witnesses")


def canonical_line(case: dict) -> str:
    ordered = {key: case[key] for key in TOP_LEVEL_ORDER if key in case}
    ordered.update((key, case[key]) for key in sorted(case) if key not in ordered)
    for key in ("reference", "witnesses"):
        if key in ordered:
            ordered[key] = _with_sorted_keys(ordered[key])
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def _with_sorted_keys(value):
    if isinstance(value, dict):
        return {key: _with_sorted_keys(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_with_sorted_keys(item) for item in value]
    return value


def is_compressed(path: Path) -> bool:
    return path.name.endswith(COMPRESSED_SUFFIX)


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: Path) -> bool:
    """A file Git LFS has not fetched: its pointer stands where the content would be."""
    with path.open("rb") as handle:
        return handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def repository_root(path: Path) -> Path | None:
    """The root of the git checkout that holds the file at ``path``; None outside a checkout, or without git."""
    try:
        found = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path.parent, capture_output=True, text=True)
    except OSError:
        return None
    return Path(found.stdout.strip()) if found.returncode == 0 else None


def lfs_include(paths: Sequence[Path], root: Path | None) -> str:
    """The sets' paths from the repository root, comma-joined: what ``git lfs pull --include`` matches, wherever in
    the checkout it runs. Outside a checkout, the paths as given."""
    if root is None:
        return ",".join(str(path) for path in paths)
    return ",".join(path.resolve().relative_to(root.resolve()).as_posix() for path in paths)


def lfs_pull_command(paths: Sequence[Path], root: Path | None) -> str:
    """The command that fetches these sets. ``--exclude ''`` clears ``.lfsconfig``'s ``fetchexclude``, which names
    every benchmark set and would otherwise win over ``--include``, so that the pull fetched nothing."""
    return f"git lfs pull --include '{lfs_include(paths, root)}' --exclude ''"


def plain_text(path: Path) -> str:
    """A fixture file's lines as text, whichever form it is stored in."""
    data = path.read_bytes()
    if data.startswith(LFS_POINTER_PREFIX):
        command = lfs_pull_command([path], repository_root(path))
        raise ValueError(f"{path} is a Git LFS pointer; fetch it first: {command}")
    if is_compressed(path):
        data = zstandard.ZstdDecompressor().decompress(data)
    return data.decode("utf-8")


def read_fixture_file(path: Path) -> dict[str, dict]:
    cases: dict[str, dict] = {}
    for number, case in jsonl.loads(plain_text(path), path):
        if case["id"] in cases:
            raise ValueError(f"{path}:{number}: duplicate id {case['id']}")
        cases[case["id"]] = case
    return cases


def write_fixture_file(path: Path, cases: dict[str, dict]) -> None:
    lines = []
    for case_id in sorted(cases):
        case = cases[case_id]
        errors = sorted(validator().iter_errors(case), key=lambda e: list(e.path))
        if errors:
            where = "/".join(str(p) for p in errors[0].path) or "(root)"
            raise ValueError(f"{case_id}: does not match the case schema at {where}: {errors[0].message}")
        lines.append(canonical_line(case))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(f"{line}\n" for line in lines).encode("utf-8")
    if is_compressed(path):
        if path.is_file() and not is_lfs_pointer(path) and plain_text(path).encode("utf-8") == data:
            return  # the same content keeps its bytes, so a compressor upgrade makes no new LFS object
        data = zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(data)
    path.write_bytes(data)
