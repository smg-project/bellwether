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
from functools import cache
from importlib import resources
from pathlib import Path

import zstandard
from jsonschema import Draft202012Validator

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


def plain_text(path: Path) -> str:
    """A fixture file's lines as text, whichever form it is stored in."""
    data = path.read_bytes()
    if is_compressed(path):
        data = zstandard.ZstdDecompressor().decompress(data)
    return data.decode("utf-8")


def read_fixture_file(path: Path) -> dict[str, dict]:
    cases: dict[str, dict] = {}
    for number, raw in enumerate(plain_text(path).splitlines(), start=1):
        if raw.strip():
            case = json.loads(raw)
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
        data = zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(data)
    path.write_bytes(data)
