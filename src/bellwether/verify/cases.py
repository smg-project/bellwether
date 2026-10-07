"""The render cases a verify run sends: each model's set files, and the cases in them read a piece at a time.

Anything that keeps verify from reading every case stops the run before the first request: passing over a set,
a line or a case would verify fewer cases than the fixtures hold and still pass.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Iterator
from pathlib import Path

import zstandard

from bellwether.manifest import Manifest
from bellwether.record import sets as set_tables
from bellwether.storage import (
    COMPRESSED_SUFFIX,
    LFS_POINTER_PREFIX,
    is_compressed,
    lfs_pull_command,
    repository_root,
)
from bellwether.unpack import set_files

from .render import CannotVerify

CHUNK = 1 << 16  # plain bytes read from a set file at a time, in either form
HEAD = max(len(LFS_POINTER_PREFIX), 18)  # enough for a Git LFS pointer's first line and any zstd frame header
SUFFIXES = {"plain": ".jsonl", "zstd": COMPRESSED_SUFFIX}  # sets.toml's form -> the set file's suffix


def render_sets(manifest: Manifest, names: list[str] | None = None) -> list[tuple[str, Path]]:
    """``(set, file)`` for each of the model's render sets, in either form, or for the named ones only.

    Every render set the model's ``sets.toml`` lists, of those selected, must be in the checkout, in the form it gives.
    """
    found = [(name, path) for kind, name, path in set_files(manifest) if kind == "render"]
    if names is not None:
        found = [(name, path) for name, path in found if name in names]
    listing = manifest.path.parent / set_tables.FILE
    try:
        tables = tomllib.loads(listing.read_text(encoding="utf-8")).get("render", {}) if listing.is_file() else {}
    except (OSError, ValueError) as err:
        raise CannotVerify(f"{listing}: {err}") from None
    if not isinstance(tables, dict):
        raise CannotVerify(f"{listing}: [render] is not a table of sets")
    for name, table in sorted(tables.items()):
        if names is not None and name not in names:
            continue
        suffix = SUFFIXES.get(table.get("form")) if isinstance(table, dict) else None
        if suffix is None:
            raise CannotVerify(f"{listing}: render.{name} has no form of plain or zstd")
        expected = manifest.path.parent / "render" / f"{name}{suffix}"
        if not expected.is_file():
            raise CannotVerify(f"{listing} lists the render set {name}, but {expected} is not there")
    return found


def read_cases(path: Path) -> Iterator[tuple[int, dict]]:
    """``(line number, case)`` for each case in a set file, plain or compressed, read a piece at a time.

    The file must be read whole: a set Git LFS has not fetched stops the run with the command that fetches it, as
    does one that is not zstd or ends inside a frame, and a line that is not JSON or not a case verify can send.
    """
    for number, raw in enumerate(_lines(path), start=1):
        if not raw.strip():
            continue
        try:
            case = json.loads(raw)
        except ValueError:
            raise CannotVerify(f"{path}:{number}: not a JSON line") from None
        problem = _problem(case)
        if problem is not None:
            raise CannotVerify(f"{path}:{number}: {problem}")
        yield number, case


def _problem(case: object) -> str | None:
    """What keeps verify from sending and judging a fixture line, or None when nothing does."""
    if not isinstance(case, dict) or not isinstance(case.get("id"), str):
        return "not a case with an id"
    if not isinstance(case.get("request"), dict):
        return f"{case['id']} has no request"
    reference = case.get("reference")
    ids = reference.get("input_ids") if isinstance(reference, dict) else None
    if not isinstance(ids, list) or not all(type(i) is int for i in ids):
        return f"{case['id']}: the reference has no list of integer input_ids"
    return None


def _lines(path: Path) -> Iterator[bytes]:
    """The file's plain lines; the last one may lack its newline."""
    pending = b""
    for chunk in _plain_chunks(path):
        pending += chunk
        *lines, pending = pending.split(b"\n")
        yield from lines
    if pending:
        yield pending


def _plain_chunks(path: Path) -> Iterator[bytes]:
    """The file's plain content, ``CHUNK`` bytes at a time, decompressing a ``.jsonl.zst`` set as it is read.

    zstd's stream reader takes a file cut short for a shorter one, so a set must be what ``record`` writes, one frame
    that declares its content size, and the plain bytes must come to that size. What one compressed byte expands to
    depends on the data, so the reads are bounded by what they return, not by what they consume.
    """
    try:
        with path.open("rb") as raw:
            head = raw.read(HEAD)
            if head.startswith(LFS_POINTER_PREFIX):
                command = lfs_pull_command([path], repository_root(path))
                raise CannotVerify(f"{path} is a Git LFS pointer; fetch it first: {command}")
            raw.seek(0)
            if not is_compressed(path):
                while chunk := raw.read(CHUNK):
                    yield chunk
                return
            declared = zstandard.frame_content_size(head)
            if declared < 0:
                raise CannotVerify(f"{path}: its zstd frame does not declare its content size, as record's always do")
            reader = zstandard.ZstdDecompressor().stream_reader(raw)  # one frame
            plain = 0
            while chunk := reader.read(CHUNK):
                plain += len(chunk)
                yield chunk
            if plain != declared:
                raise CannotVerify(f"{path} is cut short: {plain} of the {declared} plain bytes its frame declares")
    except zstandard.ZstdError as err:
        raise CannotVerify(f"{path} is not a zstd stream verify can read: {err}") from None
    except OSError as err:
        raise CannotVerify(f"{path}: {err}") from None
