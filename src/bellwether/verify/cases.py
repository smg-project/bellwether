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
from bellwether.record.fixtures import (
    COMPRESSED_SUFFIX,
    LFS_POINTER_PREFIX,
    is_compressed,
    lfs_pull_command,
    repository_root,
)
from bellwether.unpack import set_files

from .render import CannotVerify

CHUNK = 1 << 20  # bytes read from a set file at a time
SUFFIXES = {"plain": ".jsonl", "zstd": COMPRESSED_SUFFIX}  # sets.toml's form -> the set file's suffix


def render_sets(manifest: Manifest) -> list[tuple[str, Path]]:
    """``(set, file)`` for each of the model's render sets, in either form.

    Every render set the model's ``sets.toml`` lists must be in the checkout, in the form it gives.
    """
    found = [(name, path) for kind, name, path in set_files(manifest) if kind == "render"]
    listing = manifest.path.parent / set_tables.FILE
    try:
        tables = tomllib.loads(listing.read_text(encoding="utf-8")).get("render", {}) if listing.is_file() else {}
    except (OSError, ValueError) as err:
        raise CannotVerify(f"{listing}: {err}") from None
    if not isinstance(tables, dict):
        raise CannotVerify(f"{listing}: [render] is not a table of sets")
    for name, table in sorted(tables.items()):
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
    """The file's plain content, decompressing a ``.jsonl.zst`` set as it is read.

    zstd's stream reader takes a file cut short for a shorter one, so each frame is decompressed on its own and must
    reach its end.
    """
    try:
        with path.open("rb") as raw:
            if raw.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
                command = lfs_pull_command([path], repository_root(path))
                raise CannotVerify(f"{path} is a Git LFS pointer; fetch it first: {command}")
            raw.seek(0)
            if not is_compressed(path):
                while chunk := raw.read(CHUNK):
                    yield chunk
                return
            frame = zstandard.ZstdDecompressor().decompressobj()
            while chunk := raw.read(CHUNK):
                while chunk:
                    if frame.eof:  # the bytes after a frame's end start the next frame
                        frame = zstandard.ZstdDecompressor().decompressobj()
                    yield frame.decompress(chunk)
                    chunk = frame.unused_data if frame.eof else b""
            if not frame.eof:
                raise CannotVerify(f"{path} ends inside a zstd frame: the set is cut short")
    except zstandard.ZstdError as err:
        raise CannotVerify(f"{path} is not a zstd stream verify can read: {err}") from None
    except OSError as err:
        raise CannotVerify(f"{path}: {err}") from None
