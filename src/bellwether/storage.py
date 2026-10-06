"""Set files in their two storage forms: plain JSON Lines, and JSON Lines compressed with zstd and kept in Git LFS.

A set is plain (``<set>.jsonl``) or compressed (``<set>.jsonl.zst``); both hold the same lines, and every reader takes
either. A file Git LFS has not fetched holds its pointer instead, and reading it names the command that fetches it. A
set is written whole or not at all (``write_whole``), as the importers' pinned files are.

It is a top-level module, as ``jsonl`` is, so that both the recorder and the importers can import it: importing anything
under ``bellwether.record`` runs its ``__init__``, which loads the oracles and their third-party dependencies, and the
importers import nothing beyond the standard library. Nor does this module until a compressed set is read or written.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

COMPRESSED_SUFFIX = ".jsonl.zst"
# Level 19, one thread: the compressed bytes are a function of the content for a given zstandard version, which
# uv.lock pins. sets.toml and the import checks compare plain content, so nothing depends on them.
ZSTD_LEVEL = 19


def is_compressed(path: Path) -> bool:
    return path.name.endswith(COMPRESSED_SUFFIX)


def stem(path: Path) -> str | None:
    """The set's name: the file's name without ``.jsonl`` or ``.jsonl.zst``; None for a file that is not a set."""
    for suffix in (COMPRESSED_SUFFIX, ".jsonl"):
        if path.name.endswith(suffix):
            return path.name.removesuffix(suffix)
    return None


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
    every benchmark fixture set and would otherwise win over ``--include``, so that the pull fetched nothing."""
    return f"git lfs pull --include '{lfs_include(paths, root)}' --exclude ''"


def plain_bytes(path: Path) -> bytes:
    """A set file's content as plain JSON Lines, whichever form it is stored in.

    A Git LFS pointer is refused with the command that fetches it, and a compressed file zstd cannot read, such as one a
    write left cut short, by its path.
    """
    data = path.read_bytes()
    if data.startswith(LFS_POINTER_PREFIX):
        command = lfs_pull_command([path], repository_root(path))
        raise ValueError(f"{path} is a Git LFS pointer; fetch it first: {command}")
    if is_compressed(path):
        import zstandard

        try:
            data = zstandard.ZstdDecompressor().decompress(data)
        except zstandard.ZstdError as err:
            raise ValueError(f"{path} cannot be decompressed: {err}") from err
    return data


def plain_text(path: Path) -> str:
    """A set file's lines as text, whichever form it is stored in."""
    return plain_bytes(path).decode("utf-8")


def write(path: Path, data: bytes) -> None:
    """Write a set's plain content to ``path`` in the form the name says: as it is, or compressed for ``.jsonl.zst``.

    The file is replaced whole (``write_whole``), so a write cut short leaves the old one. A compressed file that
    already holds ``data`` keeps its bytes, so a compressor upgrade makes no new LFS object; one that cannot be read, a
    Git LFS pointer or a frame zstd cannot decompress, is replaced.
    """
    if is_compressed(path):
        try:
            if path.is_file() and plain_bytes(path) == data:
                return
        except ValueError:
            pass  # a Git LFS pointer, or a frame zstd cannot decompress: replaced below
        import zstandard

        data = zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(data)
    write_whole(path, data)


def write_whole(path: Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` through ``<name>.partial``, renamed into place once it is written whole.

    A write cut short leaves the file at ``path`` as it was, and no partial file beside it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        partial.write_bytes(data)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return path
