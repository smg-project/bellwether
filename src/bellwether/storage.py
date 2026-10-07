"""Set files in their two storage forms: plain JSON Lines, and JSON Lines compressed with zstd and kept in Git LFS.

A set is plain (``<set>.jsonl``) or compressed (``<set>.jsonl.zst``); both hold the same lines, and every reader takes
either. A file Git LFS has not fetched holds its pointer instead, and reading it names the command that fetches it. A
set is written whole or not at all (``write_whole``), as the importers' pinned files are. ``write`` and ``holds`` take a
set's content whole, or in ``Pieces`` read as they are asked for, so that a large set is written and compared without a
copy of it whole.

It is a top-level module, as ``jsonl`` is, so that both the recorder and the importers can import it: importing anything
under ``bellwether.record`` runs its ``__init__``, which loads the oracles and their third-party dependencies, and the
importers import nothing beyond the standard library. Nor does this module until a compressed set is read or written.
"""

from __future__ import annotations

import io
import os
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol, TextIO

COMPRESSED_SUFFIX = ".jsonl.zst"
# Level 19, one thread: the compressed bytes are a function of the content for a given zstandard version, which
# uv.lock pins. sets.toml and the import checks compare plain content, so nothing depends on them.
ZSTD_LEVEL = 19


# How much of a set file holds reads and decompresses at a time.
READ_SIZE = 1 << 20


class Pieces(Protocol):
    """A set's plain content in pieces, read again as often as it is asked for, and its size in bytes: what ``write``
    and ``holds`` take in place of the content, so that no copy of it is made whole."""

    def __iter__(self) -> Iterator[bytes]: ...

    def __len__(self) -> int: ...


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


def _pointer_refused(path: Path) -> ValueError:
    return ValueError(f"{path} is a Git LFS pointer; fetch it first: {lfs_pull_command([path], repository_root(path))}")


def plain_bytes(path: Path) -> bytes:
    """A set file's content as plain JSON Lines, whichever form it is stored in.

    A Git LFS pointer is refused with the command that fetches it, and a compressed file zstd cannot read, such as one a
    write left cut short, by its path.
    """
    data = path.read_bytes()
    if data.startswith(LFS_POINTER_PREFIX):
        raise _pointer_refused(path)
    if is_compressed(path):
        import zstandard

        try:
            data = zstandard.ZstdDecompressor().decompress(data)
        except zstandard.ZstdError as err:
            raise ValueError(f"{path} cannot be decompressed: {err}") from err
    return data


def _plain_pieces(path: Path) -> Iterator[bytes]:
    """A set file's plain content, a piece at a time, whichever form it is stored in, refused as ``plain_bytes`` refuses
    it: a Git LFS pointer, and a frame zstd cannot decompress or that ends before its last block, as one a write left
    cut short does."""
    if is_lfs_pointer(path):
        raise _pointer_refused(path)
    with path.open("rb") as handle:
        if not is_compressed(path):
            while piece := handle.read(READ_SIZE):
                yield piece
            return
        import zstandard

        frame = zstandard.ZstdDecompressor().decompressobj()
        try:
            while piece := handle.read(READ_SIZE):
                if plain := frame.decompress(piece):
                    yield plain
            if not frame.eof:
                raise zstandard.ZstdError("decompression error: did not decompress full frame")
        except zstandard.ZstdError as err:
            raise ValueError(f"{path} cannot be decompressed: {err}") from err


def holds(path: Path, data: bytes | Pieces) -> bool:
    """Whether the set file at ``path`` holds ``data`` as its plain content, whichever form it is stored in.

    The file is read and decompressed a piece at a time and compared as it comes, to its end even once it differs, so
    that a file zstd cannot read is always named, by ``plain_bytes``'s reason, as is a Git LFS pointer.
    """
    stored = _plain_pieces(path)
    same, buffer = True, memoryview(b"")
    for piece in [data] if isinstance(data, bytes | bytearray | memoryview) else data:
        piece = memoryview(piece)
        while piece:
            if not buffer:
                buffer = memoryview(next(stored, b""))
                if not buffer:
                    return False
            taken = min(len(buffer), len(piece))
            same = same and buffer[:taken] == piece[:taken]
            buffer, piece = buffer[taken:], piece[taken:]
    rest = len(buffer) + sum(len(piece) for piece in stored)
    return same and rest == 0


def plain_text(path: Path) -> str:
    """A set file's lines as text, whichever form it is stored in."""
    return plain_bytes(path).decode("utf-8")


def open_text(path: Path) -> TextIO:
    """A set file's lines as a text stream, whichever form it is stored in: read and decompressed as they are asked
    for, so a reader that needs the first lines does not load the set whole.

    Only "\\n" ends a line (``newline="\\n"``), as in ``jsonl``: U+2028 and its kind stay inside the string that holds
    them. A Git LFS pointer is refused as ``plain_bytes`` refuses it.
    """
    if is_lfs_pointer(path):
        raise _pointer_refused(path)
    if not is_compressed(path):
        return path.open(encoding="utf-8", newline="\n")
    import zstandard

    return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(path.open("rb")), encoding="utf-8", newline="\n")


def write(path: Path, data: bytes | Pieces) -> None:
    """Write a set's plain content to ``path`` in the form the name says: as it is, or compressed for ``.jsonl.zst``.

    The file is replaced whole (``write_whole``), so a write cut short leaves the old one. A compressed file that
    already holds ``data`` keeps its bytes, so a compressor upgrade makes no new LFS object; one that cannot be read, a
    Git LFS pointer or a frame zstd cannot decompress, is replaced. ``data`` in ``Pieces`` is compared, compressed and
    written a piece at a time. Whole or in pieces, it is compressed by zstd's stream writer, told the size first, so the
    bytes are a function of the content however it is given: zstd's one-shot compress gives content past its window
    (8 MiB at level 19) other bytes.
    """
    whole = isinstance(data, bytes | bytearray | memoryview)
    if is_compressed(path):
        try:
            if path.is_file() and (plain_bytes(path) == data if whole else holds(path, data)):
                return
        except ValueError:
            pass  # a Git LFS pointer, or a frame zstd cannot decompress: replaced below
        import zstandard

        if not whole:
            _write_pieces(path, data, zstandard.ZstdCompressor(level=ZSTD_LEVEL))
            return
        compressed = io.BytesIO()
        with zstandard.ZstdCompressor(level=ZSTD_LEVEL).stream_writer(compressed, size=len(data), closefd=False) as w:
            w.write(data)
        data = compressed.getvalue()
    if whole:
        write_whole(path, data)
    else:
        _write_pieces(path, data, None)


def _write_pieces(path: Path, data: Pieces, compressor) -> None:
    """``write_whole`` for content in pieces: written to ``<name>.partial`` as they come, compressed when a compressor
    is given, which is told the size first so that the frame carries it, then renamed into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("wb") as handle:
            if compressor is None:
                for piece in data:
                    handle.write(piece)
            else:
                with compressor.stream_writer(handle, size=len(data), closefd=False) as writer:
                    for piece in data:
                        writer.write(piece)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


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
