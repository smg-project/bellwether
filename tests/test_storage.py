import pathlib
import random
import re

import pytest

from bellwether import storage

LINES = b'{"name": "a", "request": {}}\n{"name": "b", "request": {}}\n'
MORE = LINES + b'{"name": "c", "request": {}}\n'


def write_half_then_fail(self, data):
    """``Path.write_bytes`` cut short: half the bytes reach the file, then the disk is full."""
    with self.open("wb") as handle:
        handle.write(data[: len(data) // 2])
    raise OSError(28, "No space left on device")


def cut_short(path: pathlib.Path) -> None:
    """Keep the first half of a file's bytes, as a write cut short leaves it."""
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])


@pytest.mark.parametrize("name", ["set.jsonl", "set.jsonl.zst"])
def test_a_write_cut_short_leaves_the_old_file_whole_and_no_partial_file(tmp_path, monkeypatch, name):
    path = tmp_path / name
    storage.write(path, LINES)
    before = path.read_bytes()
    monkeypatch.setattr(pathlib.Path, "write_bytes", write_half_then_fail)
    with pytest.raises(OSError, match="No space left on device"):
        storage.write(path, MORE)
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert [file.name for file in tmp_path.iterdir()] == [name]


def test_a_compressed_set_cut_short_is_replaced_by_the_next_write(tmp_path):
    path = tmp_path / "set.jsonl.zst"
    storage.write(path, LINES)
    cut_short(path)
    storage.write(path, LINES)
    assert storage.plain_bytes(path) == LINES


def test_a_compressed_set_zstd_cannot_read_is_named_by_its_path(tmp_path):
    path = tmp_path / "set.jsonl.zst"
    storage.write(path, LINES)
    cut_short(path)
    with pytest.raises(ValueError, match=re.escape(f"{path} cannot be decompressed: ")):
        storage.plain_bytes(path)


class Chunks:
    """Bytes in pieces, read as often as asked, with their size: what ``storage.write`` and ``storage.holds`` stream."""

    def __init__(self, data: bytes, piece: int = 7, fail_at: int | None = None):
        self.data, self.piece, self.fail_at = data, piece, fail_at

    def __iter__(self):
        for start in range(0, len(self.data), self.piece):
            if self.fail_at is not None and start >= self.fail_at:
                raise OSError(28, "No space left on device")
            yield self.data[start : start + self.piece]

    def __len__(self):
        return len(self.data)


@pytest.mark.parametrize("name", ["set.jsonl", "set.jsonl.zst"])
def test_a_set_written_in_pieces_holds_the_bytes_a_whole_write_gives(tmp_path, name):
    whole, pieces = tmp_path / "whole" / name, tmp_path / "pieces" / name
    storage.write(whole, MORE)
    storage.write(pieces, Chunks(MORE))
    assert pieces.read_bytes() == whole.read_bytes()
    assert storage.plain_bytes(pieces) == MORE


def past_the_window() -> bytes:
    """8.7 MB of set lines, past zstd's window at level 19 (8 MiB), each with one of a few long tool lists, as a large
    set's cases repeat their tools: content zstd's one-shot compress gives other bytes than its stream writer does."""
    rng = random.Random(0)
    words = [f"w{n}".encode() for n in range(5000)]
    tools = [b" ".join(rng.choices(words, k=100)) for _ in range(50)]
    return b"".join(
        b'{"name": "%d", "tools": "%s", "request": "%s"}\n'
        % (n, rng.choice(tools), b" ".join(rng.choices(words, k=30)))
        for n in range(11_000)
    )


def test_a_compressed_set_past_zstds_window_has_the_same_bytes_written_whole_or_in_pieces(tmp_path):
    data = past_the_window()
    assert len(data) > 8 << 20
    whole, pieces = tmp_path / "whole.jsonl.zst", tmp_path / "pieces.jsonl.zst"
    storage.write(whole, data)
    storage.write(pieces, Chunks(data, piece=1 << 16))
    assert pieces.read_bytes() == whole.read_bytes()


@pytest.mark.parametrize("name", ["set.jsonl", "set.jsonl.zst"])
def test_a_write_in_pieces_cut_short_leaves_the_old_file_whole_and_no_partial_file(tmp_path, name):
    path = tmp_path / name
    storage.write(path, LINES)
    before = path.read_bytes()
    with pytest.raises(OSError, match="No space left on device"):
        storage.write(path, Chunks(MORE, fail_at=len(MORE) // 2))
    assert path.read_bytes() == before
    assert [file.name for file in tmp_path.iterdir()] == [name]


def test_a_compressed_set_that_already_holds_the_pieces_keeps_its_bytes(tmp_path, monkeypatch):
    path = tmp_path / "set.jsonl.zst"
    storage.write(path, MORE)
    monkeypatch.setattr(storage, "write_whole", None)  # any write would call one of them
    monkeypatch.setattr(storage, "_write_pieces", None)
    storage.write(path, Chunks(MORE))
    assert storage.plain_bytes(path) == MORE


@pytest.mark.parametrize("name", ["set.jsonl", "set.jsonl.zst"])
def test_holds_compares_a_set_files_plain_content_with_pieces(tmp_path, name):
    path = tmp_path / name
    storage.write(path, MORE)
    assert storage.holds(path, Chunks(MORE))
    assert storage.holds(path, Chunks(MORE, piece=1000))
    assert not storage.holds(path, Chunks(LINES))
    assert not storage.holds(path, Chunks(MORE + b"\n"))
    assert not storage.holds(path, Chunks(MORE.replace(b'"c"', b'"d"')))


def test_holds_names_a_frame_cut_short_and_a_git_lfs_pointer_as_plain_bytes_does(tmp_path):
    path = tmp_path / "set.jsonl.zst"
    storage.write(path, MORE * 1000)
    cut_short(path)
    with pytest.raises(ValueError, match=re.escape(f"{path} cannot be decompressed: ")):
        storage.holds(path, Chunks(MORE * 1000, piece=4096))
    path.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n")
    with pytest.raises(ValueError, match=re.escape(f"{path} is a Git LFS pointer; fetch it first: ")):
        storage.holds(path, Chunks(MORE))
