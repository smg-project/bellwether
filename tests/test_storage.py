import pathlib
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
