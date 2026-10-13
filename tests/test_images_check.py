import hashlib
import json

from image_files import jpeg, png

from bellwether.cli import NOT_IMPLEMENTED, main
from bellwether.images import problems


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pointer(data: bytes) -> bytes:
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha(data)}\nsize {len(data)}\n".encode()


def line(dataset: str, index: int, data: bytes, ext: str) -> dict:
    name = f"{dataset}-{index:05d}-{sha(data)[:8]}"
    return {"name": name, "path": f"images/{dataset}/{name}.{ext}", "bytes": len(data), "sha256": sha(data)}


def corpus_with(tmp_path, as_pointer: bool):
    """A corpus of one set with two images, one as content and one as a Git LFS pointer when asked, and its licence."""
    corpus = tmp_path / "corpus"
    first, second = png(4, 4), jpeg(6, 6)
    lines = [line("ds", 1, first, "png"), line("ds", 2, second, "jpg")]
    (corpus / "image").mkdir(parents=True)
    (corpus / "image" / "ds.jsonl").write_text("".join(json.dumps(item) + "\n" for item in lines))
    (corpus / "images" / "ds").mkdir(parents=True)
    (corpus / lines[0]["path"]).write_bytes(first)
    (corpus / lines[1]["path"]).write_bytes(pointer(second) if as_pointer else second)
    (corpus / "licenses").mkdir()
    (corpus / "licenses" / "ds-LICENSE").write_bytes(b"licence\n")
    return corpus, lines, first, second


def test_a_corpus_of_content_or_pointers_passes_and_the_command_counts_it(tmp_path, capsys):
    for as_pointer in (False, True):
        corpus, _, first, second = corpus_with(tmp_path / str(as_pointer), as_pointer)
        assert problems(corpus) == []
        assert main(["images", "check", "--corpus", str(corpus)]) == 0
        assert capsys.readouterr().out.splitlines() == [
            f"{corpus}: 2 images in 1 sets, {len(first) + len(second)} bytes"
        ]


def test_every_kind_of_disagreement_is_named(tmp_path, capsys):
    corpus, lines, first, second = corpus_with(tmp_path, as_pointer=True)
    (corpus / lines[0]["path"]).write_bytes(png(5, 5))  # other content
    (corpus / lines[1]["path"]).write_bytes(pointer(second)[:-1] + b"9\n")  # the pointer's size disagrees
    (corpus / "images" / "ds" / "stray.png").write_bytes(b"x")
    (corpus / "images" / "other").mkdir()
    (corpus / "licenses" / "ds-LICENSE").unlink()
    set_file = corpus / "image" / "ds.jsonl"
    third = dict(lines[0], path="images/elsewhere/x.png")
    missing = line("ds", 4, jpeg(9, 9), "jpg")
    text = [json.dumps(item) for item in [*lines, lines[0], third, missing]]
    set_file.write_text("".join(item + "\n" for item in text))
    assert problems(corpus) == [
        f"{lines[0]['path']}: sha256 {sha(png(5, 5))}, the set says {sha(first)}",
        f"{lines[1]['path']}: {len(second) * 10 + 9} bytes, the set says {len(second)}",
        f"{set_file}:3: name {lines[0]['name']} is already used in {set_file}",
        f"{set_file}:3: path {lines[0]['path']} is already used in {set_file}",
        f"{set_file}:4: name {lines[0]['name']} is already used in {set_file}",
        f"{set_file}:4: path images/elsewhere/x.png is not under images/ds/",
        f"{missing['path']}: the set names it, the checkout lacks it",
        "images/ds/stray.png: no line of ds.jsonl names it",
        "licenses/ds-LICENSE: the licence copy of ds.jsonl is missing",
        "images/other: no set image/other.jsonl names it",
    ]
    assert main(["images", "check", "--corpus", str(corpus)]) == 1
    assert capsys.readouterr().err.splitlines()[-1] == "bellwether images check: 10 problem(s)"


def test_a_set_that_does_not_read_is_named_and_an_empty_corpus_passes(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    (corpus / "image").mkdir(parents=True)
    (corpus / "image" / "ds.jsonl").write_text('{"name": "Bad Name"}\n')
    assert problems(corpus) == [
        f"{corpus / 'image' / 'ds.jsonl'}:1: a line is an object whose `name` is a lowercase slug"
    ]
    assert problems(tmp_path / "nothing") == []
    assert main(["images", "check", "--corpus", str(tmp_path / "nothing")]) == 0
    assert capsys.readouterr().out.splitlines() == [f"{tmp_path / 'nothing'}: 0 images in 0 sets, 0 bytes"]


def test_images_without_a_subcommand_prints_help_and_does_nothing(capsys):
    assert main(["images"]) == NOT_IMPLEMENTED
    assert "check" in capsys.readouterr().out
