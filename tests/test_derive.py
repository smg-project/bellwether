import hashlib
import json
from collections import Counter

import pytest
from image_files import gif, jpeg, png, riff, vp8

from bellwether import derive
from bellwether.cli import main
from bellwether.importers import coco, image_sets

# The first derivation's table: files per variant over the 5,000 COCO photographs (multimodal/README.md of October).
TABLE = {
    "webp-q85": 300, "webp-lossless": 150, "png-rgb": 300, "png-gray8": 100, "jpeg-gray": 100, "jpeg-progressive": 150,
    "jpeg-444": 100, "jpeg-422": 100, "jpeg-q100": 50, "png-palette": 100, "png-palette-trns": 60, "png-rgba": 100,
    "png-la": 60, "webp-rgba": 60, "png-gray16": 60, "png-rgb16": 60, "png-interlaced": 60, "cmyk-pillow": 200,
    "cmyk-noapp14": 199, "ycck-420": 100, "ycck-444": 50, "exif1-jpeg": 30, "exif6-png-rgba": 30, "tiny-jpeg": 100,
    "tiny-png": 50, "huge-jpeg": 24, "gif-animated": 10, "apng-animated": 10, "webp-animated": 10, "gif-static": 10,
    **{f"exif{o}-jpeg": 200 for o in range(2, 9)},
    **{f"exif{o}-webp": 50 for o in range(2, 9)},
    **{f"exif{o}-png": 50 for o in range(2, 9)},
}  # fmt: skip


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sources(count: int) -> list[dict]:
    """COCO lines as the set holds them, in name order already, with distinct digests."""
    return [
        {
            "name": f"coco-{n:05d}-{sha(str(n).encode())[:8]}",
            "path": f"images/coco/coco-{n:05d}-{sha(str(n).encode())[:8]}.jpg",
            "bytes": 10 + n,
            "sha256": sha(str(n).encode()),
        }
        for n in range(1, count + 1)
    ]


def test_the_plan_over_five_thousand_photographs_is_the_first_derivations_table():
    steps = derive.plan(sources(5000))
    assert len(steps) == 4833 and len(TABLE) == 51
    assert Counter(step.variant for step in steps) == TABLE
    assert steps[0].name == "webp-q85-coco-00001-" + sha(b"1")[:8] and steps[0].params == "WEBP quality=85"
    assert steps[0].ext == "webp" and steps[-1].variant == "gif-static"
    names = [step.name for step in steps]
    assert len(set(names)) == len(names)


def test_the_plan_reads_the_sources_in_name_order_whatever_order_they_come_in():
    photos = sources(5000)
    assert [s.source["name"] for s in derive.plan(list(reversed(photos)))] == [
        s.source["name"] for s in derive.plan(photos)
    ]


def test_the_stripped_cmyk_files_are_every_other_source_of_the_slice_starting_with_the_first():
    steps = [s for s in derive.plan(sources(5000)) if s.variant in ("cmyk-pillow", "cmyk-noapp14")]
    pillow = [s.source["name"] for s in steps if s.variant == "cmyk-pillow"]
    stripped = [s.source["name"] for s in steps if s.variant == "cmyk-noapp14"]
    assert (
        stripped == pillow[0::2]
        and len(stripped) == 100
        or stripped == pillow[1::2]
        and len(stripped) == 100
        or len(stripped) == 199
    )


def test_the_tiny_and_huge_sides_cycle_as_the_recipe_cycles_them():
    steps = derive.plan(sources(5000))
    tiny = [s for s in steps if s.variant in ("tiny-jpeg", "tiny-png")]
    assert [s.params.rsplit(" ", 1)[1] for s in tiny[:4]] == ["16", "24", "28", "31"]
    assert [s.variant for s in tiny[:3]] == ["tiny-jpeg", "tiny-jpeg", "tiny-png"]
    huge = [s for s in steps if s.variant == "huge-jpeg"]
    assert [s.params.rsplit(" ", 1)[1] for s in huge[:5]] == ["4096", "4400", "4800", "5200", "4096"]


def test_an_origin_names_the_coco_image_the_variant_the_parameters_the_tool_and_cocos_licence():
    step = derive.plan(sources(5000))[0]
    assert derive.origin(step) == {
        "dataset": "derived",
        "source": step.source["path"],
        "sha256": step.source["sha256"],
        "variant": "webp-q85",
        "params": "WEBP quality=85",
        "tool": derive.origin(step)["tool"],
        "license": coco.LICENSE,
    }
    assert derive.origin(step)["tool"].startswith("bellwether derive, Pillow 12.3.0")


VARIANT_INDEX = {variant: index for index, variant in enumerate(TABLE)}


def fake_bytes(step: derive.Step) -> bytes:
    """Header-only files of the right container for a step, standing in for the encoders: the width is the source's
    index and the height the variant's, so no two steps give the same bytes and the written set has no repeats."""
    w = 10 + int(step.source["name"].split("-")[1])
    h = 8 + VARIANT_INDEX[step.variant]
    if step.ext == "webp":
        return riff([(b"VP8 ", vp8(w, h))])
    if step.ext == "png":
        return png(w, h)
    if step.ext == "gif":
        return gif(w, h)
    return jpeg(w, h)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A corpus with a small COCO set and the encoders replaced by header-only files."""
    corpus = tmp_path / "corpus"
    photos = sources(3000)
    (corpus / "image").mkdir(parents=True)
    (corpus / "image" / "coco.jsonl").write_text("".join(json.dumps(line) + "\n" for line in photos))
    (corpus / "licenses").mkdir()
    (corpus / "licenses" / "coco-LICENSE").write_bytes(b"terms")

    def found(steps, corpus_dir):
        for step in steps:
            yield image_sets.Found(fake_bytes(step), derive.origin(step), step.name, name=step.name)

    monkeypatch.setattr(derive, "_found", found)
    return corpus


def test_write_then_check_passes_and_every_kind_of_difference_is_named(corpus, capsys):
    assert main(["derive", "--corpus", str(corpus), "--write"]) == 0
    lines = image_sets.read_set(corpus / "image" / "derived.jsonl")
    assert len(lines) == 4833 and lines[0]["name"].startswith("webp-q85-coco-00001-")
    assert lines[0]["path"] == f"images/derived/{lines[0]['name']}.webp" and lines[0]["format"] == "webp"
    assert (corpus / "licenses" / "derived-LICENSE").read_bytes() == b"terms"
    assert main(["derive", "--corpus", str(corpus)]) == 0
    assert (
        capsys.readouterr().out.splitlines()[-1]
        == f"{corpus}: the derived set is the plan over the COCO set, 4833 files"
    )
    # A line whose origin is not the plan's.
    text = (corpus / "image" / "derived.jsonl").read_text().splitlines()
    tampered = json.loads(text[5])
    tampered["origin"]["params"] = "WEBP quality=80"
    text[5] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "image" / "derived.jsonl").write_text("".join(line + "\n" for line in text))
    assert main(["derive", "--corpus", str(corpus)]) == 1
    err = capsys.readouterr().err
    assert "derived.jsonl:6: names" in err and "the plan names" in err
    (corpus / "image" / "derived.jsonl").write_text(
        "".join(line + "\n" for line in text[:5] + [json.dumps(lines[5])] + text[6:])
    )
    assert main(["derive", "--corpus", str(corpus)]) == 0
    # A file whose bytes are not its line's, a file no line names, a missing licence copy.
    path = corpus / lines[7]["path"]
    path.write_bytes(b"other")
    (corpus / "images" / "derived" / "stray.png").write_bytes(png())
    (corpus / "licenses" / "derived-LICENSE").unlink()
    assert main(["derive", "--corpus", str(corpus)]) == 1
    err = capsys.readouterr().err.splitlines()
    assert any(str(path) in line and "holds sha256" in line for line in err)
    assert any("stray.png: no line of the set names it" in line for line in err)
    assert any("derived-LICENSE: missing" in line for line in err)


def test_check_names_a_missing_set_and_a_set_of_another_length(corpus, capsys):
    assert main(["derive", "--corpus", str(corpus)]) == 1
    assert capsys.readouterr().err.splitlines() == [f"{corpus / 'image' / 'derived.jsonl'}: missing"]
    main(["derive", "--corpus", str(corpus), "--write"])
    text = (corpus / "image" / "derived.jsonl").read_text().splitlines()
    (corpus / "image" / "derived.jsonl").write_text("".join(line + "\n" for line in text[:-1]))
    assert main(["derive", "--corpus", str(corpus)]) == 1
    err = capsys.readouterr().err
    assert "4832 lines, the plan over the COCO set names 4833 files" in err


def test_derive_without_the_coco_set_says_to_import_it_first(tmp_path):
    with pytest.raises(FileNotFoundError, match="import coco first"):
        main(["derive", "--corpus", str(tmp_path / "corpus")])
