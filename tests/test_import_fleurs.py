import hashlib
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from audio_files import flac, opus, wav

from bellwether.cli import main
from bellwether.importers import audio_sets, fleurs, github, hf

LICENSE_TEXT = b"Creative Commons Attribution 4.0 International\n\n(the body of the text)\n"
CARD = b"---\nlicense:\n- cc-by-4.0\n---\n# FLEURS\n"
FILE = "parquet-data/en_us/test-00000-of-00001.parquet"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table(clips: list[bytes | None]) -> pa.Table:
    """Rows of one utterance each, in the ``audio`` column as the dataset stores it, null where absent."""
    kind = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    cells = [({"bytes": data, "path": None} if data else None) for data in clips]
    return pa.table({"id": pa.array([f"r{n}" for n in range(len(clips))]), "audio": pa.array(cells, type=kind)})


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """The pinned shard served from ``tmp_path``, with the card and the licence text, through the real ``hf.fetch``
    and ``github.fetch`` signatures with no network."""
    first, second, third = flac(), opus(), wav()
    pq.write_table(table([first, first, second, b"not a clip", None, third, second]), tmp_path / "shard.parquet")
    files = {FILE: tmp_path / "shard.parquet"}
    served = {}

    def fetch_hf(repo, revision, filename, sha256, cache):
        assert (repo, revision) == (fleurs.REPO, fleurs.REVISION) and cache == tmp_path / "cache"
        served[filename] = sha256
        if filename == hf.CARD:
            path = tmp_path / "README.md"
            path.write_bytes(CARD)
            return path
        return files[filename]

    def fetch_github(owner, repo, commit, path, sha256, cache):
        assert (owner, repo, commit, path) == (
            fleurs.LICENSE_OWNER,
            fleurs.LICENSE_REPO,
            fleurs.LICENSE_COMMIT,
            fleurs.LICENSE_PATH,
        )
        assert (sha256, cache) == (fleurs.LICENSE_SHA256, tmp_path / "cache")
        (tmp_path / "LICENSE").write_bytes(LICENSE_TEXT)
        return tmp_path / "LICENSE"

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    monkeypatch.setattr(github, "fetch", fetch_github)
    return {"first": first, "second": second, "third": third, "served": served, "files": files}


def test_the_shard_is_pinned_by_a_full_commit_and_a_sha256():
    assert re.fullmatch(r"[0-9a-f]{40}", fleurs.REVISION)
    assert list(fleurs.FILES) == [FILE] and re.fullmatch(r"[0-9a-f]{64}", fleurs.FILES[FILE])
    assert fleurs.SOURCE == f"hf:datasets/google/fleurs@{fleurs.REVISION}"
    assert fleurs.CARD_LICENSE == ["cc-by-4.0"] and fleurs.LICENSE == "CC-BY-4.0"


def test_read_gives_every_utterance_in_row_order_with_its_origin(pinned):
    found = list(fleurs.read(pinned["files"]))
    assert [(f.origin["file"], f.origin["row"]) for f in found] == [
        (FILE, 0),
        (FILE, 1),
        (FILE, 2),
        (FILE, 3),
        (FILE, 5),
        (FILE, 6),
    ]
    assert found[0].data == pinned["first"] and found[2].data == pinned["second"]
    assert found[0].origin == {
        "dataset": "fleurs",
        "source": fleurs.SOURCE,
        "sha256": fleurs.FILES[FILE],
        "file": FILE,
        "row": 0,
        "column": "audio",
        "license": "CC-BY-4.0",
    }
    assert found[0].notes == "FLEURS test row 0, audio"
    assert found[4].notes == "FLEURS test row 5, audio"


def test_the_licence_text_passes_and_another_text_is_refused():
    fleurs.check_license(LICENSE_TEXT)
    with pytest.raises(ValueError, match=re.escape("not the CC BY 4.0 licence text")):
        fleurs.check_license(b"Apache License\n")


def test_import_writes_the_set_the_clips_and_the_licence_copy_and_check_passes_then_fails(pinned, tmp_path, capsys):
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert main(["import", "fleurs", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr().out.splitlines()
    first, second, third = pinned["first"], pinned["second"], pinned["third"]
    lines = audio_sets.read_set(corpus / "audio" / "fleurs.jsonl")
    assert [line["name"] for line in lines] == [
        f"fleurs-00001-{sha(first)[:8]}",
        f"fleurs-00002-{sha(second)[:8]}",
        f"fleurs-00003-{sha(third)[:8]}",
    ]
    assert lines[0]["path"] == f"audios/fleurs/fleurs-00001-{sha(first)[:8]}.flac"
    assert lines[1]["path"].endswith(".opus") and lines[2]["path"].endswith(".wav")
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "fleurs-LICENSE").read_bytes() == LICENSE_TEXT
    assert pinned["served"][hf.CARD] == fleurs.CARD_SHA256 and pinned["served"][FILE] == fleurs.FILES[FILE]
    assert out == [
        f"no line for FLEURS test row 1, audio: it repeats fleurs-00001-{sha(first)[:8]}",
        f"no line for FLEURS test row 6, audio: it repeats fleurs-00002-{sha(second)[:8]}",
        "no line for FLEURS test row 3, audio: not a FLAC, WAV, Ogg, MPEG audio, MP4 or M4A file",
        f"{corpus / 'audio' / 'fleurs.jsonl'}: 3 clips, {len(first) + len(second) + len(third)} bytes, "
        "2 left out as repeats, 1 rows with no clip",
    ]
    assert main(["import", "fleurs", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"{corpus}: the FLEURS set equals a fresh import of {fleurs.SOURCE}"
    ]
    text = (corpus / "audio" / "fleurs.jsonl").read_text().splitlines()
    tampered = json.loads(text[2])
    tampered["sample_rate"] = 8000
    text[2] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "audio" / "fleurs.jsonl").write_text("".join(line + "\n" for line in text))
    assert main(["import", "fleurs", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 1
    assert capsys.readouterr().err.splitlines() == [
        f"{corpus / 'audio' / 'fleurs.jsonl'}:3: differs from a fresh import"
    ]


def test_a_card_whose_licence_statement_changes_stops_the_import_until_it_is_reviewed(pinned, tmp_path, monkeypatch):
    def fetch_hf(repo, revision, filename, sha256, cache):
        path = tmp_path / "README.md"
        path.write_bytes(b"---\nlicense: mit\n---\n")
        return path

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    with pytest.raises(ValueError, match=re.escape("the card's license is 'mit', not the reviewed ['cc-by-4.0']")):
        main(["import", "fleurs", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
