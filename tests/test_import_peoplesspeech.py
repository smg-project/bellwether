import hashlib
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from audio_files import flac, opus, wav

from bellwether.cli import main
from bellwether.importers import audio_sets, github, hf, peoplesspeech

LICENSE_TEXT = b"Creative Commons Attribution 4.0 International\n\n(the body of the text)\n"
CARD = (
    b"---\nlicense:\n"
    b"- cc-by-2.0\n"
    b"- cc-by-2.5\n"
    b"- cc-by-3.0\n"
    b"- cc-by-4.0\n"
    b"- cc-by-sa-3.0\n"
    b"- cc-by-sa-4.0\n---\n# People's Speech\n"
)
FILE = "clean/test-00000-of-00009.parquet"


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
        assert (repo, revision) == (peoplesspeech.REPO, peoplesspeech.REVISION) and cache == tmp_path / "cache"
        served[filename] = sha256
        if filename == hf.CARD:
            path = tmp_path / "README.md"
            path.write_bytes(CARD)
            return path
        return files[filename]

    def fetch_github(owner, repo, commit, path, sha256, cache):
        assert (owner, repo, commit, path) == (
            peoplesspeech.LICENSE_OWNER,
            peoplesspeech.LICENSE_REPO,
            peoplesspeech.LICENSE_COMMIT,
            peoplesspeech.LICENSE_PATH,
        )
        assert (sha256, cache) == (peoplesspeech.LICENSE_SHA256, tmp_path / "cache")
        (tmp_path / "LICENSE").write_bytes(LICENSE_TEXT)
        return tmp_path / "LICENSE"

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    monkeypatch.setattr(github, "fetch", fetch_github)
    # The window covers the whole seven-row shard here; the window test narrows it.
    monkeypatch.setattr(peoplesspeech, "SHARDS", ((FILE, 0, 7),))
    return {"first": first, "second": second, "third": third, "served": served, "files": files}


def test_the_shard_is_pinned_by_a_full_commit_and_a_sha256():
    assert re.fullmatch(r"[0-9a-f]{40}", peoplesspeech.REVISION)
    assert list(peoplesspeech.FILES) == [FILE] and re.fullmatch(r"[0-9a-f]{64}", peoplesspeech.FILES[FILE])
    assert peoplesspeech.SOURCE == f"hf:datasets/MLCommons/peoples_speech@{peoplesspeech.REVISION}"
    assert (
        peoplesspeech.CARD_LICENSE
        == ["cc-by-2.0", "cc-by-2.5", "cc-by-3.0", "cc-by-4.0", "cc-by-sa-3.0", "cc-by-sa-4.0"]
        and peoplesspeech.LICENSE == "CC-BY-4.0"
    )
    assert peoplesspeech.SHARDS == ((FILE, 0, 1000),)


def test_read_gives_every_utterance_in_row_order_with_its_origin(pinned):
    found = list(peoplesspeech.read(pinned["files"]))
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
        "dataset": "peoplesspeech",
        "source": peoplesspeech.SOURCE,
        "sha256": peoplesspeech.FILES[FILE],
        "file": FILE,
        "row": 0,
        "column": "audio",
        "license": "CC-BY-4.0",
    }
    assert found[0].notes == "People's Speech test row 0, audio"
    assert found[4].notes == "People's Speech test row 5, audio"


def test_read_keeps_to_the_window_of_rows(pinned, monkeypatch):
    monkeypatch.setattr(peoplesspeech, "SHARDS", ((FILE, 2, 3),))
    assert [f.origin["row"] for f in peoplesspeech.read(pinned["files"])] == [2, 3]
    monkeypatch.setattr(peoplesspeech, "SHARDS", ((FILE, 5, 1), (FILE, 0, 1)))
    assert [f.origin["row"] for f in peoplesspeech.read(pinned["files"])] == [5, 0]


def test_the_licence_text_passes_and_another_text_is_refused():
    peoplesspeech.check_license(LICENSE_TEXT)
    with pytest.raises(ValueError, match=re.escape("not the CC BY 4.0 licence text")):
        peoplesspeech.check_license(b"Apache License\n")


def test_import_writes_the_set_the_clips_and_the_licence_copy_and_check_passes_then_fails(pinned, tmp_path, capsys):
    corpus, cache = tmp_path / "corpus", tmp_path / "cache"
    assert main(["import", "peoplesspeech", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr().out.splitlines()
    first, second, third = pinned["first"], pinned["second"], pinned["third"]
    lines = audio_sets.read_set(corpus / "audio" / "peoplesspeech.jsonl")
    assert [line["name"] for line in lines] == [
        f"peoplesspeech-00001-{sha(first)[:8]}",
        f"peoplesspeech-00002-{sha(second)[:8]}",
        f"peoplesspeech-00003-{sha(third)[:8]}",
    ]
    assert lines[0]["path"] == f"audios/peoplesspeech/peoplesspeech-00001-{sha(first)[:8]}.flac"
    assert lines[1]["path"].endswith(".opus") and lines[2]["path"].endswith(".wav")
    assert (corpus / lines[0]["path"]).read_bytes() == first
    assert (corpus / "licenses" / "peoplesspeech-LICENSE").read_bytes() == LICENSE_TEXT
    assert (
        pinned["served"][hf.CARD] == peoplesspeech.CARD_SHA256 and pinned["served"][FILE] == peoplesspeech.FILES[FILE]
    )
    assert out == [
        f"no line for People's Speech test row 1, audio: it repeats peoplesspeech-00001-{sha(first)[:8]}",
        f"no line for People's Speech test row 6, audio: it repeats peoplesspeech-00002-{sha(second)[:8]}",
        "no line for People's Speech test row 3, audio: not a FLAC, WAV, Ogg, MPEG audio, MP4 or M4A file",
        f"{corpus / 'audio' / 'peoplesspeech.jsonl'}: 3 clips, {len(first) + len(second) + len(third)} bytes, "
        "2 left out as repeats, 1 rows with no clip",
    ]
    assert main(["import", "peoplesspeech", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"{corpus}: the People's Speech set equals a fresh import of {peoplesspeech.SOURCE}"
    ]
    text = (corpus / "audio" / "peoplesspeech.jsonl").read_text().splitlines()
    tampered = json.loads(text[2])
    tampered["sample_rate"] = 8000
    text[2] = json.dumps(tampered, ensure_ascii=False)
    (corpus / "audio" / "peoplesspeech.jsonl").write_text("".join(line + "\n" for line in text))
    assert main(["import", "peoplesspeech", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 1
    assert capsys.readouterr().err.splitlines() == [
        f"{corpus / 'audio' / 'peoplesspeech.jsonl'}:3: differs from a fresh import"
    ]


def test_a_card_whose_licence_statement_changes_stops_the_import_until_it_is_reviewed(pinned, tmp_path, monkeypatch):
    def fetch_hf(repo, revision, filename, sha256, cache):
        path = tmp_path / "README.md"
        path.write_bytes(b"---\nlicense: mit\n---\n")
        return path

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    refused = f"the card's license is 'mit', not the reviewed {peoplesspeech.CARD_LICENSE!r}"
    with pytest.raises(ValueError, match=re.escape(refused)):
        main(["import", "peoplesspeech", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
