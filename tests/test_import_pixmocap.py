import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from image_files import jpeg, png

from bellwether.cli import main
from bellwether.importers import github, hf, image_sets, pixmocap

ODC_TEXT = b"# ODC Attribution License (ODC-By)\n\n### Preamble\n"
CARD = b"---\nlicense: odc-by\n---\n# PixMo-Cap\n"
FILE = "data/train-00000-of-00004.parquet"
URLS = [
    "https://pictures.example/a.png",
    "https://pictures.example/b.jpg",
    "https://pictures.example/gone.png",
    "https://pictures.example/page.html",
    None,
    "https://pictures.example/a-again.png",
    "https://pictures.example/f.png",
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def table(urls: list[str | None]) -> pa.Table:
    """Rows of one URL each, as the dataset stores them, with a caption and transcripts beside."""
    return pa.table(
        {
            "image_url": pa.array(urls, type=pa.string()),
            "caption": pa.array(["a caption"] * len(urls)),
            "transcripts": pa.array([["t"]] * len(urls)),
        }
    )


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """The pinned file served from ``tmp_path`` with the card and the licence text through the real ``hf.fetch`` and
    ``github.fetch`` signatures, and the web served from a dict: two pictures, a 404, a page that is not an image, a
    null URL, a repeat of the first picture's bytes, and a third picture. The window covers the seven rows."""
    first, second, third = png(8, 6), jpeg(10, 4), png(3, 3)
    pq.write_table(table(URLS), tmp_path / "shard.parquet")
    files = {FILE: tmp_path / "shard.parquet"}
    web = {
        URLS[0]: ("200", first),
        URLS[1]: ("200", second),
        URLS[2]: ("404", None),
        URLS[3]: ("not-an-image", None),
        URLS[5]: ("200", first),
        URLS[6]: ("200", third),
    }
    fetched: list[str] = []
    served = {}

    def fetch_hf(repo, revision, filename, sha256, cache):
        assert (repo, revision) == (pixmocap.REPO, pixmocap.REVISION) and cache == tmp_path / "cache"
        served[filename] = sha256
        if filename == hf.CARD:
            path = tmp_path / "README.md"
            path.write_bytes(CARD)
            return path
        return files[filename]

    def fetch_github(owner, repo, commit, path, sha256, cache):
        assert (owner, repo, commit, path) == (
            pixmocap.LICENSE_OWNER,
            pixmocap.LICENSE_REPO,
            pixmocap.LICENSE_COMMIT,
            pixmocap.LICENSE_PATH,
        )
        assert (sha256, cache) == (pixmocap.LICENSE_SHA256, tmp_path / "cache")
        target = tmp_path / "ODC-By-1.0.txt"
        target.write_bytes(ODC_TEXT)
        return target

    def fetch_url(url):
        fetched.append(url)
        return web[url]

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    monkeypatch.setattr(github, "fetch", fetch_github)
    monkeypatch.setattr(pixmocap, "fetch_url", fetch_url)
    monkeypatch.setattr(pixmocap, "SHARDS", ((FILE, 0, 7),))
    return {"first": first, "second": second, "third": third, "served": served, "fetched": fetched, "web": web}


def test_the_shard_is_pinned_by_a_full_commit_and_a_sha256():
    assert len(pixmocap.REVISION) == 40 and int(pixmocap.REVISION, 16) >= 0
    assert pixmocap.SOURCE == f"hf:datasets/{pixmocap.REPO}@{pixmocap.REVISION}"
    assert list(pixmocap.FILES) == [FILE] and len(pixmocap.FILES[FILE]) == 64
    assert pixmocap.SHARDS == ((FILE, 0, 2000),)
    assert len(pixmocap.CARD_SHA256) == 64 and len(pixmocap.LICENSE_SHA256) == 64


def test_parquet_urls_keeps_to_the_window_and_skips_null_urls(pinned, tmp_path):
    rows = list(pixmocap.parquet_urls(tmp_path / "shard.parquet", 0, 7))
    assert [row for row, _ in rows] == [0, 1, 2, 3, 5, 6]
    assert list(pixmocap.parquet_urls(tmp_path / "shard.parquet", 2, 2)) == [(2, URLS[2]), (3, URLS[3])]


def test_the_licence_text_passes_and_another_text_is_refused():
    pixmocap.check_license(ODC_TEXT)
    with pytest.raises(ValueError, match="not the ODC Attribution License"):
        pixmocap.check_license(b"MIT License\n")


def test_import_fetches_once_writes_the_set_and_the_table_and_check_passes_then_fails(pinned, tmp_path, capsys):
    corpus = tmp_path / "corpus"
    cache = tmp_path / "cache"
    assert main(["import", "pixmocap", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    out = capsys.readouterr()
    lines = image_sets.read_set(corpus / "image" / "pixmocap.jsonl")
    assert [line["sha256"] for line in lines] == [sha(pinned["first"]), sha(pinned["second"]), sha(pinned["third"])]
    assert [line["origin"]["row"] for line in lines] == [0, 1, 6]
    assert lines[0]["origin"]["url"] == URLS[0] and len(lines[0]["origin"]["fetched"]) == 10
    assert lines[0]["notes"] == "PixMo-Cap train row 0, image_url"
    assert (corpus / "licenses" / "pixmocap-LICENSE").read_bytes() == ODC_TEXT
    table_lines = (corpus / "sources" / "pixmocap.tsv").read_text(encoding="utf-8").splitlines()
    assert table_lines[0] == "file\trow\turl\tfetched\tstatus\tsha256\tbytes"
    assert [line.split("\t")[1:2] + line.split("\t")[4:5] for line in table_lines[1:]] == [
        ["0", "200"],
        ["1", "200"],
        ["2", "404"],
        ["3", "not-an-image"],
        ["5", "200"],
        ["6", "200"],
    ]
    assert "1 left out as repeats" in out.out and "6 rows, 4 pictures, 2 misses" in out.out
    assert sorted(pinned["fetched"]) == sorted(u for u in URLS if u)
    # --check reads the table and the files and fetches nothing
    pinned["fetched"].clear()
    assert main(["import", "pixmocap", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    assert pinned["fetched"] == []
    # a second import fetches nothing either: the table's outcomes stand, the bytes come from the committed files
    assert main(["import", "pixmocap", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    assert pinned["fetched"] == []
    assert image_sets.read_set(corpus / "image" / "pixmocap.jsonl") == lines
    # a replaced picture fails the check by its identity; a table row rewritten fails it too
    (corpus / lines[2]["path"]).write_bytes(png(4, 4))
    assert main(["import", "pixmocap", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 1
    assert "holds sha256" in capsys.readouterr().err
    (corpus / lines[2]["path"]).write_bytes(pinned["third"])
    text = (corpus / "sources" / "pixmocap.tsv").read_text(encoding="utf-8")
    (corpus / "sources" / "pixmocap.tsv").write_text(text.replace("\t404\t", "\t500\t"), encoding="utf-8")
    assert main(["import", "pixmocap", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    (corpus / "sources" / "pixmocap.tsv").write_text(
        text.replace(URLS[1], "https://other.example/x.jpg"), encoding="utf-8"
    )
    assert main(["import", "pixmocap", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 1
    assert "names" in capsys.readouterr().err


def test_a_later_window_appends_after_the_earlier_rows_and_a_pointer_set_refuses_to_rebuild(
    pinned, tmp_path, monkeypatch
):
    corpus = tmp_path / "corpus"
    cache = tmp_path / "cache"
    monkeypatch.setattr(pixmocap, "SHARDS", ((FILE, 0, 2),))
    assert main(["import", "pixmocap", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    assert len(image_sets.read_set(corpus / "image" / "pixmocap.jsonl")) == 2
    monkeypatch.setattr(pixmocap, "SHARDS", ((FILE, 0, 2), (FILE, 2, 5)))
    pinned["fetched"].clear()
    assert main(["import", "pixmocap", "--corpus", str(corpus), "--cache", str(cache)]) == 0
    assert pinned["fetched"] == [URLS[2], URLS[3], URLS[5], URLS[6]]
    lines = image_sets.read_set(corpus / "image" / "pixmocap.jsonl")
    assert [line["name"].split("-")[1] for line in lines] == ["00001", "00002", "00003"]
    assert main(["import", "pixmocap", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    # the set's files as Git LFS pointers: --check still passes; a rebuild needs the bytes and says so
    for line in lines:
        path = corpus / line["path"]
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{line['sha256']}\nsize {line['bytes']}\n"
        path.write_bytes(pointer.encode())
    assert main(["import", "pixmocap", "--corpus", str(corpus), "--cache", str(cache), "--check"]) == 0
    with pytest.raises(ValueError, match="fetch the set's files first"):
        main(["import", "pixmocap", "--corpus", str(corpus), "--cache", str(cache)])


def test_a_card_whose_licence_statement_changes_stops_the_import_until_it_is_reviewed(pinned, tmp_path, monkeypatch):
    def fetch_hf(repo, revision, filename, sha256, cache):
        path = tmp_path / "README.md"
        path.write_bytes(b"---\nlicense: cc-by-4.0\n---\n")
        return path

    monkeypatch.setattr(hf, "fetch", fetch_hf)
    with pytest.raises(ValueError, match="review it before importing"):
        main(["import", "pixmocap", "--corpus", str(tmp_path / "corpus"), "--cache", str(tmp_path / "cache")])
    assert json.loads(json.dumps({"ok": True}))["ok"]
