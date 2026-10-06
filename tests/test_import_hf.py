import hashlib

import huggingface_hub
import pytest

from bellwether.importers import corpus_sets, hf

REVISION = "a" * 40


def serve_file(path, calls: list):
    def download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(path)

    return download


def test_fetch_returns_the_cached_file_whose_hash_is_the_pinned_one(tmp_path, monkeypatch):
    (tmp_path / "rows.parquet").write_bytes(b"rows")
    calls: list = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", serve_file(tmp_path / "rows.parquet", calls))
    path = hf.fetch("org/data", REVISION, "data/rows.parquet", hashlib.sha256(b"rows").hexdigest())
    assert path == tmp_path / "rows.parquet"
    assert calls == [("org/data", "data/rows.parquet", {"repo_type": "dataset", "revision": REVISION})]


def test_fetch_refuses_a_file_that_is_not_the_pinned_one(tmp_path, monkeypatch):
    (tmp_path / "rows.parquet").write_bytes(b"tampered")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", serve_file(tmp_path / "rows.parquet", []))
    with pytest.raises(ValueError, match=f"org/data@{REVISION} data/rows.parquet: sha256 .* is not the pinned"):
        hf.fetch("org/data", REVISION, "data/rows.parquet", hashlib.sha256(b"rows").hexdigest())


CARD = "---\nconfigs:\n  - config_name: default\n    license: mit\nlicense: apache-2.0\n---\n\n# A dataset\n"


def test_card_license_is_the_top_level_license_of_the_front_matter():
    assert hf.card_license(CARD) == "apache-2.0"
    assert hf.card_license(CARD.replace("\n", "\r\n")) == "apache-2.0"


def test_a_license_line_outside_the_front_matter_or_nested_in_it_does_not_count():
    assert hf.card_license("---\nconfigs:\n  - license: mit\n---\nlicense: mit\n") is None
    assert hf.card_license("# A dataset\nlicense: mit\n") is None
    assert hf.card_license("") is None


def test_a_line_separator_inside_the_front_matter_does_not_start_a_line():
    assert hf.card_license("---\npretty_name: a license: mit\n---\n") is None


def test_check_card_license_passes_the_reviewed_license_and_refuses_any_other():
    hf.check_card_license("org/data", CARD, "apache-2.0")
    with pytest.raises(
        ValueError, match=r"org/data README.md: the card's license is 'apache-2.0', not the reviewed 'mit'"
    ):
        hf.check_card_license("org/data", CARD, "mit")


def test_check_card_license_with_none_reviewed_refuses_a_card_that_states_one():
    hf.check_card_license("org/data", "---\npretty_name: a\n---\n", None)
    with pytest.raises(ValueError, match=r"the card's license is 'apache-2.0', not the reviewed None"):
        hf.check_card_license("org/data", CARD, None)


def test_write_refuses_a_source_past_the_limit_and_writes_nothing(tmp_path, monkeypatch):
    sets = {("render", "x-a"): [{"name": "x-a-0"}], ("parse", "x-a"): [{"name": "x-a-0"}]}
    size = len(corpus_sets.text(sets["render", "x-a"]).encode()) * 2
    monkeypatch.setattr(corpus_sets, "LIMIT", size - 1)
    with pytest.raises(ValueError, match=f"the x-\\* sets take {size} bytes, past the {size - 1}"):
        corpus_sets.write(sets, tmp_path, "x-")
    assert not any(tmp_path.rglob("*.jsonl"))
    monkeypatch.setattr(corpus_sets, "LIMIT", size)
    assert len(corpus_sets.write(sets, tmp_path, "x-")) == 2


def test_the_limit_is_fifty_megabytes():
    assert corpus_sets.LIMIT == 50_000_000
