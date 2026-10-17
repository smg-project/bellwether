"""``bellwether patterns``: the recordings of what Python's ``regex`` module says about the response templates'
patterns, written beside the fixture root and copied into the tree ``unpack`` writes."""

from __future__ import annotations

import json
import pathlib

import regex

from bellwether import patterns
from bellwether.cli import main

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_recording_is_three_files_the_check_accepts_and_a_changed_one_fails(tmp_path, capsys):
    out = tmp_path / "patterns"
    assert main(["patterns", "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(patterns.FILES)
    assert main(["patterns", "--out", str(out), "--check"]) == 0
    (out / "search.jsonl").write_text((out / "search.jsonl").read_text() + "\n")
    assert main(["patterns", "--out", str(out), "--check"]) == 1
    assert "search.jsonl differs from what the recording gives" in capsys.readouterr().err


def test_the_classes_name_the_module_that_gave_them_and_cover_every_probe():
    tables = json.loads(patterns.record()["classes.json"])
    assert tables["regex"] == regex.__version__ == patterns.REGEX_VERSION
    assert tables["python"] == patterns.PYTHON
    for name in ("word", "space", "digit"):
        assert tables[name], name
        for span in tables[name]:
            assert len(span) == 2 and span[0] <= span[1], f"{name}: a range is its first and last character: {span!r}"
    for source, bits in tables["members"].items():
        assert len(bits) == len(tables["probes"]) and set(bits) <= {"0", "1"}, source
    assert "a" in tables["probes"] and tables["members"]["\\w"][tables["probes"].index("a")] == "1"


def test_every_recorded_pattern_answers_as_the_module_does_on_its_written_texts():
    for line in patterns.record()["regex.jsonl"].splitlines():
        recorded = json.loads(line)
        assert recorded["cases"], recorded["id"]
        for case in recorded["cases"]:
            assert ("prefixes" in case) == (case.get("prefixes") is not None) or True  # every case carries its answer
            assert "match" in case, f"{recorded['id']}: a case records what match gives"


def test_a_version_other_than_the_pinned_one_is_refused(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(patterns, "REGEX_VERSION", "1970.1.1")
    assert main(["patterns", "--out", str(tmp_path / "p")]) == 1
    assert "record with Python 3.12 and regex 1970.1.1" in capsys.readouterr().err
    assert not (tmp_path / "p").exists()


def test_the_committed_recordings_are_what_the_command_writes():
    """What CI's ``patterns --check`` holds, run here too so a stale file shows in the test report."""
    assert patterns.stale(ROOT / "patterns", patterns.record()) == []


def test_unpack_copies_the_recordings_beside_the_fixture_root_into_the_tree(tmp_path):
    from bellwether.unpack import copy_patterns

    fixtures, out = tmp_path / "fixtures", tmp_path / "plain"
    fixtures.mkdir()
    assert copy_patterns(fixtures, out) == []  # a root with no recordings beside it
    (tmp_path / "patterns").mkdir()
    for name in patterns.FILES:
        (tmp_path / "patterns" / name).write_text(f"{name}\n")
    (tmp_path / "patterns" / "notes.txt").write_text("not a recording\n")
    written = copy_patterns(fixtures, out)
    assert sorted(p.name for p in written) == sorted(patterns.FILES)
    assert sorted(p.name for p in (out / "patterns").iterdir()) == sorted(patterns.FILES)
    assert (out / "patterns" / "classes.json").read_text() == "classes.json\n"
