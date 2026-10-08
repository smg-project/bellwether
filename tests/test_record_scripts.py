import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
REVISION = "a" * 40


def load(name: str):
    """``scripts/<name>.py`` as a module."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(root: pathlib.Path, slug: str, model: str, inputs: list[str], group: str | None = None) -> None:
    directory = root / "fixtures" / slug
    directory.mkdir(parents=True)
    rows = "\n".join(f'"{name}" = "{"0" * 64}"' for name in inputs)
    member = f'group = "{group}"\n' if group else ""
    text = f'model = "{model}"\nrevision = "{REVISION}"\n{member}tier = 1\n\n[inputs]\n{rows}\n'
    (directory / "manifest.toml").write_text(text)


def test_the_plan_takes_every_group_once_and_leaves_vendor_code_to_the_sandbox(tmp_path):
    plan = load("record_plan")
    manifest(tmp_path, "b-group", "org/B", ["config.json", "tokenizer.json"])
    manifest(tmp_path, "a-group", "org/A", ["tokenizer_config.json"])
    manifest(tmp_path, "vendor", "org/V", ["config.json", "tokenization_v.py"])
    manifest(tmp_path, "a-member", "org/A-Small", ["tokenizer_config.json"], group="a-group")
    found, vendor = plan.groups(tmp_path, [])
    assert found == [
        {"slug": "a-group", "model": "org/A", "revision": REVISION},
        {"slug": "b-group", "model": "org/B", "revision": REVISION},
    ]
    assert vendor == ["vendor"]


def test_the_plan_takes_the_groups_named_and_refuses_a_name_with_no_manifest(tmp_path):
    plan = load("record_plan")
    manifest(tmp_path, "a-group", "org/A", ["config.json"])
    manifest(tmp_path, "b-group", "org/B", ["config.json"])
    found, _ = plan.groups(tmp_path, ["b-group"])
    assert [group["slug"] for group in found] == ["b-group"]
    with pytest.raises(SystemExit, match="no manifest for c-group"):
        plan.groups(tmp_path, ["b-group", "c-group"])
    manifest(tmp_path, "a-member", "org/A-Small", ["config.json"], group="a-group")
    with pytest.raises(SystemExit, match="a-member is recorded with its group's primary, a-group"):
        plan.groups(tmp_path, ["a-member"])


def test_the_sets_sets_toml_lists_are_left_out_and_the_rest_cut_into_calls(tmp_path):
    record = load("record_sources")
    parse = tmp_path / "corpus" / "parse"
    parse.mkdir(parents=True)
    for name in ("tau2-run-0", "tau2-run-1", "tau2-run-2", "tau2-run-3", "tau2-run-4", "glaive-v2-00"):
        (parse / f"{name}.jsonl.zst").write_bytes(b"")
    sets_toml = tmp_path / "sets.toml"
    sets_toml.write_text("[parse.tau2-run-1]\ncases = 1\n\n[render.tau2-run-2]\ncases = 1\n")
    calls = record.calls(tmp_path / "corpus", "parse", "tau2", 2, record.listed(sets_toml, "parse"))
    assert [[path.name for path in call] for call in calls] == [
        ["tau2-run-0.jsonl.zst", "tau2-run-2.jsonl.zst"],
        ["tau2-run-3.jsonl.zst", "tau2-run-4.jsonl.zst"],
    ]
    assert record.listed(tmp_path / "missing.toml", "parse") == set()


def test_the_calls_of_every_source_and_kind_queue_largest_first(tmp_path):
    record = load("record_sources")
    sizes = {
        ("render", "glaive-v2-00"): 5,
        ("render", "glaive-v2-01"): 5,
        ("parse", "glaive-v2-00"): 7,
        ("render", "tau2-run-0"): 20,
        ("parse", "tau2-run-0"): 30,
        ("render", "swehero-00"): 50,
        ("parse", "swehero-00"): 60,
    }
    for (kind, name), size in sizes.items():
        directory = tmp_path / "corpus" / kind
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.jsonl.zst").write_bytes(b"x" * size)
    queue = record.queue(tmp_path / "corpus", ["glaive-v2", "tau2", "swehero"], tmp_path / "sets.toml")
    assert [(kind, [path.name.removesuffix(".jsonl.zst") for path in call]) for kind, call in queue] == [
        ("parse", ["swehero-00"]),
        ("render", ["swehero-00"]),
        ("parse", ["tau2-run-0"]),
        ("render", ["tau2-run-0"]),
        ("render", ["glaive-v2-00", "glaive-v2-01"]),
        ("parse", ["glaive-v2-00"]),
    ]


def test_a_call_reads_a_corpus_directory_that_links_its_own_sets_alone(tmp_path):
    record = load("record_sources")
    source = tmp_path / "corpus" / "render"
    source.mkdir(parents=True)
    for name in ("glaive-v2-00", "glaive-v2-01", "tau2-run-0"):
        (source / f"{name}.jsonl.zst").write_bytes(name.encode())
    call = [source / "glaive-v2-01.jsonl.zst"]
    with record.call_corpus("render", call) as directory:
        linked = sorted(directory.joinpath("render").iterdir())
        assert [path.name for path in linked] == ["glaive-v2-01.jsonl.zst"]
        assert linked[0].read_bytes() == b"glaive-v2-01"
    assert not directory.exists()


def test_a_call_that_refused_cases_is_not_a_failure_and_one_that_crashed_is():
    record = load("record_sources")
    out = "/f/render/glaive-v2-00.jsonl.zst: 4990 cases recorded\n/f/render/glaive-v2-01.jsonl.zst: 10 cases recorded\n"
    refused = "not recorded m/render/a: ValueError: the template trims the message's content\n"
    assert record.outcome(1, out, refused) == (5000, 1, False)
    assert record.outcome(0, out, "") == (5000, 0, False)
    crashed = "Traceback (most recent call last):\n  ...\nMemoryError\n"
    assert record.outcome(1, "", crashed) == (0, 0, True)
    assert record.outcome(-9, out, "") == (5000, 0, True)
    # Status 1 stands for refused cases only when some were refused: an error the recorder reports and stops on is not.
    stopped = "bellwether record: cannot read the oracle inputs of org/M at 0123abcd: the cache holds no list\n"
    assert record.outcome(1, "", stopped) == (0, 0, True)


def test_the_largest_recorder_is_measured_from_the_children_that_ended():
    record = load("record_sources")
    subprocess.run([sys.executable, "-c", "held = b'x' * (64 << 20)"], check=True)
    assert record.largest_child_gb() >= 0.06
