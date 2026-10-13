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
    assert vendor == [{"slug": "vendor", "model": "org/V", "revision": REVISION}]


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


def test_a_source_with_no_corpus_set_is_refused_and_a_finished_one_is_not(tmp_path):
    record = load("record_sources")
    for kind in ("render", "parse"):
        (tmp_path / "corpus" / kind).mkdir(parents=True)
        (tmp_path / "corpus" / kind / "glaive-v2-00.jsonl.zst").write_bytes(b"x")
    sets_toml = tmp_path / "sets.toml"
    sets_toml.write_text("[parse.glaive-v2-00]\ncases = 1\n\n[render.glaive-v2-00]\ncases = 1\n")
    assert record.queue(tmp_path / "corpus", ["glaive-v2"], sets_toml) == []
    with pytest.raises(SystemExit, match="no corpus set for swehro under"):
        record.queue(tmp_path / "corpus", ["glaive-v2", "swehro"], sets_toml)


def test_a_source_named_again_is_recorded_again_but_not_a_third_time(tmp_path):
    """After a recorder fix, ``--again`` records a source whose sets ``sets.toml`` already lists; a job run again
    leaves out the sets this run has already recorded again."""
    record = load("record_sources")
    for kind in ("render", "parse"):
        directory = tmp_path / "corpus" / kind
        directory.mkdir(parents=True)
        for name in ("tau2-run-0", "tau2-run-1", "glaive-v2-00"):
            (directory / f"{name}.jsonl.zst").write_bytes(b"x")
    sets_toml = tmp_path / "sets.toml"
    sets_toml.write_text(
        "".join(
            f"[{kind}.{name}]\ncases = 1\n\n"
            for kind in ("render", "parse")
            for name in ("tau2-run-0", "tau2-run-1", "glaive-v2-00")
        )
    )
    corpus = tmp_path / "corpus"
    assert record.queue(corpus, ["tau2", "glaive-v2"], sets_toml) == []
    queue = record.queue(corpus, ["tau2", "glaive-v2"], sets_toml, again={"tau2"}, rewritten={"parse": {"tau2-run-0"}})
    assert sorted((kind, [path.name.removesuffix(".jsonl.zst") for path in call]) for kind, call in queue) == [
        ("parse", ["tau2-run-1"]),
        ("render", ["tau2-run-0", "tau2-run-1"]),
    ]


def test_the_sets_a_run_recorded_again_are_the_ones_its_commits_changed(tmp_path):
    record = load("record_sources")
    group = tmp_path / "fixtures" / "g"
    (group / "parse").mkdir(parents=True)
    (group / "parse" / "tau2-run-0.jsonl.zst").write_bytes(b"old")
    (group / "parse" / "tau2-run-1.jsonl.zst").write_bytes(b"old")
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "dispatched"], check=True)
    dispatched = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    assert record.rewritten(group, dispatched) == {}
    (group / "parse" / "tau2-run-0.jsonl.zst").write_bytes(b"new")
    (group / "render").mkdir()
    (group / "render" / "tau2-run-1.jsonl").write_bytes(b"new")
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "record g: tau2"], check=True)
    assert record.rewritten(group, dispatched) == {"parse": {"tau2-run-0"}, "render": {"tau2-run-1"}}


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


def test_plain_sets_and_a_set_named_for_its_source_are_recorded_too(tmp_path):
    record = load("record_sources")
    files = {
        ("render", "bfcl-simple.jsonl"): 10,
        ("render", "bfcl-multiple.jsonl"): 10,
        ("parse", "bfcl-simple.jsonl"): 8,
        ("render", "common.jsonl"): 1,
        ("parse", "common.jsonl"): 1,
        ("render", "glaive-v2-00.jsonl.zst"): 5,
    }
    for (kind, name), size in files.items():
        directory = tmp_path / "corpus" / kind
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(b"x" * size)
    sets_toml = tmp_path / "sets.toml"
    sets_toml.write_text("[render.common]\ncases = 1\n")
    queue = record.queue(tmp_path / "corpus", ["common", "bfcl", "glaive-v2"], sets_toml)
    assert [(kind, [record.set_name(path) for path in call]) for kind, call in queue] == [
        ("render", ["bfcl-multiple", "bfcl-simple"]),
        ("parse", ["bfcl-simple"]),
        ("render", ["glaive-v2-00"]),
        ("parse", ["common"]),
    ]
    with pytest.raises(SystemExit, match="no corpus set for comm under"):
        record.queue(tmp_path / "corpus", ["comm"], sets_toml)


def test_the_changed_sets_are_cut_into_shards_under_the_budget_in_path_order():
    shards = load("set_shards").shards
    sized = [
        ("fixtures/a/parse/x.jsonl.zst", 3),
        ("fixtures/a/render/x.jsonl.zst", 3),
        ("fixtures/b/parse/y.jsonl.zst", 5),
    ]
    assert shards(sized, budget=6) == [
        ["fixtures/a/parse/x.jsonl.zst", "fixtures/a/render/x.jsonl.zst"],
        ["fixtures/b/parse/y.jsonl.zst"],
    ]
    # A set past the budget on its own is a shard of its own.
    assert shards([("big", 10), ("small", 1)], budget=6) == [["big"], ["small"]]
    assert shards([], budget=6) == []


def test_a_sets_size_is_the_one_its_git_lfs_pointer_names(tmp_path):
    size = load("set_shards").size
    pointer = tmp_path / "x.jsonl.zst"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 12345\n")
    assert size(pointer) == 12345


def test_the_sandbox_plan_takes_the_vendor_groups_and_refuses_a_runner_group(tmp_path, monkeypatch, capsys):
    plan = load("record_plan")
    manifest(tmp_path, "a-group", "org/A", ["tokenizer_config.json"])
    manifest(tmp_path, "vendor", "org/V", ["config.json", "tokenization_v.py"])
    monkeypatch.setattr(plan, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["record_plan.py", "--sandbox"])
    assert plan.main() == 0
    out = capsys.readouterr().out.splitlines()
    assert out == [f'groups=[{{"slug":"vendor","model":"org/V","revision":"{REVISION}"}}]', "count=1"]
    monkeypatch.setattr(sys, "argv", ["record_plan.py", "--sandbox", "a-group"])
    with pytest.raises(
        SystemExit, match="a-group: no vendor code among its oracle inputs; record it with the record workflow"
    ):
        plan.main()


def test_a_sandbox_call_runs_sandbox_record_with_the_memory_limit(tmp_path):
    record = load("record_sources")
    plain = record.command("org/M", "parse", tmp_path / "f", tmp_path / "c", ["tau2-run-0"], None)
    assert plain[1:] == [
        "record", "--model", "org/M", "--kind", "parse", "--oracle", "reference",
        "--fixtures", str(tmp_path / "f"), "--corpus", str(tmp_path / "c"), "--set", "tau2-run-0",
    ]  # fmt: skip
    boxed = record.command("org/M", "parse", tmp_path / "f", tmp_path / "c", ["tau2-run-0"], "20g")
    assert boxed[1:] == [
        "sandbox-record", "--model", "org/M", "--kind", "parse",
        "--fixtures", str(tmp_path / "f"), "--corpus", str(tmp_path / "c"), "--memory", "20g", "--set", "tau2-run-0",
    ]  # fmt: skip
