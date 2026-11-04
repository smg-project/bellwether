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


def manifest_for(
    fixtures: pathlib.Path, model: str, inputs: dict, group: str | None = None, encoder: str | None = None
):
    path = fixtures / model.split("/")[-1].lower() / "manifest.toml"
    path.parent.mkdir(parents=True)
    lines = [f'model = "{model}"', f'revision = "{REVISION}"', *([f'group = "{group}"'] if group else [])]
    lines += [f'encoder = "{encoder}"'] if encoder else []
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


def test_the_context_carries_the_engines_copy_of_a_prompt_encoder_read_only(listed, snapshot, tmp_path):
    corpus, context = tmp_path / "corpus", tmp_path / "context"
    (corpus / "render").mkdir(parents=True)
    context.mkdir()
    checked = tmp_path / "cache" / "deepseek_x_encoding.py"
    checked.parent.mkdir()
    checked.write_text("def encode_messages(messages, thinking_mode): ...\n")
    engine = ("vllm/tokenizers/deepseek_x_encoding.py", checked)
    sandbox.stage(listed, snapshot, "render", tmp_path / "fixtures", corpus, context, engine)
    staged = context / "engine" / "vllm" / "tokenizers" / "deepseek_x_encoding.py"
    assert staged.read_bytes() == checked.read_bytes()
    dockerfile = (context / "Dockerfile").read_text()
    assert f"COPY --chown=root:root engine {sandbox.ENGINE_DIR}\nRUN chmod -R a-w {sandbox.ENGINE_DIR}\n" in dockerfile
    assert sandbox.ENGINE_DIR == "/engine"


@pytest.fixture
def encoder_snapshot(tmp_path) -> pathlib.Path:
    """A checkpoint that ships a prompt encoder and no template, of an architecture the engine renders with a mode."""
    directory = tmp_path / "encoder-snapshot"
    (directory / "encoding").mkdir(parents=True)
    (directory / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<eos>"}))
    (directory / "config.json").write_text(
        json.dumps({"model_type": "toy", "architectures": ["DeepseekV4ForCausalLM"]})
    )
    (directory / "encoding" / "encoding_toy.py").write_text("def encode_messages(messages, thinking_mode): ...\n")
    return directory


def test_the_engines_copy_is_fetched_at_its_pin_for_a_checkpoint_with_a_prompt_encoder(
    encoder_snapshot, tmp_path, monkeypatch
):
    calls = []
    fetched = tmp_path / "fetched.py"
    fetched.write_text("engine copy")
    monkeypatch.setattr(sandbox, "fetch_pinned", lambda *args: calls.append(args) or fetched)
    inputs = oracle_inputs(str(encoder_snapshot), "local")
    manifest = manifest_for(tmp_path / "fixtures", "acme/X-4", inputs, encoder="deepseek_v4")
    pin = vendor.engine_encoder("deepseek_v4")
    assert sandbox.engine_copy(manifest, encoder_snapshot) == (pin.path, fetched)
    assert calls == [("vllm-project", "vllm", pin.commit, pin.path, pin.sha256)]


def test_a_manifest_whose_encoder_line_is_not_the_architectures_mode_is_refused_before_any_fetch(
    encoder_snapshot, tmp_path, monkeypatch
):
    # The recording plan routes a group to the sandbox by the manifest's line, so a line that disagrees with the
    # checkpoint's architecture is a manifest to rewrite, not a recording to make.
    monkeypatch.setattr(sandbox, "fetch_pinned", lambda *args: pytest.fail("fetched"))
    inputs = oracle_inputs(str(encoder_snapshot), "local")
    unnamed = manifest_for(tmp_path / "fixtures", "acme/X-6", inputs)
    with pytest.raises(sandbox.Refused, match="names None as the engine's encoder mode .*takes 'deepseek_v4'"):
        sandbox.engine_copy(unnamed, encoder_snapshot)
    other = manifest_for(tmp_path / "fixtures", "acme/X-7", inputs, encoder="deepseek_v32")
    with pytest.raises(sandbox.Refused, match="`bellwether manifests` writes the line"):
        sandbox.engine_copy(other, encoder_snapshot)


def test_the_engines_copy_is_fetched_for_an_encoder_architecture_that_ships_no_module(
    encoder_snapshot, tmp_path, monkeypatch
):
    # DeepSeek-V3.2-Exp ships a template and no encoding/ module; the engine renders its architecture with its copy.
    (encoder_snapshot / "encoding" / "encoding_toy.py").unlink()
    (encoder_snapshot / "encoding").rmdir()
    (encoder_snapshot / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<eos>", "chat_template": "{{ messages }}"})
    )
    (encoder_snapshot / "config.json").write_text(
        json.dumps({"model_type": "toy", "architectures": ["DeepseekV32ForCausalLM"]})
    )
    calls = []
    fetched = tmp_path / "fetched.py"
    fetched.write_text("engine copy")
    monkeypatch.setattr(sandbox, "fetch_pinned", lambda *args: calls.append(args) or fetched)
    inputs = oracle_inputs(str(encoder_snapshot), "local")
    manifest = manifest_for(tmp_path / "fixtures", "acme/X-32", inputs, encoder="deepseek_v32")
    assert not any(name.endswith(".py") for name in manifest.inputs)
    pin = vendor.engine_encoder("deepseek_v32")
    assert sandbox.engine_copy(manifest, encoder_snapshot) == (pin.path, fetched)
    assert calls == [("vllm-project", "vllm", pin.commit, pin.path, pin.sha256)]


def test_a_checkpoint_without_a_prompt_encoder_needs_no_engine_copy(listed, snapshot, monkeypatch):
    monkeypatch.setattr(sandbox, "fetch_pinned", lambda *args: pytest.fail("fetched"))
    assert sandbox.engine_copy(listed, snapshot) is None


def test_an_encoder_of_an_architecture_without_an_engine_mode_is_refused_before_any_fetch(
    encoder_snapshot, tmp_path, monkeypatch
):
    monkeypatch.setattr(sandbox, "fetch_pinned", lambda *args: pytest.fail("fetched"))
    (encoder_snapshot / "config.json").write_text(json.dumps({"architectures": ["SomeOtherForCausalLM"]}))
    manifest = manifest_for(tmp_path / "fixtures", "acme/X-5", oracle_inputs(str(encoder_snapshot), "local"))
    with pytest.raises(sandbox.Refused, match="no tokenizer mode of its own"):
        sandbox.engine_copy(manifest, encoder_snapshot)


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


@pytest.mark.parametrize("status, copied", [(0, True), (1, True), (124, False), (137, False), (143, False)])
def test_the_group_comes_back_only_from_a_run_that_finished(listed, tmp_path, monkeypatch, status, copied):
    # 1 is the recorder refusing some cases, its usual outcome; a run the time or memory limit stopped, or one that was
    # stopped, leaves the host's fixtures as they were.
    calls = []

    def docker(*args, **kwargs):
        calls.append(args[0])
        if args[0] == "cp":
            (pathlib.Path(args[2]) / "manifest.toml").write_text("from the run")
        return subprocess.CompletedProcess(args, 0)

    group = listed.path.parent
    monkeypatch.setattr(sandbox, "checked_snapshot", lambda manifest: tmp_path)
    monkeypatch.setattr(sandbox, "base_image", lambda: "sha256:base")
    monkeypatch.setattr(sandbox, "stage", lambda *args: None)
    monkeypatch.setattr(sandbox, "docker", docker)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, status))
    namespace = type("Args", (), {"model": "acme/X-1", "kind": "render", "fixtures": group.parent, "corpus": tmp_path})
    assert sandbox.run(namespace) == status
    assert ("cp" in calls) == copied
    assert ((group / "manifest.toml").read_text() == "from the run") == copied
    assert sorted(p.name for p in group.parent.iterdir()) == [group.name]


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
        ("--env", f"{vendor.ENGINE_DIR_ENV}={sandbox.ENGINE_DIR}"),
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


def test_named_sets_reach_the_recorder_and_the_memory_limit_is_the_one_given(listed):
    # Benchmark-scale recording runs a source's sets a few at a time, as the record runners do, with room for them.
    args = sandbox.run_args(
        "run-1", "run-1-work", "run-image", "sha256:base", listed, "render", ["common", "bfcl-simple"], "20g"
    )
    assert ("--memory", "20g") in set(zip(args, args[1:], strict=False))
    assert "6g" not in args
    command = args[args.index("run-image") + 1 :]
    assert command[-4:] == ["--set", "common", "--set", "bfcl-simple"]


def sets_toml(group: pathlib.Path, tables: dict[str, int]) -> None:
    fields = f'form = "plain"\nrejected = 0\nplain_bytes = 3\nplain_sha256 = "{"0" * 64}"\n'
    text = "".join(f"[render.{name}]\ncases = {cases}\n{fields}\n" for name, cases in tables.items())
    (group / "sets.toml").write_text(text)


def test_a_run_of_named_sets_brings_back_those_sets_alone(tmp_path):
    # Calls at once each bring back their own sets: a call's copy of the group holds the others as they were when it
    # started, so the host's other sets and tables stay as the host holds them now.
    from bellwether.record import sets as set_tables

    group = tmp_path / "fixtures" / "x-1"
    (group / "render").mkdir(parents=True)
    (group / "render" / "other.jsonl").write_text("host's\n")
    (group / "render" / "named.jsonl").write_text("old\n")
    (group / "render" / "gone.jsonl").write_text("old\n")
    sets_toml(group, {"other": 7, "named": 1, "gone": 1})
    fresh = sandbox.landing(group)
    (fresh / "render").mkdir()
    (fresh / "render" / "other.jsonl").write_text("a call's stale copy\n")
    (fresh / "render" / "named.jsonl.zst").write_bytes(b"new")
    sets_toml(fresh, {"other": 6, "named": 2})
    sandbox.bring_back(fresh, group, "render", ["named", "gone"])
    files = sorted(p.relative_to(group).as_posix() for p in group.rglob("*") if p.is_file() and p.suffix != ".lock")
    assert files == ["render/named.jsonl.zst", "render/other.jsonl", "sets.toml"]
    assert (group / "render" / "other.jsonl").read_text() == "host's\n"
    tables = set_tables.read(group / "sets.toml")
    assert {name: table["cases"] for (_, name), table in tables.items()} == {"other": 7, "named": 2}


def test_a_run_of_named_sets_from_the_command_line_keeps_the_hosts_other_sets(listed, tmp_path, monkeypatch):
    group = listed.path.parent
    (group / "render").mkdir()
    (group / "render" / "other.jsonl").write_text("host's\n")
    sets_toml(group, {"other": 7})

    def docker(*args, **kwargs):
        if args[0] == "cp":
            landed = pathlib.Path(args[2])
            (landed / "render").mkdir()
            (landed / "render" / "named.jsonl").write_text("new\n")
            sets_toml(landed, {"named": 3})
        return subprocess.CompletedProcess(args, 0)

    ran = []
    monkeypatch.setattr(sandbox, "checked_snapshot", lambda manifest: tmp_path)
    monkeypatch.setattr(sandbox, "base_image", lambda: "sha256:base")
    monkeypatch.setattr(sandbox, "stage", lambda *args: None)
    monkeypatch.setattr(sandbox, "docker", docker)
    monkeypatch.setattr(
        subprocess, "run", lambda args, **kwargs: ran.append(args) or subprocess.CompletedProcess(args, 0)
    )
    namespace = type(
        "Args",
        (),
        {
            "model": "acme/X-1",
            "kind": "render",
            "fixtures": group.parent,
            "corpus": tmp_path,
            "sets": ["named"],
            "memory": "20g",
        },
    )
    assert sandbox.run(namespace) == 0
    assert ran[0][-2:] == ["--set", "named"] and ("--memory", "20g") in set(zip(ran[0], ran[0][1:], strict=False))
    assert (group / "render" / "other.jsonl").read_text() == "host's\n"
    assert (group / "render" / "named.jsonl").read_text() == "new\n"
    assert (group / "manifest.toml").is_file()
