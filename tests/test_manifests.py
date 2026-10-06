import json
import pathlib
import shutil

import pytest

from bellwether.cli import main
from bellwether.inputs import oracle_inputs
from bellwether.manifest import load_manifest

PINNED = "1" * 40

# A manifest as a person wrote it before groups: a comment, parser names, no tier and no inputs.
WRITTEN_BY_HAND = """model    = "{model}"
revision = "{revision}"   # the fixtures were recorded here

# A note a person wrote.
[authority]
render = ["hf-template"]

[smg]
tool_parser = "qwen"

[engines]
vllm = {{ tool_parser = "hermes" }}
"""


@pytest.fixture
def hub(tmp_path, tiny_model) -> pathlib.Path:
    """Three checkpoints as directories. alpha and beta differ only in a sampling default, so their oracle inputs are
    equal; gamma's chat template differs."""
    root = tmp_path / "hub"
    for name, temperature in (("alpha", 0.6), ("beta", 0.9), ("gamma", 0.6)):
        shutil.copytree(tiny_model, root / name)
        (root / name / "generation_config.json").write_text(json.dumps({"eos_token_id": 2, "temperature": temperature}))
    config = json.loads((root / "gamma" / "tokenizer_config.json").read_text())
    config["chat_template"] += "{{ '' }}"
    (root / "gamma" / "tokenizer_config.json").write_text(json.dumps(config))
    return root


def manifests(tmp_path: pathlib.Path, *rows: tuple) -> int:
    """``bellwether manifests`` over these rows, written as the list file with a header comment on line 1."""
    listed = tmp_path / "models.tsv"
    listed.write_text("# model\trevision\tdownloads\ttier\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))
    return main(["manifests", "--models", str(listed), "--fixtures", str(tmp_path / "fixtures")])


def write_by_hand(fixtures: pathlib.Path, slug: str, model: str, *, recorded: bool, revision: str = "local") -> str:
    """A manifest from before groups; ``recorded`` gives it the ``sets.toml`` that recording writes."""
    (fixtures / slug).mkdir(parents=True)
    text = WRITTEN_BY_HAND.format(model=model, revision=revision)
    (fixtures / slug / "manifest.toml").write_text(text)
    if recorded:
        (fixtures / slug / "sets.toml").write_text("")
    return text


def tree(directory: pathlib.Path) -> dict[str, bytes]:
    return {p.relative_to(directory).as_posix(): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def test_manifests_groups_checkpoints_by_their_oracle_inputs(tmp_path, hub):
    alpha, beta, gamma = (str(hub / name) for name in ("alpha", "beta", "gamma"))
    assert manifests(tmp_path, (alpha, "local", 10, 1), (beta, "local", 50, 1), (gamma, "local", 5, 3)) == 0
    fixtures = tmp_path / "fixtures"
    primary, member, other = (load_manifest(fixtures / slug / "manifest.toml") for slug in ("beta", "alpha", "gamma"))
    # beta has more downloads than alpha, so the group takes its slug; alpha's manifest names it.
    assert (primary.model, primary.revision, primary.group, primary.tier) == (beta, "local", None, 1)
    assert (member.model, member.revision, member.group, member.tier) == (alpha, "local", "beta", 1)
    assert (other.model, other.group, other.tier) == (gamma, None, 3)
    assert member.inputs == primary.inputs == oracle_inputs(beta, "local") != other.inputs
    assert primary.authority == member.authority == other.authority == {}
    assert (primary.smg, primary.engines, member.smg, member.engines) == ({}, {}, {}, {})
    assert list(tree(fixtures)) == ["alpha/manifest.toml", "beta/manifest.toml", "gamma/manifest.toml"]


def test_manifests_prints_each_group_with_its_members_and_their_tiers(tmp_path, hub, capsys):
    alpha, beta, gamma = (str(hub / name) for name in ("alpha", "beta", "gamma"))
    assert manifests(tmp_path, (alpha, "local", 10, 1), (beta, "local", 50, 2), (gamma, "local", 5, 3)) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"beta: {beta} (tier 2), {alpha} (tier 1)",
        f"gamma: {gamma} (tier 3)",
        "3 checkpoints in 2 checkpoint groups",
    ]


def test_manifests_writes_the_same_files_when_run_again(tmp_path, hub):
    rows = [(str(hub / name), "local", downloads, 1) for name, downloads in (("alpha", 10), ("beta", 50), ("gamma", 5))]
    assert manifests(tmp_path, *rows) == 0
    first = tree(tmp_path / "fixtures")
    assert manifests(tmp_path, *reversed(rows)) == 0
    assert tree(tmp_path / "fixtures") == first


def test_a_recorded_group_keeps_its_slug_and_every_line_a_person_wrote(tmp_path, hub):
    alpha, beta = str(hub / "alpha"), str(hub / "beta")
    fixtures = tmp_path / "fixtures"
    written = write_by_hand(fixtures, "alpha", alpha, recorded=True)
    assert manifests(tmp_path, (alpha, "local", 10, 1), (beta, "local", 50, 1)) == 0
    assert load_manifest(fixtures / "beta" / "manifest.toml").group == "alpha"
    text = (fixtures / "alpha" / "manifest.toml").read_text()
    kept = written.replace("recorded here\n", "recorded here\ntier     = 1\n")
    assert text.startswith(kept + "\n[inputs]")
    primary = load_manifest(fixtures / "alpha" / "manifest.toml")
    assert (primary.group, primary.inputs) == (None, oracle_inputs(alpha, "local"))


def test_a_new_manifest_has_no_authority_order_and_an_existing_one_keeps_its_own(tmp_path, hub):
    # The design's two sources of truth replaced the order the manifests written before groups share; restating an
    # order for each checkpoint is the sponsor's call (AGENTS.md), so the command writes none.
    alpha, gamma = str(hub / "alpha"), str(hub / "gamma")
    fixtures = tmp_path / "fixtures"
    write_by_hand(fixtures, "alpha", alpha, recorded=False)
    assert manifests(tmp_path, (alpha, "local", 10, 1), (gamma, "local", 5, 3)) == 0
    created = fixtures / "gamma" / "manifest.toml"
    assert "[authority]" not in created.read_text()
    assert load_manifest(created).authority == {}
    assert load_manifest(fixtures / "alpha" / "manifest.toml").authority == {"render": ["hf-template"]}


def test_a_group_recorded_nowhere_takes_its_most_downloaded_member_as_primary(tmp_path, hub):
    alpha, beta = str(hub / "alpha"), str(hub / "beta")
    fixtures = tmp_path / "fixtures"
    write_by_hand(fixtures, "alpha", alpha, recorded=False, revision=PINNED)
    assert manifests(tmp_path, (alpha, "local", 10, 1), (beta, "local", 50, 1)) == 0
    member = load_manifest(fixtures / "alpha" / "manifest.toml")
    # Nothing was recorded at the old revision, so it moves to the listed one.
    assert (member.group, member.revision) == ("beta", "local")
    assert load_manifest(fixtures / "beta" / "manifest.toml").group is None


def test_an_unrecorded_group_follows_its_downloads_until_it_is_recorded(tmp_path, hub):
    alpha, beta = str(hub / "alpha"), str(hub / "beta")
    fixtures = tmp_path / "fixtures"
    assert manifests(tmp_path, (alpha, "local", 10, 1), (beta, "local", 50, 1)) == 0
    assert manifests(tmp_path, (alpha, "local", 90, 1), (beta, "local", 50, 1)) == 0
    assert load_manifest(fixtures / "alpha" / "manifest.toml").group is None
    assert load_manifest(fixtures / "beta" / "manifest.toml").group == "alpha"
    assert not [line for line in (fixtures / "alpha" / "manifest.toml").read_text().splitlines() if "group " in line]


def test_manifests_refuses_a_list_that_leaves_out_a_checkpoint_with_a_manifest(tmp_path, hub, capsys):
    alpha, beta = str(hub / "alpha"), str(hub / "beta")
    fixtures = tmp_path / "fixtures"
    assert manifests(tmp_path, (alpha, "local", 10, 1), (beta, "local", 50, 1)) == 0
    before = tree(fixtures)
    # Grouped alone, alpha would become a second primary beside beta, whose inputs are the same.
    assert manifests(tmp_path, (alpha, "local", 10, 1)) == 1
    assert f"the list leaves out {beta}, which has {fixtures / 'beta' / 'manifest.toml'}" in capsys.readouterr().err
    assert tree(fixtures) == before


def test_manifests_refuses_to_move_a_recorded_group_to_another_revision(tmp_path, hub, capsys):
    alpha = str(hub / "alpha")
    fixtures = tmp_path / "fixtures"
    write_by_hand(fixtures, "alpha", alpha, recorded=True, revision=PINNED)
    before = tree(fixtures)
    assert manifests(tmp_path, (alpha, "local", 10, 1)) == 1
    assert f"{fixtures / 'alpha' / 'manifest.toml'} pins {PINNED}, where its fixtures were recorded" in (
        capsys.readouterr().err
    )
    assert tree(fixtures) == before


def test_manifests_refuses_to_merge_two_recorded_groups(tmp_path, hub, capsys):
    alpha, beta = str(hub / "alpha"), str(hub / "beta")
    fixtures = tmp_path / "fixtures"
    write_by_hand(fixtures, "alpha", alpha, recorded=True)
    write_by_hand(fixtures, "beta", beta, recorded=True)
    before = tree(fixtures)
    assert manifests(tmp_path, (alpha, "local", 10, 1), (beta, "local", 50, 1)) == 1
    assert "the recorded groups alpha and beta have equal oracle inputs" in capsys.readouterr().err
    assert tree(fixtures) == before


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (("acme/Tiny",), "expected model, revision, downloads and tier, separated by tabs"),
        (("acme/Tiny", "main", 1, 1), "revision 'main' is not a commit"),
        (("acme/Tiny", PINNED, "many", 1), "downloads must be a whole number, got 'many'"),
        (("acme/Tiny", PINNED, 1, 4), "tier must be 1, 2 or 3, got '4'"),
    ],
)
def test_manifests_rejects_a_malformed_row_naming_its_line(tmp_path, row, message, capsys):
    assert manifests(tmp_path, row) == 1
    assert f"{tmp_path / 'models.tsv'}:2: {message}" in capsys.readouterr().err
    assert not (tmp_path / "fixtures").exists()


def test_manifests_rejects_a_checkpoint_listed_twice(tmp_path, capsys):
    row = ("acme/Tiny", PINNED, 1, 1)
    assert manifests(tmp_path, row, row) == 1
    assert f"{tmp_path / 'models.tsv'}:3: acme/Tiny is listed twice" in capsys.readouterr().err


def test_manifests_names_each_checkpoint_it_cannot_read_and_writes_nothing(tmp_path, hub, monkeypatch, capsys):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    assert manifests(tmp_path, (str(hub / "alpha"), "local", 1, 1), ("acme/missing", PINNED, 1, 1)) == 1
    assert f"cannot read the oracle inputs of acme/missing at {PINNED}: FileNotFoundError" in capsys.readouterr().err
    assert not (tmp_path / "fixtures").exists()


def test_manifests_refuses_two_checkpoints_that_take_one_slug(tmp_path, tiny_model, capsys):
    one, two = tmp_path / "one" / "tiny", tmp_path / "two" / "tiny"
    shutil.copytree(tiny_model, one)
    shutil.copytree(tiny_model, two)
    assert manifests(tmp_path, (str(one), "local", 1, 1), (str(two), "local", 1, 1)) == 1
    assert f"{two} and {one} both take the slug tiny" in capsys.readouterr().err
    assert not (tmp_path / "fixtures").exists()
