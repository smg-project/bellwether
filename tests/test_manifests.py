import errno
import json
import pathlib
import shutil

import pytest

from bellwether.cli import main
from bellwether.groups import INPUTS_HEADER, LIST, existing_manifests, is_recorded, rank, read_list
from bellwether.inputs import NARROWED, oracle_inputs
from bellwether.manifest import load_manifest

ROOT = pathlib.Path(__file__).resolve().parents[1]

PINNED = "1" * 40
DAY = "2026-10-06"  # the day the downloads were read

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


HEADER = "# model\trevision\tdownloads\tday\ttier\n"


def manifests(tmp_path: pathlib.Path, *rows: tuple) -> int:
    """``bellwether manifests`` over these rows, written as the list file with a header comment on line 1."""
    listed = tmp_path / "models.tsv"
    listed.write_text(HEADER + "".join("\t".join(map(str, r)) + "\n" for r in rows))
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
    assert (
        manifests(tmp_path, (alpha, "local", 10, DAY, 1), (beta, "local", 50, DAY, 1), (gamma, "local", 5, DAY, 3)) == 0
    )
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
    assert (
        manifests(tmp_path, (alpha, "local", 10, DAY, 1), (beta, "local", 50, DAY, 2), (gamma, "local", 5, DAY, 3)) == 0
    )
    assert capsys.readouterr().out.splitlines() == [
        f"beta: {beta} (tier 2), {alpha} (tier 1)",
        f"gamma: {gamma} (tier 3)",
        "3 checkpoints in 2 checkpoint groups",
    ]


def test_manifests_writes_the_same_files_when_run_again(tmp_path, hub):
    rows = [
        (str(hub / name), "local", downloads, DAY, 1) for name, downloads in (("alpha", 10), ("beta", 50), ("gamma", 5))
    ]
    assert manifests(tmp_path, *rows) == 0
    first = tree(tmp_path / "fixtures")
    assert manifests(tmp_path, *reversed(rows)) == 0
    assert tree(tmp_path / "fixtures") == first


def test_a_recorded_group_keeps_its_slug_and_every_line_a_person_wrote(tmp_path, hub):
    alpha, beta = str(hub / "alpha"), str(hub / "beta")
    fixtures = tmp_path / "fixtures"
    written = write_by_hand(fixtures, "alpha", alpha, recorded=True)
    assert manifests(tmp_path, (alpha, "local", 10, DAY, 1), (beta, "local", 50, DAY, 1)) == 0
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
    assert manifests(tmp_path, (alpha, "local", 10, DAY, 1), (gamma, "local", 5, DAY, 3)) == 0
    created = fixtures / "gamma" / "manifest.toml"
    assert "[authority]" not in created.read_text()
    assert load_manifest(created).authority == {}
    assert load_manifest(fixtures / "alpha" / "manifest.toml").authority == {"render": ["hf-template"]}


def test_a_group_recorded_nowhere_takes_its_most_downloaded_member_as_primary(tmp_path, hub):
    alpha, beta = str(hub / "alpha"), str(hub / "beta")
    fixtures = tmp_path / "fixtures"
    write_by_hand(fixtures, "alpha", alpha, recorded=False, revision=PINNED)
    assert manifests(tmp_path, (alpha, "local", 10, DAY, 1), (beta, "local", 50, DAY, 1)) == 0
    member = load_manifest(fixtures / "alpha" / "manifest.toml")
    # Nothing was recorded at the old revision, so it moves to the listed one.
    assert (member.group, member.revision) == ("beta", "local")
    assert load_manifest(fixtures / "beta" / "manifest.toml").group is None


def test_an_unrecorded_group_follows_its_downloads_until_it_is_recorded(tmp_path, hub):
    alpha, beta = str(hub / "alpha"), str(hub / "beta")
    fixtures = tmp_path / "fixtures"
    assert manifests(tmp_path, (alpha, "local", 10, DAY, 1), (beta, "local", 50, DAY, 1)) == 0
    assert manifests(tmp_path, (alpha, "local", 90, DAY, 1), (beta, "local", 50, DAY, 1)) == 0
    assert load_manifest(fixtures / "alpha" / "manifest.toml").group is None
    assert load_manifest(fixtures / "beta" / "manifest.toml").group == "alpha"
    assert not [line for line in (fixtures / "alpha" / "manifest.toml").read_text().splitlines() if "group " in line]


def test_manifests_refuses_a_list_that_leaves_out_a_checkpoint_with_a_manifest(tmp_path, hub, capsys):
    alpha, beta = str(hub / "alpha"), str(hub / "beta")
    fixtures = tmp_path / "fixtures"
    assert manifests(tmp_path, (alpha, "local", 10, DAY, 1), (beta, "local", 50, DAY, 1)) == 0
    before = tree(fixtures)
    # Grouped alone, alpha would become a second primary beside beta, whose inputs are the same.
    assert manifests(tmp_path, (alpha, "local", 10, DAY, 1)) == 1
    assert f"the list leaves out {beta}, which has {fixtures / 'beta' / 'manifest.toml'}" in capsys.readouterr().err
    assert tree(fixtures) == before


def test_manifests_refuses_to_move_a_recorded_group_to_another_revision(tmp_path, hub, capsys):
    alpha = str(hub / "alpha")
    fixtures = tmp_path / "fixtures"
    write_by_hand(fixtures, "alpha", alpha, recorded=True, revision=PINNED)
    before = tree(fixtures)
    assert manifests(tmp_path, (alpha, "local", 10, DAY, 1)) == 1
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
    assert manifests(tmp_path, (alpha, "local", 10, DAY, 1), (beta, "local", 50, DAY, 1)) == 1
    assert "the recorded groups alpha and beta have equal oracle inputs" in capsys.readouterr().err
    assert tree(fixtures) == before


def test_manifests_checks_every_manifest_before_writing_any(tmp_path, hub, capsys):
    # alpha's recorded manifest sits in a directory whose name is not a slug, so beta, whose inputs are alpha's, could
    # not name the group: nothing is written, alpha's manifest included.
    alpha, beta = str(hub / "alpha"), str(hub / "beta")
    fixtures = tmp_path / "fixtures"
    write_by_hand(fixtures, "alpha_chat", alpha, recorded=True)
    before = tree(fixtures)
    assert manifests(tmp_path, (alpha, "local", 10, DAY, 1), (beta, "local", 50, DAY, 1)) == 1
    err = capsys.readouterr().err
    member = fixtures / "beta" / "manifest.toml"
    assert f"{member}: `group` must be the slug of its group's primary, got 'alpha_chat'" in err
    assert "bellwether manifests: nothing was written" in err
    assert tree(fixtures) == before


def test_manifests_reports_a_manifest_it_cannot_write(tmp_path, hub, monkeypatch, capsys):
    alpha, gamma = str(hub / "alpha"), str(hub / "gamma")
    write_text = pathlib.Path.write_text

    def disk_full_at_gamma(self, *args, **kwargs):
        if self.parent.name == "gamma":
            raise OSError(errno.ENOSPC, "No space left on device", str(self))
        return write_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "write_text", disk_full_at_gamma)
    assert manifests(tmp_path, (alpha, "local", 10, DAY, 1), (gamma, "local", 5, DAY, 3)) == 1
    err = capsys.readouterr().err
    assert f"No space left on device: '{tmp_path / 'fixtures' / 'gamma' / 'manifest.toml'}'" in err
    assert "bellwether manifests: stopped after writing 1 of 2 manifests, in slug order" in err


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (("acme/Tiny",), "expected model, revision, downloads, day and tier, separated by tabs"),
        (("acme/Tiny", "main", 1, DAY, 1), "revision 'main' is not a commit"),
        (
            ("acme/Tiny", PINNED, "many", DAY, 1),
            "downloads must be a whole number, or - where no count was read, got 'many'",
        ),
        (("acme/Tiny", PINNED, 1, "6 Oct", 1), "day must be the date the downloads were read, YYYY-MM-DD, got '6 Oct'"),
        (
            ("acme/Tiny", PINNED, 1, "2026-02-30", 1),
            "day must be the date the downloads were read, YYYY-MM-DD, got '2026-02-30'",
        ),
        (("acme/Tiny", PINNED, "-", DAY, 1), "a count goes with the day it was read: give both, or - for both"),
        (("acme/Tiny", PINNED, 1, "-", 1), "a count goes with the day it was read: give both, or - for both"),
        (("acme/Tiny", PINNED, 1, DAY, 4), "tier must be 1, 2 or 3, got '4'"),
    ],
)
def test_manifests_rejects_a_malformed_row_naming_its_line(tmp_path, row, message, capsys):
    assert manifests(tmp_path, row) == 1
    assert f"{tmp_path / 'models.tsv'}:2: {message}" in capsys.readouterr().err
    assert not (tmp_path / "fixtures").exists()


def test_manifests_rejects_a_checkpoint_listed_twice(tmp_path, capsys):
    row = ("acme/Tiny", PINNED, 1, DAY, 1)
    assert manifests(tmp_path, row, row) == 1
    assert f"{tmp_path / 'models.tsv'}:3: acme/Tiny is listed twice" in capsys.readouterr().err


def test_manifests_names_each_checkpoint_it_cannot_read_and_writes_nothing(tmp_path, hub, monkeypatch, capsys):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    assert manifests(tmp_path, (str(hub / "alpha"), "local", 1, DAY, 1), ("acme/missing", PINNED, 1, DAY, 1)) == 1
    assert f"cannot read the oracle inputs of acme/missing at {PINNED}: FileNotFoundError" in capsys.readouterr().err
    assert not (tmp_path / "fixtures").exists()


def test_manifests_refuses_two_checkpoints_that_take_one_slug(tmp_path, tiny_model, capsys):
    one, two = tmp_path / "one" / "tiny", tmp_path / "two" / "tiny"
    shutil.copytree(tiny_model, one)
    shutil.copytree(tiny_model, two)
    assert manifests(tmp_path, (str(one), "local", 1, DAY, 1), (str(two), "local", 1, DAY, 1)) == 1
    assert f"{two} and {one} both take the slug tiny" in capsys.readouterr().err
    assert not (tmp_path / "fixtures").exists()


def test_manifests_reads_the_list_beside_the_manifests_unless_given_another(tmp_path, hub):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "models.tsv").write_text(HEADER + f"{hub / 'alpha'}\tlocal\t10\t{DAY}\t2\n")
    assert main(["manifests", "--fixtures", str(fixtures)]) == 0
    assert load_manifest(fixtures / "alpha" / "manifest.toml").tier == 2


def test_a_checkpoint_whose_downloads_were_not_read_ranks_after_one_whose_were(tmp_path, hub):
    alpha, beta, fixtures = str(hub / "alpha"), str(hub / "beta"), tmp_path / "fixtures"
    # "-" is a count nobody read: a count of 0 still ranks above it.
    assert manifests(tmp_path, (alpha, "local", "-", "-", 1), (beta, "local", 0, DAY, 1)) == 0
    assert load_manifest(fixtures / "alpha" / "manifest.toml").group == "beta"
    # Without a count on either side, the first model id is the primary.
    assert manifests(tmp_path, (alpha, "local", "-", "-", 1), (beta, "local", "-", "-", 1)) == 0
    assert load_manifest(fixtures / "beta" / "manifest.toml").group == "alpha"


def test_the_committed_list_names_each_committed_manifest_at_its_revision_and_tier():
    listed = {checkpoint.model: checkpoint for checkpoint in read_list(ROOT / "fixtures" / LIST)}
    committed = existing_manifests(ROOT / "fixtures")
    assert sorted(listed) == sorted(committed)
    for model, manifest in committed.items():
        assert (manifest.revision, manifest.tier) == (listed[model].revision, listed[model].tier), manifest.path


def test_each_committed_group_recorded_nowhere_has_the_list_s_most_downloaded_member_as_primary():
    listed = {checkpoint.model: checkpoint for checkpoint in read_list(ROOT / "fixtures" / LIST)}
    committed = existing_manifests(ROOT / "fixtures")
    slugs = {manifest.slug: manifest for manifest in committed.values()}
    members: dict[str, list] = {}
    for manifest in committed.values():
        members.setdefault(manifest.group_slug, []).append(listed[manifest.model])
    for slug, checkpoints in members.items():
        if not is_recorded(slugs[slug]):
            assert min(checkpoints, key=rank).model == slugs[slug].model, slug


def test_the_inputs_header_names_where_the_narrowed_files_are_listed():
    # sha256sum of a narrowed file does not give its listed hash; the header says why without naming the files, so it
    # stays true as the list of narrowed files changes.
    assert "chat_template.json" in NARROWED
    assert "bellwether.inputs.NARROWED" in INPUTS_HEADER
