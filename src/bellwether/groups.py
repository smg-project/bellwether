"""bellwether manifests: group checkpoints by their oracle inputs and write each checkpoint's manifest.

Reads a list of checkpoints, ``fixtures/models.tsv`` unless ``--models`` names another, one
``model<TAB>revision<TAB>downloads<TAB>day<TAB>tier`` row each (blank lines and lines that start with ``#`` are
skipped). ``downloads`` is the Hub's count over the last 30 days and ``day`` the day it was read; both are ``-`` for a
checkpoint whose count nobody read. The list is committed beside the manifests it builds, so a rebuild is a diff of
both. The command computes each checkpoint's oracle inputs at its revision (``bellwether.inputs``), and groups the
checkpoints whose inputs are equal (docs/benchmark-sets.md, "Which models" and "Fixture ids"). A group is recorded
once, under the slug of its primary:

- a group that holds a recorded primary, one whose directory has a ``sets.toml``, keeps it, since fixture ids are
  cited by their slug and a slug never changes once recorded. Two recorded primaries whose inputs have become equal
  are refused: merging them retires one slug's ids, which is a person's call;
- otherwise the most-downloaded member is the primary, a member with a count before one without, ties going to the
  first model id, and the group takes its slug.

Each checkpoint's manifest stays where it is, or is written under its own slug. The command sets ``revision``, ``tier``,
``group`` (on members only) and ``[inputs]``, and keeps every other line of an existing manifest: comments,
``[authority]``, ``[smg]`` and ``[engines]``. A new manifest has none of these tables. The order the manifests written
before groups share predates the design's two sources of truth (docs/benchmark-sets.md, "Recording"), and a manifest's
authority order waits for the sponsor's approval (AGENTS.md); parser names stay unknown until someone maps them. A
recorded manifest's revision does not move here, since its fixtures were recorded at it. The list names every checkpoint
that has a manifest, as the design's one list of checkpoints does: grouped without it, a listed checkpoint with the same
inputs would become a second primary. Nothing is written unless every checkpoint was read and every check passed, and a
second run over the same list writes the same files.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from bellwether.inputs import oracle_inputs
from bellwether.manifest import TIERS, Manifest, is_pinned, load_manifest, load_manifests, slug_for
from bellwether.record import sets as set_tables

INPUTS_HEADER = (
    "[inputs]  # sha256 of each oracle input; a JSON file in bellwether.inputs.NARROWED over its fields only"
)
LIST = "models.tsv"  # the list of checkpoints, beside the manifests it builds
NOT_READ = "-"
_REVISION = re.compile(r"""(revision\s*=\s*)("[^"]*"|'[^']*')""")
_SET_HERE = re.compile(r"(tier|group)\s*=")


@dataclass(frozen=True)
class Checkpoint:
    model: str
    revision: str
    downloads: int | None  # the Hub's count over the last 30 days; None when nobody read it
    day: str | None  # the day the count was read, YYYY-MM-DD
    tier: int


@dataclass
class Group:
    slug: str
    primary: Checkpoint
    others: list[Checkpoint]


def read_list(path: Path) -> list[Checkpoint]:
    checkpoints: dict[str, Checkpoint] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        where = f"{path}:{number}"
        fields = [field.strip() for field in line.split("\t")]
        if len(fields) != 5:
            raise ValueError(f"{where}: expected model, revision, downloads, day and tier, separated by tabs")
        model, revision, downloads, day, tier = fields
        if not is_pinned(model, revision):
            raise ValueError(f"{where}: revision {revision!r} is not a commit; list the 40-character commit hash")
        if downloads != NOT_READ and not re.fullmatch(r"[0-9]+", downloads):
            raise ValueError(
                f"{where}: downloads must be a whole number, or - where no count was read, got {downloads!r}"
            )
        if day != NOT_READ and not _is_date(day):
            raise ValueError(f"{where}: day must be the date the downloads were read, YYYY-MM-DD, got {day!r}")
        if (downloads == NOT_READ) != (day == NOT_READ):
            raise ValueError(f"{where}: a count goes with the day it was read: give both, or - for both")
        if tier not in {str(t) for t in TIERS}:
            raise ValueError(f"{where}: tier must be 1, 2 or 3, got {tier!r}")
        if model in checkpoints:
            raise ValueError(f"{where}: {model} is listed twice")
        read = downloads != NOT_READ
        checkpoints[model] = Checkpoint(
            model, revision, int(downloads) if read else None, day if read else None, int(tier)
        )
    return list(checkpoints.values())


def _is_date(text: str) -> bool:
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", text):
        return False
    try:
        datetime.date.fromisoformat(text)
    except ValueError:
        return False
    return True


def existing_manifests(fixtures: Path) -> dict[str, Manifest]:
    return {manifest.model: manifest for manifest in load_manifests(fixtures, check_groups=False)}


def check_listed(checkpoints: list[Checkpoint], existing: dict[str, Manifest]) -> None:
    listed = {checkpoint.model for checkpoint in checkpoints}
    left_out = [manifest for model, manifest in existing.items() if model not in listed]
    if left_out:
        raise ValueError(
            "\n".join(
                f"the list leaves out {manifest.model}, which has {manifest.path}; list every checkpoint that has a "
                "manifest, so that its group is worked out with the others"
                for manifest in left_out
            )
        )


def is_recorded(manifest: Manifest) -> bool:
    """Whether fixtures were recorded under this manifest's slug: ``record`` writes a ``sets.toml`` with every set."""
    return (manifest.path.parent / set_tables.FILE).is_file()


def directories(checkpoints: list[Checkpoint], existing: dict[str, Manifest], fixtures: Path) -> dict[str, Path]:
    """Each checkpoint's manifest directory: its existing one, else its own slug's, which no other checkpoint holds."""
    holders = {manifest.path.parent: model for model, manifest in existing.items()}
    found: dict[str, Path] = {}
    for checkpoint in checkpoints:
        manifest = existing.get(checkpoint.model)
        directory = manifest.path.parent if manifest else fixtures / slug_for(checkpoint.model)
        if not directory.name:
            raise ValueError(f"{checkpoint.model} has no slug: its last path segment holds no letter or digit")
        holder = holders.setdefault(directory, checkpoint.model)
        if holder != checkpoint.model:
            raise ValueError(f"{checkpoint.model} and {holder} both take the slug {directory.name}")
        found[checkpoint.model] = directory
    return found


def check_revisions(checkpoints: list[Checkpoint], existing: dict[str, Manifest]) -> None:
    for checkpoint in checkpoints:
        manifest = existing.get(checkpoint.model)
        if manifest and manifest.revision != checkpoint.revision and is_recorded(manifest):
            raise ValueError(
                f"{manifest.path} pins {manifest.revision}, where its fixtures were recorded, and the list gives "
                f"{checkpoint.revision}; a recorded group moves only with a re-record, so list the pinned revision"
            )


def read_inputs(checkpoints: list[Checkpoint]) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Each checkpoint's oracle inputs, and a line for each checkpoint that could not be read."""
    found: dict[str, dict[str, str]] = {}
    unread: list[str] = []
    for checkpoint in checkpoints:
        where = f"{checkpoint.model} at {checkpoint.revision}"
        try:
            found[checkpoint.model] = oracle_inputs(checkpoint.model, checkpoint.revision)
        except Exception as err:  # every checkpoint that cannot be read is named before the run stops
            unread.append(f"{where}: {type(err).__name__}: {err}")
            continue
        if not found[checkpoint.model]:
            unread.append(f"{where}: none of the files the oracle reads")
    return found, unread


def rank(checkpoint: Checkpoint) -> tuple:
    """The order in which a group's members become its primary: most downloads first, a member whose count nobody
    read after every member with one, ties to the first model id."""
    return (checkpoint.downloads is None, -(checkpoint.downloads or 0), checkpoint.model)


def assign(
    checkpoints: list[Checkpoint],
    inputs: dict[str, dict[str, str]],
    existing: dict[str, Manifest],
    directory: dict[str, Path],
) -> list[Group]:
    """The checkpoint groups, sorted by slug: a group's members have equal oracle inputs."""
    members_of: dict[tuple[tuple[str, str], ...], list[Checkpoint]] = {}
    for checkpoint in checkpoints:
        members_of.setdefault(tuple(sorted(inputs[checkpoint.model].items())), []).append(checkpoint)
    groups = []
    for members in members_of.values():
        recorded = [c for c in members if c.model in existing and is_recorded(existing[c.model])]
        if len(recorded) > 1:
            slugs = " and ".join(sorted(directory[c.model].name for c in recorded))
            raise ValueError(
                f"the recorded groups {slugs} have equal oracle inputs; merging them retires one slug's fixture ids, "
                "which is a person's call"
            )
        ranked = sorted(members, key=rank)
        primary = recorded[0] if recorded else ranked[0]
        groups.append(Group(directory[primary.model].name, primary, [c for c in ranked if c is not primary]))
    return sorted(groups, key=lambda group: group.slug)


def with_fields(text: str, revision: str, tier: int, group: str | None, inputs: dict[str, str]) -> str:
    """A manifest's text with these fields set and every other line kept. The top-level keys go after ``revision``,
    before the first table, and ``[inputs]`` goes last."""
    lines = text.splitlines()
    first_table = next((i for i, line in enumerate(lines) if line.startswith("[")), len(lines))
    head = [line for line in lines[:first_table] if not _SET_HERE.match(line)]
    at = next((i for i, line in enumerate(head) if _REVISION.match(line)), None)
    if at is None:
        raise ValueError("the manifest has no `revision` line before its first table")
    head[at] = _REVISION.sub(lambda found: found.group(1) + _string(revision), head[at], count=1)
    head[at + 1 : at + 1] = [f"tier     = {tier}", *([f"group    = {_string(group)}"] if group else [])]
    body = head + _without_inputs(lines[first_table:])
    while body and not body[-1].strip():
        body.pop()
    width = max((len(_string(name)) for name in inputs), default=0)
    body += ["", INPUTS_HEADER, *(f"{_string(name):<{width}} = {_string(inputs[name])}" for name in sorted(inputs))]
    return "\n".join(body) + "\n"


def _without_inputs(lines: list[str]) -> list[str]:
    kept: list[str] = []
    skipping = False
    for line in lines:
        if line.startswith("["):
            skipping = line.split("#")[0].strip() == "[inputs]"
        if not skipping:
            kept.append(line)
    return kept


def _string(value: str) -> str:
    """A TOML basic string: JSON's escapes are TOML's."""
    return json.dumps(value, ensure_ascii=False)


def write_manifests(groups: list[Group], inputs: dict[str, dict[str, str]], directory: dict[str, Path]) -> None:
    for group in groups:
        for checkpoint in (group.primary, *group.others):
            path = directory[checkpoint.model] / "manifest.toml"
            if path.is_file():
                text = path.read_text(encoding="utf-8")
            else:
                text = f"model    = {_string(checkpoint.model)}\nrevision = {_string(checkpoint.revision)}\n"
            named = None if checkpoint is group.primary else group.slug
            fields = (checkpoint.revision, checkpoint.tier, named, inputs[checkpoint.model])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(with_fields(text, *fields), encoding="utf-8")
            written = load_manifest(path)
            if (written.revision, written.tier, written.group, written.inputs) != fields:
                raise RuntimeError(f"{path}: the manifest does not read back as written")


def describe(groups: list[Group]) -> str:
    lines = [
        f"{group.slug}: " + ", ".join(f"{c.model} (tier {c.tier})" for c in (group.primary, *group.others))
        for group in groups
    ]
    checkpoints = sum(1 + len(group.others) for group in groups)
    lines.append(f"{checkpoints} checkpoints in {len(groups)} checkpoint groups")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    try:
        checkpoints = read_list(args.models or args.fixtures / LIST)
        existing = existing_manifests(args.fixtures)
        check_listed(checkpoints, existing)
        directory = directories(checkpoints, existing, args.fixtures)
        check_revisions(checkpoints, existing)
        inputs, unread = read_inputs(checkpoints)
        if unread:
            raise ValueError("\n".join(f"cannot read the oracle inputs of {line}" for line in unread))
        groups = assign(checkpoints, inputs, existing, directory)
    except (OSError, ValueError) as err:
        for line in str(err).splitlines():
            print(f"bellwether manifests: {line}", file=sys.stderr)
        print("bellwether manifests: nothing was written", file=sys.stderr)
        return 1
    write_manifests(groups, inputs, directory)
    sys.stdout.write(describe(groups))
    return 0
