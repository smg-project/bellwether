"""Per-model manifests: the pinned revision, the checkpoint group, the authority order and the parser names.

One ``fixtures/<slug>/manifest.toml`` per checkpoint. The slug is the lowercase last path segment of
the Hugging Face id with anything but letters, digits, dots and dashes turned into a dash.

Checkpoints whose oracle inputs are equal render and parse identically, so they form one checkpoint group and are
recorded once (docs/benchmark-sets.md, "Which models"). ``[inputs]`` lists each oracle input with its sha256
(``bellwether.inputs``). The group's primary holds its fixture sets; every other member's manifest names the primary's
slug as its ``group`` and holds none. ``tier`` is the checkpoint's place in the recording order, 1 to 3, and belongs
to the checkpoint, not the group, since a group's members can sit in different tiers. ``bellwether manifests`` writes
all three. A manifest from before groups has none of them and still loads, so that the command can add them;
``record`` refuses it until it lists its inputs.

``[smg]`` and ``[engines]`` name the parsers a group's fixtures exercise, which ``gaps`` reads. They are optional: a
new group's parsers are not known until someone maps them, and a member names none, since its fixtures are its
group's.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("render", "parse", "tokenize", "detokenize")
TIERS = (1, 2, 3)
LOCAL_REVISION = "local"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SLUG = re.compile(r"[a-z0-9.-]+")


@dataclass
class Manifest:
    slug: str
    path: Path
    model: str
    revision: str
    authority: dict[str, list[str]]
    smg: dict[str, str]
    engines: dict[str, dict[str, str]]
    inputs: dict[str, str] = field(default_factory=dict)
    group: str | None = None  # the primary's slug; None for the primary itself
    tier: int | None = None

    @property
    def group_slug(self) -> str:
        """The slug the checkpoint's fixtures live under: its group's, or its own for a primary."""
        return self.group or self.slug


def slug_for(model: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", model.split("/")[-1].lower()).strip("-")


def is_pinned(model: str, revision: str) -> bool:
    """Whether ``revision`` pins ``model``: a full commit hash, never a branch or tag, which can move under a consumer
    that caches fixtures as if they were pinned. A checkpoint given as an absolute path, a directory as the tests use,
    has no commits; its revision is ``local``, which a Hub id, never an absolute path, cannot take."""
    return bool(_COMMIT.fullmatch(revision)) or (revision == LOCAL_REVISION and Path(model).is_absolute())


def load_manifest(path: Path) -> Manifest:
    try:
        data = tomllib.loads(path.read_text())
    except ValueError as err:  # not TOML, or not text: the message does not say which file
        raise ValueError(f"{path}: {err}") from None
    for key in ("model", "revision"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"{path}: `{key}` is required")
    if not is_pinned(data["model"], data["revision"]):
        raise ValueError(
            f"{path}: revision {data['revision']!r} is not a commit: pin the 40-character lowercase hexadecimal "
            "commit hash, since a branch or tag can move"
        )
    authority = data.get("authority", {})
    for kind, sources in authority.items():
        if kind not in KINDS:
            raise ValueError(f"{path}: unknown kind `{kind}` under [authority]")
        if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
            raise ValueError(f"{path}: authority.{kind} must be a list of sources")
    return Manifest(
        slug=path.parent.name,
        path=path,
        model=data["model"],
        revision=data["revision"],
        authority={k: list(v) for k, v in authority.items()},
        smg={k: v for k, v in data.get("smg", {}).items() if isinstance(v, str)},
        engines={e: dict(v) for e, v in data.get("engines", {}).items() if isinstance(v, dict)},
        inputs=_inputs(path, data.get("inputs", {})),
        group=_group(path, data.get("group")),
        tier=_tier(path, data.get("tier")),
    )


def _inputs(path: Path, inputs: object) -> dict[str, str]:
    if not isinstance(inputs, dict) or not all(
        isinstance(digest, str) and _SHA256.fullmatch(digest) for digest in inputs.values()
    ):
        raise ValueError(f"{path}: [inputs] must give each file's sha256, 64 lowercase hexadecimal digits")
    return dict(inputs)


def _group(path: Path, group: object) -> str | None:
    if group is None:
        return None
    if not isinstance(group, str) or not _SLUG.fullmatch(group):
        raise ValueError(f"{path}: `group` must be the slug of its group's primary")
    if group == path.parent.name:
        raise ValueError(f"{path}: `group` names this manifest's own directory; a group's primary has no `group`")
    return group


def _tier(path: Path, tier: object) -> int | None:
    # A TOML boolean loads as a Python bool, which is an int; a tier is never one.
    if tier is not None and (type(tier) is not int or tier not in TIERS):
        raise ValueError(f"{path}: `tier` must be 1, 2 or 3")
    return tier


def find_manifest(fixtures_dir: Path, model: str) -> Manifest:
    """The manifest whose ``model`` is ``model``; models are matched exactly, never by slug."""
    seen = []
    for path in sorted(fixtures_dir.glob("*/manifest.toml")):
        manifest = load_manifest(path)
        if manifest.model == model:
            return manifest
        seen.append(manifest.model)
    known = ", ".join(seen) or "none"
    raise FileNotFoundError(f"no manifest under {fixtures_dir} for model {model!r}; manifests exist for: {known}")
