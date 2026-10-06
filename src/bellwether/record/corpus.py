"""Request corpora: JSON Lines files of cases, shared across models or specific to one.

``corpus/<kind>/<set>.jsonl`` holds cases every model records; ``corpus/<kind>/<slug>/<set>.jsonl``
adds cases for one model to the set of the same name. A line is ``{"name", "request", "notes"}``,
plus ``"message"`` for a parse case (the assistant message the output must parse to) and ``"origin"``
for an imported one (the dataset, file and row it came from). ``name`` is a
lowercase slug that becomes the last part of the fixture id, so it is unique across every set of a
kind.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_NAME = re.compile(r"^[a-z0-9-]+$")


@dataclass
class Case:
    name: str
    request: dict
    notes: str = ""
    message: dict | None = None
    origin: dict | None = None
    source: Path | None = field(default=None, compare=False)


def read_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    names: set[str] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as err:
            raise ValueError(f"{path}:{number}: not JSON: {err}") from err
        name = data.get("name")
        if not isinstance(name, str) or not _NAME.match(name):
            raise ValueError(f"{path}:{number}: `name` must be a lowercase slug, got {name!r}")
        if name in names:
            raise ValueError(f"{path}:{number}: duplicate case name {name!r}")
        if not isinstance(data.get("request"), dict):
            raise ValueError(f"{path}:{number}: `request` must be an object")
        message = data.get("message")
        if message is not None and not isinstance(message, dict):
            raise ValueError(f"{path}:{number}: `message` must be an object")
        origin = data.get("origin")
        if origin is not None and not isinstance(origin, dict):
            raise ValueError(f"{path}:{number}: `origin` must be an object")
        names.add(name)
        cases.append(Case(name, data["request"], str(data.get("notes", "")), message, origin, path))
    return cases


def load_corpus(corpus_dir: Path, kind: str, slug: str) -> dict[str, list[Case]]:
    """Case sets for ``kind``: the shared files, then the model's own, merged by set name.

    The fixture id carries the case name but not the set name, so a name used in two files of the
    same kind, whether two sets or a shared set and a model's addition to it, is an error.
    """
    sets: dict[str, list[Case]] = {}
    owner: dict[str, Path] = {}
    for directory in (corpus_dir / kind, corpus_dir / kind / slug):
        if not directory.is_dir():
            continue
        # iterdir raises when a directory that exists cannot be read; glob would return nothing,
        # and the run would then rebuild the fixtures from an incomplete corpus.
        for path in sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == ".jsonl"):
            cases = read_cases(path)
            for case in cases:
                if case.name in owner:
                    raise ValueError(f"{path}: case name {case.name!r} is already used in {owner[case.name]}")
                owner[case.name] = path
            sets.setdefault(path.stem, []).extend(cases)
    return sets
