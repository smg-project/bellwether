"""Fixture files: JSON Lines, one case per line, sorted by id, canonical, schema-checked on write."""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from pathlib import Path

from jsonschema import Draft202012Validator


def schema_path() -> Path:
    """``case.schema.json``, packaged with the module so an installed wheel finds it too."""
    return Path(str(resources.files("bellwether") / "schemas" / "case.schema.json"))


@cache
def validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(schema_path().read_text()))


def canonical_line(case: dict) -> str:
    return json.dumps(case, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def read_fixture_file(path: Path) -> dict[str, dict]:
    cases: dict[str, dict] = {}
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        if raw.strip():
            case = json.loads(raw)
            if case["id"] in cases:
                raise ValueError(f"{path}:{number}: duplicate id {case['id']}")
            cases[case["id"]] = case
    return cases


def write_fixture_file(path: Path, cases: dict[str, dict]) -> None:
    lines = []
    for case_id in sorted(cases):
        case = cases[case_id]
        errors = sorted(validator().iter_errors(case), key=lambda e: list(e.path))
        if errors:
            where = "/".join(str(p) for p in errors[0].path) or "(root)"
            raise ValueError(f"{case_id}: does not match the case schema at {where}: {errors[0].message}")
        lines.append(canonical_line(case))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines))
