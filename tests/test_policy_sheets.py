"""The recorded policy sheets under ``policy/`` (policy/README.md).

Nothing in this repository reads them: SMG's policy controller replays them. The table in the README is then the only
statement of what each sheet is, so this holds it to the files: every sheet has a row and every row a sheet and a
recorder, and a row's commit, row counts and sha256 are the file's.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY = ROOT / "policy"
ROW = re.compile(
    r"^\| `(?P<sheet>[a-z_]+\.json)` \| [^|]+ \| `(?P<commit>[0-9a-f]{40})` "
    r"\| (?P<rows>\d+): (?P<sequence>\d+) sequence, (?P<drawn>\d+) drawn "
    r"\| `(?P<sha256>[0-9a-f]{64})` \| `(?P<recorder>[a-z_]+\.rs)` \|$"
)


def table() -> dict[str, dict[str, str]]:
    lines = (POLICY / "README.md").read_text().splitlines()
    return {row["sheet"]: row.groupdict() for line in lines if (row := ROW.match(line))}


def test_every_sheet_has_a_row_and_every_row_a_sheet_and_a_recorder():
    rows = table()
    assert sorted(rows) == sorted(path.name for path in POLICY.glob("*.json"))
    for name, row in rows.items():
        assert (POLICY / row["recorder"]).is_file(), f"{name}: no recorder {row['recorder']}"


def test_a_row_states_its_sheets_commit_row_counts_and_sha256():
    for name, row in table().items():
        data = (POLICY / name).read_bytes()
        sheet = json.loads(data)
        names = [recorded["name"] for recorded in sheet["rows"]]
        drawn = sum("counts" in recorded["outcome"] for recorded in sheet["rows"])
        assert len(set(names)) == len(names), f"{name}: a row name repeats"
        assert sheet["recorded_from"] == row["commit"], name
        stated = (int(row["rows"]), int(row["sequence"]), int(row["drawn"]))
        assert (len(names), len(names) - drawn, drawn) == stated, name
        assert hashlib.sha256(data).hexdigest() == row["sha256"], name
