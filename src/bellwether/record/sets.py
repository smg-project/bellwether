"""``sets.toml``: one table per recorded fixture set, beside the sets in ``fixtures/<slug>/``.

Each ``[<kind>.<set>]`` table says how the set is stored (``form``: ``plain`` or ``zstd``), how many cases were
recorded and rejected, and the size and sha256 of its plain content. ``count`` and reviewers read it without fetching
a compressed set, and a re-record is checked against the plain content's hash, never the compressor's bytes.
Written by ``bellwether record``, never by hand.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import tomllib
from pathlib import Path

FILE = "sets.toml"
FIELDS = ("form", "cases", "rejected", "plain_bytes", "plain_sha256")
_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def entry(form: str, plain: str, cases: int, rejected: int) -> dict:
    data = plain.encode("utf-8")
    return {
        "form": form,
        "cases": cases,
        "rejected": rejected,
        "plain_bytes": len(data),
        "plain_sha256": hashlib.sha256(data).hexdigest(),
    }


def read(path: Path) -> dict[tuple[str, str], dict]:
    """``(kind, set) -> table`` from a ``sets.toml``; empty when there is none."""
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {(kind, name): table for kind, sets in data.items() for name, table in sets.items()}


def update(path: Path, kind: str, tables: dict[str, dict | None]) -> None:
    """Put one run's ``kind`` tables, by set, into the file; a set given ``None`` loses its ``kind`` table. Every other
    table stays as the file holds it now, not as it was when the run started, since a run of the other kind for the
    same model may have written it in between. The read and the write hold an exclusive lock on ``sets.toml.lock``."""
    # fcntl.flock: bellwether runs on Linux and macOS. The lock file stays: if a run removed it, a run waiting on the
    # removed file and a run that created a new one would both hold a lock. .gitignore keeps it out of commits.
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # released when the file closes
        merged = read(path)
        for name, table in tables.items():
            if table is None:
                merged.pop((kind, name), None)
            else:
                merged[(kind, name)] = table
        write(path, merged)


def write(path: Path, tables: dict[tuple[str, str], dict]) -> None:
    """Write the tables sorted, fields in a fixed order; remove the file when there are none. The text goes to a
    partial file that then replaces the table whole, so no reader finds half of one."""
    if not tables:
        path.unlink(missing_ok=True)
        return
    lines: list[str] = []
    for kind, name in sorted(tables):
        for key in (kind, name):
            if not _BARE_KEY.match(key):
                raise ValueError(f"{path}: {key!r} cannot be a table name in sets.toml")
        lines.append(f"[{kind}.{name}]")
        table = tables[(kind, name)]
        for field in FIELDS:
            value = table[field]
            lines.append(f'{field} = "{value}"' if isinstance(value, str) else f"{field} = {value}")
        lines.append("")
    partial = path.with_name(path.name + ".partial")
    partial.write_text("\n".join(lines), encoding="utf-8")
    os.replace(partial, path)
