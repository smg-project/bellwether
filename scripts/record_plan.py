"""The checkpoint groups a ``record`` workflow run records, as the job matrix: every ``fixtures/<slug>/manifest.toml``,
or the slugs named, each as its slug, model and revision.

    python3 scripts/record_plan.py [<slug> ...] >> "$GITHUB_OUTPUT"

A member's manifest (one with ``group``) is not a job: the recorder records a group once, with its primary, and
refuses the member. A group whose oracle inputs hold the vendor's code (a ``.py`` file, Kimi-K3's) is left out and
named on stderr: it is recorded in the sandbox (``bellwether sandbox-record``), not on a runner. A slug with no
manifest, or a member's, stops the run. The script imports nothing beyond the standard library, so it runs before the
project's environment exists.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def groups(root: Path, wanted: list[str]) -> tuple[list[dict], list[str]]:
    """The groups to record, sorted by slug, and the slugs left to the sandbox."""
    found: list[dict] = []
    vendor: list[str] = []
    members: dict[str, str] = {}
    for manifest in sorted(root.glob("fixtures/*/manifest.toml")):
        slug = manifest.parent.name
        if wanted and slug not in wanted:
            continue
        data = tomllib.loads(manifest.read_text("utf-8"))
        if "group" in data:
            members[slug] = data["group"]
            continue
        if any(name.endswith(".py") for name in data.get("inputs", {})):
            vendor.append(slug)
            continue
        found.append({"slug": slug, "model": data["model"], "revision": data["revision"]})
    if wanted and members:
        named = (f"{slug} is recorded with its group's primary, {primary}" for slug, primary in members.items())
        raise SystemExit("; ".join(named))
    unknown = sorted(set(wanted) - {group["slug"] for group in found} - set(vendor))
    if unknown:
        raise SystemExit(f"no manifest for {', '.join(unknown)}")
    return found, vendor


def main() -> int:
    found, vendor = groups(ROOT, " ".join(sys.argv[1:]).split())
    for slug in vendor:
        print(f"{slug}: the vendor's code is among its oracle inputs; use `bellwether sandbox-record`", file=sys.stderr)
    print(f"groups={json.dumps(found, separators=(',', ':'))}")
    print(f"count={len(found)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
