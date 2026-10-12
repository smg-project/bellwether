"""The benchmark sets a pull request changes, cut into shards that a runner each fetches and checks on its own.

    python3 scripts/set_shards.py <base ref> --matrix      # the shards' indices as JSON, [0, 1, ...], or [] for none
    python3 scripts/set_shards.py <base ref> --shard <i>   # shard i's set files, one a line

A set is checked whole (``test_committed_fixtures_are_canonical_sorted_and_valid``), about ten minutes a compressed
gigabyte, and a runner fetches it into the working tree and the Git LFS store both, so one runner takes a few gigabytes
of sets in an hour. The sets are taken in path order, a set's size is the one its Git LFS pointer names, and a shard
closes before the set that would take it past ``--budget`` bytes; a set past the budget alone is a shard of its own.
Every job computes the same cut from the same commit, so no list of paths travels between jobs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

BUDGET = 5_000_000_000
SIZE = re.compile(r"^size (\d+)$", re.M)


def changed(base: str) -> list[str]:
    """The compressed benchmark sets the commits since ``base`` add or change, in path order."""
    out = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=d", f"{base}...HEAD", "--", "fixtures/**/*.jsonl.zst"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(out.split())


def size(path: Path) -> int:
    """The size the set's Git LFS pointer names."""
    match = SIZE.search(path.read_text())
    if match is None:
        raise SystemExit(f"{path} is not a Git LFS pointer")
    return int(match.group(1))


def shards(sized: list[tuple[str, int]], budget: int) -> list[list[str]]:
    """The sets in order, cut into shards of at most ``budget`` bytes each, a larger set alone."""
    cut: list[list[str]] = []
    current: list[str] = []
    total = 0
    for path, bytes_ in sized:
        if current and total + bytes_ > budget:
            cut.append(current)
            current, total = [], 0
        current.append(path)
        total += bytes_
    if current:
        cut.append(current)
    return cut


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("base")
    parser.add_argument("--budget", type=int, default=BUDGET)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--matrix", action="store_true")
    which.add_argument("--shard", type=int)
    args = parser.parse_args()
    cut = shards([(path, size(Path(path))) for path in changed(args.base)], args.budget)
    if args.matrix:
        print(json.dumps(list(range(len(cut)))))
    elif 0 <= args.shard < len(cut):
        print("\n".join(cut[args.shard]))
    else:
        raise SystemExit(f"no shard {args.shard}: the pull request's sets make {len(cut)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
