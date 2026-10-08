"""Record the named corpus sources for one checkpoint group, as a ``record`` workflow job does on a runner.

    uv run python scripts/record_sources.py <slug> [--sources glaive-v2 tau2 swehero] [--jobs 3]

1. **The checkpoint's files.** The oracle inputs are read once with network access, as the recorder reads them
   (``bellwether.inputs.checkpoint_dir``), which leaves them in the Hugging Face cache at the manifest's revision with
   the commit's file list beside them; the recorder then runs offline (``HF_HUB_OFFLINE=1``), and every call reads the
   same snapshot.
2. **The calls.** ``bellwether record`` runs for each source and kind, on a few sets at a time, and ``--jobs`` calls run
   at once, the largest first, so the job does not end on one large call running alone. Each call reads a corpus
   directory of its own that links only its sets, and names them with ``--set``: ``load_corpus`` reads every set file
   in the directory it is given, so a call against the whole corpus would hold all of it (17-20 GB), and ``--set``
   leaves the group's other fixture sets and their ``sets.toml`` tables as they are (the recorder locks ``sets.toml``
   while it writes its tables, so calls at once are safe). ``SETS_PER_CALL`` keeps each call's corpus under a gigabyte
   or so of plain JSON Lines, and a recorder near 3 GB.
3. **Resuming.** A set ``sets.toml`` already lists for the kind is not recorded again, so a job that is run again picks
   up where the last one stopped.

A call that refuses cases ends with status 1 and is not a failure. One that fails is: status 1 with no case refused
(an error the recorder reports and stops on, such as a checkpoint it cannot read), a traceback, or a status other than
0 and 1, such as a kill. The script then exits 1 once every other call has finished, and says how much memory the
largest recorder held.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import resource
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUFFIX = ".jsonl.zst"
# How many sets one call records: glaive-v2's 71 sets take 0.54 GB (render) and 0.70 GB (parse) as plain JSON Lines in
# all, tau2's 22 about 4.9 GB and SWE-Hero's 14 about 7.4 GB.
SETS_PER_CALL = {"glaive-v2": 71, "tau2": 4, "swehero": 1}
RECORDED = re.compile(r": (\d+) cases recorded")
PRINT = threading.Lock()


def listed(sets_toml: Path, kind: str) -> set[str]:
    """The sets of ``kind`` that ``sets.toml`` has a table for."""
    if not sets_toml.is_file():
        return set()
    return set(tomllib.loads(sets_toml.read_text("utf-8")).get(kind, {}))


def calls(corpus: Path, kind: str, source: str, per: int, done: set[str]) -> list[list[Path]]:
    """The source's set files of ``kind`` not in ``done``, in name order, cut into calls of ``per`` sets."""
    files = [
        path
        for path in sorted((corpus / kind).glob(f"{source}-*{SUFFIX}"))
        if path.name.removesuffix(SUFFIX) not in done
    ]
    return [files[start : start + per] for start in range(0, len(files), per)]


def queue(corpus: Path, sources: list[str], sets_toml: Path) -> list[tuple[str, list[Path]]]:
    """Every call the group still needs, as ``(kind, sets)``, the largest corpus first. A source with no corpus set of
    either kind is refused, so a misspelt one is not taken for one already recorded."""
    kinds = ("render", "parse")
    missing = [
        source for source in sources if not any(any((corpus / kind).glob(f"{source}-*{SUFFIX}")) for kind in kinds)
    ]
    if missing:
        raise SystemExit(f"no corpus set for {', '.join(missing)} under {corpus}")
    pending = [
        (kind, call)
        for source in sources
        for kind in kinds
        for call in calls(corpus, kind, source, SETS_PER_CALL.get(source, 1), listed(sets_toml, kind))
    ]
    return sorted(pending, key=lambda item: -sum(path.stat().st_size for path in item[1]))


@contextlib.contextmanager
def call_corpus(kind: str, call: list[Path]) -> Iterator[Path]:
    """A corpus directory that links the call's sets alone, removed afterwards (the links, not the sets)."""
    with tempfile.TemporaryDirectory(prefix="bellwether-call-") as tmp:
        directory = Path(tmp)
        (directory / kind).mkdir()
        for path in call:
            (directory / kind / path.name).symlink_to(path.resolve())
        yield directory


def outcome(status: int, stdout: str, stderr: str) -> tuple[int, int, bool]:
    """Cases recorded, cases refused, and whether the call failed: refusing cases ends a call with status 1, and so
    does an error the recorder reports and stops on, which refuses none."""
    recorded = sum(int(match.group(1)) for match in RECORDED.finditer(stdout))
    refused = sum(1 for line in stderr.splitlines() if line.startswith("not recorded "))
    crashed = status not in (0, 1) or (status == 1 and not refused) or "Traceback (most recent call last)" in stderr
    return recorded, refused, crashed


def largest_child_gb() -> float:
    """The resident set of the largest child that has ended, in GB: the number that says how many calls fit at once."""
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss  # bytes on macOS, KiB on Linux
    return peak * (1 if sys.platform == "darwin" else 1024) / 1e9


def say(message: str) -> None:
    with PRINT:
        print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def fetch_inputs(model: str, revision: str) -> None:
    """The checkpoint's oracle inputs at ``revision`` into the Hugging Face cache, with the commit's file list, which
    the recorder needs to tell a file the checkpoint does not ship from one the cache lacks when it runs offline."""
    from bellwether.inputs import checkpoint_dir

    say(f"{model} at {revision[:8]}: oracle inputs in {checkpoint_dir(model, revision)}")


def record(slug: str, model: str, kind: str, call: list[Path], fixtures: Path, failures: list[str]) -> None:
    """One ``bellwether record`` call, on a corpus directory that links its sets alone."""
    bellwether = str(Path(sys.executable).parent / "bellwether")
    names = [path.name.removesuffix(SUFFIX) for path in call]
    with call_corpus(kind, call) as directory:
        args = [bellwether, "record", "--model", model, "--kind", kind, "--oracle", "reference"]
        args += ["--fixtures", str(fixtures), "--corpus", str(directory)]
        args += [flag for name in names for flag in ("--set", name)]
        start = time.monotonic()
        env = {**os.environ, "HF_HUB_OFFLINE": "1"}
        done = subprocess.run(args, capture_output=True, text=True, env=env)
    recorded, refused, crashed = outcome(done.returncode, done.stdout, done.stderr)
    span = f"{names[0]}..{names[-1]}" if len(names) > 1 else names[0]
    say(
        f"{slug} {kind} {span}: {recorded} recorded, {refused} refused, status {done.returncode}, "
        f"{time.monotonic() - start:.0f} s"
    )
    if crashed:
        failures.append(f"{slug} {kind} {span}")
        say(f"{slug} {kind} {span} crashed:\n{done.stderr[-4000:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("slug")
    parser.add_argument("--sources", nargs="+", default=["glaive-v2", "tau2", "swehero"])
    parser.add_argument("--fixtures", type=Path, default=ROOT / "fixtures")
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus")
    parser.add_argument("--jobs", type=int, default=3, help="calls at once (default: %(default)s)")
    args = parser.parse_args()
    manifest = tomllib.loads((args.fixtures / args.slug / "manifest.toml").read_text("utf-8"))
    model, revision = manifest["model"], manifest["revision"]
    fetch_inputs(model, revision)
    pending = queue(args.corpus, args.sources, args.fixtures / args.slug / "sets.toml")
    say(f"{args.slug}: {len(pending)} call(s), {args.jobs} at once")
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(record, args.slug, model, kind, call, args.fixtures, failures) for kind, call in pending]
    for future in futures:
        future.result()  # raises what this script, not the recorder, ran into, once every call has finished
    say(f"{args.slug}: the largest recorder held {largest_child_gb():.1f} GB")
    if failures:
        say(f"{len(failures)} call(s) crashed: {'; '.join(failures)}")
        return 1
    say(f"{args.slug}: every call of {', '.join(args.sources)} finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
