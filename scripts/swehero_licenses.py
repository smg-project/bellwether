"""Build the table of license files that ``bellwether import swehero`` copies next to its sets.

    uv run python scripts/swehero_licenses.py [--check]

One copy per repository: for every repository among the rows the importer keeps, its license file and its NOTICE file,
if any, at the root of its tree at one commit. The commit is the one the repository's first kept row (in shard and row
order) names: the commit in its ``instance_id`` (``owner__repo-<commit>``), or, where the id names a pull request
(``owner__repo-<number>``, as SWE-rebench's and SWE-Gym's do), that pull request's base commit on GitHub; when the pull
request is gone, the repository's next rows are tried, up to three. When that commit has no license file at its root,
the base of the newest pull request among the repository's rows is read instead, and its license file is taken for the
earlier rows too: tornado's LICENSE is from 2013-08, and its first row's commit from 2013-01. A license file is
``LICENSE``, ``LICENCE`` or
``COPYING``, bare or with ``.txt``, ``.md`` or ``.rst``, or with a license's name after a dash, dot or underscore
(``LICENSE-MIT``, ``LICENSE-APACHE``); a symbolic link is followed to the file it names. A NOTICE file is ``NOTICE``,
bare or with one of those extensions. Each file is pinned by repository, commit, path and sha256 and named for them.

The card says the rows are under MIT, Apache-2.0, BSD-2-Clause or BSD-3-Clause. The license file decides, not the
row's label: a repository whose file at the commit is none of the four (a GNU license, CC0, a license of its own) is
left out, as is one whose commit or license file cannot be read; ``left_out`` gives each its reason, and the importer
names its rows. Code under the Apache License goes with a full text of it (Apache-2.0 4(a)): where the repository's
file only points to the License, the table joins the full text another kept repository ships (``FULL_TEXT``).

Reads go through GitHub's API (``gh api``) for pull requests and trees, and ``raw.githubusercontent.com`` for files,
into the importers' cache (``github.fetch`` reads them from there). Reading the rows needs the pinned shards in the
importers' cache, ``~/.cache/bellwether/datasets`` (``HF_HUB_OFFLINE=1`` reads them offline). The table is written to
``src/bellwether/importers/swehero_licenses.json``; ``--check`` compares instead, and exits 1 on a difference.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path, PurePosixPath

from bellwether.importers import hf, pinned, swehero

TABLE = Path(__file__).resolve().parent.parent / "src" / "bellwether" / "importers" / "swehero_licenses.json"
LICENSE_NAME = re.compile(
    r"^(licen[cs]e|copying)((\.(txt|md|rst))|([-_.](mit|apache|apache2|apache-2\.0|bsd)(\.(txt|md|rst))?))?$", re.I
)
NOTICE_NAME = re.compile(r"^notice(\.(txt|md|rst))?$", re.I)
# The repository's own license file, as against one named for a license (LICENSE-MIT) or for vendored code.
MAIN_LICENSE_NAME = re.compile(r"^(licen[cs]e|copying)(\.(txt|md|rst))?$", re.I)
CARD_LICENSES = swehero.REPOSITORY_LICENSES
# The full text of the Apache License 2.0 joined to a repository whose file only points to it: the file most kept
# repositories ship byte for byte (86 of them at the pinned commits), pinned where the first of them holds it.
FULL_TEXT = {
    "repository": "20c/ctl",
    "commit": "879af37647e61767a1ede59ffd353e4cfd27cd6f",
    "path": "LICENSE",
    "sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}
TRIES = 3
NO_LICENSE_FILE = "no license file at the root"


def license_kind(text: str) -> str:
    """The license a file states, by its words: one of the card's four, or what it is instead."""
    words = " ".join(text.split()).lower()
    for header, kind in (
        ("gnu affero general public license", "AGPL"),
        ("gnu lesser general public license", "LGPL"),
        ("gnu library general public license", "LGPL"),
        ("gnu general public license", "GPL"),
        ("mozilla public license", "MPL"),
        ("cc0 1.0 universal", "CC0"),
    ):
        if words.startswith(header):
            return kind
    if "apache license" in words and "2.0" in words:
        return "Apache-2.0"
    if "permission is hereby granted, free of charge" in words:
        return "MIT"
    if "redistribution and use in source and binary forms" in words:
        return "BSD-3-Clause" if "endorse or promote" in words else "BSD-2-Clause"
    if words.startswith("isc license") or "permission to use, copy, modify, and/or distribute this software" in words:
        return "ISC"
    return "unknown"


def is_full_apache_text(text: str) -> bool:
    words = " ".join(re.sub(r"[_*#=<>`]", " ", text).split()).lower()
    return words.startswith("apache license version 2.0, january 2004") and "terms and conditions for use" in words


def gh(path: str, cache: Path) -> dict | None:
    """``gh api path``, or None when GitHub says the thing is not there; a rate limit is waited out.

    Each answer is kept under ``cache/github-api/``: a pull request's base commit and a commit's tree do not change,
    so a run cut short, by a rate limit or otherwise, resumes without asking again."""
    kept = cache / "github-api" / f"{path}.json"
    if kept.is_file():
        found = json.loads(kept.read_text())
        return None if found == {"bellwether": "not on GitHub"} else found
    for attempt in range(5):
        done = subprocess.run(["gh", "api", path], capture_output=True, text=True)
        gone = any(word in done.stderr for word in ("Not Found", "No commit found", "422"))
        if done.returncode == 0 or gone:
            found = json.loads(done.stdout) if done.returncode == 0 else None
            kept.parent.mkdir(parents=True, exist_ok=True)
            kept.write_text(json.dumps(found if found is not None else {"bellwether": "not on GitHub"}))
            return found
        time.sleep(60 * (attempt + 1))
    raise SystemExit(f"gh api {path}: {done.stderr.strip()}")


def raw(repository: str, commit: str, path: str, cache: Path) -> bytes:
    """The bytes of ``path`` at ``commit``, read once into the cache ``github.fetch`` reads."""
    target = cache / "github" / repository / commit / path
    if not target.is_file():
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repository}/{commit}/{path}", timeout=60) as r:
            data = r.read()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return target.read_bytes()


def first_refs() -> dict[str, dict]:
    """``repository -> {"label", "refs", "newest"}``: the label, up to ``TRIES`` refs of its kept rows in shard and row
    order, and the number of the newest pull request any of its rows names (or ``None``). A ref is
    ``("commit", sha)`` or ``("pull", number)``."""
    found: dict[str, dict] = {}
    for name in swehero.SHARD_FILES:
        path = hf.fetch(swehero.REPO, swehero.REVISION, name, swehero.FILES[name])
        for row in swehero.read_rows(path):
            if row["license"] not in CARD_LICENSES:
                continue
            entry = found.setdefault(row["repo"], {"label": row["license"], "refs": [], "newest": None})
            suffix = row["instance_id"].removeprefix(row["repo"].replace("/", "__") + "-")
            ref = ("commit", suffix) if pinned.COMMIT_ID.fullmatch(suffix) else ("pull", suffix)
            if ref[0] == "pull" and not suffix.isdigit():
                ref = ("none", row["instance_id"])
            if ref not in entry["refs"] and len(entry["refs"]) < TRIES:
                entry["refs"].append(ref)
            if ref[0] == "pull" and (entry["newest"] is None or int(suffix) > int(entry["newest"])):
                entry["newest"] = suffix
    return found


def survey(repository: str, refs: list[tuple[str, str]], cache: Path, newest: str | None = None) -> dict:
    """What the repository's first readable ref holds at its root: its commit, files and license, or why none.

    When that commit has no license file at its root, the base of ``newest``, the repository's newest pull request among
    its rows, is read instead: a license file the repository added later is taken for its earlier rows too."""
    reasons = []
    for kind, value in refs:
        if kind == "none":
            reasons.append(f"{value} names neither a commit nor a pull request")
            continue
        commit = value
        if kind == "pull":
            pull = gh(f"repos/{repository}/pulls/{value}", cache)
            if pull is None:
                reasons.append(f"pull request {value} is not on GitHub")
                continue
            commit = pull["base"]["sha"]
        found = at_commit(repository, commit, f"{kind} {value}", cache)
        if found is None:
            reasons.append(f"commit {commit[:12]} is not on GitHub")
            continue
        if found.get("left_out") == NO_LICENSE_FILE and newest is not None and (kind, value) != ("pull", newest):
            pull = gh(f"repos/{repository}/pulls/{newest}", cache)
            later = pull and at_commit(repository, pull["base"]["sha"], f"pull {newest}", cache)
            if later and later.get("left_out") != NO_LICENSE_FILE:
                return later
        return found
    return {"left_out": "; ".join(reasons) or "no row names a commit or a pull request"}


def at_commit(repository: str, commit: str, origin: str, cache: Path) -> dict | None:
    """The repository's license and NOTICE files at the root of ``commit`` and the license they state, or why the
    repository is left out; ``None`` when GitHub does not have the commit. ``origin`` names the row's ref."""
    tree = gh(f"repos/{repository}/git/trees/{commit}", cache)
    if tree is None:
        return None
    entries = {entry["path"]: entry for entry in tree["tree"] if entry["type"] == "blob"}
    files = []
    for path in sorted(entries):
        role = "license" if LICENSE_NAME.match(path) else "notice" if NOTICE_NAME.match(path) else None
        if role is None:
            continue
        target = path
        if entries[path]["mode"] == "120000":  # a symbolic link: the file it names
            target = str(PurePosixPath(raw(repository, commit, path, cache).decode().strip()))
        data = raw(repository, commit, target, cache)
        files.append({"role": role, "path": target, "sha256": hashlib.sha256(data).hexdigest()})
    licenses = [f for f in files if f["role"] == "license"]
    if not licenses:
        return {"commit": commit, "from": origin, "left_out": NO_LICENSE_FILE}
    # The repository's own license file decides; others, often for vendored code, are copied beside it. With no
    # such file, the files named for licenses decide together (LICENSE-MIT and LICENSE-APACHE, a dual license).
    deciding = [f for f in licenses if MAIN_LICENSE_NAME.match(f["path"])] or licenses
    texts = [raw(repository, commit, f["path"], cache).decode("utf-8", "replace") for f in deciding]
    kinds = sorted({license_kind(text) for text in texts} - {"unknown"}) or ["unknown"]
    outside = [k for k in kinds if k not in CARD_LICENSES]
    found = {"commit": commit, "from": origin, "files": files, "license": " AND ".join(kinds)}
    if outside:
        found["left_out"] = f"its license file at {commit[:12]} is {', '.join(outside)}, not one of the card's four"
    elif "Apache-2.0" in kinds and not any(is_full_apache_text(text) for text in texts):
        found["full_text"] = True
    return found


def name_of(repository: str, commit: str, path: str) -> str:
    return f"swehero-{repository.replace('/', '-')}-{commit[:12]}-{path.replace('/', '-')}"


def build(cache: Path) -> dict:
    refs = first_refs()
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        surveys = pool.map(lambda r: survey(r, refs[r]["refs"], cache, refs[r]["newest"]), sorted(refs))
        surveyed = dict(zip(sorted(refs), surveys, strict=True))
    full_text = name_of(FULL_TEXT["repository"], FULL_TEXT["commit"], FULL_TEXT["path"])
    files, repositories, left_out = {}, {}, {}
    for repository, found in surveyed.items():
        if "left_out" in found:
            left_out[repository] = found["left_out"]
            continue
        names = []
        for f in found["files"]:
            name = name_of(repository, found["commit"], f["path"])
            pin = {"repository": repository, "commit": found["commit"], "path": f["path"], "sha256": f["sha256"]}
            files[name] = pin
            names.append(name)
        if found.get("full_text"):
            names.append(full_text)
        repositories[repository] = {
            "commit": found["commit"],
            "from": found["from"],
            "license": found["license"],
            "files": names,
        }
    if full_text not in files:
        raise SystemExit(f"{full_text}: the pinned full text of the Apache License is not among the kept files")
    if files[full_text]["sha256"] != FULL_TEXT["sha256"]:
        raise SystemExit(f"{full_text}: its sha256 is not the pinned one")
    lower = [name.lower() for name in files]
    if len(set(lower)) != len(lower):
        raise SystemExit("two copies' names differ only in case")
    return {
        "files": dict(sorted(files.items())),
        "repositories": dict(sorted(repositories.items())),
        "left_out": dict(sorted(left_out.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="compare with the committed table; exit 1 on a difference")
    parser.add_argument("--cache", type=Path, default=pinned.CACHE)
    args = parser.parse_args()
    table = build(args.cache)
    text = json.dumps(table, indent=1, ensure_ascii=False) + "\n"
    if args.check:
        if TABLE.read_text("utf-8") != text:
            print(f"{TABLE}: differs from a fresh build", file=sys.stderr)
            return 1
        print(f"{TABLE}: equals a fresh build")
        return 0
    TABLE.write_text(text, "utf-8")
    kept, out = len(table["repositories"]), len(table["left_out"])
    print(f"{TABLE}: {len(table['files'])} files for {kept} repositories; {out} left out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
