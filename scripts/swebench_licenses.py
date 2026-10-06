"""Build the table of license files that ``bellwether import swebench`` copies next to its sets.

    uv run python scripts/swebench_licenses.py <directory for bare clones> [--check]

For every row the importer reads, the license file and the NOTICE file at the root of the row's repository at the
row's base commit: ``LICENSE``, ``LICENSE.md``, ``LICENSE.rst``, ``LICENSE.txt`` or ``COPYING``, or ``LICENSE/LICENSE``
where ``LICENSE`` is a directory, and ``NOTICE``. A license file that only points to the Apache License 2.0 is joined
by the repository's own full text of it, which Apache-2.0 4(a) asks recipients be given. Each distinct file, by
repository, path and sha256, is pinned at the earliest base commit that holds it (by committer date) and named for that
commit. A repository the importer says is under a GPL version the text cannot tell (``swebench.GPL_DECLARED``) must
declare it in its packaging metadata at every base commit.

The repositories are cloned bare into the directory given, or used as they are when already there. The table is
written to ``src/bellwether/importers/swebench_licenses.json``; ``--check`` compares instead, and exits 1 on a
difference. Reading the rows needs the pinned parquet files in the Hugging Face cache (``HF_HUB_OFFLINE=1`` reads them
offline).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from bellwether.importers import hf, swebench

TABLE = Path(__file__).resolve().parent.parent / "src" / "bellwether" / "importers" / "swebench_licenses.json"
LICENSE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.rst", "LICENSE.txt", "COPYING")
NOTICE_NAME = "NOTICE"
METADATA = ("setup.cfg", "pyproject.toml", "setup.py")


def git(clone: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(clone), *args], check=True, capture_output=True).stdout


def clone_of(clones: Path, repository: str) -> Path:
    clone = clones / f"{repository.replace('/', '__')}.git"
    if not clone.is_dir():
        url = f"https://github.com/{repository}.git"
        subprocess.run(["git", "clone", "--bare", "--quiet", url, str(clone)], check=True)
    return clone


def root_entries(clone: Path, commit: str) -> dict[str, tuple[str, str]]:
    """``name -> (type, object id)`` for the entries at the root of the tree at ``commit``."""
    entries = {}
    for entry in git(clone, "ls-tree", commit).decode("utf-8").splitlines():
        meta, name = entry.split("\t", 1)
        _, kind, oid = meta.split()
        entries[name] = (kind, oid)
    return entries


def license_files(clone: Path, commit: str) -> tuple[str, str | None]:
    """The path of the license file at ``commit``, and of the NOTICE file or None."""
    entries = root_entries(clone, commit)
    found = [name for name in LICENSE_NAMES if entries.get(name, ("", ""))[0] == "blob"]
    if entries.get("LICENSE", ("", ""))[0] == "tree":
        found.append("LICENSE/LICENSE")
    if len(found) != 1:
        raise SystemExit(f"{clone.name} at {commit}: expected one license file, found {found}")
    notice = NOTICE_NAME if entries.get(NOTICE_NAME, ("", ""))[0] == "blob" else None
    return found[0], notice


def is_apache_notice(text: str) -> bool:
    words = " ".join(text.split())
    return "Licensed under the Apache License, Version 2.0" in words and "TERMS AND CONDITIONS FOR USE" not in words


def is_apache_text(text: str) -> bool:
    return " ".join(text.split()).startswith("Apache License Version 2.0, January 2004")


def build(clones: Path) -> dict:
    rows, seen = [], set()
    for source in swebench.SOURCES:
        path = hf.fetch(source.dataset_id, source.revision, source.file, source.sha256)
        for row in swebench.read_rows(path):
            if row["instance_id"] not in seen:
                seen.add(row["instance_id"])
                rows.append(row)
    pins: dict[tuple[str, str, str], tuple[int, str]] = {}  # (repository, path, sha256) -> earliest (time, commit)
    at_commit: dict[tuple[str, str], dict[str, tuple[str, str] | None]] = {}
    texts: dict[tuple[str, str, str], str] = {}
    for row in rows:
        repository, commit = row["repo"], row["base_commit"]
        if (repository, commit) in at_commit:
            continue
        clone = clone_of(clones, repository)
        when = int(git(clone, "show", "-s", "--format=%ct", commit))
        found: dict[str, tuple[str, str] | None] = {"license": None, "notice": None}
        license_path, notice_path = license_files(clone, commit)
        for role, path in (("license", license_path), ("notice", notice_path)):
            if path is None:
                continue
            content = git(clone, "show", f"{commit}:{path}")
            digest = hashlib.sha256(content).hexdigest()
            key = (repository, path, digest)
            texts[key] = content.decode("utf-8")
            if key not in pins or (when, commit) < pins[key]:
                pins[key] = (when, commit)
            found[role] = (path, digest)
        at_commit[(repository, commit)] = found
        declared = swebench.GPL_DECLARED.get(repository)
        if declared is not None:
            metadata = b"".join(
                git(clone, "show", f"{commit}:{name}") for name in METADATA if name in root_entries(clone, commit)
            )
            if declared.encode() not in metadata:
                raise SystemExit(f"{repository} at {commit}: its packaging metadata does not declare {declared}")

    def name(key: tuple[str, str, str]) -> str:
        repository, path, _ = key
        return f"swebench-{repository.replace('/', '-')}-{pins[key][1][:12]}-{path.replace('/', '-')}"

    full_texts = {}
    for key, text in sorted(texts.items(), key=lambda item: pins[item[0]]):
        if is_apache_text(text):
            full_texts.setdefault(key[0], key)
    versions: dict[tuple, list[str]] = {}
    for (repository, commit), found in at_commit.items():
        license_key = (repository, *found["license"])
        notice_key = (repository, *found["notice"]) if found["notice"] else None
        full_text = None
        if is_apache_notice(texts[license_key]):
            if repository not in full_texts:
                raise SystemExit(f"{repository} at {commit}: an Apache notice, and no full text of the License")
            full_text = name(full_texts[repository])
        version = (repository, name(license_key), name(notice_key) if notice_key else None, full_text)
        versions.setdefault(version, []).append(commit)
    named = {}
    for key, (_, commit) in pins.items():
        repository, path, digest = key
        named[name(key)] = {"repository": repository, "commit": commit, "path": path, "sha256": digest}
    ordered = sorted(versions.items(), key=lambda item: tuple(field or "" for field in item[0]))
    return {
        "files": dict(sorted(named.items())),
        "versions": [
            {
                "repository": repository,
                "license": license,
                "notice": notice,
                "full_text": full,
                "commits": sorted(commits),
            }
            for (repository, license, notice, full), commits in ordered
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("clones", type=Path, help="directory for bare clones of the SWE-bench repositories")
    parser.add_argument("--check", action="store_true", help="compare with the committed table instead of writing it")
    args = parser.parse_args()
    args.clones.mkdir(parents=True, exist_ok=True)
    table = json.dumps(build(args.clones), indent=1) + "\n"
    if args.check:
        if TABLE.read_text("utf-8") != table:
            print(f"{TABLE}: differs from the repositories' history", file=sys.stderr)
            return 1
        print(f"{TABLE}: equals the repositories' history")
        return 0
    TABLE.write_text(table, "utf-8")
    print(f"{TABLE}: {len(json.loads(table)['files'])} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
