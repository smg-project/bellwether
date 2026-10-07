"""bellwether sandbox-record: record a checkpoint whose tokenizer is the vendor's code, inside a sandbox.

The vendor-code oracle (``record.vendor``) runs code from the checkpoint's repository, so it runs only here, in a
container that holds bellwether, the checkpoint's files and the corpus, and nothing else (docs/benchmark-sets.md,
"Running vendor code"):

1. **Download, then check.** On the host, the checkpoint's files are read into the Hugging Face cache at the manifest's
   revision, with the token if there is one (``inputs.checkpoint_dir``). Every one of them, every Python file
   included, must have the sha256 the manifest lists, or nothing runs.
2. **A context of only those files.** A build context is staged with exactly the files the manifest lists, in the
   cache's layout with the commit's file list and the cache's marks for files the repository does not have, the corpus
   of the one kind recorded, and the group's fixtures.
3. **Two images.** The base image (``docker/vendor/Dockerfile``) is a pinned Python image with the packages
   ``uv.lock`` pins for bellwether and its ``vendor`` extra, installed by hash, and bellwether itself. A run image
   adds the context; it is removed after the run.
4. **The run.** ``bellwether record --oracle vendor`` runs as a user without privileges, with no network, a read-only
   root, no capabilities, and limits on CPU, memory, processes and time. It writes only to ``/work``, a volume, and
   ``/tmp``. No credential is passed in.
5. **Back out.** The group's fixtures are copied from the volume to the host and take the place of the host's copy,
   so a set the run removed is gone there too; the container, volume and run image are removed whatever the outcome.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from bellwether.inputs import checkpoint_dir, oracle_inputs
from bellwether.manifest import Manifest, find_manifest
from bellwether.record import vendor

ROOT = Path(__file__).resolve().parents[2]  # the bellwether checkout: pyproject.toml, uv.lock, src/, docker/
DOCKERFILE = ROOT / "docker" / "vendor" / "Dockerfile"
BASE_TAG = "bellwether-vendor-base"
USER = "10001:10001"
LIMITS = ("--cpus", "2", "--memory", "6g", "--pids-limit", "512")
SECONDS = 4 * 3600  # the longest a group's recording may take


class Refused(ValueError):
    """The checkpoint is not run: its files differ from the manifest, or it cannot be read."""


def checked_snapshot(manifest: Manifest) -> Path:
    """The checkpoint's snapshot directory, once every file the manifest lists matches its sha256 and no file the
    manifest does not list is an oracle input."""
    if not manifest.inputs:
        raise Refused(f"{manifest.path} lists no oracle inputs; `bellwether manifests` writes them")
    try:
        snapshot = checkpoint_dir(manifest.model, manifest.revision)
        found = oracle_inputs(manifest.model, manifest.revision)
    except (OSError, ValueError) as err:
        raise Refused(f"cannot read {manifest.model} at {manifest.revision}: {err}") from err
    differ = sorted(
        name for name in manifest.inputs.keys() | found.keys() if manifest.inputs.get(name) != found.get(name)
    )
    if differ:
        raise Refused(
            f"{manifest.model} at {manifest.revision}: these files differ from the manifest: {', '.join(differ)}"
        )
    return snapshot


def stage(manifest: Manifest, snapshot: Path, kind: str, fixtures: Path, corpus: Path, context: Path) -> None:
    """Write the run image's build context: the listed files in the Hugging Face cache's layout, the commit's file
    list, the corpus of ``kind`` and the group's fixtures."""
    repo = context / "hf" / "hub" / ("models--" + manifest.model.replace("/", "--"))
    target = repo / "snapshots" / manifest.revision
    target.mkdir(parents=True)
    for name in manifest.inputs:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snapshot / name, target / name)
    tree = snapshot.parents[1] / "trees" / f"{manifest.revision}.json"
    if tree.is_file():
        (repo / "trees").mkdir()
        shutil.copyfile(tree, repo / "trees" / tree.name)
    # The cache's marks for files the repository does not have: offline, the oracles take a marked file as absent.
    absent = snapshot.parents[1] / ".no_exist" / manifest.revision
    if absent.is_dir():
        shutil.copytree(absent, repo / ".no_exist" / manifest.revision)
    shutil.copytree(corpus / kind, context / "work" / "corpus" / kind)
    shutil.copytree(fixtures / manifest.slug, context / "work" / "fixtures" / manifest.slug)
    (context / "Dockerfile").write_text(
        f"FROM {BASE_TAG}\nCOPY --chown=root:root hf /hf\nCOPY --chown={USER} work /work\nRUN chmod -R a-w /hf\n"
    )


def run_args(name: str, volume: str, image: str, base_digest: str, manifest: Manifest, kind: str) -> list[str]:
    """``docker run`` for one recording: no network, a read-only root, an unprivileged user, no capabilities, limits,
    and no credential in the environment."""
    record = ["bellwether", "record", "--model", manifest.model, "--kind", kind, "--oracle", "vendor"]
    record += ["--fixtures", "/work/fixtures", "--corpus", "/work/corpus"]
    return [
        "docker", "run", "--name", name, "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,size=2g,mode=1777", "--mount", f"type=volume,src={volume},dst=/work",
        "--user", USER, "--cap-drop", "ALL", "--security-opt", "no-new-privileges", *LIMITS,
        "--env", f"{vendor.SANDBOX_ENV}=1", "--env", f"{vendor.IMAGE_ENV}={base_digest}",
        "--env", "HF_HOME=/hf", "--env", "HF_HUB_OFFLINE=1", "--env", "HF_MODULES_CACHE=/tmp/hf_modules",
        "--env", "HOME=/tmp", image, "timeout", str(SECONDS), *record,
    ]  # fmt: skip


def landing(group: Path) -> Path:
    """An empty directory beside ``group``, on its file system, for the group's fixtures as the run left them."""
    return Path(tempfile.mkdtemp(prefix=f".{group.name}-", dir=group.parent))


def replace_group(fresh: Path, group: Path) -> None:
    """Put ``fresh`` in ``group``'s place, so the host holds the group's fixtures as the run left them: a set the run
    removed is gone here too."""
    old = group.with_name(f".{group.name}-{uuid.uuid4().hex[:12]}")
    group.rename(old)
    fresh.rename(group)
    shutil.rmtree(old)


def docker(*args: str, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    out = subprocess.DEVNULL if quiet else None
    return subprocess.run(["docker", *args], check=check, stdout=out, stderr=out)


def base_image() -> str:
    """Build the base image from ``docker/vendor/Dockerfile`` and ``uv.lock`` (cached by Docker), and return its id."""
    with tempfile.TemporaryDirectory(prefix="bellwether-base-") as tmp:
        context = Path(tmp)
        requirements = subprocess.run(
            ["uv", "export", "--frozen", "--extra", "vendor", "--no-dev", "--no-emit-project", "--format",
             "requirements-txt"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout  # fmt: skip
        (context / "requirements.txt").write_text(requirements)
        shutil.copytree(ROOT / "src", context / "src")
        for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE"):
            if (ROOT / name).is_file():
                shutil.copyfile(ROOT / name, context / name)
        shutil.copyfile(DOCKERFILE, context / "Dockerfile")
        docker("build", "--quiet", "--tag", BASE_TAG, str(context), quiet=True)
    found = subprocess.run(
        ["docker", "image", "inspect", BASE_TAG, "--format", "{{.Id}}"], check=True, capture_output=True, text=True
    )
    return found.stdout.strip()


def run(args: argparse.Namespace) -> int:
    try:
        manifest = find_manifest(args.fixtures, args.model)
        if manifest.group is not None:
            raise Refused(f"{manifest.model} is a member of checkpoint group {manifest.group}; record its primary")
        snapshot = checked_snapshot(manifest)
    except (OSError, ValueError) as err:
        print(f"bellwether sandbox-record: {err}", file=sys.stderr)
        return 1
    base = base_image()
    tag = f"bellwether-run-{uuid.uuid4().hex[:12]}"
    name, volume = tag, f"{tag}-work"
    try:
        with tempfile.TemporaryDirectory(prefix="bellwether-run-") as tmp:
            stage(manifest, snapshot, args.kind, args.fixtures, args.corpus, Path(tmp))
            docker("build", "--quiet", "--tag", tag, tmp, quiet=True)
        status = subprocess.run(run_args(name, volume, tag, base, manifest, args.kind)).returncode
        fresh = landing(args.fixtures / manifest.slug)
        try:
            docker("cp", f"{name}:/work/fixtures/{manifest.slug}/.", str(fresh))
            replace_group(fresh, args.fixtures / manifest.slug)
        finally:
            shutil.rmtree(fresh, ignore_errors=True)
        return status
    finally:
        docker("rm", "--force", name, check=False, quiet=True)
        docker("volume", "rm", "--force", volume, check=False, quiet=True)
        docker("rmi", "--force", tag, check=False, quiet=True)
