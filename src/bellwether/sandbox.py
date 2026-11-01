"""bellwether sandbox-record: record a checkpoint whose tokenizer is the vendor's code, inside a sandbox.

The vendor-code oracle (``record.vendor``) runs code from the checkpoint's repository, so it runs only here, in a
container that holds bellwether, the checkpoint's files and the corpus, and nothing else (docs/benchmark-sets.md,
"Running vendor code"):

1. **Download, then check.** On the host, the checkpoint's files are read into the Hugging Face cache at the manifest's
   revision, with the token if there is one (``inputs.checkpoint_dir``). Every one of them, every Python file
   included, must have the sha256 the manifest lists, and the manifest's ``encoder`` line must name the tokenizer
   mode the checkpoint's architecture takes in the engine, or nothing runs.
2. **A context of only those files, and the engine's copy of the encoder.** A build context is staged with exactly the
   files the manifest lists, in the cache's layout with the commit's file list and the cache's marks for files the
   repository does not have, the corpus of the one kind recorded, and the group's fixtures. For a checkpoint of an
   architecture the engine renders with a prompt encoder (``inputs.encoder_mode``), a module of its own shipped or
   not, the engine's copy of that encoder (``record/engine_encoders.toml``) is fetched from the engine's repository
   at the pinned commit, checked against its sha256 (``importers.github.fetch``) and staged under ``/engine``: it is
   what the oracle renders with (``record.vendor.prompt_encoder``), the checkpoint's module never.
3. **Two images.** The base image (``docker/vendor/Dockerfile``) is a pinned Python image with the packages
   ``uv.lock`` pins for bellwether and its ``vendor`` extra, installed by hash, and bellwether itself. A run image
   adds the context; it is removed after the run.
4. **The run.** ``bellwether record --oracle vendor`` runs as a user without privileges, with no network, a read-only
   root, no capabilities, and limits on CPU, memory, processes and time. It writes only to ``/work``, a volume, and
   ``/tmp``. No credential is passed in.
5. **Back out.** When the recorder finished, refusals or not, the group's fixtures are copied from the volume to the
   host and take the place of the host's copy, so a set the run removed is gone there too; a run the time or memory
   limit stopped leaves the host's fixtures unchanged. A run of named sets (``--set``) brings back those sets alone,
   files and tables, so runs at once, each recording a few of a source's sets as the record runners do at benchmark
   scale, keep each other's. The container, volume and run image are removed whatever the outcome.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from bellwether.importers.github import fetch as fetch_pinned
from bellwether.inputs import checkpoint_dir, encoder_mode, oracle_inputs, vendor_encoder
from bellwether.manifest import Manifest, find_manifest
from bellwether.record import sets as set_tables
from bellwether.record import vendor
from bellwether.storage import stem

ROOT = Path(__file__).resolve().parents[2]  # the bellwether checkout: pyproject.toml, uv.lock, src/, docker/
DOCKERFILE = ROOT / "docker" / "vendor" / "Dockerfile"
BASE_TAG = "bellwether-vendor-base"
USER = "10001:10001"
ENGINE_DIR = "/engine"  # where the run image holds the engine's copy of a prompt encoder (``vendor.ENGINE_DIR_ENV``)
LIMITS = ("--cpus", "2", "--pids-limit", "512")
MEMORY = "6g"  # a run's memory limit unless one is given: a benchmark-scale call of tau2's sets holds more
SECONDS = 4 * 3600  # the longest a group's recording may take
# The statuses of a run whose fixtures are copied back: the recorder finished (0), or refused some cases (1, its usual
# outcome). A run the time limit stopped (124), the memory limit killed (137) or that was stopped (143) has its
# fixtures left in the container.
FINISHED = (0, 1)


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


def engine_copy(manifest: Manifest, snapshot: Path) -> tuple[str, Path] | None:
    """The engine's copy of the prompt encoder for a checkpoint the engine's server renders with one: the copy's path
    at the engine's pinned commit and the cached file, fetched from the engine's repository and checked against the
    pinned sha256 (``vendor.engine_encoder`` for the tokenizer mode of the checkpoint's architecture,
    ``inputs.encoder_mode``; ``importers.github.fetch``), a module of the checkpoint's own shipped or not. None for a
    checkpoint of another architecture that ships no prompt encoder; one that ships a module for such an architecture
    is refused here, before any image is built, since no tokenizer mode of the engine's tells how it is called. So is
    a manifest whose ``encoder`` line is not the mode the architecture takes: the recording plan routes by that line
    (``bellwether manifests`` writes it)."""
    renderer = encoder_mode(snapshot)
    if manifest.encoder != renderer:
        raise Refused(
            f"{manifest.path} names {manifest.encoder!r} as the engine's encoder mode where the checkpoint's "
            f"architecture takes {renderer!r}; `bellwether manifests` writes the line"
        )
    if renderer is None:
        if vendor_encoder(snapshot) is None:
            return None
        raise Refused(
            f"{manifest.model} ships a prompt encoder for an architecture the engine's server renders with no "
            "tokenizer mode of its own; the vendor-code oracle cannot tell how the encoder is called"
        )
    pinned = vendor.engine_encoder(renderer)
    owner, repo = pinned.repo.split("/", 1)
    return pinned.path, fetch_pinned(owner, repo, pinned.commit, pinned.path, pinned.sha256)


def stage(
    manifest: Manifest,
    snapshot: Path,
    kind: str,
    fixtures: Path,
    corpus: Path,
    context: Path,
    engine: tuple[str, Path] | None = None,
) -> None:
    """Write the run image's build context: the listed files in the Hugging Face cache's layout, the commit's file
    list, the corpus of ``kind``, the group's fixtures and, when the checkpoint ships a prompt encoder, the engine's
    copy of it (``engine``: its path at the engine's commit and the checked file), read-only under ``ENGINE_DIR``."""
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
    dockerfile = (
        f"FROM {BASE_TAG}\nCOPY --chown=root:root hf /hf\nCOPY --chown={USER} work /work\nRUN chmod -R a-w /hf\n"
    )
    if engine is not None:
        path, checked = engine
        (context / "engine" / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checked, context / "engine" / path)
        dockerfile += f"COPY --chown=root:root engine {ENGINE_DIR}\nRUN chmod -R a-w {ENGINE_DIR}\n"
    (context / "Dockerfile").write_text(dockerfile)


def run_args(
    name: str,
    volume: str,
    image: str,
    base_digest: str,
    manifest: Manifest,
    kind: str,
    sets: list[str] | None = None,
    memory: str = MEMORY,
) -> list[str]:
    """``docker run`` for one recording: no network, a read-only root, an unprivileged user, no capabilities, limits,
    and no credential in the environment. Named ``sets`` are the only ones the recorder records."""
    record = ["bellwether", "record", "--model", manifest.model, "--kind", kind, "--oracle", "vendor"]
    record += ["--fixtures", "/work/fixtures", "--corpus", "/work/corpus"]
    record += [flag for name in sets or [] for flag in ("--set", name)]
    return [
        "docker", "run", "--name", name, "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,size=2g,mode=1777", "--mount", f"type=volume,src={volume},dst=/work",
        "--user", USER, "--cap-drop", "ALL", "--security-opt", "no-new-privileges", *LIMITS, "--memory", memory,
        "--env", f"{vendor.SANDBOX_ENV}=1", "--env", f"{vendor.IMAGE_ENV}={base_digest}",
        "--env", f"{vendor.ENGINE_DIR_ENV}={ENGINE_DIR}",
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


def bring_back(fresh: Path, group: Path, kind: str, names: list[str]) -> None:
    """Put the named sets of ``kind`` from ``fresh`` into ``group``, files and tables, and leave every other set as the
    host holds it now: calls at once each bring back their own sets, and a call's copy of the group holds the others as
    they were when it started. A named set the run left no table for is gone from the host too."""
    tables = set_tables.read(fresh / set_tables.FILE)
    for name in names:
        for path in (group / kind).glob(f"{name}.jsonl*"):
            if stem(path) == name:
                path.unlink()
        for path in (fresh / kind).glob(f"{name}.jsonl*") if (fresh / kind).is_dir() else []:
            if stem(path) == name:
                (group / kind).mkdir(exist_ok=True)
                shutil.copyfile(path, group / kind / path.name)
    set_tables.update(group / set_tables.FILE, kind, {name: tables.get((kind, name)) for name in names})


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
        engine = engine_copy(manifest, snapshot)
    except (OSError, ValueError) as err:
        print(f"bellwether sandbox-record: {err}", file=sys.stderr)
        return 1
    sets = list(getattr(args, "sets", None) or [])
    base = base_image()
    tag = f"bellwether-run-{uuid.uuid4().hex[:12]}"
    name, volume = tag, f"{tag}-work"
    try:
        with tempfile.TemporaryDirectory(prefix="bellwether-run-") as tmp:
            stage(manifest, snapshot, args.kind, args.fixtures, args.corpus, Path(tmp), engine)
            docker("build", "--quiet", "--tag", tag, tmp, quiet=True)
        memory = getattr(args, "memory", None) or MEMORY
        status = subprocess.run(run_args(name, volume, tag, base, manifest, args.kind, sets, memory)).returncode
        if status not in FINISHED:
            print(
                f"bellwether sandbox-record: the run stopped with status {status}; the host's fixtures are unchanged",
                file=sys.stderr,
            )
            return status
        fresh = landing(args.fixtures / manifest.slug)
        try:
            docker("cp", f"{name}:/work/fixtures/{manifest.slug}/.", str(fresh))
            if sets:
                bring_back(fresh, args.fixtures / manifest.slug, args.kind, sets)
            else:
                replace_group(fresh, args.fixtures / manifest.slug)
        finally:
            shutil.rmtree(fresh, ignore_errors=True)
        return status
    finally:
        docker("rm", "--force", name, check=False, quiet=True)
        docker("volume", "rm", "--force", volume, check=False, quiet=True)
        docker("rmi", "--force", tag, check=False, quiet=True)
