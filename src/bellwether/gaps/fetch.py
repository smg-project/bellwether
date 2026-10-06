"""Fetch an engine's registry files from GitHub at a pinned ref, into a checkout-shaped cache."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import httpx

from .registries import SGLANG_SRT

FILES: dict[str, tuple[str, list[str]]] = {
    "vllm": (
        "vllm-project/vllm",
        [
            "vllm/tool_parsers/__init__.py",
            "vllm/reasoning/__init__.py",
            "vllm/renderers/registry.py",
            "vllm/tokenizers/registry.py",
        ],
    ),
    "sglang": (
        "sgl-project/sglang",
        [
            f"{SGLANG_SRT}/function_call/parser_names.py",
            f"{SGLANG_SRT}/parser/reasoning_parser_names.py",
            f"{SGLANG_SRT}/function_call/function_call_parser.py",
            f"{SGLANG_SRT}/parser/reasoning_parser.py",
        ],
    ),
}

# Directories whose matching modules are native renderers or tokenizers (SGLang has no registry
# for them): the directory is listed through the contents API and the matching files fetched.
DIRS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "sglang": [
        (f"{SGLANG_SRT}/parser", ("_renderer.py", "_tokenizer.py")),
        (f"{SGLANG_SRT}/tokenizer", ("_tokenizer.py",)),
    ],
}

_API = {"Accept": "application/vnd.github+json"}


def write_whole(dest: Path, content: bytes) -> None:
    """Write ``content`` to a temporary file next to ``dest`` and move it into place.

    A run cut off mid-write then leaves no file at ``dest``, so the next run downloads it again
    instead of reading a partial file as if it were complete.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=f".{dest.name}.", delete=False) as part:
        part_path = Path(part.name)
        try:
            part.write(content)
            part.close()
            os.replace(part_path, dest)
        finally:
            part_path.unlink(missing_ok=True)


def fetch_registry_files(engine: str, ref: str, cache_dir: Path, client: httpx.Client | None = None) -> Path:
    """Download the registry files for ``engine`` at ``ref`` and return the cache root.

    A branch or tag is resolved to a commit first so the cache is keyed by what was read; files
    already present are kept, and a directory is listed again until every file it matched is on disk. The returned
    directory carries ``COMMIT.txt`` like an offline copy.
    """
    repo, files = FILES[engine]
    own_client = client is None
    client = client or httpx.Client(timeout=30, follow_redirects=True)
    try:
        sha = ref
        if not re.fullmatch(r"[0-9a-f]{40}", ref):
            resp = client.get(f"https://api.github.com/repos/{repo}/commits/{ref}", headers=_API)
            resp.raise_for_status()
            sha = resp.json()["sha"]
        target = cache_dir / engine / sha
        wanted = list(files)
        # A directory counts as listed only once every file it matched has been written: the
        # marker is created after the downloads, so an interrupted run lists again next time.
        pending: list[tuple[Path, list[str]]] = []
        for rel_dir, suffixes in DIRS.get(engine, []):
            marker = target / rel_dir / ".listed"
            if marker.is_file():
                wanted.extend(f"{rel_dir}/{name}" for name in marker.read_text().split())
                continue
            resp = client.get(
                f"https://api.github.com/repos/{repo}/contents/{rel_dir}", params={"ref": sha}, headers=_API
            )
            if resp.status_code == 404:
                pending.append((marker, []))  # upstream has no such directory; remember that too
                continue
            resp.raise_for_status()
            names = sorted(item["name"] for item in resp.json() if item.get("type") == "file")
            matched = [name for name in names if name.endswith(suffixes)]
            wanted.extend(f"{rel_dir}/{name}" for name in matched)
            pending.append((marker, matched))
        for rel in wanted:
            dest = target / rel
            if dest.is_file():
                continue
            resp = client.get(f"https://raw.githubusercontent.com/{repo}/{sha}/{rel}")
            resp.raise_for_status()
            write_whole(dest, resp.content)
        for marker, names in pending:
            write_whole(marker, "".join(f"{name}\n" for name in names).encode())
        write_whole(target / "COMMIT.txt", f"{sha} {repo} fetched at ref {ref}\n".encode())
    finally:
        if own_client:
            client.close()
    return target
