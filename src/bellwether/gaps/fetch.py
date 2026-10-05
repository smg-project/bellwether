"""Fetch an engine's registry files from GitHub at a pinned ref, into a checkout-shaped cache."""

from __future__ import annotations

import re
from pathlib import Path

import httpx

from .registries import _SGL

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
            f"{_SGL}/function_call/parser_names.py",
            f"{_SGL}/parser/reasoning_parser_names.py",
            f"{_SGL}/function_call/function_call_parser.py",
            f"{_SGL}/parser/reasoning_parser.py",
        ],
    ),
}

# Directories whose matching modules are native renderers or tokenizers (SGLang has no registry
# for them): the directory is listed through the contents API and the matching files fetched.
DIRS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "sglang": [
        (f"{_SGL}/parser", ("_renderer.py", "_tokenizer.py")),
        (f"{_SGL}/tokenizer", ("_tokenizer.py",)),
    ],
}

_API = {"Accept": "application/vnd.github+json"}


def fetch_registry_files(engine: str, ref: str, cache_dir: Path, client: httpx.Client | None = None) -> Path:
    """Download the registry files for ``engine`` at ``ref`` and return the cache root.

    A branch or tag is resolved to a commit first so the cache is keyed by what was read; files
    already present are kept, and a directory already listed is not listed again. The returned
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
        for rel_dir, suffixes in DIRS.get(engine, []):
            listed = target / rel_dir
            if listed.is_dir():
                wanted.extend(str(p.relative_to(target)) for p in sorted(listed.glob("*.py")))
                continue
            resp = client.get(
                f"https://api.github.com/repos/{repo}/contents/{rel_dir}", params={"ref": sha}, headers=_API
            )
            listed.mkdir(parents=True, exist_ok=True)  # listed, even when upstream has no such directory
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            names = sorted(item["name"] for item in resp.json() if item.get("type") == "file")
            wanted.extend(f"{rel_dir}/{name}" for name in names if name.endswith(suffixes))
        for rel in wanted:
            dest = target / rel
            if dest.is_file():
                continue
            resp = client.get(f"https://raw.githubusercontent.com/{repo}/{sha}/{rel}")
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
        (target / "COMMIT.txt").write_text(f"{sha} {repo} fetched at ref {ref}\n")
    finally:
        if own_client:
            client.close()
    return target
