"""Every committed agentic session corpus is what its SHA256SUMS and manifest say (corpus/agentic/README.md).

Each version lives in ``corpus/agentic/v<n>/`` with ``corpus.jsonl.zst`` in Git LFS. A checkout that has not fetched it
holds its pointer, which names the compressed content's sha256: that is compared with SHA256SUMS without a download. A
checkout that has fetched it (CI pulls ``corpus/**/*.jsonl.zst``) is read a piece at a time: the compressed bytes'
sha256, the plain content's sha256, its line count against the manifest's session count, and the session ids, each one
JSON object with its turns, are compared with SHA256SUMS and manifest.json. The plain files beside it are hashed whole.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

import pytest

from bellwether import storage

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENTIC = ROOT / "corpus" / "agentic"
VERSIONS = sorted(p for p in AGENTIC.glob("v*") if p.is_dir()) if AGENTIC.is_dir() else []
POINTER = re.compile(rb"\Aversion https://git-lfs\.github\.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n\Z")
READ_SIZE = 1 << 20

pytestmark = pytest.mark.parametrize("version", VERSIONS, ids=[p.name for p in VERSIONS])


def sums(version: pathlib.Path) -> dict[str, str]:
    """``file -> sha256`` as SHA256SUMS lists them (``sha256sum`` format, paths relative to the version's directory)."""
    lines = (version / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    return {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in lines if line}


def manifest(version: pathlib.Path) -> dict:
    return json.loads((version / "manifest.json").read_text(encoding="utf-8"))


def test_sha256sums_names_the_corpus_in_both_forms_and_its_manifest(version):
    listed = sums(version)
    assert {"corpus.jsonl", "corpus.jsonl.zst", "manifest.json", "MANIFEST.md", "REPORT.md"} <= set(listed)


def test_the_plain_files_beside_the_corpus_are_what_sha256sums_says(version):
    for name, digest in sums(version).items():
        path = version / name
        if name.startswith("corpus.jsonl") or not path.is_file():
            continue  # the corpus is checked below; the tools (../tools/) are listed for provenance and may be absent
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


def test_the_manifest_records_the_build_the_sums_name(version):
    built, listed = manifest(version), sums(version)
    assert built["version"] == version.name
    assert built["sessions"] == len(built["per_session"])
    assert built["sources"]["generator"]["sha256"] == listed["../tools/gen_corpus.py"]
    builder = built["sources"]["builder"]["file"]
    assert built["sources"]["builder"]["sha256"] == listed[f"../tools/{builder}"]


def test_the_compressed_corpus_is_the_pointer_or_the_content_sha256sums_names(version):
    compressed_path, listed = version / "corpus.jsonl.zst", sums(version)
    if storage.is_lfs_pointer(compressed_path):
        match = POINTER.match(compressed_path.read_bytes())
        assert match is not None, f"{compressed_path} begins like a Git LFS pointer but is not one"
        assert match.group(1).decode() == listed["corpus.jsonl.zst"]
        pytest.skip(f"{compressed_path.relative_to(ROOT)} is a Git LFS pointer whose sha256 agrees; fetch it to read")
    zstandard = pytest.importorskip("zstandard")
    compressed, plain = hashlib.sha256(), hashlib.sha256()
    frame = zstandard.ZstdDecompressor().decompressobj()
    lines, ids, rest = 0, set(), b""
    with compressed_path.open("rb") as handle:
        while piece := handle.read(READ_SIZE):
            compressed.update(piece)
            data = frame.decompress(piece)
            plain.update(data)
            rest += data
            *whole, rest = rest.split(b"\n")
            for line in whole:
                session = json.loads(line)
                assert session["turns"], session["id"]
                ids.add(session["id"])
                lines += 1
    assert frame.eof and rest == b"", "the compressed file ends before its last block or without a final newline"
    assert compressed.hexdigest() == listed["corpus.jsonl.zst"]
    assert plain.hexdigest() == listed["corpus.jsonl"]
    assert lines == manifest(version)["sessions"] == len(ids)
