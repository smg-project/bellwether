"""The multimodal suite's commands on a small suite written here: two images (one as content, one as a Git LFS pointer),
one family, a reference set imported from a recorder's lines, a dump compared with it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bellwether.cli import main
from bellwether.multimodal import references, suite, verify

RED = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077053e0000000c4944415408d763f8cfc00000030101"
    "0018dd8db00000000049454e44ae426082"
)


def pointer_for(data: bytes) -> bytes:
    oid = hashlib.sha256(data).hexdigest()
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {len(data)}\n".encode()


def write_suite(root: Path, pointer: bool = True) -> dict[str, bytes]:
    """A suite of two images: ``a.png`` as content, ``b.png`` as a pointer (its content when ``pointer`` is False)."""
    a, b = RED, RED + b"\x00"
    (root / "images" / "tiny").mkdir(parents=True)
    (root / "images" / "tiny" / "a.png").write_bytes(a)
    (root / "images" / "tiny" / "b.png").write_bytes(pointer_for(b) if pointer else b)
    header = "id\tpath\tsource\tsplit\tgroup\tvariant\tclass\tformat\tw\th\tbytes\tsha256\n"
    rows = [
        f"a\timages/tiny/a.png\ttiny\ttest\tsource\t-\ttiny/png\tpng\t1\t1\t{len(a)}\t{hashlib.sha256(a).hexdigest()}\n",
        f"b\timages/tiny/b.png\ttiny\ttest\tsource\t-\ttiny/png\tpng\t1\t1\t{len(b)}\t{hashlib.sha256(b).hexdigest()}\n",
    ]
    (root / "manifest.tsv").write_text(header + "".join(rows))
    (root / "tiers").mkdir()
    (root / "tiers" / "corpus-50.txt").write_text("images/tiny/a.png\nimages/tiny/b.png\n")
    (root / "tiers" / "corpus-500.txt").write_text("images/tiny/a.png\nimages/tiny/b.png\n")
    (root / "tiers" / "all.txt").write_text("images/tiny/a.png\nimages/tiny/b.png\n")
    (root / "sources").mkdir()
    (root / "sources" / "tiny.tsv").write_text("id\tsha256\tbytes\tpath\torigin\na\tx\t1\timages/tiny/a.png\tfile:0\n")
    family = root / "families" / "fam"
    family.mkdir(parents=True)
    config = b'{"patch_size": 14}'
    (family / "preprocessor_config.json").write_bytes(config)
    (family / "family.toml").write_text(
        'smg_processor = "fam"\nmodel = "org/Model"\nrevision = "0123456789abcdef0123456789abcdef01234567"\n'
        f'[inputs]\n"preprocessor_config.json" = "{hashlib.sha256(config).hexdigest()}"\n'
    )
    return {"a": a, "b": b}


def recorder_lines(names: list[str], **overrides) -> list[dict]:
    lines = [{"meta": {"vllm": "0.31.0", "transformers": "5.17.0"}}]
    for name in names:
        line = {
            "name": name,
            "path": f"/somewhere/{name}",
            "w": 1,
            "h": 1,
            "mode": "RGB",
            "grid": [[1, 2, 2]],
            "shape": [4, 1176],
            "rgb_sha": "r" * 64,
            "u8_sha": "u" * 64,
            "bf16_sha": "b" * 64,
            "f32_sha": "f" * 64,
            "load_ms": 1.0,
        }
        line.update(overrides)
        lines.append(line)
    return lines


def write_jsonl(path: Path, lines: list[dict]) -> Path:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def test_manifest_tiers_and_pointers_read_as_the_suite_holds_them(tmp_path):
    data = write_suite(tmp_path)
    images = suite.read_manifest(tmp_path)
    assert [image.name for image in images] == ["a.png", "b.png"]
    assert suite.identity(tmp_path / "images/tiny/b.png") == (hashlib.sha256(data["b"]).hexdigest(), len(data["b"]))
    assert suite.pointer(tmp_path / "images/tiny/a.png") is None
    assert [image.id for image in suite.tier_images(tmp_path, "corpus-50")] == ["a", "b"]
    assert suite.read_families(tmp_path)["fam"].smg_processor == "fam"


def test_check_passes_on_pointers_and_names_each_problem(tmp_path, capsys):
    write_suite(tmp_path)
    assert main(["multimodal", "check", "--root", str(tmp_path)]) == 0
    (tmp_path / "images" / "tiny" / "a.png").write_bytes(RED + b"\x01")
    (tmp_path / "tiers" / "corpus-50.txt").write_text("images/tiny/a.png\nimages/none.png\n")
    assert main(["multimodal", "check", "--root", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "images/tiny/a.png: sha256" in err
    assert "images/none.png is not in the manifest" in err


def test_import_references_writes_the_canonical_set_and_its_table(tmp_path):
    write_suite(tmp_path)
    recorded = write_jsonl(tmp_path / "rec.jsonl", recorder_lines(["a.png", "b.png"]))
    assert (
        main(
            [
                "multimodal",
                "import-references",
                "--root",
                str(tmp_path),
                "--family",
                "fam",
                "--source",
                "vllm",
                "--from",
                str(recorded),
            ]
        )
        == 0
    )
    lines = references.read_references(tmp_path, "fam", "vllm")
    assert set(lines) == {"a.png", "b.png"}
    assert lines["a.png"]["path"] == "images/tiny/a.png"
    assert lines["a.png"]["grid"] == [1, 2, 2]
    assert (
        lines["a.png"]["rgb_sha256"] == "r" * 64 and "rgb_sha" not in lines["a.png"] and "load_ms" not in lines["a.png"]
    )
    assert list(lines["a.png"]) == sorted(lines["a.png"])
    table = suite.read_sets(tmp_path / "references" / "fam" / "sets.toml")[("vllm", "fam")]
    assert table["cases"] == 2 and table["errors"] == 0 and table["form"] == "zstd"
    assert table["provenance"]["vllm"] == "0.31.0"
    assert main(["multimodal", "check", "--root", str(tmp_path)]) == 0


def test_import_refuses_a_file_the_manifest_does_not_list(tmp_path):
    write_suite(tmp_path)
    recorded = write_jsonl(tmp_path / "rec.jsonl", recorder_lines(["zzz.png"]))
    with pytest.raises(ValueError, match="manifest does not list"):
        references.import_references(tmp_path, "fam", "vllm", recorded)


def test_verify_matches_differs_and_knows(tmp_path, capsys):
    write_suite(tmp_path)
    references.import_references(
        tmp_path, "fam", "vllm", write_jsonl(tmp_path / "rec.jsonl", recorder_lines(["a.png", "b.png"]))
    )
    same = recorder_lines(["a.png", "b.png"])[1:]
    for line in same:
        line["rgb_sha256"] = line.pop("rgb_sha")  # the dump spells the hashes with their full names
    dump = write_jsonl(tmp_path / "dump.jsonl", same)
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "multimodal",
                "verify",
                "--root",
                str(tmp_path),
                "--tier",
                "corpus-50",
                "--dump",
                f"fam={dump}",
                "--report",
                str(report),
            ]
        )
        == 0
    )
    assert json.loads(report.read_text())["families"]["fam"]["total"]["match"] == 2
    same[1]["bf16_sha"] = "x" * 64
    dump = write_jsonl(tmp_path / "dump2.jsonl", same)
    assert (
        main(
            [
                "multimodal",
                "verify",
                "--root",
                str(tmp_path),
                "--tier",
                "corpus-50",
                "--dump",
                f"fam={dump}",
                "--report",
                str(report),
                "--junit",
                str(tmp_path / "j.xml"),
            ]
        )
        == 1
    )
    result = json.loads(report.read_text())["families"]["fam"]
    assert result["differences"][0]["name"] == "b.png" and result["differences"][0]["differs"] == ["bf16"]
    assert "b.png: bf16" in (tmp_path / "j.xml").read_text()
    (tmp_path / "known.toml").write_text(
        '[[known]]\nfamily = "fam"\nnames = ["b.png"]\nreason = "staged apart"\nissue = ""\n'
    )
    assert main(["multimodal", "verify", "--root", str(tmp_path), "--tier", "corpus-50", "--dump", f"fam={dump}"]) == 0
    assert "known 1" in capsys.readouterr().out


def test_compare_reads_a_reference_without_u8_as_the_engine_path_gives_it():
    reference = {
        "w": 1,
        "h": 1,
        "shape": [4, 1176],
        "rgb_sha256": "r" * 64,
        "u8_sha256": None,
        "bf16_sha256": "b" * 64,
        "f32_sha256": "f" * 64,
    }
    dumped = {
        "w": 1,
        "h": 1,
        "shape": [4, 1176],
        "rgb_sha256": "r" * 64,
        "u8_sha256": "u" * 64,
        "bf16_sha256": "b" * 64,
        "f32_sha256": "f" * 64,
    }
    assert verify.compare(reference, dumped) == []
    dumped["shape"] = [8, 1176]
    assert verify.compare(reference, dumped) == ["shape", "bf16", "f32"]


def test_cross_check_counts_agreement_per_class(tmp_path, capsys):
    write_suite(tmp_path)
    references.import_references(
        tmp_path, "fam", "vllm", write_jsonl(tmp_path / "v.jsonl", recorder_lines(["a.png", "b.png"]))
    )
    references.import_references(
        tmp_path, "fam", "hf", write_jsonl(tmp_path / "h.jsonl", recorder_lines(["a.png", "b.png"], f32_sha="g" * 64))
    )
    assert main(["multimodal", "cross-check", "--root", str(tmp_path)]) == 0
    assert "agree 0, differ 2" in capsys.readouterr().out
    assert main(["multimodal", "cross-check", "--root", str(tmp_path), "--strict"]) == 1


def test_count_writes_and_checks_the_readme_tables(tmp_path, capsys):
    write_suite(tmp_path)
    readme = tmp_path / "README.md"
    from bellwether.multimodal.count import README_BEGIN, README_END

    readme.write_text(f"# suite\n\n{README_BEGIN}\n\nold\n{README_END}\n\nafter\n")
    assert main(["multimodal", "count", "--root", str(tmp_path), "--readme", str(readme), "--check"]) == 1
    assert main(["multimodal", "count", "--root", str(tmp_path), "--readme", str(readme)]) == 0
    text = readme.read_text()
    assert "| tiny | 2 |" in text and text.endswith("after\n") and "old" not in text
    assert main(["multimodal", "count", "--root", str(tmp_path), "--readme", str(readme), "--check"]) == 0


def test_fetch_names_the_pull_command_for_the_tier(tmp_path, capsys):
    write_suite(tmp_path)
    assert main(["multimodal", "fetch", "--root", str(tmp_path), "--tier", "corpus-50", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("git lfs pull --include '") and "b.png" in out and "a.png" not in out


def test_the_command_group_runs_on_its_own(tmp_path, capsys):
    from bellwether.multimodal.__main__ import main as alone

    write_suite(tmp_path)
    assert alone(["check", "--root", str(tmp_path)]) == 0
    assert "tables agree" in capsys.readouterr().out
