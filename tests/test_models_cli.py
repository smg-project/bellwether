"""Tests for ``bellwether models`` end to end: pinned files from a filled cache, the Hub faked."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

import bellwether.models as models_command
from bellwether.cli import main
from bellwether.models import registry
from bellwether.models.hub import Details
from bellwether.models.pins import SGLANG, VLLM

DATA = Path(__file__).parent / "data" / "models"


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A registry cache holding the excerpts where the pinned files go, pinned by the excerpts' own sha256.

    The pinned files are checked on every use, so the excerpts stand in for them only under pins of their own.
    """
    root = tmp_path / "cache"
    excerpts = {"tests/models/registry.py": DATA / "vllm-registry-excerpt.py.txt"} | {
        page: DATA / f"sglang-{Path(page).stem}-excerpt.mdx" for page in SGLANG.files
    }
    for pin in (VLLM, SGLANG):
        files = {}
        for rel in pin.files:
            content = excerpts[rel].read_bytes()
            path = root / pin.engine / pin.commit / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            files[rel] = hashlib.sha256(content).hexdigest()
        monkeypatch.setattr(registry, pin.engine.upper(), replace(pin, files=files, dirs={}))  # no code registry here
    return root


def test_help_names_the_models_command(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "models" in capsys.readouterr().out


def test_registry_only_writes_the_list_and_says_what_is_in_it(cache: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "lists" / "models.jsonl"
    assert main(["models", "--registry-only", "--out", str(out), "--cache", str(cache)]) == 0
    header, *lines = out.read_text().splitlines()
    assert json.loads(header) == {  # what the list was built from: the pins, and no Hub
        "registries": {
            "vllm": {"repo": "vllm-project/vllm", "ref": "v0.31.0", "commit": VLLM.commit},
            "sglang": {"repo": "sgl-project/sglang", "ref": "7d22b7a8", "commit": SGLANG.commit},
        },
        "hub": None,
    }
    rows = [json.loads(line) for line in lines]
    assert len(rows) == 28
    assert rows[0]["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert {row["status"] for row in rows} == {"unchecked", "no-checkpoint-named"}
    printed = capsys.readouterr()
    assert f"28 rows in {out}" in printed.out
    assert "tier 1: 2, tier 2: 0, tier 3: 3, left to the Hub: 23" in printed.out
    assert {row["tier"] for row in rows} == {1, 3, None}
    assert "text: 8, multimodal: 7, left to the Hub: 13" in printed.out
    assert "vllm FunAudioChatForConditionalGeneration: 'funaudiochat' is not a Hugging Face id" in printed.err
    assert "tier 1 names missing from the list: MiniMaxAI/MiniMax-M3, tencent/Hy4-preview" in printed.err
    assert "gpt-oss, set aside: lmsys/gpt-oss-20b-bf16, openai/gpt-oss-120b, openai/gpt-oss-20b" in printed.err


class OneModelHub:
    """A Hub that has Qwen3-8B and nothing else, and lists nothing."""

    def list_models(self, org: str) -> list:
        return []

    def model(self, model_id: str, revision: str | None = None, tokenizer: str | None = None) -> Details | None:
        if model_id != "Qwen/Qwen3-8B":
            return None
        return Details(model_id, "b968826d", date(2025, 4, 27), 900, (), ("Qwen3ForCausalLM",), True, False)


def test_without_registry_only_the_hub_is_asked(cache: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(models_command, "HfHub", OneModelHub)
    out = tmp_path / "models.jsonl"
    assert main(["models", "--out", str(out), "--cache", str(cache)]) == 0
    rows = {row["model"]: row for row in map(json.loads, out.read_text().splitlines()[1:])}
    assert (rows["Qwen/Qwen3-8B"]["revision"], rows["Qwen/Qwen3-8B"]["status"]) == ("b968826d", "pending")
    assert rows["Qwen/Qwen3-0.6B"]["status"] == "not-on-hub"
    assert "Qwen: 0 listed, 0 added" in capsys.readouterr().err


def test_the_registry_only_list_goes_to_the_committed_file_and_a_hub_list_to_runs_under_its_date(
    cache: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["models", "--registry-only", "--cache", str(cache)]) == 0
    committed = (tmp_path / "models.jsonl").read_text()
    monkeypatch.setattr(models_command, "HfHub", OneModelHub)
    assert main(["models", "--cache", str(cache)]) == 0
    today = datetime.now(UTC).date().isoformat()
    header = json.loads((tmp_path / "runs" / f"models-{today}.jsonl").read_text().splitlines()[0])
    assert header["hub"] == today  # the day the Hub was read: tier 2 and downloads depend on it
    assert (tmp_path / "models.jsonl").read_text() == committed  # a Hub run leaves the committed list alone


def test_check_compares_a_fresh_registry_only_list_with_the_committed_one(cache: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "models.jsonl"
    argv = ["models", "--registry-only", "--cache", str(cache), "--out", str(out)]
    assert main(argv) == 0
    capsys.readouterr()
    assert main([*argv, "--check"]) == 0
    assert "equals a fresh build" in capsys.readouterr().out
    tampered = out.read_text().replace('"unchecked"', '"pending"', 1)
    out.write_text(tampered)
    assert main([*argv, "--check"]) == 1
    assert "differs from a fresh build" in capsys.readouterr().err
    assert out.read_text() == tampered  # a check writes nothing
    assert main(["models", "--check", "--cache", str(cache), "--out", str(out)]) == 2  # a Hub list changes daily
    assert "--registry-only" in capsys.readouterr().err


def test_a_pinned_file_out_of_reach_fails_the_run_with_what_to_do(tmp_path: Path, capsys) -> None:
    checkout = tmp_path / "vllm"
    checkout.mkdir()
    subprocess.run(["git", "-C", str(checkout), "init", "-q"], check=True)
    argv = ["models", "--registry-only", "--cache", str(tmp_path / "empty"), "--vllm-src", str(checkout)]
    assert main([*argv, "--out", str(tmp_path / "models.jsonl")]) == 1
    assert f"has no tests/models/registry.py at {VLLM.commit}" in capsys.readouterr().err
    assert not (tmp_path / "models.jsonl").exists()
