"""Tests for ``bellwether models`` end to end: pinned files from a filled cache, the Hub faked."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import date
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
        monkeypatch.setattr(registry, pin.engine.upper(), replace(pin, files=files))
    return root


def test_help_names_the_models_command(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "models" in capsys.readouterr().out


def test_registry_only_writes_the_list_and_says_what_is_in_it(cache: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "lists" / "models.jsonl"
    assert main(["models", "--registry-only", "--out", str(out), "--cache", str(cache)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 28
    assert rows[0]["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert {row["status"] for row in rows} == {"unchecked", "no-checkpoint-named"}
    printed = capsys.readouterr()
    assert f"28 rows in {out}" in printed.out
    assert "tier 1: 2, tier 2: 0, tier 3: 3, left to the Hub: 23" in printed.out
    assert {row["tier"] for row in rows} == {1, 3, None}
    assert "text: 17, multimodal: 11" in printed.out
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
    rows = {row["model"]: row for row in map(json.loads, out.read_text().splitlines())}
    assert (rows["Qwen/Qwen3-8B"]["revision"], rows["Qwen/Qwen3-8B"]["status"]) == ("b968826d", "pending")
    assert rows["Qwen/Qwen3-0.6B"]["status"] == "not-on-hub"
    assert "Qwen: 0 listed, 0 added" in capsys.readouterr().err


def test_a_pinned_file_out_of_reach_fails_the_run_with_what_to_do(tmp_path: Path, capsys) -> None:
    checkout = tmp_path / "vllm"
    checkout.mkdir()
    subprocess.run(["git", "-C", str(checkout), "init", "-q"], check=True)
    argv = ["models", "--registry-only", "--cache", str(tmp_path / "empty"), "--vllm-src", str(checkout)]
    assert main([*argv, "--out", str(tmp_path / "models.jsonl")]) == 1
    assert f"has no tests/models/registry.py at {VLLM.commit}" in capsys.readouterr().err
    assert not (tmp_path / "models.jsonl").exists()
