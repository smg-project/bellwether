"""Tests for ``bellwether gaps`` on small synthetic registry trees (no engine code is copied here)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from bellwether.cli import main
from bellwether.gaps.fetch import fetch_registry_files
from bellwether.gaps.manifests import read_manifests
from bellwether.gaps.matrix import build_matrix, render_json, render_markdown
from bellwether.gaps.names import Aliases, default_aliases_path, normalize
from bellwether.gaps.registries import read_sglang, read_smg, read_vllm


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def vllm_root(tmp_path: Path) -> Path:
    root = tmp_path / "vllm"
    write(root, "COMMIT.txt", "abc123 2026-10-04 a commit subject\n")
    write(
        root,
        "vllm/tool_parsers/__init__.py",
        "_TOOL_PARSERS_TO_REGISTER = {\n"
        '    "hermes": ("hermes_tool_parser", "HermesParser"),\n'
        '    "qwen3_xml": ("qwen3_tool_parser", "Qwen3XmlParser"),\n'
        '    "qwen3_coder": ("qwen3_tool_parser", "Qwen3XmlParser"),\n'
        '    "openai": ("gptoss_tool_parser", "GptOssParser"),\n'
        "}\n",
    )
    write(
        root,
        "vllm/reasoning/__init__.py",
        '_REASONING_PARSERS_TO_REGISTER = {"deepseek_r1": ("r1", "R1Parser"), "qwen3": ("q3", "Qwen3Reasoning")}\n',
    )
    write(
        root,
        "vllm/renderers/registry.py",
        '_VLLM_RENDERERS = {"hf": ("hf", "HfRenderer"), "kimi_k3": ("kimi_k3", "KimiK3Renderer")}\n',
    )
    write(
        root,
        "vllm/tokenizers/registry.py",
        '_VLLM_TOKENIZERS = {"hf": ("hf", "CachedHfTokenizer"), "kimi_k3": ("hf", "CachedHfTokenizer")}\n',
    )
    return root


@pytest.fixture
def sglang_root(tmp_path: Path) -> Path:
    root = tmp_path / "sglang"
    srt = "python/sglang/srt"
    write(
        root,
        f"{srt}/function_call/parser_names.py",
        'TOOL_CALL_PARSER_NAMES = [\n    "hermes",\n    "qwen25",\n    "qwen3_coder",\n    "gpt-oss",\n]\n',
    )
    write(
        root,
        f"{srt}/function_call/function_call_parser.py",
        "class FunctionCallParser:\n"
        "    ToolCallParserEnum: dict[str, type] = {\n"
        '        "hermes": HermesDetector,\n'
        '        "qwen25": Qwen25Detector,\n'
        '        "qwen3_coder": Qwen3CoderDetector,\n'
        '        "gpt-oss": GptOssDetector,\n'
        "    }\n",
    )
    write(root, f"{srt}/parser/reasoning_parser_names.py", 'REASONING_PARSER_NAMES = ["deepseek-r1", "qwen3"]\n')
    write(
        root,
        f"{srt}/parser/reasoning_parser.py",
        'class ReasoningParser:\n    DetectorMap = {"deepseek-r1": R1Detector, "qwen3": Qwen3Detector}\n',
    )
    write(root, f"{srt}/parser/inkling_renderer.py", "class InklingRenderer:\n    pass\n")
    write(root, f"{srt}/parser/inkling_tokenizer.py", "class InklingTokenizer:\n    pass\n")
    write(root, f"{srt}/tokenizer/tiktoken_tokenizer.py", "class TiktokenTokenizer:\n    pass\n")
    return root


@pytest.fixture
def smg_root(tmp_path: Path) -> Path:
    root = tmp_path / "smg"
    write(
        root,
        "crates/tool_parser/src/factory.rs",
        "impl ParserFactory {\n"
        "    pub fn new() -> Self {\n"
        '        registry.register_parser("qwen", || Box::new(QwenParser::new()));\n'
        "        registry.register_parser_with_structural_tag(\n"
        '            "mistral",\n'
        "            || Box::new(MistralParser::new()),\n"
        "            MistralParser::build_structural_tag,\n"
        "        );\n"
        '        registry.register_parser("qwen_xml", || Box::new(QwenXmlParser::new()));\n'
        '        registry.register_parser("sarashina", || Box::new(SarashinaParser::new()));\n'
        '        registry.map_model("qwen*", "qwen");\n'
        '        registry.map_model("Qwen/Qwen3-Coder*", "qwen_xml");\n'
        "    }\n}\n",
    )
    write(
        root,
        "crates/reasoning_parser/src/factory.rs",
        '        registry.register_parser("deepseek_r1", || Box::new(DeepSeekR1Parser::new()));\n'
        '        registry.register_parser("deepseek_v31", || {\n'
        "            let config = ParserConfig { think_start_token: None };\n"
        '            Box::new(BaseReasoningParser::new(config).with_model_type("deepseek_v31".to_string()))\n'
        "        });\n"
        '        registry.register_pattern("deepseek-r1", "deepseek_r1");\n'
        '        registry.register_pattern("deepseek-v3.1", "deepseek_v31");\n',
    )
    write(
        root,
        "crates/tokenizer/src/huggingface.rs",
        "#[derive(Debug, Clone, Copy)]\nenum Renderer {\n    Jinja,\n    DeepseekV32,\n"
        "    DeepseekV4(deepseek_v4::EffortEncoding), // comment\n}\n",
    )
    write(root, "crates/tokenizer/src/encoders/mod.rs", "pub mod kimi_k3_xtml;\n")
    write(root, "crates/tokenizer/src/encoders/kimi_k3_xtml.rs", "// encoder\n")
    write(root, "crates/tokenizer/src/encoders/deepseek_v32.rs", "// encoder behind Renderer::DeepseekV32\n")
    write(root, "crates/tokenizer/src/encoders/deepseek_common.rs", "// shared helpers\n")
    write(
        root,
        "crates/tokenizer/src/factory.rs",
        "pub enum TokenizerType {\n    HuggingFace(String),\n    Mock,\n    Tiktoken(String),\n}\n",
    )
    write(root, "model_gateway/src/routers/grpc/harmony/mod.rs", "//! Harmony pipeline\n")
    return root


@pytest.fixture
def aliases_path(tmp_path: Path) -> Path:
    return write(
        tmp_path,
        "aliases.toml",
        "schema_version = 1\n"
        '[tool.vllm]\nopenai = "gptoss"\n'
        '[tool.sglang]\nqwen25 = "hermes"\n'
        '[tool.smg]\nqwen = "hermes"\nqwen_xml = "qwen3xml"\nharmony = "gptoss"\n'
        '[renderer.smg]\njinja = "hf"\nkimi_k3_xtml = "kimik3"\n'
        '[tokenizer.smg]\nhuggingface = "hf"\n',
    )


def test_vllm_reader_reads_the_four_tables(vllm_root: Path) -> None:
    reg = read_vllm([vllm_root])
    assert reg.commit == "abc123"
    assert reg.names("tool") == ["hermes", "openai", "qwen3_coder", "qwen3_xml"]
    assert reg.names("reasoning") == ["deepseek_r1", "qwen3"]
    assert reg.names("renderer") == ["hf", "kimi_k3"]
    assert reg.names("tokenizer") == ["hf", "kimi_k3"]
    impls = {e.name: e.impl for e in reg.entries if e.kind == "tool"}
    assert impls["qwen3_xml"] == impls["qwen3_coder"] == "Qwen3XmlParser"
    assert reg.notes == []


def test_vllm_reader_notes_a_missing_table(tmp_path: Path) -> None:
    root = tmp_path / "partial"
    write(root, "vllm/tool_parsers/__init__.py", '_TOOL_PARSERS_TO_REGISTER = {"hermes": ("h", "HermesParser")}\n')
    reg = read_vllm([root])
    assert reg.names("tool") == ["hermes"]
    assert reg.names("reasoning") == []
    assert any("vllm/reasoning/__init__.py not found" in n for n in reg.notes)
    assert reg.commit == "unknown"


def test_readers_take_each_file_from_the_first_root_that_has_it(vllm_root: Path, tmp_path: Path) -> None:
    extra = tmp_path / "fetched"
    write(extra, "vllm/tool_parsers/__init__.py", '_TOOL_PARSERS_TO_REGISTER = {"other": ("o", "OtherParser")}\n')
    reg = read_vllm([vllm_root, extra])
    assert "other" not in reg.names("tool")  # the first root wins
    partial = tmp_path / "partial"
    write(partial, "vllm/reasoning/__init__.py", '_REASONING_PARSERS_TO_REGISTER = {"only": ("o", "OnlyParser")}\n')
    reg = read_vllm([partial, vllm_root])
    assert reg.names("reasoning") == ["only"]
    assert reg.names("tool") == ["hermes", "openai", "qwen3_coder", "qwen3_xml"]


def test_sglang_reader_names_impls_and_native_modules(sglang_root: Path) -> None:
    reg = read_sglang([sglang_root])
    tools = {e.name: e.impl for e in reg.entries if e.kind == "tool"}
    assert tools == {
        "hermes": "HermesDetector",
        "qwen25": "Qwen25Detector",
        "qwen3_coder": "Qwen3CoderDetector",
        "gpt-oss": "GptOssDetector",
    }
    assert reg.names("reasoning") == ["deepseek-r1", "qwen3"]
    renderers = {e.name: e.impl for e in reg.entries if e.kind == "renderer"}
    assert renderers == {"hf": "jinja chat template", "inkling": "InklingRenderer"}
    tokenizers = {e.name: e.impl for e in reg.entries if e.kind == "tokenizer"}
    assert tokenizers == {
        "hf": "transformers tokenizer",
        "inkling": "InklingTokenizer",
        "tiktoken": "TiktokenTokenizer",
    }


def test_sglang_reader_falls_back_to_the_class_map_for_names(tmp_path: Path) -> None:
    root = tmp_path / "sglang"
    write(
        root,
        "python/sglang/srt/function_call/function_call_parser.py",
        'class P:\n    ToolCallParserEnum = {"hermes": HermesDetector}\n',
    )
    reg = read_sglang([root])
    assert reg.names("tool") == ["hermes"]
    assert any("taken from ToolCallParserEnum" in n for n in reg.notes)


def test_smg_reader_registrations_patterns_enums_and_harmony(smg_root: Path) -> None:
    reg = read_smg([smg_root])
    tools = {e.name: e.impl for e in reg.entries if e.kind == "tool"}
    assert tools == {
        "qwen": "QwenParser",
        "mistral": "MistralParser",  # the multi-line structural-tag form
        "qwen_xml": "QwenXmlParser",
        "sarashina": "SarashinaParser",
        "harmony": "HarmonyDetector",
    }
    assert reg.model_patterns["tool"] == {"qwen": ["qwen*"], "qwen_xml": ["Qwen/Qwen3-Coder*"]}
    reasoning = {e.name: e.impl for e in reg.entries if e.kind == "reasoning"}
    assert reasoning == {
        "deepseek_r1": "DeepSeekR1Parser",
        "deepseek_v31": "BaseReasoningParser",
        "harmony": "HarmonyDetector",
    }
    assert reg.model_patterns["reasoning"] == {"deepseek_r1": ["deepseek-r1"], "deepseek_v31": ["deepseek-v3.1"]}
    renderers = {e.name: e.impl for e in reg.entries if e.kind == "renderer"}
    assert renderers == {
        "jinja": "Renderer::Jinja",
        "deepseek_v32": "Renderer::DeepseekV32",
        "deepseek_v4": "Renderer::DeepseekV4",
        "kimi_k3_xtml": "encoders::kimi_k3_xtml",
    }
    assert reg.names("tokenizer") == ["huggingface", "tiktoken"]  # Mock is a test double


def test_normalize_drops_case_and_separators() -> None:
    assert normalize("minimax-m2") == normalize("minimax_m2") == normalize("MiniMaxM2") == "minimaxm2"
    assert normalize("kimi_k3") == normalize("kimik3")


def test_aliases_reject_a_target_that_is_not_normalized(tmp_path: Path) -> None:
    path = write(tmp_path, "bad.toml", 'schema_version = 1\n[tool.smg]\nqwen = "Hermes"\n')
    with pytest.raises(ValueError, match="not a normalized id"):
        Aliases.load(path)
    path = write(tmp_path, "kind.toml", 'schema_version = 1\n[grammar.smg]\nx = "y"\n')
    with pytest.raises(ValueError, match="unknown kind"):
        Aliases.load(path)


def test_shipped_aliases_load_and_only_map_to_normalized_ids() -> None:
    aliases = Aliases.load(default_aliases_path())
    assert aliases.schema_version == 1
    assert aliases.canonical("tool", "smg", "qwen_xml") == "qwen3xml"
    assert aliases.canonical("tool", "vllm", "qwen3_xml") == "qwen3xml"
    assert aliases.canonical("tool", "sglang", "qwen3_coder") == "qwen3xml"  # one XML-parameter format
    assert aliases.canonical("tool", "vllm", "kimi_k3") == aliases.canonical("tool", "sglang", "kimi_k3") == "kimik3"


def _matrix(vllm_root: Path, sglang_root: Path, smg_root: Path, aliases_path: Path, manifests=()):
    registries = {"vllm": read_vllm([vllm_root]), "sglang": read_sglang([sglang_root]), "smg": read_smg([smg_root])}
    return build_matrix(registries, Aliases.load(aliases_path), list(manifests))


def test_matrix_merges_through_aliases_and_classifies_rows(vllm_root, sglang_root, smg_root, aliases_path) -> None:
    matrix = _matrix(vllm_root, sglang_root, smg_root, aliases_path)
    tools = matrix.kinds["tool"]
    rows = {r.key: r for r in tools.rows}
    assert rows["hermes"].cells == {
        "vllm": [("hermes", "HermesParser")],
        "sglang": [("hermes", "HermesDetector"), ("qwen25", "Qwen25Detector")],
        "smg": [("qwen", "QwenParser")],
    }
    assert rows["hermes"].label == "hermes"
    assert rows["gptoss"].cells["smg"] == [("harmony", "HarmonyDetector")]
    assert rows["gptoss"].cells["sglang"] == [("gpt-oss", "GptOssDetector")]
    assert tools.counts == {"vllm": 4, "sglang": 4, "smg": 5}
    assert tools.engine_only == ["qwen3coder"]  # vLLM qwen3_coder and SGLang qwen3_coder, no SMG alias
    assert tools.smg_only == ["mistral", "sarashina"]  # neither synthetic engine registers them
    assert tools.engines_differ == ["qwen3xml"]  # vLLM has it, SGLang does not; SMG-only rows do not count
    assert ("vllm", "Qwen3XmlParser", ["qwen3coder", "qwen3xml"]) in tools.shared_impl
    renderers = {r.key: r for r in matrix.kinds["renderer"].rows}
    assert renderers["hf"].cells["smg"] == [("jinja", "Renderer::Jinja")]
    assert renderers["kimik3"].cells == {
        "vllm": [("kimi_k3", "KimiK3Renderer")],
        "sglang": [],
        "smg": [("kimi_k3_xtml", "encoders::kimi_k3_xtml")],
    }
    assert matrix.kinds["tokenizer"].engine_only == ["inkling", "kimik3"]  # SMG has hf and tiktoken only


def test_manifests_mark_rows_with_fixtures(vllm_root, sglang_root, smg_root, aliases_path, tmp_path) -> None:
    fixtures = tmp_path / "fixtures"
    write(
        fixtures,
        "qwen3-coder-30b/manifest.toml",
        'model = "Qwen/Qwen3-Coder-30B"\n[smg]\ntool_parser = "qwen_xml"\nreasoning_parser = "deepseek_r1"\n',
    )
    manifests = read_manifests(fixtures)
    assert manifests[0].smg == {"tool": "qwen_xml", "reasoning": "deepseek_r1"}
    matrix = _matrix(vllm_root, sglang_root, smg_root, aliases_path, manifests)
    rows = {r.key: r for r in matrix.kinds["tool"].rows}
    assert rows["qwen3xml"].fixtures == ["qwen3-coder-30b"]
    assert rows["hermes"].fixtures == []


def test_json_is_canonical_and_stable(vllm_root, sglang_root, smg_root, aliases_path) -> None:
    matrix = _matrix(vllm_root, sglang_root, smg_root, aliases_path)
    first, second = render_json(matrix), render_json(matrix)
    assert first == second
    data = json.loads(first)
    assert data["schema_version"] == 1
    assert list(data) == sorted(data)
    assert data["sources"]["vllm"]["commit"] == "abc123"
    assert data["sources"]["smg"]["model_patterns"]["tool"] == {"qwen": ["qwen*"], "qwen_xml": ["Qwen/Qwen3-Coder*"]}
    assert first.endswith("\n")


def test_markdown_lists_the_gaps(vllm_root, sglang_root, smg_root, aliases_path) -> None:
    text = render_markdown(_matrix(vllm_root, sglang_root, smg_root, aliases_path))
    assert "## Tool-call parsers (vLLM 4, SGLang 4, SMG 5 names; " in text
    assert (
        "| hermes | hermes (HermesParser) | hermes (HermesDetector), qwen25 (Qwen25Detector) | qwen (QwenParser) |  |"
        in text
    )
    assert "Engine names with no SMG counterpart (1): qwen3coder" in text
    assert "- vLLM `Qwen3XmlParser`: qwen3coder, qwen3xml" in text


def test_cli_gaps_runs_on_checkouts(vllm_root, sglang_root, smg_root, aliases_path, tmp_path, capsys) -> None:
    out = tmp_path / "out" / "gaps.json"
    argv = [
        "gaps",
        "--vllm-src",
        str(vllm_root),
        "--sglang-src",
        str(sglang_root),
        "--smg-src",
        str(smg_root),
        "--aliases",
        str(aliases_path),
        "--fixtures",
        str(tmp_path / "none"),
        "--format",
        "json",
        "--out",
        str(out),
    ]
    assert main(argv) == 0
    assert json.loads(out.read_text())["kinds"]["tool"]["engine_only"] == ["qwen3coder"]
    assert main(argv[:-4]) == 0
    assert "# Coverage matrix" in capsys.readouterr().out


def test_cli_gaps_requires_an_engine_source(smg_root, capsys) -> None:
    assert main(["gaps", "--smg-src", str(smg_root)]) == 2
    assert "--vllm-src or --vllm-ref" in capsys.readouterr().err


def test_fetch_resolves_a_ref_and_caches_files(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.host == "api.github.com":
            return httpx.Response(200, json={"sha": "f" * 40})
        return httpx.Response(200, content=b"_TOOL_PARSERS_TO_REGISTER = {}\n")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    root = fetch_registry_files("vllm", "v9.9.9", tmp_path / "cache", client=client)
    assert root == tmp_path / "cache" / "vllm" / ("f" * 40)
    assert (root / "vllm/tool_parsers/__init__.py").read_bytes() == b"_TOOL_PARSERS_TO_REGISTER = {}\n"
    assert (root / "COMMIT.txt").read_text().startswith("f" * 40)
    assert len(calls) == 5  # one resolve, four files
    fetch_registry_files("vllm", "f" * 40, tmp_path / "cache", client=client)
    assert len(calls) == 5  # a full sha with cached files makes no request


def test_fetch_lists_native_modules_for_sglang(tmp_path: Path) -> None:
    listed: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/contents/" in url:
            listed.append(url)
            if "/srt/tokenizer?" in url:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(
                200,
                json=[
                    {"type": "file", "name": "inkling_renderer.py"},
                    {"type": "file", "name": "reasoning_parser.py"},
                    {"type": "dir", "name": "chat_parsing"},
                ],
            )
        return httpx.Response(200, content=b"x = 1\n")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    root = fetch_registry_files("sglang", "e" * 40, tmp_path / "cache", client=client)
    srt = root / "python/sglang/srt"
    assert (srt / "parser/inkling_renderer.py").is_file()
    assert not (srt / "parser/other.py").exists()
    assert len(listed) == 2
    fetch_registry_files("sglang", "e" * 40, tmp_path / "cache", client=client)
    assert len(listed) == 2  # a listed directory is not listed again


def test_fetch_lists_again_after_an_interrupted_download(tmp_path: Path) -> None:
    listed: list[str] = []
    fail_once = {"inkling_renderer.py": True}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/contents/" in url:
            listed.append(url)
            return httpx.Response(200, json=[{"type": "file", "name": "inkling_renderer.py"}])
        if url.endswith("inkling_renderer.py") and fail_once.pop("inkling_renderer.py", False):
            return httpx.Response(502, content=b"bad gateway")
        return httpx.Response(200, content=b"x = 1\n")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        fetch_registry_files("sglang", "d" * 40, tmp_path / "cache", client=client)
    root = fetch_registry_files("sglang", "d" * 40, tmp_path / "cache", client=client)
    assert (root / "python/sglang/srt/parser/inkling_renderer.py").is_file()
    assert len(listed) == 4  # two directories, listed on both runs: the first run wrote no marker
    fetch_registry_files("sglang", "d" * 40, tmp_path / "cache", client=client)
    assert len(listed) == 4
