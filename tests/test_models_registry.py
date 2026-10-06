"""Tests for reading the engines' registries: vLLM's test registry and SGLang's supported-models docs.

The excerpts under ``tests/data/models/`` are copied verbatim from the pinned files. The tests at
the end read the pinned files themselves when a run of ``bellwether models`` has cached them, and
are skipped otherwise.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import httpx
import pytest

from bellwether.models.pins import (
    SGLANG,
    SGLANG_MODELS,
    SGLANG_PAGES,
    SGLANG_PROCESSORS,
    VLLM,
    Pin,
    PinError,
    pinned_root,
)
from bellwether.models.registry import read_sglang, read_sglang_code, read_vllm

DATA = Path(__file__).parent / "data" / "models"
TEXT = "_TEXT_GENERATION_EXAMPLE_MODELS"
MULTIMODAL = "_MULTIMODAL_EXAMPLE_MODELS"
GENERATIVE_PAGE = "docs/docs/supported-models/generative_models.mdx"
MULTIMODAL_PAGE = "docs/docs/supported-models/multimodal_language_models.mdx"
DIFFUSION_PAGE = "docs/docs/supported-models/diffusion_language_models.mdx"


def excerpt(name: str) -> str:
    return (DATA / name).read_text()


def by_name(entries, table: str) -> dict:
    return {e.name: e for e in entries if e.table == table}


def test_vllm_reader_takes_the_default_and_the_extras_in_either_form() -> None:
    text = by_name(read_vllm(excerpt("vllm-registry-excerpt.py.txt")), TEXT)
    assert text["AfmoeForCausalLM"].checkpoints == ("arcee-ai/Trinity-Nano-Preview",)
    assert text["BloomForCausalLM"].checkpoints == ("bigscience/bloom-560m", "bigscience/bloomz-1b1")
    assert text["GPTBigCodeForCausalLM"].checkpoints == (
        "bigcode/starcoder",
        "bigcode/tiny_starcoder_py",
        "bigcode/gpt_bigcode-santacoder",
    )
    assert text["Qwen4ExpForCausalLM"].checkpoints == ("",)  # kept as written; the rows skip it


def test_vllm_reader_marks_what_is_generative_and_what_is_multimodal() -> None:
    entries = read_vllm(excerpt("vllm-registry-excerpt.py.txt"))
    assert {e.table for e in entries} == {
        TEXT,
        MULTIMODAL,
        "_EMBEDDING_EXAMPLE_MODELS",
        "_SPECULATIVE_DECODING_EXAMPLE_MODELS",
    }
    text, multimodal = by_name(entries, TEXT), by_name(entries, MULTIMODAL)
    assert all(e.generative and not e.multimodal for e in text.values())
    assert multimodal["Qwen2VLForConditionalGeneration"].generative
    assert multimodal["Qwen2VLForConditionalGeneration"].multimodal
    assert multimodal["Qwen3_5ForConditionalGeneration"].checkpoints == ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B")
    assert not multimodal["JinaVLForRanking"].generative
    assert not multimodal["Qwen3ASRForcedAlignerForTokenClassification"].generative
    embedding = by_name(entries, "_EMBEDDING_EXAMPLE_MODELS")
    assert embedding["Qwen2VLForConditionalGeneration"].checkpoints == ("MrLight/dse-qwen2-2b-mrl-v1",)
    assert not embedding["Qwen2VLForConditionalGeneration"].generative
    assert not by_name(entries, "_SPECULATIVE_DECODING_EXAMPLE_MODELS")["DFlashDraftModel"].generative
    assert {e.engine for e in entries} == {"vllm"}


def test_vllm_reader_refuses_a_checkpoint_it_cannot_read_as_written() -> None:
    source = (
        '_TEXT_GENERATION_EXAMPLE_MODELS = {"XForCausalLM": _HfExamplesInfo(SOME_NAME)}\n'
        "_MULTIMODAL_EXAMPLE_MODELS = {}\n"
    )
    with pytest.raises(ValueError, match="XForCausalLM"):
        read_vllm(source)


def test_vllm_reader_keeps_what_vllm_needs_to_load_a_checkpoint_vendor_code_a_revision_a_tokenizer() -> None:
    source = (  # written as vLLM's registry writes them at the pin
        "_TEXT_GENERATION_EXAMPLE_MODELS = {\n"
        '    "KimiK3ForCausalLM": _HfExamplesInfo("moonshotai/Kimi-K3", trust_remote_code=True),\n'
        '    "Qwen3ForCausalLM": _HfExamplesInfo("Qwen/Qwen3-8B"),\n'
        "}\n"
        "_MULTIMODAL_EXAMPLE_MODELS = {\n"
        '    "Ernie4_5_VLMoeForConditionalGeneration": _HfExamplesInfo(\n'
        '        "baidu/ERNIE-4.5-VL-28B-A3B-PT",\n'
        "        trust_remote_code=True,\n"
        '        revision="refs/pr/17",\n'
        "    ),\n"
        '    "Moondream3ForCausalLM": _HfExamplesInfo(\n'
        '        "moondream/moondream3-preview",\n'
        '        tokenizer="moondream/starmie-v1",\n'
        "        trust_remote_code=True,\n"
        "    ),\n"
        "}\n"
    )
    entries = {e.name: e for e in read_vllm(source)}
    assert [entries[name].vendor_code for name in entries] == [True, False, True, True]
    assert entries["Ernie4_5_VLMoeForConditionalGeneration"].revision == "refs/pr/17"
    assert entries["Moondream3ForCausalLM"].tokenizer == "moondream/starmie-v1"
    assert (entries["Qwen3ForCausalLM"].revision, entries["Qwen3ForCausalLM"].tokenizer) == (None, None)


def test_vllm_reader_refuses_a_load_setting_it_cannot_read_as_written() -> None:
    source = (
        '_TEXT_GENERATION_EXAMPLE_MODELS = {"XForCausalLM": _HfExamplesInfo("org/x", trust_remote_code=REMOTE)}\n'
        "_MULTIMODAL_EXAMPLE_MODELS = {}\n"
    )
    with pytest.raises(ValueError, match="XForCausalLM"):
        read_vllm(source)


def test_vllm_reader_refuses_a_registry_without_its_generative_tables() -> None:
    with pytest.raises(ValueError, match="_MULTIMODAL_EXAMPLE_MODELS"):
        read_vllm('_TEXT_GENERATION_EXAMPLE_MODELS = {"XForCausalLM": _HfExamplesInfo("org/x")}\n')


def test_sglang_reader_takes_every_id_in_the_example_column() -> None:
    entries = read_sglang(excerpt("sglang-generative_models-excerpt.mdx"), GENERATIVE_PAGE)
    assert [(e.name, e.checkpoints) for e in entries] == [
        ("DeepSeek (v1, v2, v3/R1)", ("deepseek-ai/DeepSeek-R1",)),
        ("GPT-OSS", ("openai/gpt-oss-20b", "openai/gpt-oss-120b")),
        (
            "Qwen (3.5, 3, 3MoE, 3Next, 2.5, 2 series)",
            ("Qwen/Qwen3.5-397B-A17B", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-Next-80B-A3B-Instruct"),
        ),
        (
            "Granite 3.0, 3.1 MoE (IBM)",
            ("ibm-granite/granite-3.0-3b-a800m-instruct", "ibm-granite/granite-3.1-3b-a800m-instruct"),
        ),
    ]
    assert all(
        e.engine == "sglang" and e.generative and not e.multimodal and e.table == GENERATIVE_PAGE for e in entries
    )


def test_sglang_reader_reads_every_table_on_a_page_whatever_its_example_column_is_called() -> None:
    multimodal = read_sglang(excerpt("sglang-multimodal_language_models-excerpt.mdx"), MULTIMODAL_PAGE)
    assert [(e.name, e.checkpoints) for e in multimodal] == [
        ("Qwen-VL", ("Qwen/Qwen3-VL-235B-A22B-Instruct",)),
        ("JetVLM", ()),
        ("Whisper", ("openai/whisper-large-v3",)),
        ("Qwen3-ASR (0.6B, 1.7B)", ("Qwen/Qwen3-ASR-1.7B",)),
    ]
    assert all(e.multimodal for e in multimodal)
    diffusion = read_sglang(excerpt("sglang-diffusion_language_models-excerpt.mdx"), DIFFUSION_PAGE)
    assert [e.checkpoints for e in diffusion] == [("inclusionAI/LLaDA2.0-flash",), ("JetLM/SDAR-8B-Chat",)]
    assert not any(e.multimodal for e in diffusion)


def test_sglang_reader_takes_the_id_after_prose_in_a_cell() -> None:
    page = (
        "<table><thead><tr><th>Family</th><th>Example Identifier</th></tr></thead><tbody><tr>"
        "<td><strong>A</strong> (x &amp; y)</td><td><em>e.g.</em> <code>org/a-1</code></td></tr></tbody></table>"
    )
    [entry] = read_sglang(page, MULTIMODAL_PAGE)
    assert (entry.name, entry.checkpoints) == ("A (x & y)", ("org/a-1",))


def test_sglang_reader_refuses_a_table_without_an_example_column() -> None:
    page = (
        "<table><thead><tr><th>Family</th><th>Id</th></tr></thead>"
        "<tbody><tr><td>A</td><td>`o/a`</td></tr></tbody></table>"
    )
    with pytest.raises(ValueError, match="example column"):
        read_sglang(page, GENERATIVE_PAGE)


def test_sglang_reader_refuses_a_page_without_its_table() -> None:
    with pytest.raises(ValueError, match="no table"):
        read_sglang("<html><body>429 Too Many Requests</body></html>", GENERATIVE_PAGE)


def test_sglang_code_names_each_modules_entry_class_and_the_classes_its_multimodal_processors_serve() -> None:
    models = {  # in the forms SGLang's modules use at the pin
        "qwen3.py": b"class Qwen3ForCausalLM(nn.Module):\n    pass\n\n\nEntryClass = Qwen3ForCausalLM\n",
        "zaya.py": (  # a module may begin with a byte-order mark
            b"\xef\xbb\xbf# Copyright\nclass ZayaForCausalLM: ...\nEntryClass = [ZayaForCausalLM]\n"
        ),
        "moss_vl.py": b"class MossVLForConditionalGeneration: ...\nEntryClass = (MossVLForConditionalGeneration,)\n",
        "dots3.py": (
            b"from sglang.srt.models.dots3_common.model import Dots3NoteForCausalLM\n"
            b"EntryClass = Dots3NoteForCausalLM\n"
        ),
        "renamed.py": b"from sglang.srt.models.base import BaseForCausalLM as Renamed\nEntryClass = [Renamed]\n",
        "utils.py": b"def helper():\n    return 1\n",  # no EntryClass: not a model module
    }
    processors = {
        "moss_vl.py": (
            b"from sglang.srt.models.moss_vl import MossVLForConditionalGeneration as MossVL\n"
            b"class MossVLProcessor(BaseMultimodalProcessor):\n    models = [MossVL]\n"
        ),
        "glm4v.py": (
            b"from sglang.srt.models.glm4v import Glm4vForConditionalGeneration\n"
            b"class Glm4vImageProcessor(BaseMultimodalProcessor):\n"
            b"    models = [m for m in [Glm4vForConditionalGeneration] if m is not None]\n"
        ),
        "executor.py": b"class Executor:\n    pass\n",  # serves no model
        "base_processor.py": b"class BaseMultimodalProcessor:\n    models = []\n",  # the base serves none itself
    }
    served = read_sglang_code(models, processors)
    assert served.architectures == {
        "Qwen3ForCausalLM",
        "ZayaForCausalLM",
        "MossVLForConditionalGeneration",
        "Dots3NoteForCausalLM",
        "BaseForCausalLM",  # SGLang registers a class by its own name, whatever the module calls it
    }
    assert served.multimodal == {"MossVLForConditionalGeneration", "Glm4vForConditionalGeneration"}


@pytest.mark.parametrize(
    ("models", "processors", "module"),
    [
        ({"weird.py": b"EntryClass = make_classes()\n"}, {}, "weird.py"),
        ({"a.py": b"class A: ...\nEntryClass = A\n"}, {"odd.py": b"class P:\n    models = [load()]\n"}, "odd.py"),
    ],
)
def test_sglang_code_refuses_what_it_cannot_read_as_written(models: dict, processors: dict, module: str) -> None:
    with pytest.raises(ValueError, match=module):
        read_sglang_code(models, processors)


def test_the_pins_are_the_designs_refs() -> None:
    assert (VLLM.ref, VLLM.commit) == ("v0.31.0", "db9527a46873454610df6dbedf79a36d6bf1a7f6")
    assert tuple(VLLM.files) == ("tests/models/registry.py",)
    assert SGLANG.commit == "7d22b7a8750f53a04e41a5a5671f9a56ab6cd001"
    assert tuple(SGLANG.files) == tuple(SGLANG_PAGES) == (GENERATIVE_PAGE, MULTIMODAL_PAGE, DIFFUSION_PAGE)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git(repo: Path, *args: str) -> str:
    identity = ("-c", "user.name=bellwether tests", "-c", "user.email=tests@example.invalid")
    out = subprocess.run(["git", "-C", str(repo), *identity, *args], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_pinned_files_come_from_a_checkout_at_the_pinned_commit_not_its_working_tree(tmp_path: Path) -> None:
    repo = tmp_path / "checkout"
    (repo / "tests/models").mkdir(parents=True)
    git(repo, "init", "-q")
    (repo / "tests/models/registry.py").write_text("pinned = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "pinned")
    pinned = git(repo, "rev-parse", "HEAD")
    (repo / "tests/models/registry.py").write_text("later = 2\n")
    git(repo, "commit", "-q", "-am", "later")
    (repo / "tests/models/registry.py").write_text("uncommitted = 3\n")
    pin = Pin("vllm", "example/vllm", "v1", pinned, {"tests/models/registry.py": sha256("pinned = 1\n")})
    root = pinned_root(pin, tmp_path / "cache", checkout=repo)
    assert root == tmp_path / "cache" / "vllm" / pinned
    assert (root / "tests/models/registry.py").read_text() == "pinned = 1\n"


def test_a_checkout_without_the_pinned_commit_is_reported(tmp_path: Path) -> None:
    repo = tmp_path / "checkout"
    repo.mkdir()
    git(repo, "init", "-q")
    pin = Pin("vllm", "example/vllm", "v1", "f" * 40, {"tests/models/registry.py": sha256("pinned = 1\n")})
    with pytest.raises(PinError, match="f{40}"):
        pinned_root(pin, tmp_path / "cache", checkout=repo)


def test_pinned_files_are_downloaded_once_and_then_read_from_the_cache(tmp_path: Path) -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, content=b"pinned = 1\n")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    pin = Pin("sglang", "example/sglang", "abc", "a" * 40, {f"docs/{n}.mdx": sha256("pinned = 1\n") for n in "ab"})
    root = pinned_root(pin, tmp_path / "cache", client=client)
    assert urls == [f"https://raw.githubusercontent.com/example/sglang/{'a' * 40}/docs/{n}.mdx" for n in "ab"]
    assert (root / "docs/b.mdx").read_text() == "pinned = 1\n"
    pinned_root(pin, tmp_path / "cache", client=client)
    assert len(urls) == 2  # a full cache makes no request


def test_a_failed_download_says_how_to_read_the_files_offline(tmp_path: Path) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404)))
    pin = Pin("vllm", "example/vllm", "v1", "b" * 40, {"tests/models/registry.py": sha256("pinned = 1\n")})
    with pytest.raises(PinError, match="--vllm-src"):
        pinned_root(pin, tmp_path / "cache", client=client)
    assert not (tmp_path / "cache" / "vllm" / ("b" * 40) / "tests/models/registry.py").exists()


def serving(content: bytes, urls: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, content=content)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_a_cached_file_that_is_not_the_pinned_content_is_fetched_again(tmp_path: Path) -> None:
    pin = Pin("vllm", "example/vllm", "v1", "c" * 40, {"tests/models/registry.py": sha256("pinned = 1\n")})
    stale = tmp_path / "cache" / "vllm" / ("c" * 40) / "tests/models/registry.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("<html>an error page saved in its place</html>")
    urls: list[str] = []
    root = pinned_root(pin, tmp_path / "cache", client=serving(b"pinned = 1\n", urls))
    assert (root / "tests/models/registry.py").read_text() == "pinned = 1\n"
    assert len(urls) == 1


def test_a_downloaded_file_that_is_not_the_pinned_content_is_refused(tmp_path: Path) -> None:
    pin = Pin("vllm", "example/vllm", "v1", "d" * 40, {"tests/models/registry.py": sha256("pinned = 1\n")})
    with pytest.raises(PinError, match="sha256"):
        pinned_root(pin, tmp_path / "cache", client=serving(b"<html>429 Too Many Requests</html>", []))
    assert not (tmp_path / "cache" / "vllm" / ("d" * 40) / "tests/models/registry.py").exists()


def test_a_checkout_file_that_is_not_the_pinned_content_is_refused(tmp_path: Path) -> None:
    repo = tmp_path / "checkout"
    (repo / "tests/models").mkdir(parents=True)
    git(repo, "init", "-q")
    (repo / "tests/models/registry.py").write_text("other = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "other")
    pin = Pin("vllm", "example/vllm", "v1", git(repo, "rev-parse", "HEAD"), {"tests/models/registry.py": sha256("x")})
    with pytest.raises(PinError, match="sha256"):
        pinned_root(pin, tmp_path / "cache", checkout=repo)


def sha256sum(files: dict[str, str]) -> str:
    """A directory's pin as the sha256sum tool would give it: one "<sha256>  <name>" line per module, by name."""
    lines = "".join(f"{sha256(text)}  {name}\n" for name, text in sorted(files.items()))
    return sha256(lines)


MODULES = {"a.py": "EntryClass = A\n", "b.py": "EntryClass = B\n"}


def test_a_pinned_directory_comes_from_a_checkout_as_its_top_level_modules(tmp_path: Path) -> None:
    repo = tmp_path / "checkout"
    models = repo / "python/pkg/models"
    (models / "sub").mkdir(parents=True)
    git(repo, "init", "-q")
    for name, text in MODULES.items():
        (models / name).write_text(text)
    (models / "sub" / "__init__.py").write_text("")  # a subpackage, which SGLang's registry skips
    (models / "README.md").write_text("not a module")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "pinned")
    pin = Pin(
        "sglang", "example/sglang", "x", git(repo, "rev-parse", "HEAD"), {}, {"python/pkg/models": sha256sum(MODULES)}
    )
    root = pinned_root(pin, tmp_path / "cache", checkout=repo)
    assert {p.name for p in (root / "python/pkg/models").glob("*.py")} == {"a.py", "b.py"}


def github(listing: list[dict], files: dict[str, str], urls: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        if request.url.host == "api.github.com":
            return httpx.Response(200, json=listing)
        return httpx.Response(200, content=files[request.url.path.rsplit("/", 1)[1]].encode())

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_a_pinned_directory_from_github_is_listed_once_and_then_read_from_the_cache(tmp_path: Path) -> None:
    listing = [
        {"name": "a.py", "type": "file"},
        {"name": "b.py", "type": "file"},
        {"name": "sub", "type": "dir"},
        {"name": "README.md", "type": "file"},
    ]
    urls: list[str] = []
    pin = Pin("sglang", "example/sglang", "x", "e" * 40, {}, {"python/pkg/models": sha256sum(MODULES)})
    root = pinned_root(pin, tmp_path / "cache", client=github(listing, MODULES, urls))
    assert urls[0] == f"https://api.github.com/repos/example/sglang/contents/python/pkg/models?ref={'e' * 40}"
    assert len(urls) == 3  # the listing, then a.py and b.py
    assert (root / "python/pkg/models/b.py").read_text() == "EntryClass = B\n"
    pinned_root(pin, tmp_path / "cache", client=github(listing, MODULES, urls))
    assert len(urls) == 3  # a full cache, checked whole, makes no request


def test_the_contents_api_alone_is_asked_with_the_workflow_token_when_one_is_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_example")
    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.headers.get("Authorization")))
        if request.url.host == "api.github.com":
            return httpx.Response(200, json=[{"name": "a.py", "type": "file"}, {"name": "b.py", "type": "file"}])
        return httpx.Response(200, content=MODULES[request.url.path.rsplit("/", 1)[1]].encode())

    pin = Pin("sglang", "example/sglang", "x", "b" * 40, {}, {"python/pkg/models": sha256sum(MODULES)})
    pinned_root(pin, tmp_path / "cache", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert seen == [
        ("api.github.com", "Bearer ghs_example"),  # 60 listings an hour without one, from a runner's shared address
        ("raw.githubusercontent.com", None),
        ("raw.githubusercontent.com", None),
    ]


def test_a_pinned_directory_that_is_not_the_pinned_content_is_refused(tmp_path: Path) -> None:
    listing = [{"name": "a.py", "type": "file"}, {"name": "b.py", "type": "file"}]
    changed = {**MODULES, "b.py": "EntryClass = C\n"}
    pin = Pin("sglang", "example/sglang", "x", "f" * 40, {}, {"python/pkg/models": sha256sum(MODULES)})
    with pytest.raises(PinError, match="sha256"):
        pinned_root(pin, tmp_path / "cache", client=github(listing, changed, []))
    assert not (tmp_path / "cache" / "sglang" / ("f" * 40) / "python/pkg/models").exists()


def test_a_cached_directory_with_a_changed_module_is_fetched_again(tmp_path: Path) -> None:
    listing = [{"name": "a.py", "type": "file"}, {"name": "b.py", "type": "file"}]
    pin = Pin("sglang", "example/sglang", "x", "a" * 40, {}, {"python/pkg/models": sha256sum(MODULES)})
    root = pinned_root(pin, tmp_path / "cache", client=github(listing, MODULES, []))
    (root / "python/pkg/models/a.py").write_text("tampered = 1\n")
    urls: list[str] = []
    pinned_root(pin, tmp_path / "cache", client=github(listing, MODULES, urls))
    assert len(urls) == 3
    assert (root / "python/pkg/models/a.py").read_text() == "EntryClass = A\n"


def cached(pin: Pin, rel: str) -> str:
    path = Path.home() / ".cache" / "bellwether" / "registries" / pin.engine / pin.commit / rel
    if not path.is_file():
        pytest.skip(f"{pin.engine} at {pin.ref} is not cached; `bellwether models` caches it")
    return path.read_text()


def test_vllm_at_the_pinned_tag_has_the_designs_architecture_counts() -> None:
    entries = read_vllm(cached(VLLM, "tests/models/registry.py"))
    text, multimodal = by_name(entries, TEXT), by_name(entries, MULTIMODAL)
    assert (len(text), len(multimodal)) == (134, 130)  # docs/benchmark-sets.md, "Which models"
    assert all(e.generative for e in text.values())
    assert {name for name, e in multimodal.items() if not e.generative} == {
        "JinaVLForRanking",
        "Qwen3ASRForcedAlignerForTokenClassification",
    }
    assert text["Qwen3ForCausalLM"].checkpoints == ("Qwen/Qwen3-8B",)
    assert multimodal["DeepseekV41ForCausalLM"].checkpoints == ("deepseek-ai/DeepSeek-V4.1-Flash",)


def test_sglang_at_the_pinned_commit_yields_every_id_its_pages_format_as_code() -> None:
    code_ids = re.compile(r"(?:`|<code>)([A-Za-z0-9][\w.-]*/[\w.-]+)(?:`|</code>)")
    for page in SGLANG.files:
        source = cached(SGLANG, page)
        entries = read_sglang(source, page)
        assert [c for e in entries for c in e.checkpoints] == code_ids.findall(source)
    multimodal = read_sglang(cached(SGLANG, MULTIMODAL_PAGE), MULTIMODAL_PAGE)
    assert [e.name for e in multimodal if not e.checkpoints] == ["JetVLM", "JetVLM"]


def cached_modules(pin: Pin, rel_dir: str) -> dict[str, bytes]:
    root = Path.home() / ".cache" / "bellwether" / "registries" / pin.engine / pin.commit / rel_dir
    if not (root / ".listed").is_file():
        pytest.skip(f"{pin.engine}'s {rel_dir} at {pin.ref} is not cached; `bellwether models` caches it")
    return {name: (root / name).read_bytes() for name in (root / ".listed").read_text().split()}


def test_sglang_code_at_the_pinned_commit_serves_the_five_architectures_its_docs_name_no_checkpoint_for() -> None:
    served = read_sglang_code(cached_modules(SGLANG, SGLANG_MODELS), cached_modules(SGLANG, SGLANG_PROCESSORS))
    assert len(served.architectures) == 251
    five = {
        "ZayaForCausalLM",
        "Sarashina2VisionForCausalLM",
        "POINTSV15ChatModel",
        "YiVLForCausalLM",
        "MossVLForConditionalGeneration",
    }
    assert five <= served.architectures
    assert {"Sarashina2VisionForCausalLM", "POINTSV15ChatModel", "MossVLForConditionalGeneration"} <= served.multimodal
    assert {"Glm4vForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"} <= served.multimodal
