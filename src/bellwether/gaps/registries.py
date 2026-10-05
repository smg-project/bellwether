"""Read the parser, renderer and tokenizer registries of vLLM, SGLang and SMG from source.

Python registries are read with ``ast`` and never imported, so no engine has to be installed;
the Rust factories are read with regular expressions over the source text. A reader accepts
several roots and takes each file from the first root that has it, so a partial copy and a set
of files fetched at a pinned ref can be combined.
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("tool", "reasoning", "renderer", "tokenizer")
SYSTEMS = ("vllm", "sglang", "smg")

_SGL = "python/sglang/srt"


@dataclass(frozen=True, order=True)
class Entry:
    """One registered name: which system, which kind, and what it resolves to."""

    system: str
    kind: str
    name: str
    impl: str
    source: str


@dataclass
class Registry:
    system: str
    roots: list[Path]
    commit: str
    entries: list[Entry] = field(default_factory=list)
    model_patterns: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def find(self, rel: str) -> Path | None:
        for root in self.roots:
            path = root / rel
            if path.is_file():
                return path
        return None

    def add(self, entry: Entry) -> None:
        """Keep the first entry per (kind, name); later roots never override earlier ones."""
        if any(e.kind == entry.kind and e.name == entry.name for e in self.entries):
            return
        self.entries.append(entry)

    def names(self, kind: str) -> list[str]:
        return sorted({e.name for e in self.entries if e.kind == kind})


def describe_commit(roots: list[Path]) -> str:
    """The commit a checkout or copy is at: ``COMMIT.txt`` first, then git, else ``unknown``."""
    for root in roots:
        marker = root / "COMMIT.txt"
        if marker.is_file():
            words = marker.read_text().split()
            if words:
                return words[0]
        if (root / ".git").exists():
            try:
                out = subprocess.run(
                    ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
                )
            except (OSError, subprocess.CalledProcessError):
                continue
            return out.stdout.strip()
    return "unknown"


# --- Python registries, read with ast ---------------------------------------------------------


def _assigned_value(tree: ast.AST, var: str) -> ast.AST | None:
    """The value assigned to ``var`` anywhere in the module, at module level or in a class body."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == var for t in node.targets):
                return node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == var:
                return node.value
    return None


def _impl_of(node: ast.AST) -> str:
    """The implementation a registry value names: the class in a (module, class) tuple, or the name."""
    if isinstance(node, ast.Tuple) and node.elts:
        last = node.elts[-1]
        if isinstance(last, ast.Constant) and isinstance(last.value, str):
            return last.value
    if isinstance(node, ast.Name):
        return node.id
    return ast.unparse(node)


def _dict_literal(tree: ast.AST, var: str) -> dict[str, str] | None:
    value = _assigned_value(tree, var)
    if not isinstance(value, ast.Dict):
        return None
    table: dict[str, str] = {}
    for key, val in zip(value.keys, value.values, strict=True):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            table[key.value] = _impl_of(val)
    return table


def _list_literal(tree: ast.AST, var: str) -> list[str] | None:
    value = _assigned_value(tree, var)
    if not isinstance(value, ast.List | ast.Tuple):
        return None
    return [e.value for e in value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]


def _first_class(path: Path) -> str:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef):
            return node.name
    return ""


_VLLM_TABLES = (
    ("tool", "vllm/tool_parsers/__init__.py", "_TOOL_PARSERS_TO_REGISTER"),
    ("reasoning", "vllm/reasoning/__init__.py", "_REASONING_PARSERS_TO_REGISTER"),
    ("renderer", "vllm/renderers/registry.py", "_VLLM_RENDERERS"),
    ("tokenizer", "vllm/tokenizers/registry.py", "_VLLM_TOKENIZERS"),
)


def read_vllm(roots: list[Path]) -> Registry:
    """vLLM keeps one ``name -> (module, class)`` dict per kind; the class is the implementation."""
    reg = Registry("vllm", roots, describe_commit(roots))
    for kind, rel, var in _VLLM_TABLES:
        path = reg.find(rel)
        if path is None:
            reg.notes.append(f"{rel} not found: no {kind} names read")
            continue
        table = _dict_literal(ast.parse(path.read_text()), var)
        if table is None:
            reg.notes.append(f"{rel}: no dict named {var}")
            continue
        for name, impl in table.items():
            reg.add(Entry("vllm", kind, name, impl, rel))
    return reg


def _sglang_parsers(reg: Registry, kind: str, names_rel: str, names_var: str, impl_rel: str, impl_var: str) -> None:
    impls: dict[str, str] = {}
    impl_path = reg.find(impl_rel)
    if impl_path is not None:
        impls = _dict_literal(ast.parse(impl_path.read_text()), impl_var) or {}
    else:
        reg.notes.append(f"{impl_rel} not found: {kind} implementations unknown")
    names_path = reg.find(names_rel)
    if names_path is not None:
        names = _list_literal(ast.parse(names_path.read_text()), names_var) or []
        source = names_rel
    else:
        names = list(impls)
        source = impl_rel
        reg.notes.append(
            f"{names_rel} not found: {kind} names taken from {impl_var}"
            if names
            else f"{names_rel} not found: no {kind} names read"
        )
    for name in names:
        reg.add(Entry("sglang", kind, name, impls.get(name, ""), source))


def read_sglang(roots: list[Path]) -> Registry:
    """SGLang keeps dependency-free name lists for the CLI and class maps in the parser modules.

    No registry exists for renderers or tokenizers. The Hugging Face chat template and tokenizer
    are the defaults (``hf`` here); a native renderer is a module ``parser/<name>_renderer.py``
    the serving code selects, and a native tokenizer a module ``<name>_tokenizer.py`` under
    ``parser/`` or ``tokenizer/``.
    """
    reg = Registry("sglang", roots, describe_commit(roots))
    _sglang_parsers(
        reg,
        "tool",
        f"{_SGL}/function_call/parser_names.py",
        "TOOL_CALL_PARSER_NAMES",
        f"{_SGL}/function_call/function_call_parser.py",
        "ToolCallParserEnum",
    )
    _sglang_parsers(
        reg,
        "reasoning",
        f"{_SGL}/parser/reasoning_parser_names.py",
        "REASONING_PARSER_NAMES",
        f"{_SGL}/parser/reasoning_parser.py",
        "DetectorMap",
    )
    reg.add(Entry("sglang", "renderer", "hf", "jinja chat template", f"{_SGL}/parser/jinja_template_utils.py"))
    reg.add(Entry("sglang", "tokenizer", "hf", "transformers tokenizer", f"{_SGL}/utils/hf_transformers_utils.py"))
    globs = (
        ("renderer", f"{_SGL}/parser", "*_renderer.py"),
        ("tokenizer", f"{_SGL}/parser", "*_tokenizer.py"),
        ("tokenizer", f"{_SGL}/tokenizer", "*_tokenizer.py"),
    )
    seen_dir = False
    for kind, rel_dir, pattern in globs:
        for root in reg.roots:
            directory = root / rel_dir
            if not directory.is_dir():
                continue
            seen_dir = True
            for path in sorted(directory.glob(pattern)):
                name = path.stem[: -len(pattern) + 4]  # strip "_renderer" / "_tokenizer"
                reg.add(Entry("sglang", kind, name, _first_class(path), str(path.relative_to(root))))
    if not seen_dir:
        reg.notes.append(f"{_SGL}/parser not found: native renderers and tokenizers not enumerated")
    return reg


# --- SMG, read from the Rust sources ---------------------------------------------------------

_REGISTER = re.compile(r'register_parser(?:_with_structural_tag)?\(\s*"([^"]+)"')
_BOX = re.compile(r"Box::new\(\s*([A-Za-z0-9_]+)\s*::")
_MAP_MODEL = re.compile(r'map_model\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)')
_PATTERN = re.compile(r'register_pattern\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)')
_ENUM = r"enum\s+{name}\s*\{{([^}}]*)\}}"


def _rust_registrations(text: str) -> list[tuple[str, str]]:
    """``(name, constructor type)`` for every ``register_parser*("name", || Box::new(Type::...))``.

    The constructor is the first ``Box::new(Type::`` between one registration and the next, which
    covers the one-line form, the structural-tag form and the block-bodied closures.
    """
    matches = list(_REGISTER.finditer(text))
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        box = _BOX.search(text, m.end(), end)
        out.append((m.group(1), box.group(1) if box else ""))
    return out


def _enum_variants(text: str, name: str) -> list[str]:
    m = re.search(_ENUM.format(name=name), text, re.DOTALL)
    if not m:
        return []
    variants = []
    for line in m.group(1).splitlines():
        line = line.split("//")[0].strip().rstrip(",")
        if not line:
            continue
        variants.append(line.split("(")[0].strip())
    return variants


def _snake(camel: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", camel).lower()


def _patterns(text: str, regex: re.Pattern[str]) -> dict[str, list[str]]:
    table: dict[str, list[str]] = {}
    for pattern, name in regex.findall(text):
        table.setdefault(name, []).append(pattern)
    return table


def read_smg(roots: list[Path]) -> Registry:
    """SMG registers parsers in two Rust factories; renderers and tokenizer types are enums.

    The Harmony pipeline that serves gpt-oss models is not a factory name, so it is reported as
    the name ``harmony`` for both parser kinds when its module exists.
    """
    reg = Registry("smg", roots, describe_commit(roots))
    for kind, rel, pattern_regex in (
        ("tool", "crates/tool_parser/src/factory.rs", _MAP_MODEL),
        ("reasoning", "crates/reasoning_parser/src/factory.rs", _PATTERN),
    ):
        path = reg.find(rel)
        if path is None:
            reg.notes.append(f"{rel} not found: no {kind} names read")
            continue
        text = path.read_text()
        for name, impl in _rust_registrations(text):
            reg.add(Entry("smg", kind, name, impl, rel))
        reg.model_patterns[kind] = _patterns(text, pattern_regex)

    rel = "crates/tokenizer/src/huggingface.rs"
    path = reg.find(rel)
    if path is None:
        reg.notes.append(f"{rel} not found: no renderer names read")
    else:
        for variant in _enum_variants(path.read_text(), "Renderer"):
            reg.add(Entry("smg", "renderer", _snake(variant), f"Renderer::{variant}", rel))
    for root in reg.roots:
        encoders = root / "crates/tokenizer/src/encoders"
        if not encoders.is_dir():
            continue
        for path in sorted(encoders.glob("*.rs")):
            if path.stem == "mod" or path.stem.endswith("_common"):
                continue
            if any(e.kind == "renderer" and e.name == path.stem for e in reg.entries):
                continue  # the enum variant above already names this encoder
            reg.add(Entry("smg", "renderer", path.stem, f"encoders::{path.stem}", str(path.relative_to(root))))

    rel = "crates/tokenizer/src/factory.rs"
    path = reg.find(rel)
    if path is None:
        reg.notes.append(f"{rel} not found: no tokenizer names read")
    else:
        for variant in _enum_variants(path.read_text(), "TokenizerType"):
            if variant == "Mock":
                continue  # a test double, not a tokenizer mode
            reg.add(Entry("smg", "tokenizer", variant.lower(), f"TokenizerType::{variant}", rel))

    rel = "model_gateway/src/routers/grpc/harmony/mod.rs"
    if reg.find(rel) is not None:
        for kind in ("tool", "reasoning"):
            reg.add(Entry("smg", kind, "harmony", "HarmonyDetector", rel))
    return reg
