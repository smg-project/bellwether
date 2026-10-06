"""Read the engines' lists of supported models from source: vLLM's test registry, SGLang's docs and code.

vLLM's ``tests/models/registry.py`` keeps one table per kind of model, mapping each architecture to
the checkpoints its tests load. It is read with ``ast`` and never imported, so no engine has to be
installed. SGLang's supported-models pages name checkpoints, every id in each table's example
column; its code names architectures, each model module's ``EntryClass``, and its multimodal
processors name the architectures they serve. Both are read here, the code with ``ast`` too.

These are the engines' model registries; ``bellwether gaps`` reads their parser, renderer and
tokenizer registries (``gaps/registries.py``), which are other files with other shapes.
"""

from __future__ import annotations

import ast
import html
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from .pins import (
    SGLANG,
    SGLANG_MODELS,
    SGLANG_PAGES,
    SGLANG_PROCESSORS,
    VLLM,
    VLLM_REGISTRY,
    listed_modules,
    pinned_root,
)
from .rules import GENERATIVE_TABLES, is_generative_architecture

MULTIMODAL_TABLE = "_MULTIMODAL_EXAMPLE_MODELS"


@dataclass(frozen=True)
class Entry:
    """An architecture (vLLM) or a model family (SGLang's docs) and the checkpoints named for it."""

    engine: str
    name: str
    table: str  # vLLM's registry table, or SGLang's docs page
    generative: bool
    multimodal: bool
    checkpoints: tuple[str, ...]  # as written, placeholders included; vLLM's default comes first
    vendor_code: bool = False  # vLLM loads it with trust_remote_code=True
    revision: str | None = None  # the revision of the default checkpoint vLLM loads, when it names one
    tokenizer: str | None = None  # the repository vLLM takes the default checkpoint's tokenizer from, if not its own


@dataclass(frozen=True)
class Served:
    """The architectures an engine's code serves, and those of them it serves with images, audio or video."""

    architectures: frozenset[str]
    multimodal: frozenset[str]


NOTHING_SERVED = Served(frozenset(), frozenset())


def read_pinned(
    cache: Path, vllm_src: Path | None = None, sglang_src: Path | None = None, client: httpx.Client | None = None
) -> tuple[list[Entry], Served]:
    """Every entry of both engines at the pinned commits, and what SGLang's code serves."""
    vllm = pinned_root(VLLM, cache, vllm_src, client)
    sglang = pinned_root(SGLANG, cache, sglang_src, client)
    entries = read_vllm((vllm / VLLM_REGISTRY).read_text())
    for page in SGLANG.files:
        entries.extend(read_sglang((sglang / page).read_text(), page))
    if not SGLANG.dirs:
        return entries, NOTHING_SERVED
    return entries, read_sglang_code(listed_modules(sglang / SGLANG_MODELS), listed_modules(sglang / SGLANG_PROCESSORS))


def read_sglang_code(models: dict[str, bytes], processors: dict[str, bytes]) -> Served:
    """What SGLang's code serves: each model module's ``EntryClass``, and what its multimodal processors name.

    SGLang's ``models/registry.py`` imports every module of ``sglang.srt.models`` and registers each
    class its ``EntryClass`` names, a class or a list of them, under the class's own name; each
    multimodal processor lists the model classes it serves in ``models``. Both are read from the
    modules' text, so a form other than these is refused rather than guessed at.
    """
    architectures: set[str] = set()
    for module, source in sorted(models.items()):
        tree = ast.parse(source)
        original = _imported_names(tree)
        assigned = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and _assigns(n, "EntryClass")]
        if any(node not in tree.body for node in assigned):
            raise ValueError(f"SGLang's models/{module}: EntryClass is assigned where this reader does not read it")
        for node in assigned:
            architectures.update(original.get(name, name) for name in _class_names(module, node.value))
    multimodal: set[str] = set()
    for module, source in sorted(processors.items()):
        tree = ast.parse(source)
        model_classes = {
            local: name for local, (origin, name) in _imports(tree).items() if origin.startswith("sglang.srt.models")
        }
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            for statement in cls.body:
                if isinstance(statement, ast.Assign) and _assigns(statement, "models"):
                    if isinstance(statement.value, ast.List | ast.Tuple) and not statement.value.elts:
                        continue  # the base processor's default: it serves no model itself
                    names = {
                        model_classes[node.id]
                        for node in ast.walk(statement.value)
                        if isinstance(node, ast.Name) and node.id in model_classes
                    }
                    if not names:
                        written = ast.unparse(statement.value)
                        raise ValueError(
                            f"SGLang's processors/{module}: {cls.name}.models names no model class: {written}"
                        )
                    multimodal |= names
    return Served(frozenset(architectures), frozenset(multimodal))


def _assigns(node: ast.Assign, name: str) -> bool:
    return any(isinstance(target, ast.Name) and target.id == name for target in node.targets)


def _imports(tree: ast.Module) -> dict[str, tuple[str, str]]:
    """Each name a module imports, by the name it is bound to: the module it comes from and its own name."""
    return {
        alias.asname or alias.name: (node.module or "", alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


def _imported_names(tree: ast.Module) -> dict[str, str]:
    return {local: name for local, (_, name) in _imports(tree).items()}


def _class_names(module: str, value: ast.expr) -> list[str]:
    elements = [value] if isinstance(value, ast.Name) else getattr(value, "elts", None)
    if not isinstance(value, ast.Name | ast.List | ast.Tuple) or not all(isinstance(e, ast.Name) for e in elements):
        written = ast.unparse(value)
        raise ValueError(f"SGLang's models/{module}: EntryClass is {written}, not a class or a list of classes")
    return [element.id for element in elements]


def read_vllm(source: str) -> list[Entry]:
    """One entry per architecture in every table; a pinned file that changed shape is an error, not a gap."""
    tables = _example_tables(ast.parse(source))
    missing = sorted(GENERATIVE_TABLES - tables.keys())
    if missing:
        raise ValueError(f"vLLM's registry has no table {', '.join(missing)}; the file changed shape")
    return [
        _vllm_entry(table, key.value, call)
        for table, node in tables.items()
        for key, call in zip(node.keys, node.values, strict=True)
    ]


def _example_tables(tree: ast.Module) -> dict[str, ast.Dict]:
    """Module-level dicts from architecture names to calls; the merged ``**`` tables are not among them."""
    tables: dict[str, ast.Dict] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        value = node.value
        if (
            isinstance(value, ast.Dict)
            and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in value.keys)
            and all(isinstance(v, ast.Call) for v in value.values)
        ):
            tables[node.targets[0].id] = value
    return tables


def _vllm_entry(table: str, architecture: str, call: ast.Call) -> Entry:
    keywords = {k.arg: k.value for k in call.keywords}
    return Entry(
        engine="vllm",
        name=architecture,
        table=table,
        generative=is_generative_architecture(table, architecture),
        multimodal=table == MULTIMODAL_TABLE,
        checkpoints=_checkpoints(architecture, call),
        vendor_code=_literal(architecture, keywords, "trust_remote_code", bool) or False,
        revision=_literal(architecture, keywords, "revision", str),
        tokenizer=_literal(architecture, keywords, "tokenizer", str),
    )


def _literal(architecture: str, keywords: dict[str | None, ast.expr], name: str, kind: type) -> object:
    """A load setting as the registry writes it; one that is not a literal would need vLLM's code run to know."""
    node = keywords.get(name)
    if node is None or (isinstance(node, ast.Constant) and node.value is None):
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, kind):
        return node.value
    raise ValueError(f"vLLM's registry: {architecture} gives {name} as {ast.unparse(node)}, not as a literal")


def _checkpoints(architecture: str, call: ast.Call) -> tuple[str, ...]:
    """The default checkpoint, then the extras in their order, each given by position or by keyword."""
    keywords = {k.arg: k.value for k in call.keywords}
    default = call.args[0] if call.args else keywords.get("default")
    extras = call.args[1] if len(call.args) > 1 else keywords.get("extras")
    names = [_string(architecture, default)]
    if extras is not None:
        if not isinstance(extras, ast.Dict):
            raise ValueError(f"vLLM's registry: the extras of {architecture} are not written out")
        names.extend(_string(architecture, value) for value in extras.values)
    return tuple(names)


def _string(architecture: str, node: ast.expr | None) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    written = ast.unparse(node) if node is not None else "nothing"
    raise ValueError(f"vLLM's registry: {architecture} names a checkpoint as {written}, not as a string")


_TABLE = re.compile(r"<table\b.*?</table>", re.S)
_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S)
_HEADER = re.compile(r"<th\b[^>]*>(.*?)</th>", re.S)
_CELL = re.compile(r"<td\b[^>]*>(.*?)</td>", re.S)
_CODE = re.compile(r"`([^`]+)`|<code>(.*?)</code>", re.S)
_MARKUP = re.compile(r"<[^>]+>|\*\*")


def read_sglang(source: str, page: str) -> list[Entry]:
    """One entry per table row, in page order; a row the docs give no example for has no checkpoints.

    A page without a table changed shape, or is not the page: an error, never an empty list.
    """
    multimodal = SGLANG_PAGES[page]
    tables = _TABLE.findall(source)
    if not tables:
        raise ValueError(f"SGLang's {page}: no table; the page changed shape or is not the page")
    entries = []
    for table in tables:
        column = _example_column(page, table)
        for row in _ROW.findall(table):
            cells = _CELL.findall(row)
            if not cells:
                continue  # the header row
            ids = tuple((code or tag).strip() for code, tag in _CODE.findall(cells[column]))
            entries.append(Entry("sglang", _text(cells[0]), page, True, multimodal, ids))
    return entries


def _example_column(page: str, table: str) -> int:
    """The column the docs head "Example HuggingFace Identifier", "Example Identifier" or "Example Model"."""
    headers = [_text(h) for h in _HEADER.findall(table)]
    for index, header in enumerate(headers):
        if header.startswith("Example"):
            return index
    raise ValueError(f"SGLang's {page}: a table without an example column (headers {headers})")


def _text(cell: str) -> str:
    return " ".join(html.unescape(_MARKUP.sub("", cell)).split())
