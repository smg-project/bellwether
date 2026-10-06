"""Read the engines' lists of supported models from source: vLLM's test registry and SGLang's docs.

vLLM's ``tests/models/registry.py`` keeps one table per kind of model, mapping each architecture to
the checkpoints its tests load. It is read with ``ast`` and never imported, so no engine has to be
installed. SGLang keeps no such table in code; its supported-models pages are read instead, taking
every id in each table's example column.
"""

from __future__ import annotations

import ast
import html
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from .pins import SGLANG, SGLANG_PAGES, VLLM, VLLM_REGISTRY, pinned_root
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


def read_pinned(
    cache: Path, vllm_src: Path | None = None, sglang_src: Path | None = None, client: httpx.Client | None = None
) -> list[Entry]:
    """Every entry of both engines at the pinned commits."""
    vllm = pinned_root(VLLM, cache, vllm_src, client)
    sglang = pinned_root(SGLANG, cache, sglang_src, client)
    entries = read_vllm((vllm / VLLM_REGISTRY).read_text())
    for page in SGLANG.files:
        entries.extend(read_sglang((sglang / page).read_text(), page))
    return entries


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
