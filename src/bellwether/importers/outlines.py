"""Outlines: parse cases whose assistant message combines GSM8K's text with BFCL's calls, in every outline a parser
must handle.

An outline is which parts an assistant message holds, in the order a model writes them: reasoning
(``reasoning_content``), prose (``content``) and calls (``tool_calls``). BFCL's parse cases have one outline, calls
alone, and GSM8K's have reasoning and prose but no calls. Each set here holds one outline (``OUTLINES``): reasoning
only, prose only, reasoning then prose, reasoning then calls, prose then calls, and all three.

The cases are built by the two importers' builders from their pinned sources, the BFCL wheel and GSM8K's test file,
never from the committed corpus, so ``--check`` depends on the pins alone. BFCL's parse cases of ``CATEGORIES`` are
interleaved, one from each category in turn, so that every category is among the first cases; GSM8K's rows stay in file
order. The i-th BFCL case is paired with the i-th GSM8K row, the shorter list cycled, for at most ``SIZE`` pairs, and
pair i gives case i of every set, so the six sets hold the same pairs:

- the request and the calls are the BFCL case's, so the calls answer the request; the sets without calls keep the
  request and its tools, since a model may answer without calling;
- the reasoning is GSM8K's worked solution as the GSM8K importer gives it, calculator annotations included;
- the prose is GSM8K's final answer, as written after ``#### ``, in a fixed sentence (``PROSE``). The solution's own
  last line is part of the reasoning, carries annotations, and does not always hold the final answer as written.

The prose is synthetic and the text does not answer the request: what a case probes is its outline. ``origin`` keeps
both sources' origins whole, as their importers write them, under ``parts``: the BFCL case's, then the GSM8K row's. Each
source's license is checked by its importer on every import. This module, like the importers it builds on, imports
nothing beyond the standard library.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from . import bfcl, corpus_sets, gsm8k

DATASET = "outlines"
SIZE = 1000  # pairs, and so cases per set
# BFCL's categories whose parse cases carry calls, in the order of bfcl.CATEGORIES; Java and JavaScript are left out.
CATEGORIES = (
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
)
SPLIT = "test"  # GSM8K's split the text comes from
PROSE = "The answer is {}."  # the prose: GSM8K's final answer, as written after "#### ", in a fixed sentence
# Each outline by the parts its message holds, in the order a model writes them; one set per outline.
OUTLINES = {
    "reasoning-only": ("reasoning",),
    "prose-only": ("prose",),
    "reasoning-prose": ("reasoning", "prose"),
    "reasoning-calls": ("reasoning", "calls"),
    "prose-calls": ("prose", "calls"),
    "reasoning-prose-calls": ("reasoning", "prose", "calls"),
}


def message_for(outline: str, reasoning: str, prose: str, calls: list[dict]) -> dict:
    """The assistant message of one outline, in the shapes the other parse cases use.

    ``content`` is ``""`` when the outline has no prose but has reasoning or calls, as in BFCL's messages; the
    prose-only message has no ``reasoning_content`` key, as in GSM8K's content messages.
    """
    parts = OUTLINES[outline]
    message: dict = {}
    if "reasoning" in parts:
        message["reasoning_content"] = reasoning
    message["content"] = prose if "prose" in parts else ""
    if "calls" in parts:
        message["tool_calls"] = calls
    return message


def interleave(lists: list[list]) -> list:
    """The items of every list, one from each in turn, in list order; a list that runs out is passed over."""
    longest = max((len(items) for items in lists), default=0)
    return [items[index] for index in range(longest) for items in lists if index < len(items)]


def pair(calls: list, texts: list, size: int) -> list[tuple]:
    """The i-th call with the i-th text, the shorter list cycled, for at most ``size`` pairs."""
    if not calls or not texts:
        raise ValueError(f"nothing to pair: {len(calls)} BFCL parse cases and {len(texts)} GSM8K rows")
    count = min(size, max(len(calls), len(texts)))
    return [(calls[index % len(calls)], texts[index % len(texts)]) for index in range(count)]


def set_name(outline: str) -> str:
    return f"{DATASET}-{outline}"


def build_sets(wheel: zipfile.ZipFile, test_file: bytes, size: int | None = None) -> dict[tuple[str, str], list[dict]]:
    """One parse set per outline, from the BFCL wheel and GSM8K's test file through their importers' builders.

    The calls are BFCL's parse cases of ``CATEGORIES``, interleaved; the texts are GSM8K's reasoning-content parse cases
    of
    ``SPLIT``. Each pair gives the i-th case of every set: the request and calls of the call, the reasoning and prose of
    the text.
    """
    found = bfcl.build_sets(wheel, categories=CATEGORIES)
    calls = interleave([found.get(("parse", bfcl.set_name(category)), []) for category in CATEGORIES])
    texts = gsm8k.build_sets({SPLIT: test_file})[("parse", gsm8k.set_name(SPLIT, "reasoning-content"))]
    sets: dict[tuple[str, str], list[dict]] = {("parse", set_name(outline)): [] for outline in OUTLINES}
    for index, (bfcl_line, gsm8k_line) in enumerate(pair(calls, texts, SIZE if size is None else size)):
        reasoning = gsm8k_line["message"]["reasoning_content"]
        prose = PROSE.format(gsm8k_line["message"]["content"])
        origin = {"dataset": DATASET, "parts": [bfcl_line["origin"], gsm8k_line["origin"]]}
        for outline in OUTLINES:
            name = set_name(outline)
            message = message_for(outline, reasoning, prose, bfcl_line["message"]["tool_calls"])
            notes = f"outline {outline}: {bfcl_line['notes']}, {gsm8k_line['notes']}"
            line = {"name": f"{name}-{index}", "request": bfcl_line["request"], "message": message, "notes": notes}
            sets[("parse", name)].append({**line, "origin": origin})
    return sets


def write_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> list[Path]:
    """Write every set, and remove ``outlines-*`` files no outline writes any more."""
    return corpus_sets.write(sets, corpus_dir, f"{DATASET}-")


def check_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> list[str]:
    """One line per set file that differs from a fresh import; empty when the corpus is what the import writes."""
    return corpus_sets.check(sets, corpus_dir, f"{DATASET}-", "outline")


def run(args: argparse.Namespace) -> int:
    from . import pypi

    gsm8k.check_license(gsm8k.fetch(gsm8k.LICENSE_FILE, gsm8k.LICENSE_SHA256, args.cache))
    test_file = gsm8k.fetch(gsm8k.data_file(SPLIT), gsm8k.SHA256[SPLIT], args.cache)
    with zipfile.ZipFile(pypi.fetch(bfcl.PROJECT, bfcl.VERSION, bfcl.WHEEL, bfcl.SHA256, cache=args.cache)) as wheel:
        bfcl.check_license(wheel)
        sets = build_sets(wheel, test_file)
    if args.check:
        problems = check_sets(sets, args.corpus)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"{args.corpus}: the outline sets equal a fresh import of {bfcl.SOURCE} and {gsm8k.SOURCE}")
        return 1 if problems else 0
    for (kind, name), lines in sorted(sets.items()):
        print(f"{args.corpus / kind / f'{name}.jsonl'}: {len(lines)} cases")
    write_sets(sets, args.corpus)
    return 0
