"""Shapes: parse cases whose assistant message combines GSM8K's text with BFCL's calls, in every message shape a parser
must handle.

A message shape is which parts an assistant message holds, in the order a model writes them: reasoning
(``reasoning_content``), content (``content``) and calls (``tool_calls``). BFCL's parse cases have one shape, calls
alone, and GSM8K's have two, reasoning then content and content alone. Each set here holds one shape (``SHAPES``),
named by its parts in order: reasoning, content, reasoning-content, reasoning-calls, content-calls and
reasoning-content-calls.

The cases are built by the two importers' builders from their pinned sources, the BFCL wheel and GSM8K's test file,
never from the committed corpus, so ``--check`` depends on the pins alone. BFCL's parse cases of ``CATEGORIES`` are
interleaved, one from each category in turn, so that every category is among the first cases; GSM8K's rows stay in file
order. The i-th BFCL case is paired with the i-th GSM8K row, the shorter list cycled, for at most ``SIZE`` pairs, and
pair i gives case i of every set, so the six sets hold the same pairs:

- the request and the calls are the BFCL case's, so the calls answer the request; the sets without calls keep the
  request and its tools, since a model may answer without calling;
- the reasoning and the content are GSM8K's worked solution as the GSM8K importer gives it, as written and with its
  calculator annotations, split at its last line (``split_solution``): the lines before it are the reasoning, and the
  last line is the content.

Why the last line. GSM8K writes one step per line, so a line is the dataset's own unit of text, where a last sentence
would need a sentence splitter, which a title such as "Mr." defeats. The last step is the one that states the result, as
an answer does after reasoning. The two parts together are the solution as written, so a message holds no text that
GSM8K did not write, and none twice. The final answer after ``#### `` is not used: on its own it is a bare number, the
same in many rows, and GSM8K's own reasoning sets hold it already as their content. A row whose solution would leave
either part empty is left out with its reason.

The text does not answer the request: what a case probes is its shape. ``origin`` keeps both sources' origins whole, as
their importers write them, under ``parts``: the BFCL case's, then the GSM8K row's. Each source's license is checked by
its importer on every import. This module, like the importers it builds on, imports nothing beyond the standard library.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from . import bfcl, corpus_sets, gsm8k

DATASET = "shapes"
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
# Each message shape by the parts its message holds, in the order a model writes them; one set per shape.
SHAPES = (
    "reasoning",
    "content",
    "reasoning-content",
    "reasoning-calls",
    "content-calls",
    "reasoning-content-calls",
)


def message_for(shape: str, reasoning: str, content: str, calls: list[dict]) -> dict:
    """The assistant message of one shape, in the form the other parse cases use.

    ``content`` is ``""`` when the shape has no content but has reasoning or calls, as in BFCL's messages; the
    content-only message has no ``reasoning_content`` key, as in GSM8K's content messages.
    """
    parts = shape.split("-")
    message: dict = {}
    if "reasoning" in parts:
        message["reasoning_content"] = reasoning
    message["content"] = content if "content" in parts else ""
    if "calls" in parts:
        message["tool_calls"] = calls
    return message


def split_solution(solution: str) -> tuple[str, str]:
    """The reasoning and the content of a GSM8K worked solution: the lines before its last line, and its last line.

    Both keep the text as written, calculator annotations included. ``gsm8k.Unusable`` when either would be empty.
    """
    steps, _, last = solution.rpartition("\n")
    if not steps.strip():
        raise gsm8k.Unusable("the solution has no text before its last line")
    if not last.strip():
        raise gsm8k.Unusable("the solution's last line is empty")
    return steps, last


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


def set_name(shape: str) -> str:
    return f"{DATASET}-{shape}"


def build_sets(
    wheel: zipfile.ZipFile, test_file: bytes, size: int | None = None, skipped: list[tuple[str, str]] | None = None
) -> dict[tuple[str, str], list[dict]]:
    """One parse set per message shape, from the BFCL wheel and GSM8K's test file through their importers' builders.

    The calls are BFCL's parse cases of ``CATEGORIES``, interleaved; a category without any stops the import. The texts
    are the solutions of GSM8K's reasoning-content parse cases of ``SPLIT``, each split by ``split_solution``. Each pair
    gives the i-th case of every set: the request and calls of the call, the reasoning and content of the text.

    Every row the import reads and does not use is appended to ``skipped`` as ``(what, why)``: by name, each BFCL row
    without a parse case and each GSM8K row without a case, with their importers' reasons; then the parse cases of each
    BFCL category and the GSM8K rows that come after the pairs, each as one run in file order (``after_pairs``).
    """
    bfcl_skipped: list[tuple[str, str]] = []
    found = bfcl.build_sets(wheel, categories=CATEGORIES, skipped=bfcl_skipped)
    missing = [category for category in CATEGORIES if ("parse", bfcl.set_name(category)) not in found]
    if missing:
        raise ValueError(f"no parse case in BFCL {', '.join(missing)}: the shapes take calls from every category")
    cases = {category: found[("parse", bfcl.set_name(category))] for category in CATEGORIES}
    calls = interleave(list(cases.values()))
    owners = interleave([[category] * len(lines) for category, lines in cases.items()])  # each call's category
    gsm8k_skipped: list[tuple[str, str]] = []
    gsm8k_lines = gsm8k.build_sets({SPLIT: test_file}, gsm8k_skipped)[
        ("parse", gsm8k.set_name(SPLIT, "reasoning-content"))
    ]
    texts = []  # (GSM8K line, reasoning, content)
    for gsm8k_line in gsm8k_lines:
        try:
            texts.append((gsm8k_line, *split_solution(gsm8k_line["message"]["reasoning_content"])))
        except gsm8k.Unusable as err:
            gsm8k_skipped.append((f"{SPLIT} row {gsm8k_line['origin']['row']}", str(err)))
    pairs = pair(calls, texts, SIZE if size is None else size)
    sets: dict[tuple[str, str], list[dict]] = {("parse", set_name(shape)): [] for shape in SHAPES}
    for index, (bfcl_line, (gsm8k_line, reasoning, content)) in enumerate(pairs):
        origin = {"dataset": DATASET, "parts": [bfcl_line["origin"], gsm8k_line["origin"]]}
        for shape in SHAPES:
            name = set_name(shape)
            message = message_for(shape, reasoning, content, bfcl_line["message"]["tool_calls"])
            notes = f"shape {shape}: {bfcl_line['notes']}, {gsm8k_line['notes']}"
            line = {"name": f"{name}-{index}", "request": bfcl_line["request"], "message": message, "notes": notes}
            sets[("parse", name)].append({**line, "origin": origin})
    if skipped is not None:
        skipped.extend((f"BFCL {row}", why) for row, why in bfcl_skipped)
        skipped.extend((f"GSM8K {row}", why) for row, why in gsm8k_skipped)
        count = len(pairs)
        skipped.extend(after_pairs(calls[count:], owners[count:], [text[0] for text in texts[count:]], count))
    return sets


def after_pairs(calls: list[dict], owners: list[str], texts: list[dict], count: int) -> list[tuple[str, str]]:
    """The cases left after the ``count`` pairs, as ``(what, why)``: one run per BFCL category, then GSM8K's rows.

    ``owners`` names each call's category. A category's calls after the pairs are its last parse cases, in file order,
    and so are the GSM8K rows, so each run is named by how many it holds and its first and last row.
    """
    why = f"after the first {count} pairs"
    runs = []
    for category in CATEGORIES:
        rows = [line["origin"]["row"] for line, owner in zip(calls, owners, strict=True) if owner == category]
        if rows:
            runs.append((run_of(rows, "BFCL", f"BFCL {category} parse cases"), why))
    rows = [line["origin"]["row"] for line in texts]
    if rows:
        runs.append((run_of(rows, f"GSM8K {SPLIT} row", f"GSM8K {SPLIT} rows"), why))
    return runs


def run_of(rows: list, one: str, many: str) -> str:
    """A single row by name, or a run of rows by how many it holds and its first and last."""
    return f"{one} {rows[0]}" if len(rows) == 1 else f"{len(rows)} {many}, {rows[0]} to {rows[-1]}"


def write_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> list[Path]:
    """Write every set, and remove ``shapes-*`` files no message shape writes any more."""
    return corpus_sets.write(sets, corpus_dir, f"{DATASET}-")


def check_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> list[str]:
    """One line per set file that differs from a fresh import; empty when the corpus is what the import writes."""
    return corpus_sets.check(sets, corpus_dir, f"{DATASET}-", "message shape")


def report_skipped(skipped: list[tuple[str, str]]) -> None:
    """Print every row and run of rows the import left out with its reason, each on its own line, none summed away."""
    for what, why in skipped:
        print(f"no case for {what}: {why}")


def run(args: argparse.Namespace) -> int:
    from . import pypi

    gsm8k.check_license(gsm8k.fetch(gsm8k.LICENSE_FILE, gsm8k.LICENSE_SHA256, args.cache))
    test_file = gsm8k.fetch(gsm8k.data_file(SPLIT), gsm8k.SHA256[SPLIT], args.cache)
    with zipfile.ZipFile(pypi.fetch(bfcl.PROJECT, bfcl.VERSION, bfcl.WHEEL, bfcl.SHA256, cache=args.cache)) as wheel:
        bfcl.check_license(wheel)
        skipped: list[tuple[str, str]] = []
        sets = build_sets(wheel, test_file, skipped=skipped)
    if args.check:
        problems = check_sets(sets, args.corpus)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"{args.corpus}: the shapes sets equal a fresh import of {bfcl.SOURCE} and {gsm8k.SOURCE}")
        return 1 if problems else 0
    for (kind, name), lines in sorted(sets.items()):
        print(f"{args.corpus / kind / f'{name}.jsonl'}: {len(lines)} cases")
    report_skipped(skipped)
    write_sets(sets, args.corpus)
    return 0
