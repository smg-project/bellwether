"""MGSM's grade school math problems in eleven languages as corpus sets: per language, a render set and a parse set of
the answer, and a parse set of the worked exemplars.

The data is Google Research's url-nlp repository at a pinned commit, directory ``mgsm/``: GSM8K's first 250 test
problems, translated, as one ``mgsm_<lang>.tsv`` per language (no header, one ``question<TAB>answer`` line per
problem), and ``exemplars.py``, the eight worked exemplars per language as Python source. Each file is fetched by its
path at the commit and checked against its sha256. This module, like the fetcher and the set writer it uses, imports
nothing beyond the standard library.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import sys
from pathlib import Path

from . import corpus_sets, github

OWNER = "google-research"
REPO = "url-nlp"
COMMIT = "3622039cf51f7eeffa58b957332a8e8c337981d7"
SOURCE = f"github:{OWNER}/{REPO}@{COMMIT}"
LICENSE = "CC-BY-4.0"
DATA = "mgsm"
LICENSE_FILE = f"{DATA}/LICENSE"
LICENSE_SHA256 = "c97deeeca4ae375a0334bc7f7af5f707aabfcec959c53c783b9f8771d28fd5b3"
# The reviewed LICENSE file's first line.
LICENSE_TITLE = "Creative Commons Attribution 4.0 International Public License (CC-BY)"
EXEMPLARS_FILE = f"{DATA}/exemplars.py"
EXEMPLARS_SHA256 = "239dda1557bb2ba76b71e5ef744ddd0b454da0b453e5f8b909498ceff690a919"
ANSWER = "answer"  # the parse set per language, of the answer alone
EXEMPLAR_SET = "mgsm-exemplars"
# Each language's file, by the sha256 of its bytes, in the order the sets are built.
SHA256 = {
    "bn": "6b00bc7cc635547e866989284afa924d789b5affa2eb8de623c385dd943ad977",
    "de": "4dfea30fede44b813e2e496f5f0534049e83c7cc8aed6e00600b30ec62053626",
    "en": "50021d0f28cc957edcb44e7806425b1c7fbd648ddcb9e0a8ec689d10e57d40fa",
    "es": "5bd27ebdf00140cec845c5298dc17715f5cd57d8edda6df1976db40bf3f0750a",
    "fr": "36207c1c03fd7cd3ea491441eb755408199f94e4a647851df531bbaedc99d606",
    "ja": "59a2b50debe77981fd784cb3b2bef1505e3abf2a37116dc9d7a366ab029b4637",
    "ru": "6bd30fd2e80c5bac23f566fb4ae0e6a55a19578401fbf9a01fb21beb3645fef8",
    "sw": "2bac828d77229e65d7c7197b1ad4a2cb5b1fe99b163f2cbdd66501de6a2115c4",
    "te": "dd6b1452c244bb2e4ba254c01ea84137f80c1cefc57d69c23b0ed88b9c0f36b7",
    "th": "f3932dc5ad8e9d0ea82b017adc1e1461dd647af861e7166d6741986602a0cfd6",
    "zh": "b2fa63151022370a0de1f4211c8c284eae74b0f5a3b003b1d5982c0d4a73f661",
}

# An integer in ASCII digits, its thousands grouped by commas or not: four of the 250 answers are "2,125", "114,200",
# "276,000" and "5,600" in every language, as GSM8K writes them.
INTEGER = re.compile(r"-?([0-9]{1,3}(,[0-9]{3})+|[0-9]+)")

# How exemplars.py writes each language's exemplars: the label before the question, the label before the worked
# answer, and the words that open the answer's final sentence, which states the exemplar's number.
FORMS = {
    "bn": ("প্রশ্ন: ", "ধাপে ধাপে উত্তর: ", "উত্তর হল "),
    "de": ("Frage: ", "Schritt-für-Schritt-Antwort: ", "Die Antwort "),
    "en": ("Question: ", "Step-by-Step Answer: ", "The answer is "),
    "es": ("Pregunta: ", "Respuesta paso a paso: ", "La respuesta es "),
    "fr": ("Question : ", "Réponse étape par étape : ", "La réponse est "),
    "ja": ("問題：", "ステップごとの答え：", "答えは"),
    "ru": ("Задача: ", "Пошаговое решение: ", "Ответ — "),
    "sw": ("Swali: ", "Jibu la Hatua kwa Hatua: ", "Jibu ni "),
    "te": ("ప్రశ్న: ", "దశలవారీగా సమాధానం: ", "సమాధానం "),
    "th": ("โจทย์: ", "คำตอบทีละขั้นตอน: ", "คำตอบคือ "),
    "zh": ("问题：", "逐步解答：", "答案是 "),
}


class Unusable(ValueError):
    """An MGSM row or exemplar that cannot become a case."""


def check_license(text: bytes) -> None:
    """Refuse a LICENSE file other than the one this importer was reviewed for.

    The sha256 pins the reviewed text. Its first line, the license's title, is checked as well, so that moving the pins
    to another commit whose license is no longer CC-BY-4.0 cannot pass on an updated hash alone.
    """
    where = f"{LICENSE_FILE} at {COMMIT}"
    if text.decode("utf-8").split("\n", 1)[0] != LICENSE_TITLE:
        raise ValueError(f"{where}: not the CC-BY-4.0 license; review it before importing")
    digest = hashlib.sha256(text).hexdigest()
    if digest != LICENSE_SHA256:
        raise ValueError(f"{where}: sha256 {digest} is not the reviewed {LICENSE_SHA256}; review it before importing")


def read_rows(data: bytes) -> list[tuple[int, list[str]]]:
    """Each line of a language's file with its 0-based line index, split at its tabs, in file order.

    The file has no header and no quoting, so a line is split at its tabs and nothing else: a quote is part of the
    question it is in. Only "\\n" ends a line, as in the corpus readers, and an empty line, such as the piece after the
    last "\\n", is no row.
    """
    lines = data.decode("utf-8").split("\n")
    return [(index, line.split("\t")) for index, line in enumerate(lines) if line]


def question_and_answer(fields: list[str]) -> tuple[str, str]:
    """A row's question and answer, as written; ``Unusable`` unless the row is ``question<TAB>answer``.

    The question must not be empty and the answer must be an integer, which is kept with any commas it is written with.
    """
    if len(fields) != 2:
        raise Unusable(f"the line holds {len(fields) - 1} tabs, not the one between the question and the answer")
    question, answer = fields
    if not question.strip():
        raise Unusable("the question is empty")
    if not INTEGER.fullmatch(answer):
        raise Unusable(f"the answer {answer!r} is not an integer")
    return question, answer


def data_file(lang: str) -> str:
    return f"{DATA}/mgsm_{lang}.tsv"


def set_name(lang: str, shape: str = "") -> str:
    return f"mgsm-{lang}-{shape}" if shape else f"mgsm-{lang}"


def origin(file: str, sha256: str, row: int | str) -> dict:
    """Where a case came from, in the GSM8K importer's key order: the file and the row inside it."""
    return {"dataset": "mgsm", "source": SOURCE, "sha256": sha256, "file": file, "row": row, "license": LICENSE}


def language_sets(
    files: dict[str, bytes], skipped: list[tuple[str, str]] | None = None
) -> dict[tuple[str, str], list[dict]]:
    """Corpus lines per ``(kind, set name)`` from each language's file: a render set and an answer parse set.

    Each set has a line per row, in file order. The answer parse case's message is the answer, as written, as
    ``content``: the short output of a model that does not think. A row that cannot become a case is left out of both
    sets of its language, so that the two hold the same rows, and is appended to ``skipped`` with its reason.
    """
    sets: dict[tuple[str, str], list[dict]] = {}
    for lang, data in files.items():
        render = sets[("render", set_name(lang))] = []
        answers = sets[("parse", set_name(lang, ANSWER))] = []
        for row, fields in read_rows(data):
            try:
                question, answer = question_and_answer(fields)
            except Unusable as err:
                if skipped is not None:
                    skipped.append((f"{lang} row {row}", str(err)))
                continue
            request = {"messages": [{"role": "user", "content": question}]}
            tail = {"notes": f"MGSM {lang} row {row}", "origin": origin(data_file(lang), SHA256[lang], row)}
            render.append({"name": f"{set_name(lang)}-{row}", "request": request, **tail})
            name = f"{set_name(lang, ANSWER)}-{row}"
            answers.append({"name": name, "request": request, "message": {"content": answer}, **tail})
    return sets


def read_exemplars(data: bytes) -> tuple[dict[str, dict[str, dict[str, str]]], list[int]]:
    """``MGSM_EXEMPLARS`` and ``EXEMPLAR_NUMBER_ANSWERS`` from exemplars.py, read as literals: the file is never run.

    Its source is parsed, and only the values assigned to those two names are evaluated, by ``ast.literal_eval``, which
    refuses anything but a literal.
    """
    values = {}
    for node in ast.parse(data.decode("utf-8"), filename=EXEMPLARS_FILE).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            values[node.targets[0].id] = node.value
    for name in ("MGSM_EXEMPLARS", "EXEMPLAR_NUMBER_ANSWERS"):
        if name not in values:
            raise ValueError(f"{EXEMPLARS_FILE} at {COMMIT}: no {name}")
    return ast.literal_eval(values["MGSM_EXEMPLARS"]), ast.literal_eval(values["EXEMPLAR_NUMBER_ANSWERS"])


def split_exemplar(lang: str, exemplar: dict[str, str], number: int) -> tuple[str, str, str]:
    """An exemplar's question, worked solution and final sentence, by its language's form; ``Unusable`` off that form.

    The labels are dropped. The final sentence runs from the last place its opening words appear to the end of the
    answer, as written, and must state ``number``. The solution is the text between the answer's label and the final
    sentence, without the whitespace that separates them.
    """
    if lang not in FORMS:
        raise Unusable(f"no form is known for the language {lang!r}")
    question_label, answer_label, final_words = FORMS[lang]
    if not (isinstance(exemplar, dict) and isinstance(exemplar.get("q"), str) and isinstance(exemplar.get("a"), str)):
        raise Unusable("the exemplar is not {'q': question, 'a': answer}")
    if not exemplar["q"].startswith(question_label):
        raise Unusable(f"the question does not start with {question_label!r}")
    question = exemplar["q"][len(question_label) :]
    if not question.strip():
        raise Unusable("the question is empty")
    if not exemplar["a"].startswith(answer_label):
        raise Unusable(f"the answer does not start with {answer_label!r}")
    body = exemplar["a"][len(answer_label) :]
    at = body.rfind(final_words)
    if at < 0:
        raise Unusable(f"the answer has no final sentence opening with {final_words!r}")
    solution, final = body[:at].rstrip(), body[at:]
    if not solution:
        raise Unusable("the answer has no worked solution before its final sentence")
    stated = re.search(r"[0-9]+", final[len(final_words) :])
    if stated is None or int(stated.group()) != number:
        raise Unusable(f"the final sentence {final!r} does not state the exemplar's number {number}")
    return question, solution, final


def exemplar_set(data: bytes, skipped: list[tuple[str, str]] | None = None) -> list[dict]:
    """The exemplars' parse cases, from exemplars.py: a line per exemplar, in the file's order.

    The request is the exemplar's question; the message is the output of a model that thinks, the worked solution as
    ``reasoning_content`` and the final sentence as ``content``. An exemplar's number is the one
    ``EXEMPLAR_NUMBER_ANSWERS`` gives its key: the first for ``"1"``. An exemplar that cannot become a case is left out
    and appended to ``skipped`` with its reason.
    """
    exemplars, numbers = read_exemplars(data)
    number_of = {str(index): number for index, number in enumerate(numbers, start=1)}
    lines = []
    for lang, items in exemplars.items():
        for key, exemplar in items.items():
            try:
                if key not in number_of:
                    raise Unusable(f"EXEMPLAR_NUMBER_ANSWERS has no number for the key {key!r}")
                question, solution, final = split_exemplar(lang, exemplar, number_of[key])
            except Unusable as err:
                if skipped is not None:
                    skipped.append((f"exemplar {lang} {key}", str(err)))
                continue
            lines.append(
                {
                    "name": f"{EXEMPLAR_SET}-{lang}-{key}",
                    "request": {"messages": [{"role": "user", "content": question}]},
                    "message": {"reasoning_content": solution, "content": final},
                    "notes": f"MGSM {lang} exemplar {key}",
                    "origin": origin(EXEMPLARS_FILE, EXEMPLARS_SHA256, f"MGSM_EXEMPLARS[{lang!r}][{key!r}]"),
                }
            )
    return lines


def build_sets(
    files: dict[str, bytes], exemplars: bytes, skipped: list[tuple[str, str]] | None = None
) -> dict[tuple[str, str], list[dict]]:
    """Corpus lines per ``(kind, set name)``: each language's render and answer sets, then the exemplars' parse set."""
    return {**language_sets(files, skipped), ("parse", EXEMPLAR_SET): exemplar_set(exemplars, skipped)}


def write_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> list[Path]:
    """Write every set, and remove ``mgsm-*`` files the import no longer writes."""
    return corpus_sets.write(sets, corpus_dir, "mgsm-")


def check_sets(sets: dict[tuple[str, str], list[dict]], corpus_dir: Path) -> list[str]:
    """One line per set file that differs from a fresh import; empty when the corpus is what the import writes."""
    return corpus_sets.check(sets, corpus_dir, "mgsm-", "MGSM file")


def fetch(path: str, sha256: str, cache: Path) -> bytes:
    """The bytes of ``path`` in the repository at the pinned commit, checked against ``sha256``."""
    return github.fetch(OWNER, REPO, COMMIT, path, sha256, cache=cache).read_bytes()


def run(args: argparse.Namespace) -> int:
    check_license(fetch(LICENSE_FILE, LICENSE_SHA256, args.cache))
    files = {lang: fetch(data_file(lang), sha256, args.cache) for lang, sha256 in SHA256.items()}
    exemplars = fetch(EXEMPLARS_FILE, EXEMPLARS_SHA256, args.cache)
    skipped: list[tuple[str, str]] = []
    sets = build_sets(files, exemplars, skipped)
    if args.check:
        problems = check_sets(sets, args.corpus)
        for problem in problems:
            print(problem, file=sys.stderr)
        if not problems:
            print(f"{args.corpus}: the MGSM sets equal a fresh import of {SOURCE}")
        return 1 if problems else 0
    for (kind, name), lines in sorted(sets.items()):
        print(f"{args.corpus / kind / f'{name}.jsonl'}: {len(lines)} cases")
    corpus_sets.report_skipped(skipped)
    write_sets(sets, args.corpus)
    return 0
