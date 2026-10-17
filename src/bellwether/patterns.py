"""bellwether patterns: what Python's ``regex`` module says about the patterns response templates write.

A Hugging Face response template writes the parts of a tool call as regular expressions, compiled by transformers with
the ``regex`` module and ``DOTALL``. SMG's Symphony lowers each part into its own matcher and runs it while the output
streams, so it needs the module's answers to hold itself to: this command records them, at the ``regex`` version
``uv.lock`` pins, and ``bellwether unpack`` puts the recordings beside the models' sets, under ``patterns/`` of the
tree consumers read. The parts recorded are every pattern a call shape runs in the public response templates, each
named with the template it comes from, and a few patterns on abstract tags for what no public template writes:

- the header after a call's opening marker, which captures the tool's name;
- the text before a call's closing marker, and the closing pattern whole;
- a parameter's tag, with its key, its value and the way the value ends.

``regex.jsonl`` records, per pattern, ``regex.match`` (anchored at the text's start, under ``DOTALL``, with the
character before the text given, as ``\\b`` reads it) over written texts and over seeded variations of them: whether
it matches, where the match ends, the spans of ``name``, ``key`` and ``value``, and which of the value's ends closed
it; for the written texts also the answer for every prefix taken as the whole text. Offsets are UTF-8 bytes.
``search.jsonl`` records, for the text before a call's closing marker (the trailer), where a region closes: the start
``regex.search`` gives for the trailer followed by the marker at the end of a text (the marker appended and ``\\Z``
after it, or the trailer alone at the end for a marker that is a stop token, which the stream never shows), over
written texts, every prefix of them, and seeded variations. ``classes.json`` records the module's ``\\w``, ``\\s``
and ``\\d`` as ranges over every code point, and the membership of a set of probe characters in every class the
patterns use, with the ``regex`` and Python versions that gave them.

The recording is a function of the ``regex`` version alone, so ``--check`` (CI) records again and compares the three
files byte for byte, as the import checks do for the corpus; a version other than ``REGEX_VERSION`` is refused, since
the module's tables move with its Unicode data.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import regex

REGEX_VERSION = "2026.9.29"
PYTHON = "3.12"
FILES = ("regex.jsonl", "search.jsonl", "classes.json")

TRANSFORMERS = "transformers v5.18.0"
SERVE = f"{TRANSFORMERS} src/transformers/cli/serving/utils.py"
HF_TESTS = f"{TRANSFORMERS} tests/utils/test_chat_parsing.py"
TRL = "huggingface/trl trl/chat_template_utils.py at e4edcf7d"
ABSTRACT = "abstract tags: a matcher test, from no template"


def entry(id_, source, before, provenance, texts, pattern=None, ends=()):
    """One pattern: its id, its source as the template writes it, the character before the text it
    runs on (the last of the marker it follows), its public source, and its written texts. A
    pattern with value ends gives `pattern` (the part before the ends) and the `ends`, each a
    consumed part with an optional `(?=...)` part, which together spell `source`."""
    pattern = source if pattern is None else pattern
    ends = list(ends)
    spelled = pattern + (ends[0] if len(ends) == 1 else "(?:" + "|".join(ends) + ")" if ends else "")
    assert spelled == source, (id_, spelled, source)
    return {
        "id": id_,
        "source": source,
        "before": before,
        "provenance": provenance,
        "texts": texts,
        "pattern": pattern,
        "ends": ends,
    }


GEMMA4 = (
    "google/gemma-4-31B-it tokenizer_config.json at 842da379 (response_template); "
    f"{HF_TESTS} (gemma4_template); {SERVE} (gemma4); {TRL} (gemma4_template)"
)
MUSE = "meta-models/Muse-Glimmer-30B tokenizer_config.json at a4e59da5 (response_template)"
MUSE_PARAMETER = r'<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)</atem:parameter>'

PATTERNS = [
    entry(
        "gemma4/header",
        r"(?P<name>\w+)",
        ":",
        GEMMA4,
        [
            'get_weather{location:<|"|>Paris<|"|>}',
            "get_weather",
            "get-weather{",
            "{",
            "",
            " get{",
            "caf\xe9{",
            "e\u0301x{",
            "x\xb2{",
            "\u65e5\u672c{",
            "a\xa0b{",
            "\U0001d400b{",
            "_1{",
        ],
    ),
    entry(
        "muse-glimmer/header",
        r'\b[^>]*?\bname="(?P<name>[^"]+)">',
        "e",
        MUSE,
        [
            ' name="get_weather">',
            ' name="get_weather">{"city": "Paris"}',
            ' id="1" name="get_weather">',
            'name="x">',
            ' name="">',
            ' name="a"',
            ">",
            ' foo name="a">',
            '\tname="a">',
            ' name="a b">',
            ' xname="a">',
            '\xa0name="a">',
            '\u0301name="a">',
            ' name="\xe4">',
        ],
    ),
    entry(
        "muse-glimmer/parameter",
        MUSE_PARAMETER,
        None,
        MUSE,
        [
            '<atem:parameter name="city">Paris</atem:parameter>',
            '<atem:parameter name="city" type="string">Paris</atem:parameter>',
            '<atem:parameter name="n">5</atem:parameter><atem:parameter name="m">6</atem:parameter>',
            '<atem:parameter name="city">Pa</atem:par',
            '<atem:parameters name="x">1</atem:parameter>',
            '<atem:parameter  name="a b">x y</atem:parameter>',
            '<atem:parameter name="q">a b\tc</atem:parameter>',
            '<atem:parameter name="">1</atem:parameter>',
            '<atem:parameter type="int" name="n">7</atem:parameter>',
        ],
        pattern=r'<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)',
        ends=[r"</atem:parameter>"],
    ),
    entry(
        "serve-qwen3_5/header",
        r"\s*<function=(?P<name>[^>\n]+)>",
        ">",
        f"{SERVE} (qwen3_5)",
        [
            "\n<function=get_weather>\n",
            "<function=get_weather>",
            "  \n <function=a b>",
            "<function=>",
            "<function=get\nweather>",
            "<function=get_weather",
            "\n<parameter=x>",
            "<function=\u540d\u524d>",
            "\xa0<function=a>",
            "\x1c<function=a>",
            "\u2028<function=a>",
            "\x85<function=a>",
        ],
    ),
    entry(
        "serve-qwen3_5/trailer",
        r"</function>\s*",
        ">",
        f"{SERVE} (qwen3_5)",
        [
            "</function>\n",
            "</function>\n</tool_call>",
            "</function>",
            "</function",
            "</tool_call>",
            "</function> \t\n x",
        ],
    ),
    entry(
        "serve-qwen3_5/close",
        r"</function>\s*</tool_call>",
        ">",
        f"{SERVE} (qwen3_5)",
        [
            "</function>\n</tool_call>",
            "</function></tool_call>",
            "</function>\n",
            "</tool_call>",
            "</function> \n x</tool_call>",
            "</function>\u2028</tool_call>",
        ],
    ),
    entry(
        "serve-qwen3_5/parameter",
        r"<parameter=(?P<key>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>",
        None,
        f"{SERVE} (qwen3_5); {TRL} (qwen3_5_template, nemotron_3_template)",
        [
            "<parameter=city>\nParis\n</parameter>",
            "<parameter=city>Paris</parameter>",
            "<parameter=note>\n  two\n  lines \n</parameter>",
            "<parameter=x>\n</parameter>",
            "<parameter=x>a</parameter>b</parameter>",
            "<parameter=x>a",
            "<parameter=a b>1</parameter>",
            "<parameter=x\ny>1</parameter>",
            "<parameter=x>\xa0v\xa0</parameter>",
            "<parameter=x>\x1cv</parameter>",
        ],
        pattern=r"<parameter=(?P<key>[^>\n]+)>\s*(?P<value>.*?)",
        ends=[r"\s*</parameter>"],
    ),
    entry(
        "hf-qwen3/header",
        r"\s*<function=(?P<name>\w+)>",
        ">",
        f"{HF_TESTS} (qwen3_template, _XML_STRING_ARGS_TEMPLATE)",
        [
            "\n<function=get_weather>\n",
            "<function=get_weather>",
            "<function=get-weather>",
            "<function=>",
            " <function=caf\xe9>",
            "<function=x\xb2>",
            "<function=get_weather",
        ],
    ),
    entry(
        "hf-qwen3/parameter",
        r"<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>",
        None,
        f"{HF_TESTS} (qwen3_template, _XML_STRING_ARGS_TEMPLATE)",
        [
            "<parameter=city>\nParis\n</parameter>",
            "<parameter=n>5</parameter>",
            "<parameter=a-b>1</parameter>",
            "<parameter=x>\n\n</parameter>",
            "<parameter=x> a </parameter> b </parameter>",
            "<parameter=x>1</param",
        ],
        pattern=r"<parameter=(?P<key>\w+)>\s*(?P<value>.*?)",
        ends=[r"\s*</parameter>"],
    ),
    entry(
        "hf-kv-lines/header",
        r"\s*<function=(?P<name>\w+)>\n",
        ">",
        f"{HF_TESTS} (_KV_LINES_TOOLS_TEMPLATE)",
        [
            "\n<function=get_weather>\ncity: Paris\n",
            "<function=get_weather>",
            "<function=get_weather> \n",
            "<function=get_weather>\r\n",
        ],
    ),
    entry(
        "trl-qwen3_5/header",
        r"\s*<function=(?P<name>[^\n>]+)>",
        ">",
        f"{TRL} (qwen3_5_template, nemotron_3_template)",
        [
            "\n<function=get_weather>\n",
            "<function=a b>",
            "<function=a\nb>",
            "<function=",
            "<function=\xe9>",
        ],
    ),
    entry(
        "hf-gpt-oss/header",
        r"(?P<name>\w+).*?<\|message\|>",
        ".",
        f"{HF_TESTS} (gpt_oss_template)",
        [
            'get_weather<|message|>{"city": "Paris"}',
            "get_weather <|constrain|>json<|message|>{}",
            "get_weather json<|message|>",
            "<|message|>",
            "get_weather<|mess",
            "get_weather<|message|",
            "a<|message|>b<|message|>",
            "x\n<|message|>",
            "x\xb2<|message|>",
        ],
    ),
    entry(
        "trl-gptoss/header",
        r"(?P<name>\S+?)<\|channel\|>commentary\s+json<\|message\|>",
        ".",
        f"{TRL} (gptoss_template)",
        [
            "get_weather<|channel|>commentary json<|message|>{}",
            "get_weather<|channel|>commentary\njson<|message|>",
            "get_weather <|channel|>commentary json<|message|>",
            "get_weather<|channel|>commentaryjson<|message|>",
            "a<|channel|>commentary json<|message|>x<|channel|>commentary json<|message|>",
            "a<|channel|>analysis<|message|>",
            "a<|channel|>commentary \xa0json<|message|>",
            "a<|channel|>commentary json<|mess",
        ],
    ),
    entry(
        "trl-llama3/header",
        r'\s*"(?P<name>[^"]+)",\s*"parameters":\s*',
        ":",
        f"{TRL} (llama3_template)",
        [
            ' "get_weather", "parameters": {"city": "Paris"}}',
            '"get_weather","parameters":{}',
            ' "get_weather", "arguments": {}',
            ' "a\\"b", "parameters": {}',
            ' "x",\n  "parameters":\n {',
            ' "x", "parameters": ',
            ' "", "parameters": {}',
            ' "x" , "parameters": {}',
        ],
    ),
    entry(
        "trl-llama3/close",
        r"\}<\|eot_id\|>\s*",
        "}",
        f"{TRL} (llama3_template)",
        [
            "}<|eot_id|>",
            "}<|eot_id|>\n\n",
            "} <|eot_id|>",
            "}<|eot",
            "}}<|eot_id|>",
        ],
    ),
    entry(
        "trl-glm4moe/header",
        r"(?P<name>\S+)\n?",
        ">",
        f"{TRL} (glm4moe_template)",
        [
            "get_weather\n<arg_key>city</arg_key>",
            "get_weather<arg_key>city</arg_key>\n<arg_value>",
            "get_weather",
            "\nget",
            "get_weather \n",
            "\u65e5\u672c\n",
            "a\xa0b\n",
            "a\x1cb\n",
        ],
    ),
    entry(
        "trl-glm4moe/parameter",
        r"<arg_key>(?P<key>[^<]+)</arg_key>\s*\n<arg_value>(?P<value>.*?)</arg_value>",
        None,
        f"{TRL} (glm4moe_template)",
        [
            "<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>",
            "<arg_key>city</arg_key><arg_value>Paris</arg_value>",
            "<arg_key>city</arg_key>\n\n<arg_value>Paris</arg_value>",
            "<arg_key>city</arg_key> \n <arg_value>Paris</arg_value>",
            "<arg_key>a<b</arg_key>\n<arg_value>1</arg_value>",
            "<arg_key>n</arg_key>\n<arg_value>1</arg_value>\n<arg_key>m</arg_key>",
            "<arg_key>n</arg_key>\n<arg_value>1</arg_val",
        ],
        pattern=r"<arg_key>(?P<key>[^<]+)</arg_key>\s*\n<arg_value>(?P<value>.*?)",
        ends=[r"</arg_value>"],
    ),
    entry(
        "abstract/optional-and-bounded",
        r"(?:<a>)?(?P<name>\w{2,4})y",
        None,
        ABSTRACT,
        ["<a>abcy", "abcdy", "abcdey", "<a>ay", "ay", "<a>xyy", "<ab>y", "abc"],
    ),
    entry(
        "abstract/lazy-bounded",
        r"(?P<name>\w{2,4}?)y",
        None,
        ABSTRACT,
        ["abyy", "abcy", "ay", "abcdey", "yyy", "ab"],
    ),
    entry(
        "abstract/two-optionals",
        r"(?:ab)?(?:a)?b",
        None,
        ABSTRACT,
        ["abab", "abb", "ab", "b", "aab", "ac", "a"],
    ),
    entry(
        "abstract/boundaries",
        r"\b(?P<name>\w+)\b\s*=",
        " ",
        ABSTRACT,
        ["key = 1", "key=1", "key\xa0=", "k\u0301ey =", "=", "key", "key x="],
    ),
    entry(
        "abstract/filler",
        r"[^<]*?<a>",
        None,
        ABSTRACT,
        ["x y<a>", "<a>", "x<b><a>", "x<a<a>", "\n\n<a>z"],
    ),
    entry(
        "abstract/lookahead-end",
        r"<k=(?P<key>\w+)>(?P<value>.*?)(?:</k>|(?=<k=))",
        None,
        ABSTRACT,
        [
            "<k=a>1</k>",
            "<k=a>1<k=b>2</k>",
            "<k=a>1<kx</k>",
            "<k=a></k>",
            "<k=a><k=b>",
            "<k=a>1",
            "<k=a>1<k",
        ],
        pattern=r"<k=(?P<key>\w+)>(?P<value>.*?)",
        ends=[r"</k>", r"(?=<k=)"],
    ),
    entry(
        "abstract/lookahead-end-trimmed",
        r"<k=(?P<key>\w+)>\s*(?P<value>.*?)(?:\s*</k>|\s*(?=<k=))",
        None,
        ABSTRACT,
        [
            "<k=a> 1 </k>",
            "<k=a>\n1\n<k=b>2</k>",
            "<k=a>  </k>",
            "<k=a> 1 \n",
            "<k=a>1 <k=",
        ],
        pattern=r"<k=(?P<key>\w+)>\s*(?P<value>.*?)",
        ends=[r"\s*</k>", r"\s*(?=<k=)"],
    ),
]

# Characters every pattern's variations draw on besides its own texts': the markers' punctuation,
# whitespace of every kind the classes tell apart, and word characters of every width. A newline
# comes only from a pattern's own texts.
SHARED = ' \t<>"=/az_0\xe9\u0301\xb2\xa0\x1c\x85\u2028\u4e2d\U0001d400\U0001f600'
VARIATIONS = 48
RESPELLED = 24
SEED = 2719
SEARCH_SEED = 3145
SLOTS = ("name", "key", "value")


def utf8(text, upto):
    """The byte offset of character `upto` of `text`."""
    return len(text[:upto].encode("utf-8"))


def compiled(entry_):
    """The pattern with each end in a group of its own, so a match says which end closed it, and
    the pattern as written; both must give the same match."""
    ends = entry_["ends"]
    if not ends:
        marked = entry_["source"]
    else:
        marked = entry_["pattern"] + "(?:" + "|".join(f"(?P<end{i}>{end})" for i, end in enumerate(ends)) + ")"
    return regex.compile(marked, regex.DOTALL), regex.compile(entry_["source"], regex.DOTALL)


def answer(patterns, before, text):
    """`match` at the start of `text`, after `before`: None, or where the match ends, its spans
    and the end that closed a value."""
    marked, plain = patterns
    whole = (before or "") + text
    at = len(before or "")
    m = marked.match(whole, at)
    check = plain.match(whole, at)
    assert (m is None) == (check is None), text
    if m is None:
        return None
    assert m.span() == check.span(), text
    out = {"end": utf8(text, m.end() - at)}
    for slot in SLOTS:
        if slot in m.groupdict():
            assert m.span(slot) == check.span(slot), (text, slot)
            if m.span(slot) != (-1, -1):
                start, end = m.span(slot)
                out[slot] = [utf8(text, start - at), utf8(text, end - at)]
    closed = [i for i in range(10) if f"end{i}" in m.groupdict() and m.span(f"end{i}") != (-1, -1)]
    if closed:
        out["ending"] = closed[0]
    return out


def variation(rng, text, alphabet):
    """`text` edited one to three times: a character taken out, put in or changed, the text cut, or
    a slice of it repeated."""
    chars = list(text)
    for _ in range(rng.randrange(1, 4)):
        kind = rng.randrange(5)
        if kind == 0 and chars:
            del chars[rng.randrange(len(chars))]
        elif kind == 1:
            chars.insert(rng.randrange(len(chars) + 1), rng.choice(alphabet))
        elif kind == 2 and chars:
            chars[rng.randrange(len(chars))] = rng.choice(alphabet)
        elif kind == 3 and chars:
            chars = chars[: rng.randrange(len(chars) + 1)]
        elif kind == 4 and chars:
            start = rng.randrange(len(chars))
            stop = rng.randrange(start, len(chars) + 1)
            chars[start:start] = chars[start:stop]
    return "".join(chars)


# A word as a spell checker splits text into words. A variation keeps only the words its pattern's
# written texts have, single characters and words without an ASCII letter, so that the recorded
# texts hold no misspelling the edits made; a cut inside a word is already covered by the answers
# recorded for every prefix of a written text.
WORD = regex.compile(r"[\w\-'\u2019]+")


def spelled_as_written(text, written):
    return all(len(word) == 1 or word in written or not regex.search(r"[A-Za-z]", word) for word in WORD.findall(text))


def respelled(rng, text, found, alphabet):
    """`text` with one captured span (in bytes, as `answer` gives it) spelled anew: up to six
    characters of `alphabet`, so the capture's edges meet every class the alphabet holds."""
    slots = [slot for slot in SLOTS if found and slot in found]
    if not slots:
        return None
    start, end = found[rng.choice(slots)]
    raw = text.encode("utf-8")
    spelled = "".join(rng.choice(alphabet) for _ in range(rng.randrange(7)))
    return (raw[:start] + spelled.encode("utf-8") + raw[end:]).decode("utf-8")


def record_patterns():
    rng = random.Random(SEED)
    lines = []
    for entry_ in PATTERNS:
        patterns = compiled(entry_)
        before = entry_["before"]
        alphabet = sorted(set("".join(entry_["texts"]) + SHARED))
        cases = []
        for text in entry_["texts"]:
            prefixes = [answer(patterns, before, text[:upto]) for upto in range(len(text) + 1)]
            cases.append({"text": text, "match": prefixes[-1], "prefixes": prefixes})
        seen = set(entry_["texts"])
        written = set(WORD.findall(" ".join(entry_["texts"])))
        found = [
            (case["text"], case["match"])
            for case in cases
            if case["match"] and any(slot in case["match"] for slot in SLOTS)
        ]
        made = respelt = tries = 0
        while made < VARIATIONS or (found and respelt < RESPELLED):
            tries += 1
            assert tries < 100_000, f"{entry_['id']}: too few distinct variations"
            if made < VARIATIONS:
                text = variation(rng, rng.choice(entry_["texts"]), alphabet)
            else:
                text = respelled(rng, *rng.choice(found), alphabet)
            if text is None or text in seen or not spelled_as_written(text, written):
                continue
            seen.add(text)
            if made < VARIATIONS:
                made += 1
            else:
                respelt += 1
            cases.append({"text": text, "match": answer(patterns, before, text)})
        line = {
            "id": entry_["id"],
            "source": entry_["source"],
            "before": before,
            "provenance": entry_["provenance"],
            "cases": cases,
        }
        lines.append(json.dumps(line, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return "\n".join(lines) + "\n"


# The trailers a call shape searches for before a region's closing marker, each with the marker
# (empty for a stop token, whose text the stream never shows), the character before the text it
# runs on, and its public source; and a few on abstract tags.
SEARCHES = [
    {
        "id": "serve-qwen3_5/trailer",
        "source": r"</function>\s*",
        "terminal": "</tool_call>",
        "before": ">",
        "provenance": f"{SERVE} (qwen3_5)",
        "texts": [
            "\n</function>\n",
            "</function>",
            "</function> \t\n ",
            "x</function>\n",
            "</function>\n</function>\n",
            "</function>x",
            "",
            "\n",
            "</function",
            "</function>\u2028\x85",
            "</function>\xa0",
        ],
    },
    {
        "id": "trl-llama3/trailer",
        "source": r"\}",
        "terminal": "",
        "before": '"',
        "provenance": f"{TRL} (llama3_template)",
        "texts": ["}", "}}", ' "x"}', "}\n", "", '{"a": 1}}', "} }"],
    },
    {
        "id": "abstract/word-then-boundary",
        "source": r"\w+\b",
        "terminal": "<a>",
        "before": " ",
        "provenance": ABSTRACT,
        "texts": ["ab", "a b", "ab ", "", "_9", "x\xb2", "e\u0301"],
    },
    {
        "id": "abstract/boundary-before-a-word",
        "source": r"\s*\bz",
        "terminal": "z",
        "before": "a",
        "provenance": ABSTRACT,
        "texts": ["z", " z", "az", "a z", "zz", "\nz"],
    },
    {
        "id": "abstract/lazy-tag",
        "source": r"<[^>]*?>\s*",
        "terminal": "</a>",
        "before": None,
        "provenance": ABSTRACT,
        "texts": ["<b>", "<b> <c>\n", "x<b", "<>", "<<b>>", "<b>x"],
    },
]


def search_answer(pattern, terminal, before, text):
    """The start, in bytes of `text`, of the earliest match of `pattern` that ends where `text`
    ends, `terminal` after it; None when there is none."""
    whole = (before or "") + text + terminal
    at = len(before or "")
    m = pattern.search(whole, at)
    return None if m is None else utf8(text, m.start() - at)


def record_searches():
    rng = random.Random(SEARCH_SEED)
    lines = []
    for entry_ in SEARCHES:
        source, terminal, before = entry_["source"], entry_["terminal"], entry_["before"]
        pattern = regex.compile(f"(?:{source}){regex.escape(terminal)}\\Z", regex.DOTALL)
        alphabet = sorted(set("".join(entry_["texts"]) + SHARED))
        written = set(WORD.findall(" ".join(entry_["texts"])))
        cases = []
        for text in entry_["texts"]:
            prefixes = [search_answer(pattern, terminal, before, text[:upto]) for upto in range(len(text) + 1)]
            cases.append({"text": text, "start": prefixes[-1], "prefixes": prefixes})
        seen = set(entry_["texts"])
        made = tries = 0
        while made < VARIATIONS:
            tries += 1
            assert tries < 100_000, f"{entry_['id']}: too few distinct variations"
            text = variation(rng, rng.choice(entry_["texts"]), alphabet)
            if text in seen or not spelled_as_written(text, written):
                continue
            seen.add(text)
            made += 1
            cases.append({"text": text, "start": search_answer(pattern, terminal, before, text)})
        line = {
            "id": entry_["id"],
            "source": source,
            "terminal": terminal,
            "before": before,
            "provenance": entry_["provenance"],
            "cases": cases,
        }
        lines.append(json.dumps(line, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return "\n".join(lines) + "\n"


# Characters whose membership the test checks in every class: Latin-1 and the blocks after it,
# then characters the dialects disagree on or that sit at a range's edge.
PROBES = [chr(c) for c in range(0x300)] + [
    chr(c)
    for c in (
        0x0301,
        0x0660,
        0x0669,
        0x06F0,
        0x088F,
        0x0966,
        0x1680,
        0x180E,
        0x1ACF,
        0x2000,
        0x2007,
        0x200A,
        0x200B,
        0x2028,
        0x2029,
        0x202F,
        0x203F,
        0x2040,
        0x205F,
        0x2060,
        0x3000,
        0x4E2D,
        0xD7FF,
        0xE000,
        0xFEFF,
        0xFF10,
        0xFF21,
        0x10400,
        0x1D400,
        0x1D7CE,
        0x1F600,
        0xE0001,
        0x10FFFF,
    )
]
CLASSES = (
    "\\w",
    "\\W",
    "\\s",
    "\\S",
    "\\d",
    "\\D",
    ".",
    "[^>]",
    '[^"]',
    "[^>\\n]",
    "[^\\n>]",
    "[^<]",
)


def ranges(source):
    """The characters `source` matches, as ranges: each a string of its first and last character.
    A range runs on across the surrogates, which are no characters."""
    one = regex.compile(source, regex.DOTALL)
    out = []
    start = last = None
    for code in range(0x110000):
        if 0xD800 <= code <= 0xDFFF:
            continue
        if one.match(chr(code)) is not None:
            start = code if start is None else start
            last = code
        elif start is not None:
            out.append(chr(start) + chr(last))
            start = None
    if start is not None:
        out.append(chr(start) + chr(last))
    return out


def record_classes():
    probes = "".join(PROBES)
    members = {
        source: "".join("1" if regex.compile(source, regex.DOTALL).match(c) else "0" for c in PROBES)
        for source in CLASSES
    }
    data = {
        "regex": REGEX_VERSION,
        "python": PYTHON,
        "word": ranges(r"\w"),
        "space": ranges(r"\s"),
        "digit": ranges(r"\d"),
        "probes": probes,
        "members": members,
    }
    return json.dumps(data, ensure_ascii=True, sort_keys=True, indent=1) + "\n"


def record() -> dict[str, str]:
    """The three recordings, by file name, as ``--out`` writes them."""
    return {
        "regex.jsonl": record_patterns(),
        "search.jsonl": record_searches(),
        "classes.json": record_classes(),
    }


def version_mismatch() -> str | None:
    """Why this environment cannot record: the module or the interpreter is not the pinned one."""
    python = f"{sys.version_info[0]}.{sys.version_info[1]}"
    if regex.__version__ != REGEX_VERSION or python != PYTHON:
        return (
            f"bellwether patterns: record with Python {PYTHON} and regex {REGEX_VERSION} (what uv.lock pins), "
            f"not Python {python} and regex {regex.__version__}"
        )
    return None


def stale(out: Path, files: dict[str, str]) -> list[str]:
    """The recordings under ``out`` that differ from ``files``, or are not there."""
    return [name for name, text in files.items() if not (out / name).is_file() or (out / name).read_text() != text]


def run(args: argparse.Namespace) -> int:
    if (why := version_mismatch()) is not None:
        print(why, file=sys.stderr)
        return 1
    files = record()
    if args.check:
        differing = stale(args.out, files)
        for name in differing:
            print(f"{args.out / name} differs from what the recording gives", file=sys.stderr)
        return 1 if differing else 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (args.out / name).write_text(text)
        print(f"wrote {args.out / name}: {len(text)} bytes")
    return 0
