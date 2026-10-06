# Corpus

The requests that `bellwether record` runs through the oracles. One JSON Lines file per set:

```
corpus/
  render/
    common.jsonl         # cases every model records
    <slug>/<set>.jsonl   # cases for one model, added to the set of the same name
```

A line is `{"name": "...", "request": {...}, "notes": "..."}`. `name` is a lowercase slug and
becomes the last part of the fixture id (`<slug>/<kind>/<name>`); `request` is an OpenAI chat
completion body without `model` (the manifest supplies it); `notes` says what the case probes.

A parse case adds `"message"`: the assistant message the output must parse to (`content`,
`reasoning_content`, `tool_calls` with `function.name` and `function.arguments` as the JSON string the
parser must return, byte for byte). The round-trip oracle renders that message as the final assistant
turn of `request` through the model's template; the text after the generation prompt is the output.

Cases are inputs, so they may be written by hand or imported from a vendor catalogue; the
recorded results next to them under `fixtures/` may not.

## Imported sets

Sets named `<dataset>-<split>` are written by `bellwether import <dataset>` and are never edited by hand: a
change to one is a change to its importer, and `bellwether import <dataset> --check` (run in CI) fails when a
set differs from a fresh import. Each line adds `origin`: the dataset, its source pinned by version, the sha256 of
what was downloaded, the file and row inside it (and, for a parse case, the file its message came from), and the
dataset's license. The message-shape sets, `shapes-<shape>`, are written by `bellwether import shapes` from two
datasets; their `origin` names `shapes` and holds each source's origin under `parts`.

A parse set named for the shape of its message adds the message's parts to its name in order, in the words `reasoning`,
`content` and `calls`: `gsm8k-<split>-reasoning-content` holds reasoning, then content; `gsm8k-<split>-content` holds
content alone.

A case that repeats an earlier one is left out, since it tests nothing the first does not. It repeats one when both are
of the same kind and its `request` (render), or its `request` and `message` (parse), is the same JSON, key order
included, since a template can see key order; names, notes and origin do not count. Earlier is in the order the importer
builds its sets. The import names each case it leaves out and the case it repeats, so the counts below are of distinct
cases within one dataset: the rule compares the sets of one import, not one dataset's sets with another's. Cases that
are not repeats can still carry the same message, which is all a parser sees, so the import also prints how many
distinct messages each parse set holds, compared as the line writes them, and how many the dataset's parse sets hold
together.

| Dataset | Sets | Source | License | Attribution |
|---|---|---|---|---|
| BFCL | `bfcl-<category>` for the 17 categories of smg's weekly run. The 13 single-turn categories: 3635 render cases, one per row except 6 `live_irrelevance` rows that repeat an earlier case; 2420 parse cases, one per row with a ground truth, except 76 Java and JavaScript rows whose values are not strings (#26) and 5 rows where no call the rule builds passes BFCL's own checker (the import names them), with 2233 distinct messages among them. The four multi_turn categories, `bfcl-multi-turn-{base,miss-func,miss-param,long-context}`, first turn only: 484 render cases, one per row except 316 that repeat an earlier case; 315 parse cases, one per row whose first turn has a ground-truth call (175 miss_func and miss_param rows have none) except 310 that repeat an earlier case, with 166 distinct messages among them. The import names every row it gives no parse case | `pypi:bfcl-eval==2026.3.23`, sha256 `3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a` | Apache-2.0, checked in the wheel's METADATA on every import; the wheel has no LICENSE file, so the import copies `LICENSE` from `github:ShishirPatil/gorilla@6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`, the commit the wheel was built from (sha256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`), to `corpus/licenses/bfcl-LICENSE` | Berkeley Function Calling Leaderboard, Gorilla project, UC Berkeley: https://github.com/ShishirPatil/gorilla |
| GSM8K | `gsm8k-train` and `gsm8k-test`: 7473 and 1319 render cases, one per row; `gsm8k-<split>-reasoning-content` and `gsm8k-<split>-content`: the same rows as parse cases, one set per message shape (no row of the pinned files is left out: every row is usable, and no case repeats an earlier one; all 17584 messages are distinct) | `github:openai/grade-school-math@3101c7d5072418e28b9008a6636bde82a006892c`; sha256 `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` (`grade_school_math/data/train.jsonl`), `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` (`grade_school_math/data/test.jsonl`), `86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b` (`LICENSE`) | MIT, checked against the LICENSE file's sha256 on every import and copied to `corpus/licenses/gsm8k-LICENSE` | Training Verifiers to Solve Math Word Problems, Cobbe et al. 2021, OpenAI: https://github.com/openai/grade-school-math |
| Shapes (BFCL and GSM8K) | `shapes-<shape>` for six message shapes, `reasoning`, `content`, `reasoning-content`, `reasoning-calls`, `content-calls` and `reasoning-content-calls`: 1000 parse cases each, all distinct, with 6000 distinct messages among them, the same 1000 pairs of a BFCL parse case and a GSM8K test row in every set | BFCL's and GSM8K's pins above: the wheel, and `grade_school_math/data/test.jsonl` with `LICENSE` | Apache-2.0 for the BFCL part of each case and MIT for the GSM8K part, each checked by its importer's check on every import; the GSM8K notice is the copy in `corpus/licenses/gsm8k-LICENSE` | Berkeley Function Calling Leaderboard, Gorilla project, UC Berkeley: https://github.com/ShishirPatil/gorilla; Training Verifiers to Solve Math Word Problems, Cobbe et al. 2021, OpenAI: https://github.com/openai/grade-school-math |

A BFCL request is what the weekly run sends in function-calling mode through `OpenAICompletionsHandler`. A parse
case's message is one call per ground-truth entry, each parameter taking its first acceptable value that is not
BFCL's "may be omitted" marker, and each call held to BFCL's own parameter rules: a parameter its function does not
declare is left out when the answer allows it. `docs/benchmark-sets.md` has the rules; `scripts/bfcl_equivalence.py`
checks every request and call against BFCL's own code, in a throwaway environment.

A multi_turn case is the first turn of a BFCL multi_turn row. Its request is the first one the weekly run sends for
the row: the turn's user message, with the functions of the row's classes as tools, less any BFCL holds back until a
later turn. Its message is that turn's ground-truth calls in order, read from BFCL's Python call strings; a value
passed by position takes the parameter of the class method's `def`, as BFCL's executor binds it. Later turns are not
imported yet: each of their requests carries the results of the earlier calls, which only BFCL's own simulators
produce, and whether to run them is a decision for later.

The four multi_turn sets repeat cases, and the rule for repeats leaves out 316 of their 800 render cases and 310 of
their 625 parse cases, in the order the import builds them (base, miss_func, miss_param, long_context):

| Set | Render cases | Left out as repeats | Parse cases | Left out as repeats |
|---|---|---|---|---|
| `bfcl-multi-turn-base` | 199 | 1 | 199 | 1 |
| `bfcl-multi-turn-miss-func` | 199 | 1 | 105 | 1 |
| `bfcl-multi-turn-miss-param` | 83 | 117 | 5 | 114 |
| `bfcl-multi-turn-long-context` | 3 | 197 | 6 | 194 |

In base and in miss_func, row 43's first turn repeats row 40's. miss_func holds a function back in every row, so none
of its first requests is base's, while 117 of miss_param's 200 first requests and 114 of its 119 parse cases equal
base's. long_context sets BFCL's simulators up with long contents (`long_context=True`), which changes what the calls
return in later turns, not what the first request asks: 195 of its first requests and 191 of its parse cases equal
base's, and 2 and 3 more equal miss_param's. It keeps the first turns of rows 151, 154 and 197, and the parse cases of
those and of rows 38, 41 and 46, whose first requests are base's but whose ground-truth calls differ.

A GSM8K case is one grade school math problem. Its request is the question as a single user turn, with no system
prompt, tools or sampling parameters. Its parse cases take the assistant message from the published answer, the worked
solution with its calculator annotations (`<<16-3-4=9>>`) and then a last line `#### <final answer>`, in the two
shapes a model's output takes: `gsm8k-<split>-reasoning-content` holds the solution, annotations as written, as
`reasoning_content` and the text after `#### ` as `content`; `gsm8k-<split>-content` holds the whole answer, `#### `
line included, as `content` with no reasoning. Case names carry the split and the row's 0-based line index
(`gsm8k-test-7`, `gsm8k-test-reasoning-content-7`). A row with an empty question, whose answer does not end in a `#### `
line, or whose final answer starts or ends with whitespace, would be left out of all three sets of its split with its
reason printed; the pinned files have none. One train question (row 2381) holds two U+2028 line separators, which the
corpus keeps raw, so the readers split JSON Lines on `"\n"` only.

Each import also writes its dataset's license to `corpus/licenses/`, and `--check` compares it as it does a set. The
GSM8K sets copy the dataset's text whole, and the MIT license asks that its notice be included in all copies or
substantial portions, so `bellwether import gsm8k` writes the pinned `LICENSE` to `gsm8k-LICENSE`. The BFCL sets carry
BFCL's questions and function documentation, and Apache-2.0 asks that whoever redistributes the work or a derivative
give its recipients a copy of the License (section 4(a)), and the attributions of its NOTICE file if it has one (4(d)).
The `bfcl-eval` wheel holds no LICENSE file, only `License: Apache 2.0` in its METADATA, so `bellwether import bfcl`
writes the gorilla repository's root `LICENSE`, pinned by commit and sha256, to `bfcl-LICENSE`. The commit, `6ea57973`,
is the one the wheel was built from: the wheel's 183 files under `bfcl_eval/` are that commit's, byte for byte, and
BFCL's publish workflow names a build of `main` by its UTC date, with a serial after the day's first commit, so
`2026.3.23` is the first commit of 2026-03-23, and its only one. `berkeley-function-call-leaderboard/` has no LICENSE
of its own, and the repository has no NOTICE file. The copies sit outside the `render/` and `parse/` directories the
recorder reads, and each carries its importer's prefix.

A message shape is which parts an assistant message holds, in the order a model writes them: reasoning
(`reasoning_content`), content and calls (`tool_calls`). A shapes case combines GSM8K's text with BFCL's calls, so that
a parser meets every shape: BFCL's parse cases hold calls alone, GSM8K's reasoning and content alone. Each set is named
by the parts of its shape in order. `bellwether import shapes` builds the cases with the two importers' builders from
the pinned files, never from the committed sets, so its `--check` depends on the pins alone. BFCL's parse cases of its
eight Python categories with calls (`simple_python`, `multiple`, `parallel`, `parallel_multiple` and the four live ones)
are interleaved, one from each category in turn, and the i-th is paired with the i-th GSM8K test row, for 1000 pairs
(`SIZE` in `src/bellwether/importers/shapes.py`); a source with fewer stops the import rather than give a case or a text
twice. At the pins that is 160 cases from each of the six larger categories, the 16 of `live_parallel` and the 24 of
`live_parallel_multiple`, with test rows 0 to 999. A category with no parse case stops the import. The import prints
every row it reads and does not use, with its reason: the five BFCL rows of these categories that have no parse case, by
name (two of them, `live_simple_106-63-0` and `live_simple_112-68-0`, fall inside the range the pairs use), then the
parse cases of each category and the GSM8K rows that come after the 1000 pairs, one run each (1346 parse cases, and test
rows 1000 to 1318). A case that repeats an earlier one is left out and named, as in every import; across the six sets
none does, so each keeps its 1000 cases, and their messages are distinct too: 1000 in each set, 6000 across the six.
Pair i is case i of every set (`shapes-content-calls-7`):

- the request and the calls are the BFCL case's; the shapes without calls keep the request and its tools, since a
  model may answer without calling;
- the reasoning and the content are the GSM8K solution as `gsm8k-test-reasoning-content` holds it, as written with its
  calculator annotations, split at its last line: the lines before it are the reasoning, and the last line is the
  content (`She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.` for test row 0).

GSM8K writes one step per line, and its last line is the step that states the result: in each of the 1000 test rows the
pairs take, it holds the final answer, 979 times as written after `#### ` and 21 times with other digit grouping
(`70,000` for `70000`). A line is the dataset's own unit, where a last sentence would need a sentence splitter (nine of
those last lines hold a title such as `Mr.`). The two parts together rebuild each solution as written, so a message
holds no text GSM8K did not write, and none twice. The content sets hold 1000 distinct contents, where the final answer
alone would give 296, and `gsm8k-test-reasoning-content` holds that answer already. A row whose solution has no text
before its last line, or an empty last line, would be left out with its reason; the pinned test file has none.

A message has `content: ""` where its shape has no content but has reasoning or calls, and the content-only message has
no `reasoning_content`. The text does not answer the request: what a case probes is its shape. `origin` is
`{"dataset": "shapes", "parts": [...]}`, the BFCL case's origin and then the GSM8K row's, each as its importer writes
it, and `bellwether count` counts the cases under `shapes`.

Set names starting with `bfcl-`, `gsm8k-` or `shapes-` belong to that importer: `bellwether import bfcl` deletes any
`bfcl-*.jsonl` it did not write, `bellwether import gsm8k` any `gsm8k-*.jsonl`, and `bellwether import shapes` any
`shapes-*.jsonl`. Name hand-written sets otherwise.

`--check` reads the pinned files from `~/.cache/bellwether/datasets` and downloads them on a miss. It then needs PyPI to
still serve that exact wheel: a yanked release still does when pinned by version; a release deleted from PyPI does not,
and the check fails until the importer pins another. GSM8K's files, and the LICENSE the BFCL import copies, are read
from `raw.githubusercontent.com` at the pinned commit, which serves them as long as the repository keeps that commit.
`bellwether import shapes --check` reads the same pinned files as the other two.

`bellwether record` writes a fixture set recorded from an imported set as zstd-compressed JSON Lines in Git LFS
(`fixtures/README.md`); `bellwether unpack` gives consumers the plain files.
