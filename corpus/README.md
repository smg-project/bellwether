# Corpus

The requests that `bellwether record` runs through the oracles. One JSON Lines file per set:

```
corpus/
  render/
    common.jsonl         # cases every model records
    <slug>/<set>.jsonl   # cases for one model, added to the set of the same name
    <set>.jsonl.zst      # a set of an import past the limit below, compressed in Git LFS
```

A line is `{"name": "...", "request": {...}, "notes": "..."}`. `name` is a lowercase slug and
becomes the last part of the fixture id (`<slug>/<kind>/<name>`); `request` is an OpenAI chat
completion body without `model` (the manifest supplies it); `notes` says what the case probes.

A parse case adds `"message"`: the assistant message the output must parse to (`content`,
`reasoning_content`, `tool_calls` with `function.name` and `function.arguments` as the JSON string the
parser must return, byte for byte). Each call in it holds `type` and `function` only, never an `id`: a
parser makes up its own id, so no output can carry the one a case would name. Call ids belong to a
request's history, where a tool message's `tool_call_id` names the call it answers. The round-trip oracle
renders that message as the final assistant turn of `request` through the model's template; the text
after the generation prompt is the output.

Cases are inputs, so they may be written by hand or imported from a vendor catalogue; the
recorded results next to them under `fixtures/` may not.

## Imported sets

Sets named `<dataset>-<split>` are written by `bellwether import <dataset>` and are never edited by hand: a change to
one is a change to its importer, and `bellwether import <dataset> --check` (run in CI) fails when a set differs from a
fresh import. Each line adds `origin`: the dataset, its source pinned by version or commit, the sha256 of what was
downloaded, the file and row inside it (and, for a parse case whose message comes from another file, that file), and the
license of the row's content. The message-shape sets, `shapes-<shape>`, are written by `bellwether import shapes` from
two datasets; their `origin` names `shapes` and holds each source's origin under `parts`. When an importer writes text
into a case that the dataset does not have, `origin.written` names it.

An import's sets stay plain JSON Lines while they take at most 50,000,000 bytes in all (`corpus_sets.LIMIT`), so a
change to an importer reads as a diff of cases. Past that, the import writes every one of them whole as
`<set>.jsonl.zst`, compressed with zstd and kept in Git LFS like the benchmark fixture sets, and removes their plain
files: an import keeps one form. `record`, `count` and `--check` read either form; `--check` compares plain content,
names a set stored in the form the limit does not give it, and names a set Git LFS has not fetched with the command
that fetches it. A clone fetches these sets (`.lfsconfig` leaves out only the fixture sets), and CI pulls them before
the import checks. The files an import writes beside its sets, such as a license, stay plain and do not count.

An import and its `--check` hold one set at a time, and never a set's content whole: a set is compressed, written and
compared as its lines are encoded. An importer that cannot hold all its cases at once hands its sets to
`corpus_sets.write` and `check` as it builds them, one set at a time (glaive-v2 does, one number at a time), and the
rule for repeats below applies as they come, keeping a sha256 of each case. Its total is known only at the end, so its
sets are written plain and compressed afterwards if they pass the limit, unless the importer declares their form
(`form="zstd"` or `"plain"`), which `write` and `check` hold to the limit: a declaration the total contradicts is
refused.

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
| MGSM | `mgsm-<lang>` for 10 of its 11 languages (bn, de, es, fr, ja, ru, sw, te, th, zh): 250 render cases each, 2500 in all, one per row, and none for English (below); `mgsm-<lang>-content` for all 11: the same rows as parse cases, the final answer alone as content, 2750 cases holding 134 distinct messages; `mgsm-exemplars`: 88 parse cases, the 8 worked exemplars of each language (no row or exemplar of the pinned files is left out, and no case repeats an earlier one) | `github:google-research/url-nlp@3622039cf51f7eeffa58b957332a8e8c337981d7`, directory `mgsm/`; the sha256 of each file is listed below | CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/), checked against the LICENSE file's sha256 and first line on every import and copied to `corpus/licenses/mgsm-LICENSE`; the GSM8K problems it translates are MIT (below) | MGSM, Shi et al. 2022, Google Research: https://github.com/google-research/url-nlp |
| Hermes | `hermes-func-calling-singleturn`, `hermes-func-calling` and `hermes-glaive-func-calling`, from every row of the dataset's three function-calling configs, kept as `.jsonl.zst` in Git LFS (99.6 MB as plain JSON Lines, past the 50 MB that stay plain; 5.8 MB compressed): 20769 render cases, one per user turn or tool result an assistant turn answers; 20773 parse cases, one per assistant turn; every case distinct, with 10830 distinct messages among the parse cases (8803 of the 18746 in `hermes-glaive-func-calling`). 1836 of the 8995 rows give no case, and 1964 cases that repeat an earlier one are left out (the import names both) | `hf:datasets/NousResearch/hermes-function-calling-v1@dae3e1d28cfbcf4b915c04ea1e072030529b4bda`, each file's sha256 in the importer; the License copy from `github:apache/www-site@01b1be9fbc5cd93b6794f5653a58b9b863807f84` (`content/licenses/LICENSE-2.0.txt`), sha256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | Apache-2.0, checked in the dataset card's YAML front matter on every import. The dataset ships no LICENSE or NOTICE file, so the import writes the License as the Apache Software Foundation publishes it (the bytes of https://www.apache.org/licenses/LICENSE-2.0.txt) to `corpus/licenses/hermes-LICENSE`, which `--check` checks | Hermes Function-Calling V1, NousResearch; synthetic data led by @interstellarninja with @NousResearch, @teknium, @THEODOROS and others, and Glaive AI's Glaive Function Calling 5k, updated and cleaned: https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1 |
| SWE-bench Verified | `swebench-verified` (render), `swebench-verified-call` and `swebench-verified-content` (parse): 482 cases each, one per row outside pylint, except 8 whose patch carries code under other terms (below) | `hf:datasets/SWE-bench/SWE-bench_Verified@78f471bf655a3137b2e8a75af1501690ec009ec3`, `data/test-00000-of-00001.parquet`, sha256 `030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25` | each row's code: its repository's license at the row's base commit, read from the repository's license file there and copied to `corpus/licenses/` (below); none is established for the issue texts | SWE-bench, Jimenez et al. 2024, Princeton NLP: https://github.com/SWE-bench/SWE-bench; Verified, OpenAI |
| SWE-bench test | `swebench-test`, `swebench-test-call` and `swebench-test-content`: 1697 cases each, one per test row outside pylint that is not a Verified row, except 50 whose patch carries code under other terms. All 500 Verified rows are test rows, equal in every column the import reads, so each goes where its Verified row goes: imported once, as Verified, or left out with it. The 225 rows of the `dev` split, from six other repositories (astroid's under the LGPL among them), are not imported: the benchmark is the test split, and those repositories' licenses are not in the table | `hf:datasets/SWE-bench/SWE-bench@c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2`, `data/test-00000-of-00001.parquet`, sha256 `d4f5a245c75319fa8240c540674958c4d491e82edf274b144d43836bdcbc4567` | as above | as above |
| SWE-bench, copyleft | the rows from pylint-dev/pylint, kept apart: `swebench-verified-copyleft`, `swebench-verified-call-copyleft` and `swebench-verified-content-copyleft` (10 cases each); `swebench-test-copyleft`, `swebench-test-call-copyleft` and `swebench-test-content-copyleft` (47 each) | the two files above | GPL-2.0-or-later, copied as above | as above |
| glaive-function-calling-v2 | `glaive-v2-00` to `glaive-v2-70`, every row: 289494 render cases, one per user turn or tool result an assistant turn answers, and 306240 parse cases, one per assistant turn, with 183629 distinct messages among them, stored as zstd in Git LFS (1243 MB plain, 81.4 MB compressed). 725 rows of the file have no case, and 104913 cases that repeat an earlier one are left out (the import names both) | `hf:datasets/glaiveai/glaive-function-calling-v2@e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac`, file `glaive-function-calling-v2.json`, sha256 `e9b5d671812b5ca2fbd7b625a37d5c99a19576c37252cdc806defe256aea6dad`; the License copy from `github:apache/www-site@01b1be9fbc5cd93b6794f5653a58b9b863807f84` (`content/licenses/LICENSE-2.0.txt`), sha256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | Apache-2.0, checked in the front matter of the dataset card (`README.md`, sha256 `39c78f1f56b86fcd159cadeb8feda8a9333db6ef5ca0ce6830731ac3c666838e`) on every import. The dataset ships no LICENSE or NOTICE file, so the import writes the License as the Apache Software Foundation publishes it (the bytes of https://www.apache.org/licenses/LICENSE-2.0.txt) to `corpus/licenses/glaive-v2-LICENSE`, which `--check` checks | Glaive AI: https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2 (synthetic chats; the card names no generator) |
| SWE-Hero | `swehero-13`, from shard 13 of the training split: 267 render and 267 parse cases, three assistant turns from each of 89 trajectories (the sample below), 40.7 MB | `hf:datasets/nvidia/SWE-Hero-openhands-trajectories@150bc119e52c647216fce285fd801f16b6fd745b`: `data/train-00013-of-00014.parquet`, sha256 `936b195ed11b2b9be2e60cf9cb274dfc2f31d07745cbd6144658da988ce295a9`; `tools.json`, sha256 `d0f46e87e8d6b6d4eef8c01d6add2674ecaaff1260e08df5a4c2b162824c593c`; `README.md` (the card), sha256 `018c3ca426aa3a4f8f3dc4082f8dd1da458a60ca5ce69175c890f60b7db95578` | CC-BY-4.0, checked in the card's YAML front matter on every import. Each row's repository license (its SPDX id, kept in `origin.repository_license`) must be MIT, Apache-2.0, BSD-2-Clause or BSD-3-Clause, or the row is refused and named | SWE-Hero OpenHands trajectories, NVIDIA; generated by Qwen3-Coder-480B-A35B-Instruct: https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories |

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
of its own, and the repository has no NOTICE file. The glaive-v2 sets carry the dataset's chats and function
definitions, and the dataset ships no LICENSE or NOTICE file, only `license: apache-2.0` in its card, so
`bellwether import glaive-v2` writes the License as the Apache Software Foundation publishes it, pinned by commit and
sha256, to `glaive-v2-LICENSE`. The copies sit outside the `render/` and `parse/` directories the recorder reads, and
each carries its importer's prefix.

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

A case of `mgsm-<lang>` or `mgsm-<lang>-content` is one of GSM8K's first 250 test problems in one of eleven languages:
English, which is GSM8K's own text, and ten human translations. Its request is the question as a single user turn, as
GSM8K's is. MGSM's files give each problem's final answer and not GSM8K's worked solution, so `mgsm-<lang>-content`
holds the final answer alone as `content`, as the file writes it: four final answers in each language keep GSM8K's
thousands separators (`2,125`). A row must be `question<TAB>final answer`, split at the tab only (a quote is part of the
question), with a final answer that is an integer in ASCII digits. Any other line but an empty one, a line of spaces
included, would be left out of every set of its language with its reason printed; the pinned files have none. Case names
carry the language and the row's 0-based line index (`mgsm-ja-7`, `mgsm-ja-content-7`).

English has no render set: its 250 questions are GSM8K's first 250 test questions verbatim, so `mgsm-en-<row>` would
send what `gsm8k-test-<row>` sends, byte for byte, and `gsm8k-test` already covers them. `mgsm-en-content` stays, since
no GSM8K parse case holds a final answer alone as its message.

The eleven content sets hold one list of final answers eleven times. The 250 problems have 134 distinct final answers,
and MGSM writes each the same way in every language, so the 2750 cases hold 134 distinct messages between them: their
requests differ by language, their messages do not. The sets stay one per language, and the import prints both counts,
of cases and of distinct messages.

`mgsm-exemplars` holds the 8 worked exemplars of each language from `exemplars.py`, which the import parses as Python
source and never runs. An exemplar is a question after a label (`Question: `, `問題：`) and an answer after a label
(`Step-by-Step Answer: `): the worked solution, then a final sentence that states the exemplar's final answer (`The
answer is 11.`, `答えは11です。`, `Ответ — 11.`, `คำตอบคือ 11`). The import knows each language's wording: its two labels
and the words its final sentence opens with. A case's request is the question without its label; its message holds the
solution as `reasoning_content` and the final sentence, as written, as `content`. The final sentence starts where its
opening words last appear and must state, in ASCII digits, the final answer `EXEMPLAR_NUMBER_ANSWERS` gives the
exemplar; Telugu exemplars 3 and 4 carry words after that sentence, which stay in `content`. An exemplar that does not
read this way in its language's wording would be left out with its reason printed; the pinned file has none. Case names
carry the language and the exemplar's key (`mgsm-exemplars-ja-1`), and `origin.row` its place in the file
(`MGSM_EXEMPLARS['ja']['1']`).

The import changes MGSM's text in one way only: it drops the exemplars' two labels and any whitespace before their final
sentence. It keeps the rest as written, whitespace at the edges of a question included, and writes no text of its own:
every string in a case's request and message is MGSM's. Some of MGSM's Bengali text is not in Unicode's NFC form: 238 of
the 250 Bengali questions, and the solutions of exemplars `bn-2` to `bn-8`, write U+09DF BENGALI LETTER YYA or U+09DC
BENGALI LETTER RRA as one code point, which NFC writes as two. A tokenizer that normalizes to NFC, as Qwen3-8B's does,
encodes the NFC text; `bellwether record --kind parse` then builds those seven exemplars' output ids without the
tokenizer's Unicode normalization, so the ids decode back to MGSM's text as written (#57).

MGSM is CC-BY-4.0, which asks that a copy credit its creators, say what was changed, and give the license's text or
its URI (section 3(a)(1)): the table gives the credit and the URI, and the paragraphs above say what the import changes.
The URI alone would do, but the corpus keeps each dataset's license file next to its sets, so `bellwether import mgsm`
also writes the pinned `mgsm/LICENSE` to `corpus/licenses/mgsm-LICENSE`, and `--check` compares it as it does a set.
The problems MGSM translates are GSM8K's, under MIT, whose notice goes with the corpus as
`corpus/licenses/gsm8k-LICENSE`.

MGSM's files at `3622039c`, by sha256:

- `mgsm/LICENSE`: `c97deeeca4ae375a0334bc7f7af5f707aabfcec959c53c783b9f8771d28fd5b3`
- `mgsm/exemplars.py`: `239dda1557bb2ba76b71e5ef744ddd0b454da0b453e5f8b909498ceff690a919`
- `mgsm/mgsm_bn.tsv`: `6b00bc7cc635547e866989284afa924d789b5affa2eb8de623c385dd943ad977`
- `mgsm/mgsm_de.tsv`: `4dfea30fede44b813e2e496f5f0534049e83c7cc8aed6e00600b30ec62053626`
- `mgsm/mgsm_en.tsv`: `50021d0f28cc957edcb44e7806425b1c7fbd648ddcb9e0a8ec689d10e57d40fa`
- `mgsm/mgsm_es.tsv`: `5bd27ebdf00140cec845c5298dc17715f5cd57d8edda6df1976db40bf3f0750a`
- `mgsm/mgsm_fr.tsv`: `36207c1c03fd7cd3ea491441eb755408199f94e4a647851df531bbaedc99d606`
- `mgsm/mgsm_ja.tsv`: `59a2b50debe77981fd784cb3b2bef1505e3abf2a37116dc9d7a366ab029b4637`
- `mgsm/mgsm_ru.tsv`: `6bd30fd2e80c5bac23f566fb4ae0e6a55a19578401fbf9a01fb21beb3645fef8`
- `mgsm/mgsm_sw.tsv`: `2bac828d77229e65d7c7197b1ad4a2cb5b1fe99b163f2cbdd66501de6a2115c4`
- `mgsm/mgsm_te.tsv`: `dd6b1452c244bb2e4ba254c01ea84137f80c1cefc57d69c23b0ed88b9c0f36b7`
- `mgsm/mgsm_th.tsv`: `f3932dc5ad8e9d0ea82b017adc1e1461dd647af861e7166d6741986602a0cfd6`
- `mgsm/mgsm_zh.tsv`: `b2fa63151022370a0de1f4211c8c284eae74b0f5a3b003b1d5982c0d4a73f661`

A Hermes case is a conversation from the dataset, up to one of its turns, as OpenAI chat messages. The request's `tools`
are the row's `tools` field. The system turn loses the Hermes tool prompt, the instructions and `<tools>` list Hermes
renders from those same tools, since the checkpoint's template renders its own; nothing else is in it in any row with
tools, so those requests have no system message, and the rows without tools (865 in `glaive_func_calling`) keep theirs.
A user turn is a user message. An assistant turn's prose is its `content` (empty in a turn of calls), and each
`<tool_call>` block is one call. A parse case expects the calls without ids; in the history each call has the id
`call_<n>`, numbered across the conversation, which the dataset does not have, so a case whose request holds a call has
`"written": ["tool call ids"]` in its `origin`. Each `<tool_response>` block is a tool message answering the call in the
same position. A call's arguments and a tool message's content are the block's JSON written again with raw Unicode, as
`docs/benchmark-sets.md` has BFCL's arguments written; the dataset writes its JSON with every character past ASCII
escaped, so the two differ where such a character occurs (90 of the 10808 arguments and 167 of the 8620 tool results)
and are the same bytes elsewhere. A render case ends at each user turn or tool result an assistant turn answers, the
prompt a model goes on from; a parse case is each assistant turn, its request every message before it. `origin` names
the row by its index (`row`) and the dataset's id (`row_id`), and the turn the case ends at (`turn`).

A Hermes row with no faithful OpenAI form gives no case: calls that are not JSON, or a tool list in the system prompt
that the `tools` field lacks (between them, every structured-extraction row of both `func_calling` files), a response
without a call, a call without a response, a call to a function the row does not declare, a tool that is not an OpenAI
function, or two tools under one name (79 `glaive_func_calling` rows, 76 of them with two different definitions: a call
to that name could be held to either). `func_calling`'s rows begin as `func_calling_singleturn`'s rows of the same
index, so their opening cases repeat those, and `hermes-func-calling` holds what follows: the render case at each tool
result and the assistant turns after it. A case whose request (and, for a parse case, message) repeats an earlier Hermes
case is left out and named with the case it repeats: 1964 at this pin, 1866 of them `func_calling`'s openings and 98
from `glaive_func_calling` rows that open with the same turns. Every Hermes case is distinct, though not every expected
message: `glaive_func_calling` pairs one chat with several tool lists, and cases that differ only by their tools stay,
so the 18746 parse cases of `hermes-glaive-func-calling` expect 8803 distinct messages. In the other two parse sets each
case expects a message of its own.

A SWE-bench case is bellwether's framing, since SWE-bench has no prompt and no tools. The request is a system turn, "You
are working on the {repo} repository at commit {base_commit}.", then the problem statement verbatim as the user turn,
followed by `"\n\nHints:\n"` and the hints when they are not blank, with one tool on every case: `submit_patch`, whose
one required string parameter `patch` is "A unified diff that resolves the issue, applied at the repository's base
commit." That text is bellwether's, not the dataset's, and every line's `origin.written` lists what of it the line
holds, in the order of its fields: `system prompt`, the system turn; `hints separator`, the `"\n\nHints:\n"` before the
hints, when there are any; `submit_patch tool`, the tool; `submit_patch call`, a `-call` message's call around the gold
patch, with its empty content, its name and its argument key; `code fences`, a `-content` message's fenced `diff` block
around it. Each row gives two parse cases on that request. In `-call`, the message is `content: ""` and one
`submit_patch` call whose arguments are `{"patch": ...}` with the gold patch, written as JSON with raw Unicode. In
`-content`, the message's content is the gold patch in a fenced `diff` block; a patch without a final newline gets one
before the closing fence, and every patch at these pins has one. A row with an empty problem statement, patch, base
commit or repository is skipped and named; at these pins none is. Hints that hold only whitespace stay out of the user
turn, and the import names the 16 rows that have them. The sets take 45,795,527 bytes as plain JSON Lines, under the
50,000,000 past which an import's sets are stored compressed (`corpus_sets.LIMIT`), so they stay plain. No case repeats
another; the 4472 parse cases hold 4470 distinct messages, since pytest-dev__pytest-7236 and pytest-dev__pytest-7283
carry the same patch, on requests that differ in the base commit.

The SWE-bench dataset cards state no license, and the import checks on every run that their front matter still states
none. A row's code is under its repository's license at the row's base commit, read from the repository's own license
file there: `LICENSE`, `LICENSE.md`, `LICENSE.rst`, `COPYING`, or Matplotlib's `LICENSE/LICENSE`.
`scripts/swebench_licenses.py` finds that file and any `NOTICE` file at every base commit, in the repositories' history,
and pins each by repository, commit and sha256 in `src/bellwether/importers/swebench_licenses.json`. The import fetches
each pinned file through `importers/github.py` and copies it to
`corpus/licenses/swebench-<owner>-<repo>-<commit>-<file>`, named for the earliest base commit that holds it, and
`--check` compares the copies as it does a set. Where a license file changed across the base commits, each version is
copied and each row names its own: requests was under ISC until 2013 and under Apache-2.0 after, and most other changes
are copyright years. A license file that only points to the Apache License (requests, 2013 to 2019) goes with the
repository's full text of it, which Apache-2.0 4(a) asks for.

The license is read from the file's own words, and a file that reads as none of the licenses the import knows stops it:
BSD-3-Clause for astropy, django, scikit-learn, seaborn, flask and sympy; BSD-2-Clause for sphinx; MIT for pytest; ISC
and then Apache-2.0 for requests; Apache-2.0 for xarray; Matplotlib's own license; GPL-2.0-or-later for pylint, whose
license file is the GPL's text and whose packaging metadata declares the version. A parse line's `origin` adds the
repository, the license of its message (the gold patch), the copyright holder that the license or NOTICE file names
(xarray's README and pylint's file headers, whose license files name none), and the copies that go with it, as
`notices`. A render line holds only the issue text and its hints, comments by GitHub users under no license that is
established, so its `origin` names the repository and the license `NOASSERTION`. Sets built from copyleft code end in
`-copyleft` and hold nothing else.

58 rows are left out because their gold patch carries code that its own file, or the patch itself, puts under terms
other than the repository's license, which the copied license files do not cover: PyFITS's license in astropy's
`io/fits`; NumPy's, pandas' and seaborn's code in astropy and xarray; Python's code under the PSF license in Django,
scikit-learn and Sphinx; urllib3 and pip in requests; packaging and Matplotlib in seaborn; files under MIT and BSD in
Matplotlib; SciPy's code and one Simplified BSD file in scikit-learn; and docutils' public-domain code in Sphinx. They
were found by reading each file a patch touches, at the base commit, and every line of the patch for license and
copyright statements, and reviewing by hand those that are not the repository's own header
(`scripts/swebench_licenses.py --terms` prints them); the import names each row with its reason. A file whose header
only names another copyright holder under the repository's own license, such as the Smithsonian Astrophysical
Observatory in astropy's `io/ascii` or INRIA in scikit-learn, does not count.

A glaive-v2 case comes from one chat, whose functions sit in its system prompt. The functions become `tools`, and the
system message keeps what is neither the lead-in sentence nor a function, which leaves none in rows with functions. A
call turn becomes an assistant message with empty content and one call, whose single-quoted arguments are written as a
BFCL call's are; a function response becomes a `tool` message answering that call; a prose turn keeps its text without
the closing `<|endoftext|>`. A turn with no text is kept as written, a message whose content is `""`: the file has four,
assistant turns in rows 11267 and 33683 and user turns in rows 84134 and 97124. A parse case expects the call without an
id; in the history each call has the id `call_<n>`, the chat's calls numbered from 0, which the dataset does not have,
so a case whose request holds a call has `"written": ["tool call ids"]` in its `origin`. The ids leave the row out, so a
chat that recurs in another row gives the same requests and messages there. Each assistant turn is a parse case, whose
request is every message before it, and each user turn or tool result an assistant turn answers is a render case, the
request up to and including that turn, the prompt a model goes on from; both are named `glaive-v2-<row>-<turn>`, the
turn counted in the chat from 0. Row 69130 holds a U+0085 next-line character, which the corpus keeps raw, as GSM8K's
U+2028.

A glaive-v2 row with no faithful OpenAI form gives no case, and the import names it with its reason: 725 rows of the
file, 351 of them because they declare one function name twice, so that a call to that name could be held to either
definition (338 of them with two different definitions). A case whose request (and, for a parse case, message) repeats
an earlier glaive-v2 case is left out and named with the case it repeats; such cases come from chats that open with the
same turns. Every glaive-v2 case is distinct, though not every expected message: chats that differ in their history can
end in the same turn, and "You're welcome! If you have any other questions, feel free to ask." alone ends 7837 parse
cases. The sets hold every row, cut into sets of at most 5000 cases of either kind, a set's render and parse files
holding the same rows: 289494 render and 306240 parse cases with 183629 distinct messages, after 104913 repeats are left
out. That is 1243 MB of plain JSON Lines, past the 50 MB an import's sets stay plain, so `corpus_sets.write` stores
every set compressed in Git LFS, 81.4 MB in all.

A SWE-Hero case is one assistant turn of an OpenHands trajectory: long agent conversations on real repositories, with
prose and tool calls in the assistant turns, code and shell commands in the arguments, and tool results in the history.

- **The request** is an OpenAI chat request: the trajectory's messages before the turn, unchanged, and `tools.json`
  (checked to be OpenAI's tool shape) as `tools`. The data's tool messages carry no id, so the k-th tool message after
  an assistant turn takes the id of that turn's k-th call. A row whose results and calls do not pair is refused and
  named; only the last turn may go unanswered, since every trajectory ends with a `finish` call. Messages take
  OpenAI's key order (`role`, `content`, `tool_calls`; `role`, `tool_call_id`, `content`), not the shard's
  alphabetical struct order, and `tool_calls` only where a turn makes calls (parquet fills it with null elsewhere).
- **The parse case's message** is the turn's `content` and its calls, each argument string exactly as the data holds
  it (a row with one that is not a JSON object string is refused and named), with no call id, since a parser makes
  its own. The render case is the same request.
- **The sample.** A case for every turn, each with its whole history, would grow with the square of a trajectory's
  length: the shard's 1769 trajectories hold 116,605 assistant turns, and the request for a trajectory's last turn
  is 258 KB at the median and up to 629 KB. The import takes one row in 20 (rows 0, 20, ..., 1760: 89 trajectories
  from 75 repositories), and from each the first, middle and last of the assistant turns whose request fits in
  128,000 bytes as its corpus line writes it. A turn past the cap gives way to an earlier one; no message is ever
  cut. Requests run from 20.7 KB (the system prompt, the issue and the tools) to 128 KB, with 2 to 150 messages of
  history (median 32). Case names are `swehero-13-<row>-<turn>`, the turn being the assistant message's index in
  the trajectory, so its request holds that many messages.
- **Refusals.** The import checks every row of the shard, sampled or not. It refuses a row whose repository license
  is not MIT, Apache-2.0, BSD-2-Clause or BSD-3-Clause, whose tool results do not pair with its calls, with a call
  whose arguments are not a JSON object string (JSON as RFC 8259 has it, so `NaN` and `Infinity` are refused), or
  with calls on a message other than an assistant turn. A sampled row with no assistant turn whose request fits
  under the cap gives no case, and is named too. Every refused row is named with its reason and detail. At this pin
  none is refused: all 1769 rows pass every check, and every sampled row has turns under the cap.

Unlike the imports above, `bellwether import swehero` writes no license file yet: which of the repositories' licenses
and NOTICE files to ship with its sets is still open.

Set names starting with `bfcl-`, `glaive-v2-`, `gsm8k-`, `hermes-`, `mgsm-`, `shapes-`, `swebench-` or `swehero-` belong
to that importer: `bellwether import bfcl` deletes any `bfcl-*` set file, `.jsonl` or `.jsonl.zst`, it did not write,
`bellwether import glaive-v2` any `glaive-v2-*` one, `bellwether import gsm8k` any `gsm8k-*` one,
`bellwether import hermes` any `hermes-*` one, `bellwether import mgsm` any `mgsm-*` one, `bellwether import shapes` any
`shapes-*` one, `bellwether import swebench` any `swebench-*` one, and `bellwether import swehero` any `swehero-*` one.
Name hand-written sets otherwise.


`--check` reads the pinned files from `~/.cache/bellwether/datasets` (`--cache`) and downloads them on a miss. It then
needs PyPI to still serve that exact wheel: a yanked release still does when pinned by version; a release deleted from
PyPI does not, and the check fails until the importer pins another. GSM8K's and MGSM's files, the LICENSE the BFCL
import copies, the License the Hermes and glaive-v2 imports copy and the license files the SWE-bench import copies are
read from `raw.githubusercontent.com` at the pinned commit, which serves them as long as the repository keeps that
commit. `bellwether import shapes --check` reads the same pinned files as the BFCL and GSM8K imports. The Hermes and
SWE-bench files are read from the Hugging Face Hub at the pinned commit and kept in the same cache, under
`huggingface/`, so a cached copy also serves `HF_HUB_OFFLINE=1`. SWE-bench's parquet files are checked against their
sha256 on every read, and the Hub serves them as long as each dataset keeps its pinned commit, which a squashed history
or a deleted dataset would end. SWE-Hero's three files are read from the Hugging Face Hub at the pinned commit, into
`huggingface/` under the same cache (141 MB for the shard), and `HF_HUB_OFFLINE=1` turns a miss into an error.

The glaive-v2 file and card come through `hf.fetch`, which keeps them in the Hugging Face cache layout under the same
`~/.cache/bellwether/datasets` (`huggingface/`) and downloads them on a miss; with `HF_HUB_OFFLINE=1` it reads the
cache only. CI keeps that directory whole, the store Xet keeps a file's bytes in included, in its one cache of pinned
files.

`bellwether record` writes a fixture set recorded from an imported set as zstd-compressed JSON Lines in Git LFS
(`fixtures/README.md`); `bellwether unpack` gives consumers the plain files.
