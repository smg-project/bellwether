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
dataset's license.

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
| BFCL | `bfcl-<category>` for the 13 single-turn categories of smg's weekly run: 3635 render cases, one per row except 6 `live_irrelevance` rows that repeat an earlier case; 2420 parse cases, one per row with a ground truth, except 76 Java and JavaScript rows whose values are not strings (#26) and 5 rows where no call the rule builds passes BFCL's own checker (the import names them), with 2233 distinct messages among them | `pypi:bfcl-eval==2026.3.23`, sha256 `3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a` | Apache-2.0, checked in the wheel's METADATA on every import; the wheel has no LICENSE file, so the import copies `LICENSE` from `github:ShishirPatil/gorilla@6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`, the commit the wheel was built from (sha256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`), to `corpus/licenses/bfcl-LICENSE` | Berkeley Function Calling Leaderboard, Gorilla project, UC Berkeley: https://github.com/ShishirPatil/gorilla |
| GSM8K | `gsm8k-train` and `gsm8k-test`: 7473 and 1319 render cases, one per row; `gsm8k-<split>-reasoning-content` and `gsm8k-<split>-content`: the same rows as parse cases, one set per message shape (no row of the pinned files is left out: every row is usable, and no case repeats an earlier one; all 17584 messages are distinct) | `github:openai/grade-school-math@3101c7d5072418e28b9008a6636bde82a006892c`; sha256 `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` (`grade_school_math/data/train.jsonl`), `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` (`grade_school_math/data/test.jsonl`), `86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b` (`LICENSE`) | MIT, checked against the LICENSE file's sha256 on every import and copied to `corpus/licenses/gsm8k-LICENSE` | Training Verifiers to Solve Math Word Problems, Cobbe et al. 2021, OpenAI: https://github.com/openai/grade-school-math |

A BFCL request is what the weekly run sends in function-calling mode through `OpenAICompletionsHandler`. A parse
case's message is one call per ground-truth entry, each parameter taking its first acceptable value that is not
BFCL's "may be omitted" marker, and each call held to BFCL's own parameter rules: a parameter its function does not
declare is left out when the answer allows it. `docs/benchmark-sets.md` has the rules; `scripts/bfcl_equivalence.py`
checks every request and call against BFCL's own code, in a throwaway environment.

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

Set names starting with `bfcl-` or `gsm8k-` belong to that importer: `bellwether import bfcl` deletes any
`bfcl-*.jsonl` it did not write, and `bellwether import gsm8k` any `gsm8k-*.jsonl`. Name hand-written sets otherwise.

`--check` reads the pinned files from `~/.cache/bellwether/datasets` and downloads them on a miss. It then needs PyPI to
still serve that exact wheel: a yanked release still does when pinned by version; a release deleted from PyPI does not,
and the check fails until the importer pins another. GSM8K's files, and the LICENSE the BFCL import copies, are read
from `raw.githubusercontent.com` at the pinned commit, which serves them as long as the repository keeps that commit.

`bellwether record` writes a fixture set recorded from an imported set as zstd-compressed JSON Lines in Git LFS
(`fixtures/README.md`); `bellwether unpack` gives consumers the plain files.
