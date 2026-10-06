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

| Dataset | Sets | Source | License | Attribution |
|---|---|---|---|---|
| BFCL | `bfcl-<category>` for the 13 single-turn categories of smg's weekly run: 3641 render cases, one per row; 2420 parse cases, one per row with a ground truth, except 76 Java and JavaScript rows whose values are not strings (#26) and 5 rows whose ground truth BFCL's own checker cannot accept (the import names them) | `pypi:bfcl-eval==2026.3.23`, sha256 `3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a` | Apache-2.0, checked in the wheel's METADATA on every import | Berkeley Function Calling Leaderboard, Gorilla project, UC Berkeley: https://github.com/ShishirPatil/gorilla |

A BFCL request is what the weekly run sends in function-calling mode through `OpenAICompletionsHandler`. A parse
case's message is one call per ground-truth entry, each parameter taking its first acceptable value that is not
BFCL's "may be omitted" marker, and each call held to BFCL's own parameter rules: a parameter its function does not
declare is left out when the answer allows it. `docs/benchmark-sets.md` has the rules; `scripts/bfcl_equivalence.py`
checks every request and call against BFCL's own code, in a throwaway environment.

Set names starting with `bfcl-` belong to the importer: `bellwether import bfcl` deletes any `bfcl-*.jsonl` it did
not write. Name hand-written sets otherwise.

`--check` reads the pinned wheel from `~/.cache/bellwether/datasets` and downloads it on a miss. It then needs PyPI to
still serve that exact file: a yanked release still does when pinned by version; a release deleted from PyPI does not,
and the check fails until the importer pins another.

Until the benchmark sets' storage form lands (`docs/benchmark-sets.md`, delivery step 5), record the hand-written
sets alone: `bellwether record --model <id> --kind <kind> --oracle reference --set common`.
