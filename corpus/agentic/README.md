# Agentic session corpus

The multi-turn, tool-calling sessions a traffic driver replays against a gateway to measure it: capacity, latency and
CPU per request under agentic traffic (long shared system prompts with tool schemas, histories that grow turn by turn
with tool calls and tool results, a structured final answer). It is a benchmark input, not a `record` input:
`bellwether record`, `count`, the import checks and `images check` read `render/`, `parse/` and `image/` only and
leave this directory alone. The smg repository holds the driver and its tests; the data lives here, under a pinned tree.

```
corpus/agentic/
  README.md                     this file
  tools/gen_corpus.py           the generated share ("gen", tenant profile "coder"): one coding task per session on a
                                synthetic Python repository, sized against the real tokenizer
  tools/build_corpus_v2.py      the builder: gen + BFCL -> corpus.jsonl, manifest.json, MANIFEST.md
  sources/bfcl-26ad99ed/        the BFCL v4 multi-turn files at one gorilla commit, their LICENSE (Apache-2.0) and
                                SOURCE.txt (the sha256 of every file)
  images/benchmark-2/           the benchmark's image set: its manifest and the selection tools (its own pull request)
  v2/                           the built corpus v2 (its own pull request): corpus.jsonl.zst in Git LFS, manifest.json,
                                MANIFEST.md, REPORT.md, SHA256SUMS
```

The tools are kept byte for byte: `manifest.json` pins the generator and the builder by sha256
(`sources.generator`, `sources.builder`), so they are excluded from `ruff` rather than reformatted.

## The session record

One JSON object per line of `corpus.jsonl`:

```
{"id", "source": "gen|bfcl", "tenant_profile": "coder|bfcl", "task", "origin", "system",
 "tools": [chat-completions function tools],
 "turns": [{"index", "user" (text or null),
            "expected_tool_calls": [{"id", "name", "arguments": JSON string}],
            "tool_results": [{"tool_call_id", "name", "content"}],
            "recorded_assistant": {"content", "reasoning", "tool_calls"} | null,
            "max_tokens", "response_format" | null, "planned_prompt_tokens" (gen)}],
 "max_tokens", "thinking": bool, "json_schema": bool, "tool_results_source": "generated|synthetic", "licence",
 "salt_points", "image": {"slot", "class", "mpixels_range", "file", "note"} (15 % of the gen sessions)}
```

A driver plays a session turn by turn: the system prompt and the tools on every request, the history as the earlier
turns' user text, the recorded assistant turn (tool calls) and the tool results, and `{{SALT}}` in the first user
turn replaced per play so that no two plays share a prefix beyond the system prompt. 20 % of the non-BFCL sessions end
under `response_format` json_schema; 50 % carry `thinking`; an `image` slot asks the driver to attach one image of the
named class on the first user turn when it has the image set.

## Composition of v2 (seed 20261022)

| element | v2 | v1 |
|---|---|---|
| sessions | 1,060 | 641 |
| gen (tenant `coder`) | 900 (84.9 % by count) | 300 (46.8 %) |
| bfcl (tenant `bfcl`) | 160 = 40 per multi-turn category (15.1 %) | 160 (25.0 %) |
| cli (recorded coding-CLI sessions) | 0: absent in v2 (re-recording pending) | 181 (28.2 %) |
| turns per gen session | 4: 182, 5: 211, 6: 241, 7: 182, 9: 84 (= 3/4/5/6/8 tool turns + 1 final) | same choice and weights |
| turns per bfcl session | 1-8, p50 4 | same source |
| tool schemas per session | gen 7 (one harness prompt per tenant: 3,636 shared prefix tokens); bfcl 17-39, p50 29 | gen 7 |
| json_schema sessions | 188 = 17.7 % of all (0.2 of the non-bfcl sessions) | 89 = 13.9 % |
| thinking sessions | 549 = 51.8 % | 326 = 50.9 % |
| image slots (gen, first user turn) | 135 = 15.0 % of gen (68 screenshot, 67 document) | none |
| themes | 192 = 16 task types x 12 repository domains | one theme set |
| corpus bytes | 141,837,318 (`corpus.jsonl`, plain) | 55.6 MB |

Token counts (DeepSeek V4.1 Flash tokenizer, template-free: system + tools JSON + every message text):

| source | first prompt p50 (min-max) | last prompt p50 (min-max) | sum over turns p50 | recorded output per turn p50 |
|---|---|---|---|---|
| gen | 9,579 (6,010-14,994) | 32,166 (14,995-69,983) | 143,567 | 97 |
| bfcl | 3,834 (2,251-5,006) | 4,179 (2,369-5,293) | 14,386 | 22 |
| all | 8,786 (2,251-14,994) | 28,229 (2,369-69,983) | 128,421 | 92 |

Each gen session draws its first prompt (6,000-15,000 tokens) and its last prompt (15,000-70,000) log-uniform and the
generator sizes the synthetic texts against the real tokenizer: measured after finalize, the first prompts land
within 0.65 % of the plan (900 of 900 within 5 %), the last prompts within 8.3 % (875 of 900 within 5 %).
`v2/REPORT.md` has the calibration, the per-turn growth and the corpus-wrap arithmetic.

## Rebuilding

Run with `python -I`: the BFCL files and the tokenizer files are downloaded data, and every path is an argument.

```
python3 -I tools/gen_corpus.py --tokenizer <TOK> --selftest
python3 -I tools/build_corpus_v2.py --out <OUT> --bfcl sources/bfcl-26ad99ed --tokenizers deepseek-v4.1-flash=<TOK> \
    --gen 900 --bfcl-per-category 40 --seed 20261022 --jobs 16
```

`<TOK>` is a directory holding `tokenizer.json` of `deepseek-ai/DeepSeek-V4.1-Flash` at revision
`2cba9e42aa026125f3ed06c6d98c1db82f7ca027` (sha256 `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`;
MIT; used to count tokens, not redistributed). The builder needs the `tokenizers` package. Every text of a gen session
depends only on the seed, the session index and the pinned tokenizer; `manifest.json` records the build command, the
seed, the source pins and the sha256 of the tools that built it.

## Sources, pins and licences

- gen: synthetic, generated by `tools/gen_corpus.py`; no third-party text (this repository's licence).
- bfcl: Berkeley Function Calling Leaderboard v4 multi-turn data, `github.com/ShishirPatil/gorilla` at commit
  `26ad99ed327fc47fc7c8b91b43378c5709125a06` (Apache-2.0, `sources/bfcl-26ad99ed/LICENSE`): the questions and the
  function documentation verbatim, the ground truth as the expected calls, the tool results synthesized from the
  documented response schemas (the executable backing classes are not run). `SOURCE.txt` lists the sha256 of every
  file; the builder verifies the files against it and records the result in the manifest.
- Scrub rules applied to every text: home directories and IPv6 addresses (generic patterns; 0 hits in v2).
