# Agentic session corpus v2.1: build report

v2.1 = v2 (gen 900 + BFCL 160, unchanged byte for byte) + the `cli` tenant restored: 180 recorded coding-CLI sessions,
converted with `tools/build_corpus_v21.py` (v2's builder + v1's `convert_m27`, unchanged) and published with
`--synthetic-system-prefix` (section c). Token counts: the DeepSeek V4.1 Flash tokenizer at revision 2cba9e42
(`tokenizer.json` sha256 c90dfa01...), template-free as in v2; chars/4 beside it in MANIFEST.md.

## (a) Composition

| element | v2.1 | v2 | v1 |
|---|---|---|---|
| sessions | 1,240 | 1,060 | 641 |
| gen (tenant `coder`) | 900 (72.6 % by count) | 900 (84.9 %) | 300 (46.8 %) |
| bfcl (tenant `bfcl`) | 160 (12.9 %) | 160 (15.1 %) | 160 (25.0 %) |
| cli (tenant `cli`, recorded coding-CLI sessions) | 180 (14.5 %) | 0 | 181 (28.2 %) |
| turns per cli session | 2-14, p50 5 (886 turns; 1,250 tool calls = 1.41 per turn) | - | - |
| tool schemas per cli session | 7 | - | - |
| json_schema sessions | 217 = 17.5 % of all (0.2 of the 1,080 non-bfcl sessions) | 188 = 17.7 % | 89 = 13.9 % |
| thinking sessions | 641 = 51.7 % | 549 = 51.8 % | 326 = 50.9 % |
| image slots (gen and cli, first user turn) | 162 = 15.0 % of each tenant: 135 gen (v2's picks and slot ids, drawn from v2's stream) + 27 cli (drawn apart, slot ids after the gen ones) | 135 (gen only) | none |
| corpus bytes (plain) | 147,546,426 (6,830,979 as zstd -19) | 141,837,318 | 55.6 MB |

The cli share by count (14.5 %) is not the traffic share: the driver's tenant mix draws each session class at the
design's weight at sampling time; this count only says how many distinct cli sessions a play can draw from.

## (b) Token table per source, V4.1 Flash

| source | first prompt p50 (min-max) | last prompt p50 (min-max) | sum over turns p50 | recorded output per turn p50 |
|---|---|---|---|---|
| gen | 9,579 (6,010-14,994) | 32,166 (14,995-69,983) | 143,567 | 97 |
| bfcl | 3,834 (2,251-5,006) | 4,179 (2,369-5,293) | 14,386 | 22 |
| cli (v2.1) | 4,704 (4,676-4,782) | 5,839 (4,738-10,551) | 26,497 | 133 |
| cli (v1, 181 sessions) | 4,600 | 5,800 | - | - |
| all (v2.1) | 8,015 (2,251-14,994) | 24,565 (2,369-69,983) | 112,272 | 93 |

The gen and bfcl rows equal v2's: those sessions are the same bytes. A cli session's prefix is one shared system text
(1,271 tokens) plus one shared tool list (2,948 tokens as JSON), so every cli session of the tenant shares 4,219 prefix
tokens, and its first prompt is that prefix plus the first user turn (p50 498 tokens).

## (c) The published form of the cli tenant: synthetic_system_prefix

The recordings carry the CLI's own system prompt, tool descriptions and the context notes it injects into the first
user turn (and, rarely, into a tool result). That text is not this corpus's to publish, so the builder replaces it,
after `convert_m27` has canonicalized one system text and one tool list per tenant, by synthetic text of exactly the
same token count under the V4.1 tokenizer:

| replaced | pieces | original tokens | synthetic tokens | how |
|---|---|---|---|---|
| the system text | 1 (shared by the 180 sessions) | 1,271 | 1,271 | a synthetic harness text (the builder's own paragraphs), fitted to the count |
| the tool descriptions | 40 strings in 7 tools (one shared tool list) | 2,948 for the whole tools JSON | 2,948 | each description fitted in the context of the whole JSON, in document order; names, parameter names, types, enums, defaults and required lists kept |
| context notes in user turns | 360 blocks (the policy note x180, the per-session version-control note x180) | 40,886 | equal per user text | each block fitted in the context of its user text; tags apart from the body |
| the environment message in the first user turn | 181 blocks | 35,534 | equal per user text | as above |
| warnings inside tool results | 5 blocks | 110 | equal per tool result | as above |

The builder counts the system text, the tools JSON, every user text and every tool result as separate pieces, so every
token table is unchanged by the substitution. Checked by building both forms (the raw form stays with the recordings on
the recording host) and diffing manifest.json's per-session token vectors: 1,240 sessions, 0 differ under the V4.1
tokenizer; the chars/4 estimates differ for the 180 cli sessions, as the synthetic text has other character counts; the
1,060 gen and bfcl lines are byte-identical. manifest.json records the substitution under
`source_stats.cli.synthetic_system_prefix`: per variant the kind, the session count, both token counts and both
sha256 values (the original text's sha256 lets a holder of the raw recordings match them), plus what was kept.

Kept as recorded: the user task text (the project's own ten tasks), the tool calls and the tool results (the fixture
repository's own files and the tool outputs, with their framing sentences and the sandbox's scratch path
`/tmp/rec-cli/scratch/<session>/repo`), and the model's outputs (one reasoning text names the CLI; it is the model's own
sentence).

## (d) The recordings

180 sessions = 10 tasks x 9 runs x 2 gateway parsers (the parser under test and the control), on three seeds of the
fixture repository (runs 1-3 the v1 fixture byte for byte, 4-6 seed 2, 7-9 seed 3), one session at a time on one
DeepSeek V4.1 Flash bed behind the gateway at commit 7c8e8b7 (vLLM 0.31.0, the Rust gRPC servicer, thinking at the
CLI's default), recorded on 2026-10-22 through a local proxy that writes one JSON line per HTTP exchange. 180 of 180
sessions passed their task check on both parsers, 0 void, 0 API errors; 886 turns, prompt tokens per first request p50
4,852 as the engine counted them (usage input + cache read + cache creation) against 4,704 template-free here. The
recordings' per-turn `recorded_think_s` and `recorded_stop` fields come through the conversion. The raw recordings stay
on the recording host; this directory holds the published form only.

## (e) Sizes, commands, checks

Files: `corpus.jsonl.zst` (zstd -19 of `corpus.jsonl`; both sha256 in SHA256SUMS), `manifest.json`, `MANIFEST.md`,
`REPORT.md` (this file), `SHA256SUMS`. Verification: `sha256sum -c --ignore-missing SHA256SUMS` checks the published
files; the `corpus.jsonl` line is the checksum of the decompressed stream, `zstd -dc corpus.jsonl.zst | sha256sum`.
Tools: `../tools/build_corpus_v21.py` (pinned by sha256 in manifest.json `sources.builder`), `../tools/gen_corpus.py`
(unchanged from v2).

    <PY> -I tools/build_corpus_v21.py --out v2.1 --bfcl sources/bfcl-26ad99ed --tokenizers deepseek-v4.1-flash=<TOK> \
         --gen 900 --bfcl-per-category 40 --seed 20261022 --jobs 16 --m27 <recordings dir> --synthetic-system-prefix
    zstd -19 v2.1/corpus.jsonl

The recordings and every file of this directory were scanned for host, path, account and environment names before the
push (the builder's own scrub rules: 0 hits).

## (f) Owed

- The cli sessions' 27 image slots name an image class only (`"file": null`), as the gen slots do; the driver attaches
  an image of the class from the benchmark's image set (`../images/benchmark-2/`).
- Only the V4.1 Flash family is counted; the other families' columns return when their tokenizer files are given.
- The v2 calibration tail (25 of 900 gen last prompts within 5-8.3 % of the plan) is unchanged.
