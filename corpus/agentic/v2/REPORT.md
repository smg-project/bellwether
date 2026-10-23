# Agentic session corpus v2: build report (brief R1 of plans/benchmark-2.md, lane bench-kit-1)

Built 2026-10-22 on the CPU host with `tools/build_corpus_v2.py` (seed 20261022) from `tools/gen_corpus.py` (the replacement
generator; v1's at_corpus.py is lost) and the pinned BFCL v4 multi-turn files. Token counts: DeepSeek V4.1 Flash tokenizer at the
bellwether pin 2cba9e42 (tokenizer.json sha256 c90dfa01...), template-free as in v1; chars/4 beside it. The other five v1
families were absent (printed as such) and are not counted.

## (a) Composition, v2 vs v1

| element | v2 | v1 |
|---|---|---|
| sessions | 1,060 | 641 |
| gen (tenant `coder`) | 900 (84.9 % by count) | 300 (46.8 %) |
| bfcl (tenant `bfcl`) | 160 = 40 per category (15.1 %) | 160 (25.0 %) |
| cli (tenant `cli`, recorded coding-CLI sessions) | 0: ABSENT (recordings lost at the host move; re-recording brief pending) | 181 (28.2 %) |
| turns per gen session (tool-call turns + 1 final) | 4: 182, 5: 211, 6: 241, 7: 182, 9: 84 sessions (= 3/4/5/6/8 tool turns; realized weights .202/.234/.268/.202/.093 vs .20/.25/.25/.20/.10) | same choice and weights |
| turns per bfcl session | 1-8, p50 4 (641 turns over 160 sessions) | same source |
| tool schemas per session | gen: 7 (one harness prompt for the whole tenant: 2,521 tokens + 1,115 tokens of tool JSON = 3,636 shared prefix tokens, inside the 2-6k band); bfcl: 17-39, p50 29 | gen 7; bfcl from the docs |
| json_schema sessions (f3 schema on the final turn) | 188 = 17.7 % of all = 20.9 % of the non-bfcl sessions (share parameter 0.2) | 89 = 13.9 % of all (= 0.2 of 461 non-bfcl) |
| thinking sessions (share parameter 0.5) | 549 = 51.8 % | 326 = 50.9 % |
| image slots (gen only, first user turn) | 135 = 15.0 % of gen (68 screenshot, 67 document; `"image": {"slot": "img-<k>", "class", "mpixels_range": [0.2, 2.0], "file": null, "note"}`; slots listed in manifest.json `image_slots`) | none |
| themes (task type x repository domain) | 192 = 16 task types x 12 domains; 900 sessions spread 33-74 per task type and 63-89 per domain | one generator theme set |
| salt points | 1 per session (`{{SALT}}` nonce line, first user turn) | same |
| corpus bytes | 141.8 MB (`corpus.jsonl`) | 55.6 MB |

The 13.9 % of v1 is 0.2 of its non-bfcl sessions; v2 applies the same 0.2 to its non-bfcl sessions, which are a larger share of
the corpus, hence 17.7 % of all. A reader wanting v1's 13.9 % of all would pass `--json-schema-share 0.164`.

## (b) Token table per source, V4.1 Flash (chars/4 in MANIFEST.md)

| source | first prompt p50 (min-max) | last prompt p50 (min-max) | sum over turns p50 | recorded output per turn p50 |
|---|---|---|---|---|
| gen v2 | 9,579 (6,010-14,994) | 32,166 (14,995-69,983) | 143,567 | 97 (tool turns p10/p50/p90/max 31/89/250/509; final answers min/p50/p90/max 345/1,033/1,428/1,895) |
| gen v1 (recorded in the design) | 11,400 (6,700-15,900) | 36,400 (15,200-70,300) | - | - |
| bfcl v2 | 3,834 (2,251-5,006) | 4,179 (2,369-5,293) | 14,386 | 22 |
| bfcl v1 | 3,800 | - | - | - |
| cli v1 (absent in v2) | 4,600 | 5,800 | - | - |
| all v2 | 8,786 (2,251-14,994) | 28,229 (2,369-69,983) | 128,421 | 92 |

v2's gen medians sit at the geometric means of the design's log-uniform ranges (sqrt(6k x 15k) = 9.5k, sqrt(15k x 70k) = 32.4k):
v2 realizes the specified distributions exactly; v1's realized medians ran 15-20 % above them. Growth is front-loaded as
designed: the median prompt increment by turn index is 9.6k, 6.8k, 2.7k, 1.4k, 0.9k, 0.6k, 0.1k, 0.4k tokens (file reads early,
edits/tests/diff late). Tool-turn outputs are a little short of the 50-400 band at the low end (p10 = 31 tokens: a one-sentence
reasoning, a short sentence and a small argument object) and the final reports stop at 1,895 tokens instead of the 3,000 top of
the design band (the report material runs out for short diffs); both are stated here rather than padded.

## (c) Generator calibration (planned vs achieved, V4.1 Flash)

Each session draws its first prompt (6,000-15,000) and last prompt (15,000-70,000) log-uniform, then the generator sizes the
synthetic texts against the real tokenizer (encode of the assembled texts, prefix packing of code units, per-line fitting of the
first-turn excerpts, residual carry-over between reads, up to three correction passes). Measured after finalize (the nonce line
and, on 20 % of the sessions, the response_format JSON on the last turn):

| prompt | abs deviation p50 | p90 | max | within 5 % |
|---|---|---|---|---|
| first | 0.09 % | 0.26 % | 0.65 % | 900 / 900 |
| last | 0.05 % | 1.10 % | 8.26 % | 875 / 900 |

Before finalize (the generator's own print): first 0.10 % / 0.23 % / 0.44 %, 900/900; last 0.03 % / 0.37 % / 7.04 %, 876/900;
correction passes used: 3 for 799 sessions, 4 for 101. The 25 last-prompt misses are sessions whose drawn growth is close to the
floor of the per-read minimum file sizes (8 tool turns with a small first-to-last span, where the smallest natural files still
overshoot) or whose late reads exhausted their unit pools; all 25 are within 8.3 %. Planned per-turn prompt sizes are stored as
`planned_prompt_tokens` on every gen turn; the generator's per-session facts (planned/achieved first, final, output tokens,
task, domain, reads, change calls, passes) are in manifest.json under `source_stats.gen.calibration_per_session`.

Every Python file an agent reads in the corpus parses with `ast.parse` (checked on the 96-session dry run and in the self-test);
the edit calls' `old_text` occurs exactly once in the file state they are applied to; diffs are `difflib.unified_diff` of the
real before/after texts; pytest output line numbers come from the packed files.

## (d) Corpus-wrap arithmetic (design 1.1) for 1,060 sessions

Sessions ramp: 15 steps of 150 s at 0.2 x 1.3^k sessions/s per 3-engine arm; plays per step = rate x 150; 1 engine = x 1/3,
8 engines = x 8/3. Fixed point: 85 % of 2.121 sessions/s per 3 engines over 1,200 s.

| k | rate, 3 engines (s/s) | plays per step, 3 eng | corpora | plays, 1 eng | corpora | plays, 8 eng | corpora |
|---|---|---|---|---|---|---|---|
| 0 | 0.200 | 30 | 0.03 | 10 | 0.01 | 80 | 0.08 |
| 1 | 0.260 | 39 | 0.04 | 13 | 0.01 | 104 | 0.10 |
| 2 | 0.338 | 51 | 0.05 | 17 | 0.02 | 135 | 0.13 |
| 3 | 0.439 | 66 | 0.06 | 22 | 0.02 | 176 | 0.17 |
| 4 | 0.571 | 86 | 0.08 | 29 | 0.03 | 229 | 0.22 |
| 5 | 0.743 | 111 | 0.11 | 37 | 0.04 | 297 | 0.28 |
| 6 | 0.965 | 145 | 0.14 | 48 | 0.05 | 386 | 0.36 |
| 7 | 1.255 | 188 | 0.18 | 63 | 0.06 | 502 | 0.47 |
| 8 | 1.631 | 245 | 0.23 | 82 | 0.08 | 653 | 0.62 |
| 9 | 2.121 | 318 | 0.30 | 106 | 0.10 | 848 | 0.80 |
| 10 | 2.757 | 414 | 0.39 | 138 | 0.13 | 1,103 | 1.04 |
| 11 | 3.584 | 538 | 0.51 | 179 | 0.17 | 1,434 | 1.35 |
| 12 | 4.660 | 699 | 0.66 | 233 | 0.22 | 1,864 | 1.76 |
| 13 | 6.058 | 909 | 0.86 | 303 | 0.29 | 2,423 | 2.29 |
| 14 | 7.875 | 1,181 | 1.11 | 394 | 0.37 | 3,150 | 2.97 |

- Per step: the 3-engine ramp wraps inside one step only at k = 14 (1,181 plays); the 1-engine ramp never wraps inside a step
  (max 394 plays); the 8-engine ramp wraps inside the step from k = 10 on (1,103 plays) and three times at k = 14.
- Cumulative over the whole ramp (the corpus pointer carries across steps): 3 engines 5,019 plays = 4.73 corpora, the first wrap
  inside step 9; 1 engine 1,673 plays = 1.58 corpora, first wrap inside step 13; 8 engines 13,383 plays = 12.6 corpora, first
  wrap inside step 6.
- Fixed point at 85 % of 2.121 sessions/s: 3 engines 1.803 s/s x 1,200 s = 2,163 plays = 2.04 corpora (wraps after 588 s);
  1 engine 0.601 s/s = 721 plays = 0.68 corpora (no wrap in the window); 8 engines 4.808 s/s = 5,769 plays = 5.44 corpora (wraps
  after 220 s). Every wrap replays with a fresh salt and is counted by the driver (`--corpus-wrap`, `wrap_round`).

## (e) Sizes, commands, leak scan

Files (out/): corpus.jsonl 141,837,318 bytes (1,060 lines); manifest.json 890,780 bytes; MANIFEST.md 5,963 bytes; REPORT.md
(this file); SHA256SUMS (every file of out/ and tools/). Tools: gen_corpus.py, build_corpus_v2.py.

Commands run (paths abbreviated; `<PY>` = the lane's virtualenv python with tokenizers, run with `-I`; `<BFCL>` = the pinned
BFCL directory with SOURCE.txt; `<TOK>` = the directory with the V4.1 Flash tokenizer.json):

    <PY> -I tools/gen_corpus.py --tokenizer <TOK> --selftest                          # SELFTEST OK themes=192 shared_prefix_tokens=3636
    <PY> -I tools/build_corpus_v2.py --out <tmp> --bfcl <BFCL> --tokenizers deepseek-v4.1-flash=<TOK> --selftest   # SELFTEST OK (6 sessions)
    <PY> -I tools/build_corpus_v2.py --out out --bfcl <BFCL> --tokenizers deepseek-v4.1-flash=<TOK> \
         --gen 900 --bfcl-per-category 40 --seed 20261022 --jobs 16
      -> gen: 900 sessions in 57.7 s; tokens counted: deepseek-v4.1-flash 24,012 pieces, 34,476,693 tokens in 5.2 s;
         CORPUS v2 sessions=1060 {'gen': 900, 'bfcl': 160} json_schema=188 thinking=549 images=135 bytes=141837318 scrubbed={} in 67.1 s
    sha256sum (out/ and tools/) > out/SHA256SUMS
    leak-scan.sh text out/corpus.jsonl out/manifest.json out/MANIFEST.md out/REPORT.md out/SHA256SUMS tools/gen_corpus.py tools/build_corpus_v2.py
      -> leak-scan: clean

Scrub counters: {} (no internal marker was found in any text; the scrub rules are generic: home directories, IPv6 addresses).
Source pins in manifest.json: gorilla commit 26ad99ed327fc47fc7c8b91b43378c5709125a06 with the sha256 of all 18 BFCL files read,
verified against SOURCE.txt (True); tokenizer deepseek-ai/DeepSeek-V4.1-Flash @ 2cba9e42aa026125f3ed06c6d98c1db82f7ca027,
tokenizer.json sha256 c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b; the generator's and the builder's own
sha256; the build command line with path arguments reduced to their last component.

## (f) Owed

- The `cli` tenant: v1's 181 recorded coding-CLI sessions (real harness prompt, real tool results, 4.6k -> 5.8k tokens) are not in
  v2; a GPU lane re-records them and the builder gains the source back (`convert_m27` of v1 was dropped from this copy; it can be
  restored from v1's build_corpus.py when the recordings exist).
- The image files: 135 gen sessions carry a slot reference only (`"file": null`); the baked image set (screenshot/document classes,
  0.2-2.0 Mpixel) is fetched by another lane and the driver attaches the data URL by slot when present (decision D1 pending).
- Calibration tail: 25 of 900 last prompts miss the plan by 5-8.3 % (first prompts: none); a follow-up can raise the growth floor
  for 8-turn sessions or merge the smallest late reads.
- Only the V4.1 Flash family is counted; the other five families' columns return when their tokenizer files are on the host.
