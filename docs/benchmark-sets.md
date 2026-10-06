# Benchmark sets

Design for bellwether #23, written 2026-10-06. It records what was decided and why; the code that
follows it is reviewed against it. The measurements behind it are in the per-model report on #23:
https://github.com/smg-project/bellwether/issues/23#issuecomment-6017548219

## What this is for

Simo wants bellwether's evidence for every model to come from public benchmark data, through both
paths a gateway runs: the request rendered to prompt tokens, and the model's function calls parsed
back from its output. Thousands of cases per model, next to the hand-written sets, which stay as the
adversarial ones. Bellwether leads Symphony (ground rule 26): a Symphony format starts only once the
models that use it have these sets.

## Which models

Every generative model that vLLM or SGLang supports, except gpt-oss, which Simo set aside. Two scope
lines are assumptions Simo has not confirmed: multimodal chat models are in, with text-only cases
for now; embedding, reranking and classification models are out.

- **Where the list comes from.** `bellwether models` builds it and writes one row per checkpoint,
  from two sources:
  - the engines' registries at pinned refs: every example checkpoint that vLLM's
    `tests/models/registry.py` names for a generative architecture (134 text-generation and 130
    multimodal architectures at the 2026-10-04 ref), and every checkpoint in SGLang's
    supported-models docs;
  - the Hugging Face Hub, for each organization that publishes a registered architecture: its
    models whose `config.json` names a registered architecture and that ship a chat template.
    Two kinds are left out. Quantized and converted copies (GGUF, AWQ, GPTQ, MLX, ONNX, FP8,
    NVFP4, MXFP4, MXFP8, Int4, Int8, bitsandbytes) carry their source's files. Checkpoints created
    before 2025 are left out unless a registry names them.

  The command pins the registry refs, but the Hub only shows its present state, so what is
  reproducible is the committed list with each row's revision; a rerun shows what changed as a diff
  of that list. "Every current Qwen chat checkpoint" in tier 1 is this rule applied to the
  Qwen organization: about 45 checkpoints today.
- **Every checkpoint is a row.** Each row gives the model, its pinned revision, checkpoint group,
  tier, and a status. The status says what was recorded, what was rejected and why, or why nothing
  can be recorded yet (vendor code needed, gated, no chat template). A checkpoint that records
  nothing still has its row.
- **Checkpoint groups.** Checkpoints whose oracle inputs are byte-identical render and parse
  identically, so a group is recorded once and each checkpoint's row names its group.
  - **The oracle inputs** are every file the oracle reads for the model: the tokenizer files, the
    chat template, the generation config (for the end of turn), and, for a vendor-code oracle, the
    vendor's files. From `config.json` they take only the two fields that choose the tokenizer
    class (`model_type`, `tokenizer_class`).
  - **Membership.** The manifest lists each input with its sha256, and membership is equality of
    those lists, checked when recording. As a check on the list itself, the weekly job also
    records one other member of each group and compares the results, so a file the list misses
    shows up.
  - **The effect.** The current Qwen chat checkpoints fall into about 20 groups: the seven Qwen3
    sizes share one, and six Qwen3.5 sizes share another.
- **Order.**
  1. Simo's tier: DeepSeek-V4.1-Flash, MiniMax-M3, the latest GLM (GLM-5.3-Flash), every current
     Qwen chat checkpoint, and Hy4-preview.
  2. Every checkpoint created in the twelve months before the list is built, in order of its
     downloads over the last 30 days as the Hub reports them (ground rule 15). Kimi-K3 and
     Muse-Glimmer-30B are here.
  3. Everything else, in the same order.

  Extra work follows the same order: a tier-1 model that needs vendor code, a login or another
  reference comes before a tier-3 model that records cleanly.

## Sources

BFCL's single-turn categories come first, because they carry tool definitions and function calls,
the path Symphony is built for. After them, in #23's order: BFCL multi_turn, GSM8K, tau2-bench, the
tool-schema collections in `corpus.md`, and real model outputs.

## Importing BFCL

`bellwether import bfcl` writes corpus sets from the data inside the `bfcl-eval` wheel that smg's
weekly run pins (2026.3.23).

- **The source.** The wheel is fetched from PyPI by version, checked against a sha256 in the code
  (`3bb6dfa5…`, equal to PyPI's), and read as a zip. Nothing is installed and none of its code runs;
  its own dependencies include sentence-transformers.
- **Requests, as the weekly run sends them** through `OpenAICompletionsHandler`:
  - the case's messages, unchanged;
  - the functions, with BFCL's language hint, Java and JavaScript parameter types turned into
    strings, its type map and float note, and "." replaced by "_" in names;
  - `tools` left out when a case has none;
  - `temperature: 0.001` and `store: false`.

  Key order is kept, since templates serialize tool definitions in the order given. Unit tests pin
  each rule with hand-written expectations. A one-off comparison against BFCL's own
  `convert_to_tool` over every case is reported with the change; it runs BFCL's code, so it runs in
  a throwaway environment, never in the importer.
- **Parse cases** for the rows with a ground truth. The assistant message has `content: ""` and
  one call per ground-truth entry:
  - each parameter takes its first acceptable value that is not the "may be omitted" marker;
  - a parameter that can only be omitted is left out, and nested objects follow the same rule;
  - arguments are JSON written with raw Unicode, as the hand-written cases are.

  BFCL's checker accepts every call built this way. The irrelevance and relevance categories are
  render-only: they have no answer to round-trip.
- **Sets and provenance.** One set per BFCL category, `corpus/{render,parse}/bfcl-<category>.jsonl`;
  case names are `bfcl-<row id>`. Each line carries `origin`: dataset, source
  (`pypi:bfcl-eval==2026.3.23`), sha256, file, row id and license. Corpus lines are not covered by
  the case schema, so the schema does not change; a fixture line finds its origin through its name.
- **Determinism.** A re-import reproduces the corpus byte for byte, and CI checks that it does.
- **License.** BFCL is Apache-2.0 (the wheel's metadata and the gorilla repository).
  `corpus/README.md` lists each imported dataset with its license and attribution, and a dataset's
  license is checked before its sets land.

The single-turn categories give 3641 render cases and 2501 parse cases per checkpoint group.

## Recording

`record` stays the recorder. The probe covered 63 current checkpoints (gpt-oss set aside); 45 of
them load and record BFCL's render cases, and the parse and storage figures here are for those 45.
It found two things the round-trip oracle has to learn, both declared per model in the manifest.
Each lands as its own change, and each shows that the existing fixtures come out byte-identical.

- **How the template takes tool-call arguments.**
  - Most current templates iterate the arguments as an object: Qwen3.5 to 3.8, GLM, MiniMax, Step,
    Hy4, Gemma 4, Muse-Glimmer and others. DeepSeek's concatenate them as a string.
  - The manifest says `tool_call_arguments = "object"` (the default) or `"string"`. Qwen3-8B's
    fixtures come out byte-identical either way.
  - With objects, and strings for DeepSeek, 26 of the 45 models record all 2501 parse cases; with
    strings alone, 11 do.
  - Both engines turn the arguments into an object before rendering: vLLM in
    `vllm/entrypoints/chat_utils.py:1931` (1ad5182b), SGLang in `parse_tool_call_arguments`,
    `python/sglang/srt/entrypoints/openai/serving_chat.py:126` (7d22b7a8). So they hand
    DeepSeek's templates a form those templates cannot take: a request whose history carries a
    DeepSeek tool call fails to render, the shape behind smg #2783. That stays a finding, reported
    with the render cases. The manifest setting only says how the reference renders the round
    trip; it does not hide the finding.
- **The reference's arguments string.**
  - For templates that write JSON, it is the bytes in the output.
  - For tagged formats, it is the canonical JSON a parser builds from the tags. Whether that is the
    target, and in which canonical form, is #24; the form is Simo's call at S2.
  - Each parse line records, per call, whether the arguments string occurs verbatim in the output
    (`arguments_verbatim`). That adds a field to the case schema, so it waits for Simo's approval.
    When it is false for a template that writes JSON, the reference still holds the canonical
    string, and each such template is reported as its own kind.
  - When engine witnesses land for a tagged format, each engine's argument string is kept per case.
- **Where the assistant turn ends.** The oracle cuts the output at the tokenizer's end-of-sequence
  token, but many templates close a turn with their own marker:
  - `<|eot|>` (Muse-Glimmer), `<|endofassistant|>` (dots3), `<|im_end|>` (ERNIE, MiniCPM5),
    `</assistant>` (Laguna);
  - GLM closes a turn with no marker at all: generation stops at the next role tag;
  - Inkling's tokenizer has no end-of-sequence token.

  The output ends at the first of the model's own stop tokens (its generation config) found in the
  rendered turn, or at the turn's end when there is none, and the manifest records which.
  GLM-5.3-Flash is the first to fix, as the only one of the four weekly models without parse cases.

Rejections are never silent. A case the reference cannot take is reported with its reason, every
checkpoint's row carries its recorded and rejected counts, and each kind of rejection gets an issue.
The probe's open kinds:
- the template does not extend the generation prompt when the turn is appended (Nemotron 3,
  GLM-4.7-Flash, MiniMax-M2, Step3, Trinity; DeepSeek-R1, #14);
- Mistral wants call ids of nine letters or digits;
- K2-Horizon wants a thinking field;
- Hunyuan-A13B renders no tool calls;
- five templates drop the tool list from the prompt (DeepSeek R1, V3, V3.1, Hunyuan-A13B,
  Phi-4-mini), which needs another reference than the round trip.

Chunk plans stay as they are, the full set for every case. They are a function of the output
length, so they add no information that could go stale, and compressed they cost little.

## Fixture ids and the tree consumers read

- **A group's slug.** A group's sets live under one slug.
  - An existing model keeps its slug: the Qwen3 group is `qwen3-8b`, so ids such as
    `qwen3-8b/parse/call-unicode-arguments` stay valid.
  - A new group takes the slug of its most-downloaded member when it is first recorded.
  - A slug never changes once recorded, since ids are cited in pull requests and trackers.
- **Other members.** Each other member has a manifest that names its group and lists its own
  oracle-input hashes. It has no fixture files of its own.
- **What consumers read.** `bellwether unpack` writes `<root>/<slug>/<kind>/<set>.jsonl`, the same
  layout as `fixtures/`, with the hand-written sets beside the benchmark sets, so a consumer points
  at one root. Symphony's fixture test and smg's consumer test read plain JSON Lines after one
  command, and Symphony's dependencies do not change.

## Storage

All-model figures, from the 45-model probe (BFCL single-turn alone):

- **Plain JSON Lines:** about 52 MB per checkpoint group (32 MB render, 20 MB parse); 2.3 GB for
  the 45, and a projected 13 GB for 250 groups.
- **zstd at level 19:** about 2.3 MB per group, 23 times smaller.
- **Git's packing:** the plain files of all 45 pack into 61 MB, because the same requests recur in
  every model's files. But a checkout still writes every byte, so the working tree would be several
  gigabytes now and tens of gigabytes later, before GSM8K and tau2 add theirs.
- **The BFCL corpus:** 16 MB plain, 0.8 MB packed.

Decision:

- **Plain in git:** the hand-written sets and the corpus. The corpus is the reviewable input, so a
  change to an importer reads as a case-level diff. A source whose corpus passes 50 MB moves to the
  fixtures' form.
- **Benchmark fixture sets:** zstd-compressed JSON Lines (`<set>.jsonl.zst`), stored with Git LFS
  in this repository.
- **`sets.toml` per group, committed plain:** every set's case count, rejected count, plain size,
  and the sha256 of its plain content. `count` and the README read it without LFS, and a reviewer
  reads its diff. The weekly check compares plain-content hashes, so it does not depend on the
  compressor writing the same bytes on every machine.
- **`.lfsconfig`:** sets `lfs.fetchexclude` to the benchmark sets, so a default clone downloads
  none of them. `bellwether unpack --model` or `git lfs pull --include` fetches what a consumer asks
  for.

What it costs:

- **Storage:** about 2.3 MB per group for BFCL single-turn, a projected 0.6 GB for 250 groups. With
  BFCL multi_turn, GSM8K and tau2, an estimated 1.5 to 2 GB per full recording. LFS keeps every
  version in history, so each full re-record adds about as much again; re-records follow an oracle
  or environment change, a few a year.
- **Bandwidth:** the weekly check downloads nothing, and a pull request downloads only the sets it
  changes. CI caches the LFS objects it uses, keyed by their pointers, so an unchanged set is not
  fetched twice.
- **Past the allowance:** GitHub stops LFS uploads and downloads for the whole organization until
  the next billing cycle or until more is bought. Git itself, the code and the hand-written sets
  keep working. The organization's allowance is not known here; it is a question for Simo.

Why: plain files in git would put gigabytes, then tens of gigabytes, in every checkout. Compression
cuts LFS storage and transfer about 23 times. A clone fetches only the sets it asks for, and there
is no second repository to keep in step. A diff of thousands of fixture lines is not reviewable in
either form. Bulk sets are reviewed through `sets.toml`, by re-recording and comparing plain-content
hashes, and through the per-model table.

## Running vendor code

Some models need code from their own repositories: vendor encoders, custom tokenizers or configs.
That code runs pinned to the manifest's revision, inside a container that holds only bellwether and
the repository's files, in a step that does not hold the Hugging Face token. The files are
downloaded first, by a step that does.

## Counting

`bellwether count` prints the cases per model, kind and source (the `origin` dataset, or
hand-written) as a table or JSON, read from the `sets.toml` files. The README carries the table, and
`gaps` marks rows by it.

## CI

- **Per pull request:** ruff and pytest. The hand-written fixtures, and the benchmark sets the pull
  request changes, are validated in full; Git LFS fetches only those paths. The importers re-run,
  and the corpus must come out byte-identical.
- **Weekly:** every group is re-recorded and each set's plain-content sha256 is compared with
  `sets.toml`, which needs no download; one other member of each group is recorded and compared.
- **Credentials:** recording downloads tokenizer files, so CI uses a Hugging Face token stored as
  a repository secret, for rate limits and for gated repositories. Simo adds it; no token is ever
  asked for or written here.

## Delivery

Each step is its own pull request.

1. The BFCL importer, its corpus sets, and `count`.
2. The argument form, per model in the manifest, with the existing fixtures re-recorded
   byte-identical.
3. The end-of-turn rule, likewise, with GLM-5.3-Flash's parse cases first.
4. `bellwether models` and the committed list.
5. The storage form: zstd sets in Git LFS, `sets.toml`, `unpack` and `.lfsconfig`.
6. Tier 1: groups and manifests, its recorded sets, and the per-model table.
7. Tiers 2 and 3 in batches, with the extra work in tier order: a vendor-code oracle (DeepSeek V3.2
   and V4-Flash, Kimi-K3), tokenizers that need `tiktoken` or custom code, gated models, and a
   reference for templates that drop the tool list.
8. The next sources, from BFCL multi_turn on; #19's `tool_choice` cases are built from the BFCL
   import.

## Questions for Simo

1. The organization's Git LFS allowance, and whether to buy more when the sets pass it.
2. `arguments_verbatim` on parse lines, a case-schema addition (#24).
3. The two scope assumptions: multimodal chat models are in, with text-only cases; embedding,
   reranking and classification models are out.
