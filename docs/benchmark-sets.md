# Benchmark sets

Design for bellwether #23, written 2026-10-06. It records what was decided and why; the code that
follows it is reviewed against it.

## What this is for

Simo wants bellwether's evidence for every model to come from public benchmark data, through both
paths a gateway runs: the request rendered to prompt tokens, and the model's function calls parsed
back from its output. Thousands of cases per model, next to the hand-written sets, which stay as the
adversarial ones. Bellwether leads Symphony (ground rule 26): a Symphony format starts only once the
models that use it have these sets.

## Which models

Every generative model that vLLM or SGLang supports, except gpt-oss, which Simo set aside.

- **Where the list comes from.** The engines' own registries at pinned refs: vLLM's
  `tests/models/registry.py` (134 text-generation and 130 multimodal architectures at the 2026-10-04
  ref, each with example checkpoints) and SGLang's `python/sglang/srt/models/` and supported-models
  docs. A command reads them; the list is never kept by hand. Embedding, reranking and
  classification models are out. Multimodal chat models are in, with text-only cases for now.
- **Every model is a row.** Model, pinned revision, family, tier, and a status: what was recorded,
  what was rejected and why, or why nothing can be recorded yet (vendor code needed, gated, no chat
  template). A model that records nothing still has its row.
- **Families.** Checkpoints whose tokenizer and chat-template files are byte-identical render and
  parse identically, so a family is recorded once and each checkpoint's row points at it. The
  current Qwen chat checkpoints show the effect: about 45 of them fall into about 20 families (the
  seven Qwen3 sizes share one; six Qwen3.5 sizes share another). Membership is checked by file
  hash when recording, never assumed from a name.
- **Order.**
  1. Simo's tier: DeepSeek-V4.1-Flash, MiniMax-M3, the latest GLM (GLM-5.3-Flash), every current
     Qwen chat checkpoint, and Hy4-preview.
  2. The rest of the latest and most popular models, ranked by Hugging Face creation date and
     downloads (ground rule 15). Kimi-K3 and Muse-Glimmer-30B are here.
  3. Everything else, ranked the same way.

  Extra work follows the same order: a tier-1 model that needs vendor code, a login or another
  reference comes before a tier-3 model that records cleanly.

## Sources

BFCL's single-turn categories come first, because they carry tool definitions and function calls,
the path Symphony is built for. After them, in #23's order: BFCL multi_turn, GSM8K, tau2-bench, the
tool-schema collections in `corpus.md`, and real model outputs.

## Importing BFCL

`bellwether import bfcl` writes corpus sets from the data inside the `bfcl-eval` wheel that smg's
weekly run pins (2026.3.23).

- **The source.** The wheel is fetched from PyPI by version and checked against a sha256 in the
  code (`3bb6dfa5…`, equal to PyPI's), then read as a zip. Nothing is installed and none of its code
  runs; its own dependencies include sentence-transformers.
- **Requests as the weekly run sends them** through `OpenAICompletionsHandler`: the case's
  messages; the functions with BFCL's language hint, Java and JavaScript parameter types turned
  into strings, its type map and float note, and "." replaced by "_" in names; `tools` left out
  when a case has none; `temperature: 0.001` and `store: false`. Key order is kept, since templates
  serialize tool definitions in the order given. Unit tests pin each rule with hand-written
  expectations, and a one-off comparison against BFCL's own `convert_to_tool` over every case is
  reported with the change.
- **Parse cases** for the rows with a ground truth: the assistant message has `content: ""` and
  one call per ground-truth entry. Each parameter takes its first acceptable value that is not
  the "may be omitted" marker; a parameter that can only be omitted is left out; nested objects
  follow the same rule. Arguments are JSON written with raw Unicode, as the hand-written cases are.
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

The single-turn categories give 3641 render cases and 2501 parse cases per model.

## Recording

`record` stays the recorder. A probe over 45 current checkpoints (gpt-oss set aside) found two
things the round-trip oracle has to learn, both declared per model in the manifest:

- **How the template takes tool-call arguments.** Most current templates iterate the arguments as
  an object (Qwen3.5 to 3.8, GLM, MiniMax, Step, Hy4, Gemma 4, Muse-Glimmer and others); DeepSeek's
  concatenate them as a string. vLLM passes objects, and transformers documents objects. The
  manifest says `tool_call_arguments = "object"` (the default) or `"string"`, and the reference
  message keeps the JSON string a parser must return. Qwen3-8B's fixtures come out byte-identical
  either way. With objects, and strings for DeepSeek, 26 of the 45 models record all 2501 parse
  cases; with strings alone, 11 do.
- **Where the assistant turn ends.** The oracle cuts the output at the tokenizer's end-of-sequence
  token, but many templates end a turn with their own marker: `<|eot|>` (Muse-Glimmer),
  `<|endofassistant|>` (dots3), `<|im_end|>` (ERNIE, MiniCPM5), `</assistant>` (Laguna). GLM ends a
  turn with no marker; generation stops at the next role tag. The output ends at the first of the
  model's own stop tokens (its generation config) found in the rendered turn, or at the turn's end
  when there is none; the manifest records which. GLM-5.3-Flash is the first case to fix, as the
  only one of the four weekly models without parse cases.

Rejections are never silent. A case the reference cannot take is reported with its reason, every
model's row carries its recorded and rejected counts, and each kind of rejection gets an issue. The
probe's open kinds: the template does not extend the generation prompt when the turn is appended
(Nemotron 3, GLM-4.7-Flash, MiniMax-M2, Step3, Trinity; DeepSeek-R1, #14); Mistral wants call ids of
nine letters or digits; K2-Horizon wants a thinking field; Hunyuan-A13B renders no tool calls; and
five templates drop the tool list from the prompt (DeepSeek R1, V3, V3.1, Hunyuan-A13B, Phi-4-mini),
which needs another reference than the round trip.

Chunk plans stay as they are, the full set for every case. They are a function of the output
length, so they add no information that could go stale, and compressed they cost little.

## Storage

All-model figures, from the 45-model probe (BFCL single-turn alone):

- Plain JSON Lines take about 52 MB per model (32 MB render, 20 MB parse): 2.3 GB for the 45, and
  about 13 GB for 250 families.
- zstd at level 19 brings a model to about 2.3 MB, 23 times smaller.
- Git packs the plain files of all 45 into 61 MB, because the same requests recur in every model's
  files. But a checkout still writes every byte, so the working tree would be several gigabytes now
  and tens of gigabytes later, before GSM8K and tau2 add theirs.

Decision:

- Hand-written sets stay plain JSON Lines in git, reviewed as diffs.
- Benchmark sets are written as zstd-compressed JSON Lines (`<set>.jsonl.zst`, level 19, one thread,
  with the `zstandard` version pinned in `uv.lock`, so the bytes are a function of the content) and
  stored with Git LFS in this repository.
- `bellwether unpack` writes plain JSON Lines into a git-ignored tree for consumers: Symphony's
  fixture test and smg's consumer test point there. Bellwether's own readers take either form.

Why: plain files in git would put gigabytes, then tens of gigabytes, in every checkout; compression
cuts LFS storage and transfer about 23 times; a clone fetches only the sets it asks for
(`git lfs pull --include`), and there is no second repository to keep in step.
A diff of thousands of fixture lines is not reviewable in either form. Bulk sets are reviewed by
re-recording and comparing file hashes, and by the per-model table.

## Counting

`bellwether count` prints the cases per model, kind and source (the `origin` dataset, or
hand-written) as a table or JSON. The README carries the table, and `gaps` marks rows by it.

## CI

- **Per pull request:** ruff and pytest; the hand-written fixtures are validated in full, and the
  benchmark sets a pull request changes are validated in full. Git LFS fetches only those paths.
  The importers re-run and the corpus must come out byte-identical.
- **Weekly:** every family is re-recorded and each compressed file's sha256 is compared with its LFS
  pointer in git, which needs no download; then every set is downloaded and validated.
- **Credentials:** recording downloads tokenizer files, so CI uses a Hugging Face token stored as
  a repository secret, for rate limits and for gated repositories. Simo adds it; no token is ever
  asked for or written here.

## Delivery

1. The BFCL importer and its corpus sets, `count`, the two oracle changes, and their tests.
2. The model list read from the registries, manifests and families for tier 1, tier 1 recorded,
   and the per-model table.
3. Tiers 2 and 3 in batches, with the extra work in tier order: a vendor-code oracle (DeepSeek V3.2
   and V4-Flash, Kimi-K3), tokenizers that need `tiktoken` or custom code, gated models, and a
   reference for templates that drop the tool list.
4. The next sources, from BFCL multi_turn on.
