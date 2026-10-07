# Benchmark sets

Design for bellwether #23, written 2026-10-06. It records what was decided and why; the code that
follows it is reviewed against it. The measurements behind it are in the per-model report on #23:
https://github.com/smg-project/bellwether/issues/23#issuecomment-6017548219

## What this is for

The maintainer's statement of bellwether's job (2026-10-06):

> Given a request, what's expected tokens. Given a token, what's expected output text. With insane
> amount of data. Then smg uses it in many places. Tokenizer, detokenization, gRPC router, and
> symphony.

So bellwether holds expected values in both directions for every model: a request rendered to
prompt tokens, and output tokens turned back into text and, for parse cases, into the message a
parser returns. The data is public benchmark data, thousands of cases per model, next to the
hand-written sets, which stay as the adversarial ones.

Its consumers in smg:

- **The tokenizer crate:** encoding the prompt text gives the reference ids, and its incremental
  decode of the output ids gives `output_pieces`.
- **The gRPC router's request path:** smg's render parity test (smg #2782) replays render cases
  through the gateway's chat request processing. A scripted mock worker with capture, run against a
  real `smg` over gRPC, is the one test that sees what the router sends the engine, the
  `tool_choice` constraint included.
- **Symphony's parsers:** its fixture test replays parse cases, chunk plan by chunk plan.
  Bellwether leads Symphony (ground rule 26): a Symphony format starts only once the models that
  use it have these sets.

For now a consumer reads bellwether's main. Later: tagged releases with a note of which sets
changed, a pin file in smg holding the release and the `sets.toml` hashes, CI fetching that pin,
and each bump a pull request that shows what changed.

## Which models

Every generative model that vLLM or SGLang supports, except gpt-oss, which the maintainer set aside. Two scope
lines are assumptions the maintainer has not confirmed: multimodal chat models are in, with text-only cases
for now; embedding, reranking and classification models are out.

- **Where the list comes from.** `bellwether models` builds it and writes one row per checkpoint,
  from two sources:
  - the engines' registries at pinned refs: every example checkpoint that vLLM's
    `tests/models/registry.py` names for a generative architecture (134 text-generation and 130
    multimodal architectures at the 2026-10-04 ref), and every checkpoint in SGLang's
    supported-models docs; the architectures SGLang serves come from its code, each model module's
    `EntryClass`, since the docs name checkpoints and not architectures;
  - the Hugging Face Hub, for each organization that publishes a registered architecture: its
    models whose `config.json` names a registered architecture and that ship a chat template.
    Two kinds are left out. Quantized and converted copies (GGUF, AWQ, GPTQ, MLX, ONNX, FP8,
    NVFP4, MXFP4, MXFP8, Int4, Int8, bitsandbytes) carry their source's files. Checkpoints created
    before 2025 are left out unless a registry names them.

  The command pins the registry refs, but the Hub only shows its present state. What is committed,
  and rebuilt and compared in CI, is the list built from the registries alone (`models.jsonl`),
  whose first line names the pins; a list read from the Hub is that day's evidence, kept under
  `runs/` and published to smg-project/artifacts, each row pinning the Hub's sha of the day. "Every current Qwen chat checkpoint" in tier 1 is this rule applied to the
  Qwen organization: 64 checkpoints on 2026-10-06.
- **Every checkpoint is a row.** Each row gives the model, its pinned revision, checkpoint group,
  tier, and a status. The status says what was recorded, what was rejected and why, or why nothing
  can be recorded yet (vendor code needed, gated, no chat template). A checkpoint that records
  nothing still has its row.
- **Checkpoint groups.** Checkpoints whose oracle inputs are equal render and parse identically,
  so a group is recorded once and each checkpoint's row names its group. Equal means each file
  byte-identical, or, for a JSON file an oracle reads only in part, equal in the part it reads.
  - **The oracle inputs** are every file the oracle reads for the model: the tokenizer files, the
    chat templates (the named ones in `additional_chat_templates/` too), the generation config (for
    the end of turn), and, for a vendor-code oracle, the vendor's files. From `config.json` they
    take only the two fields that choose the tokenizer class (`model_type`, `tokenizer_class`), and
    from the generation config only its token ids (`eos_token_id`, `bos_token_id`, `pad_token_id`),
    so sampling defaults do not split a group. From `chat_template.json` they take only its
    `chat_template` value, which vLLM reads through the processor; no oracle reads the file's
    bytes. A checkpoint that ships no generation config has the
    end of turn read those ids from `config.json`, through `GenerationConfig.from_model_config`
    (`text_config` included), so there they count as part of `config.json`.
  - **Membership.** The manifest lists each input with its sha256, and membership is equality of
    those lists, checked when recording. As a check on the list itself, the weekly job also
    records one other member of each group and compares the results, so a file the list misses
    shows up.
  - **The effect,** measured on 2026-10-06: the 64 current Qwen chat checkpoints fall into 35
    groups. The six dense Qwen3 sizes share one, and the Qwen3-VL Instruct and Thinking sizes from
    2B to 32B two of four; most other shared groups are pairs or threes, such as Qwen3.5-4B and 9B,
    or the three large Qwen3.5 mixtures of experts. The 38 first measured here hashed
    `chat_template.json` by its bytes, which split three Qwen3-VL pairs whose templates differ only
    in that file's indentation and trailing whitespace.
- **Order.**
  1. The maintainer's tier: DeepSeek-V4.1-Flash, MiniMax-M3, the latest GLM (GLM-5.3-Flash), every current
     Qwen chat checkpoint, and Hy4-preview.
  2. Every checkpoint created in the twelve months before the list is built, in order of its
     downloads over the last 30 days as the Hub reports them (ground rule 15). Kimi-K3 and
     Muse-Glimmer-30B are here.
  3. Everything else, in the same order.

  Extra work follows the same order: a tier-1 model that needs vendor code, a login or another
  reference comes before a tier-3 model that records cleanly.

## Sources

BFCL's single-turn categories come first, because they carry tool definitions and function calls,
the path Symphony is built for. Every other source on #23 follows (the maintainer, 2026-10-06: "Why not all
of them"), each its own importer on BFCL's pattern, in this order:

1. GSM8K, with outlines combined from GSM8K text and BFCL calls: reasoning only, prose only,
   reasoning then prose, reasoning then calls, prose then calls, all three.
2. Agent trajectories on SWE-bench (SWE-agent, OpenHands and the published trajectory sets).
3. Model-written reasoning traces.
4. tau2-bench and BFCL multi_turn.
5. SWE-bench and SWE-bench Verified, their gold patches as large single arguments; then SWE-bench
   Pro.
6. Tool-schema collections: xLAM 60k, ToolACE, ToolBench and the MCP benchmarks.
7. JSONSchemaBench, for #19's constrained cases.
8. MGSM.

Each source is confirmed to exist as remembered, pinned, and license-checked before its sets land.
A set built from copyleft input (SWE-bench Pro's repositories, or any row whose repository is
copyleft) is kept in its own set files, named with a `-copyleft` suffix and carrying the license in
`origin`, so it can be dropped or handled apart if the repository's visibility ever changes. Gated
sources wait for a Hugging Face login on the recording machine.

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
  - a parameter the function does not declare is left out when the answer allows it, since BFCL's
    checker refuses it;
  - arguments are JSON written with raw Unicode, as the hand-written cases are.

  Each call is held to BFCL's own checker rules. A row where no call this rule builds passes the
  checker gets no parse case, and the import names it with its reason (5 rows at this pin). Java
  and JavaScript rows get a parse case only where every value is a string: BFCL's checker refuses
  any other type there, and their ground truth stores converted values. The other 76 rows stay
  render-only until their string forms land (#26). The irrelevance and relevance categories are
  render-only: they have no answer to round-trip.
- **Sets and provenance.** One set per BFCL category, `corpus/{render,parse}/bfcl-<category>.jsonl`;
  case names are `bfcl-<row id>`. Each line carries `origin`: dataset, source
  (`pypi:bfcl-eval==2026.3.23`), sha256, file, row id and license. Corpus lines are not covered by
  the case schema, so the schema does not change; a fixture line finds its origin through its name.
- **Determinism.** A re-import reproduces the corpus byte for byte, and CI checks that it does.
- **License.** BFCL is Apache-2.0 (the wheel's metadata and the gorilla repository).
  `corpus/README.md` lists each imported dataset with its license and attribution, and a dataset's
  license is checked before its sets land. The wheel holds no LICENSE file, so the import copies the
  repository's, at the commit the wheel was built from, to `corpus/licenses/bfcl-LICENSE`, and
  `--check` compares the copy as it does a set.

The single-turn categories give 3635 render cases and 2420 parse cases per checkpoint group: 2501
rows have a ground truth, and 76 Java and JavaScript rows and 5 rows no built call answers stay
render-only. Six `live_irrelevance` rows repeat an earlier case and are left out (one rule for
repeats, in `corpus_sets`).

The weekly run's four multi_turn categories, base, miss_func, miss_param and long_context, are
imported for their first turn, as `bfcl-multi-turn-<category>`. The request is the first one the
handler sends for a row: the turn's messages and no system message, with the functions of the row's
classes as tools, less those `missed_function` holds back for a later turn. The functions come from
the class-to-file map in the wheel's source, parsed with `ast`, and go through the same language
hint and `convert_to_tool`. The parse message is the turn's ground-truth calls in order: each call
string is parsed with `ast`, a value passed by position takes the parameter of the class method's
`def`, read from the class's source with `ast` (BFCL's executor runs the call on the class, and
`purchase_insurance`'s doc lists two of its parameters the other way round), and each call is held
to the same checker rules. That gives 800 render cases and 625 parse cases before the rule for
repeats; the 175 miss_func and miss_param rows whose first turn has no ground-truth call stay
render-only. Later turns wait for a decision: each of their requests carries the results of the
earlier calls, which only BFCL's simulators produce.

The rule for repeats leaves out 316 of those render cases and 310 of those parse cases, so the four
sets hold 484 and 315: 117 of miss_param's first requests and 114 of its 119 parse cases equal
base's, in base and in miss_func row 43 repeats row 40, and long_context, whose rows differ from
base's in what the simulators return in later turns, keeps 3 render and 6 parse cases
(`corpus/README.md` has the counts per set). The 315 parse cases carry 166 distinct messages.

## Recording

`record` stays the recorder. Every case is recorded from two sources of truth, neither of which
needs a model (the maintainer, 2026-10-06, #23):

- **Hugging Face:** the checkpoint's own published files.
  - transformers' `apply_chat_template` with the checkpoint's tokenizer and template gives the
    prompt ids for render and, through the round trip, the output text, ids and pieces for parse.
  - A checkpoint whose prompt format or tokenizer is the vendor's own code is rendered by that code
    instead: an encoder in place of a chat template (DeepSeek V3.2 and V4, Mistral's
    `mistral-common` checkpoints) or a custom tokenizer class (Kimi-K3). That is the vendor-code
    oracle, run as "Running vendor code" says; it is the first authority in the manifest's order,
    and for these models it is the Hugging Face side.
- **vLLM:** `vllm launch render <model>`, the GPU-less render server, in vLLM's official image
  pinned by tag and digest: `vllm/vllm-openai:v0.31.0` (db9527a4), index digest
  `sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`.
  The digest goes into each line's provenance, since a tag can move. It runs on Linux: under colima
  on the Mac that records, or on a Linux runner. vLLM code cited below is at that release.
  - `POST /v1/chat/completions/render` gives a request's prompt ids.
  - `POST /v1/completions/derender` gives the text of output ids without the parsers. For every
    chunk plan, vLLM's text for each chunk is compared with the pieces Hugging Face's incremental
    decode gives (`output_pieces`), and the whole output's text with the reference's.
  - `POST /v1/chat/completions/derender` gives vLLM's parsed message for the output ids, recorded
    beside the reference message. It is also vLLM's answer to two open policy questions, an empty
    think block (#17) and the marker probes (#16), so it is recorded first for the 16 hand-written
    parse cases.

  What exactly these return, including whether render reports the stop ids the end of a turn needs
  and how per-chunk text is obtained, is established first (delivery step 5).

**Names.** Hugging Face's result stays the line's `reference` and vLLM's is a `witness`, the case
schema's existing words; "source of truth" is the role both play, not a field. The two are equal in
authority, and neither overrides the other:

- when they agree, the case is settled;
- when they disagree, the line records both results and carries `disputed`, holding the
  disagreement's fingerprint. A consumer counts no parity against a disputed case: Symphony's
  fixture test and smg's tests report it as disputed and skip it.

Witness results and `disputed` are case-schema changes (question 4).

**Disagreements are issues.** Each kind has a stable fingerprint, `vllm:<what differs>:<group>`,
where what differs is `prompt-ids`, `rejected` (one source takes the case, the other refuses it),
`end-of-turn`, `output-text`, `chunk-text` or `message`. A rerun finds the issue a fingerprint
already has and updates its cases instead of opening another (ground rule 15). The recorder prints
each new fingerprint with its cases and a draft issue, and opens nothing itself; a person, or the
work loop under its ground rules, files one issue per fingerprint, with the fingerprint in its body.
Nothing in bellwether is configured to make the two agree: no per-model setting, and no adjustment
of either side.

What the probe over 63 current checkpoints settles this way (gpt-oss set aside; 45 of them load and
record BFCL's render cases, and the parse figures here are for those 45):

- **Tool-call arguments reach the template exactly as vLLM gives them** (`_postprocess_messages`,
  `vllm/entrypoints/chat_utils.py:2084`): arguments become the object they decode to, and anything
  that does not decode to a JSON object becomes `{}` (#29). Most current templates iterate the
  arguments as an object; with objects, 24 of the 45 record every BFCL parse case, and with the JSON
  string, 11 do. SGLang differs: `normalize_assistant_tool_call_arguments` (7d22b7a8) rejects a
  string that is not a JSON object; that is recorded, not decided. DeepSeek's templates (R1, V3,
  V3.1) concatenate a string and fail on an object. That is a finding (#27), and the vLLM recording
  shows whether the engine fails the same way.
- **The end of the assistant turn** is not a rule of bellwether's. The round trip cuts the output at
  the tokenizer's end-of-sequence token, which many templates do not write:
  - `<|eot|>` (Muse-Glimmer), `<|endofassistant|>` (dots3), `<|im_end|>` (ERNIE, MiniCPM5),
    `</assistant>` (Laguna);
  - GLM closes a turn with no marker at all;
  - Inkling's tokenizer has no end-of-sequence token.

  For these, the output ends where generation stops, and that too has two sources:
  - Hugging Face: the generation config as transformers' `generate` reads it (`eos_token_id`, which
    may be a list), with the tokenizer's own end-of-sequence token;
  - vLLM: its stop set, from whichever of its code paths yields it. The model's end-of-sequence
    ids are added by `SamplingParams.update_from_generation_config`, called from the engine's input
    processor (`vllm/v1/engine/input_processor.py:440`), while the render server builds its
    parameters with `request.to_sampling_params` alone
    (`vllm/entrypoints/scale_out/render/serving.py:125`), so render may not report them.

  They are compared like every other result. GLM-5.3-Flash comes first, as the only one of the four
  weekly models without parse cases.

- **The reference's arguments string.**
  - For templates that write JSON, it is the bytes in the output.
  - For tagged formats, it is the canonical JSON a parser builds from the tags. Whether that is the
    target, and in which canonical form, is #24; the form is the maintainer's call at S2.
  - Each parse line records, per call, whether the arguments string occurs verbatim in the output
    (`arguments_verbatim`). That adds a field to the case schema, so it waits for the maintainer's approval.
    When it is false for a template that writes JSON, the reference still holds the canonical
    string, and each such template is reported as its own kind.
  - When engine witnesses land for a tagged format, each engine's argument string is kept per case.
Rejections are never silent. A case the reference cannot take is reported with its reason, every
checkpoint's row carries its recorded and rejected counts, and each kind of rejection gets an issue.
The probe's open kinds:
- the template does not extend the generation prompt when the turn is appended (Nemotron 3,
  GLM-4.7-Flash, MiniMax-M2, Step3, Trinity; DeepSeek-R1, #14);
- Mistral wants call ids of nine letters or digits;
- K2-Horizon wants a thinking field;
- Hunyuan-A13B renders no tool calls, which is counted as not applicable instead ("Not
  applicable", below);
- five templates drop the tool list from the prompt (DeepSeek R1, V3, V3.1, Hunyuan-A13B,
  Phi-4-mini), which needs another reference than the round trip; for V3-0324 and V3.1, vLLM's
  docs give one (the next section).

Chunk plans stay as they are, the full set for every case. They are a function of the output
length, so they add no information that could go stale, and compressed they cost little.

## Second references from vLLM's example tool templates

The maintainer's decision on #59 (2026-10-06): where vLLM's docs name one of its example tool templates
(`examples/tool_chat_template_*.jinja`) for a checkpoint, bellwether records a second reference
with that template, at vLLM's pinned commit, beside the checkpoint's own template, and a
disagreement between the two is an issue. A user who follows vLLM's docs for tool calling starts
the server with that template, so its prompt is the one that user's requests get. The first
reference stays the checkpoint's own files.

- **What it is.** The case through the first reference's oracle, with the checkpoint's tokenizer at
  its pinned revision, but with the example template in place of the checkpoint's own: transformers'
  `apply_chat_template` given the template's text as `chat_template`. It is a reference, not a
  witness: no engine runs, and the only file taken from vLLM is the template.
- **When it applies.** Only where a page of vLLM's docs names a template for the checkpoint. The
  pages are those vLLM's docs build makes at the pin: every `docs/**/*.md`, and one page per example
  under `examples/<category>/` (`docs/mkdocs/gen_files/generate_examples.py`). Two of them name
  example tool templates, `docs/features/tool_calling.md` and the page made from
  `examples/tool_calling/openai_chat_completion_client_with_tools.py`; together they name 18.
  - A page names a checkpoint by its id or by an id pattern (`meta-llama/Llama-3.2-*`). Prose that
    names a family without an id ("additional Mistral function-calling models") names no
    checkpoint, since choosing its members would be bellwether's judgment, which the two-sources
    rule leaves out.
  - Ids are matched without regard to case, as the Hub resolves them: the page's
    `Salesforce/Llama-xLAM-2-8B-fc-r` is the Hub's `Salesforce/Llama-xLAM-2-8b-fc-r`.
  - A checkpoint gets one second reference per template its pages name. Mistral-7B-Instruct-v0.3
    gets two (`mistral` and `mistral_parallel`), and so does Llama-3.2-1B-Instruct (`llama3.2_json`
    through `meta-llama/Llama-3.2-*`, and `llama3.2_pythonic` by its id).
  - A template no page names makes no second reference, whatever its file name says.
    `tool_chat_template_hunyuan_a13b.jinja` and `tool_chat_template_phi4_mini.jinja` are two of
    them: no page names either, and for Hunyuan-A13B the tool calling page says the chat template is
    already included in the Hugging Face files (line 382). Question 8 asks whether that should
    change.
- **The mapping.** `src/bellwether/record/vllm_tool_templates.toml`, read by hand from the two pages
  at db9527a4. Each template carries its sha256 at the commit, and each statement that names it
  carries the checkpoints, the page and lines, and the flags the same lines give for vLLM's side:
  the tool parser, and for Mistral the Transformers tokenizer mode. The pages are pinned by sha256
  too. Each vLLM pin bump reads them again, and the table's diff is the review.
- **Which checkpoints, today.** Crossed with `models.jsonl` and the 106 manifests of #54, eight
  checkpoints get second references:

  | Checkpoint | Templates | Listed in |
  |---|---|---|
  | `swiss-ai/Apertus-8B-Instruct-2509` | `apertus` | manifest, `models.jsonl` |
  | `deepseek-ai/DeepSeek-V3-0324` | `deepseekv3` | manifest |
  | `deepseek-ai/DeepSeek-V3.1` | `deepseekv31` | manifest |
  | `Salesforce/Llama-xLAM-2-8b-fc-r` | `xlam_llama` | manifest |
  | `mistralai/Mistral-7B-Instruct-v0.3` | `mistral`, `mistral_parallel` | manifest |
  | `meta-llama/Llama-3.2-1B-Instruct` | `llama3.2_json`, `llama3.2_pythonic` | `models.jsonl` |
  | `meta-llama/Llama-3.2-11B-Vision-Instruct` | `llama3.2_json` | `models.jsonl` |
  | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | `llama4_pythonic` | `models.jsonl` |

  The three Llama checkpoints are gated on the Hub, so they wait for a login, as their first
  references do. The pages name 16 checkpoints bellwether has neither a row nor a manifest for:
  Apertus-70B-Instruct-2509, DeepSeek-R1-0528, functiongemma-270m-it, granite-3.0-8b-instruct,
  granite-20b-functioncalling, Hermes-2-Pro-Llama-3-8B, internlm2_5-7b-chat,
  Llama-3.1-8B-Instruct (and no member of `Llama-3.1-*`), Llama-3.2-3B-Instruct,
  Llama-4-Maverick-17B-128E-Instruct, ToolACE-8B, ultravox-v0_4-ToolACE-8B,
  Llama-xLAM-2-70B-fc-r, and the three Qwen-based xLAM ids, two of which are not public
  repositories. DeepSeek-R1 is not among the eight: the page names R1-0528, not R1, so R1 keeps
  its first reference alone (#27). Adding R1-0528 to the list is how it would be covered.
- **How it is recorded.** For the cases a tool template is for, in every set:
  - render cases whose request carries `tools`: the prompt ids and text the example template gives;
  - parse cases whose request carries `tools` or whose message has tool calls: the round trip with
    the example template, which gives its own output text, ids, pieces, chunk plans and end of turn
    for the same message. The stop ids are the checkpoint's, but which of them the template writes
    where the turn ends, and whether in the turn or opening the next message, is the template's, so
    a second reference from the round trip carries its own `end_of_turn`, as the first does (#43).

  Cases without tools keep their first reference alone, since the pages offer these templates for
  tool calling. Each second reference carries its own provenance: the vLLM commit, the template's
  path and sha256, and the oracle versions.
  - **Groups.** A checkpoint's example templates are oracle inputs like its own files: its manifest
    lists each with its sha256, so two checkpoints share a group only when the pages name the same
    templates for both.
  - **The date.** Four of the named templates write the day's date into the prompt through
    transformers' `strftime_now`: `apertus`, and the three Llama 3 ones unless the request sets
    `date_string`. So does Apertus-8B-Instruct-2509's own template when a request has no system
    message. A prompt that changes with the day can be neither a fixture nor compared across two
    runs, so the recorder renders every reference at one fixed date, written in provenance, and
    vLLM's side gets the same date. How vLLM's side is given it is settled with step 5's recording.
- **Where it lives.** In the case's own line, beside `reference`, as `second_references`, keyed by
  the template's file name, so one line holds every result for its case and the two references are
  compared line by line. A case the checkpoint's own template refuses, or that is not applicable to
  it, still gets a line when a second reference records it; the first reference then holds its
  reason instead of a result. "Proposed case-schema change" below says how.
- **Disagreements.** The first reference and each second reference are compared case by case, and
  each kind of difference is an issue with its fingerprint,
  `vllm-example:<template>:<what differs>:<group>`, the template named without
  `tool_chat_template_` and `.jinja`. What differs is `prompt-ids`, `rejected` (one takes the case,
  the other refuses it), `not-applicable` or `output-text`. The recorder prints each new fingerprint
  with its cases and a draft issue, and opens nothing itself, as it does for vLLM's witnesses.
  A difference does not make the case `disputed`. The two references answer different questions,
  what the checkpoint's own files give and what vLLM's documented setup gives, so neither unsettles
  the other; `disputed` stays the two sources of truth disagreeing on one template.
- **vLLM's side.** When vLLM's witnesses land (step 5), each second reference gets its own: the
  render server started with `--chat-template` pointing at the pinned template file, the tool parser
  its statement gives, and for Mistral the Transformers format's `--tokenizer_mode hf
  --config_format hf --load_format hf`. It is compared with the second reference by the two-sources
  rule, and a disagreement's fingerprint names the template too, `vllm:<what
  differs>:<group>:<template>`.
- **Consumers and `verify`.** Nothing changes by default. smg's tests, Symphony's fixture test and
  `verify` compare with `reference`, because SMG renders with the checkpoint's own template unless
  it is started with another.
  - A line kept only for its second references has no result in `reference`; a consumer skips it
    and counts it apart, as neither a pass nor a failure.
  - `verify --second-reference <template file>` compares with that second reference instead, for an
    SMG started with `--chat-template` at the same file, and its report names the template and its
    sha256. That is the setup of a user who follows vLLM's docs.
  - A parse line's second reference carries the output the template writes, so Symphony can replay
    it with the parser vLLM's docs give for that template. Whether it does is Symphony's call, asked
    in a `for:symphony` issue once the lines exist.

### Not applicable

A checkpoint whose own template renders no tool calls has nothing to say about a parse case with
tool calls, so those cases are counted as not applicable, not as refusals (#59's first proposal,
decided with the second).

- **Detection,** once per group and template, before the cases: a probe with one user message, one
  tool, and an assistant turn that calls it, rendered as it is, with the call renamed, and with one
  more argument, the two changes the round trip already makes per case. When neither change alters
  the rendered turn, the template renders no tool calls. A template that fails on the probe is not
  judged by it; its cases are recorded or refused one by one, as today.
- **Effect:** the template's parse cases with tool calls are not recorded with it. `sets.toml`
  counts them per set as `not_applicable`, beside `rejected`; `record` prints one line per group and
  template with the count and the reason, and they do not make it exit 1. The per-model table shows
  the count with its reason.
- **What stays a refusal:** a template that renders calls but loses one in a case, such as only the
  first of several. That is a defect with its own issue, and the per-case check reports it as
  today. Render cases are unaffected.
- **At the pin,** none of #59's eight groups gets a second reference: six have no tool-call format
  anywhere, and the pages name no template for Hunyuan-A13B or Phi-4-mini. All eight count their
  tool-call parse cases as not applicable, 15,551 each in #59's scale run and 124,408 in all,
  instead of refusing them.

### Proposed case-schema change

A case-schema change waits for the maintainer's approval, and witnesses are still question 4, so this is a
proposal (question 7). It is a separate commit, so the rest of this design can land without it.

- **`second_references`** on render and parse lines: an object keyed by the example template's file
  name. Each entry has `source` (`hf-template` for render and `roundtrip` for parse, the first
  reference's oracles with another template), the result fields the first reference uses for its
  kind (`input_ids` and `text`; for parse `message`, `finish_reason`, `text`, the output's own
  `ids`, `pieces` and `chunk_plans`, and from the round trip `end_of_turn`), and `provenance` with
  `vllm_commit` and the template's `chat_template_sha256`.
- **`rejected` and `not_applicable`** on `reference` and on each entry: the reason it records no
  result, at most one of the two. A first reference with one is written only beside
  `second_references`, and such a parse line carries no top-level `output_ids`, `output_pieces` or
  `chunk_plans`, since those are the first reference's output. #43's rule that a round-trip parse
  reference carries `end_of_turn` binds when the reference records the case.
- **Why an entry beside `reference`, not a witness:** a witness is an engine's result and this is
  the reference oracle's, and what a witness entry holds is still open (question 4), so a second
  reference filed as a witness would wait for it. Separate set files per template were the other
  choice. They would leave every existing line as it is, but store each request twice, need a second
  id for each case, and compare the two references across files.

### Delivery of second references

Each step is its own pull request.

1. This design and the mapping; the case-schema change once the maintainer approves it.
2. Not applicable in `record`: the probe, `not_applicable` in `sets.toml` and `count`, and #59's
   eight groups recorded again.
3. Consumers skip a line whose first reference records no result: `verify`, smg's consumer test,
   and Symphony's fixture test through a `for:symphony` issue. This lands before any such line is
   written.
4. `record` reads the mapping, fetches the pages and templates at the pin and checks their sha256
   as `models` does its registry files, lists each checkpoint's example templates among its oracle
   inputs, records the second references, and prints the fingerprints and draft issues. The fixed
   date comes with it.
5. `verify --second-reference`, for an SMG started with an example template.
6. vLLM's side of each second reference, with step 5 of Delivery.

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
- **The BFCL corpus:** 37.0 MB plain and 1.2 MB packed now, with the multi_turn first turns and
  without repeats; the single-turn sets alone are 17.9 MB plain, 0.8 MB packed.

Decision:

- **Plain in git:** the hand-written sets and the corpus. The corpus is the reviewable input, so a
  change to an importer reads as a case-level diff. An import whose sets pass 50 MB of plain JSON
  Lines in all (`corpus_sets.LIMIT`) stores every one of them in the fixtures' form instead, whole
  (the maintainer, on #45: no sampling). The corpus readers take both forms, and a clone fetches these sets,
  since `record` and the import checks read them.
- **Benchmark fixture sets:** zstd-compressed JSON Lines (`<set>.jsonl.zst`), stored with Git LFS
  in this repository.
- **`sets.toml` per group, committed plain:** every set's case count, rejected count, plain size,
  and the sha256 of its plain content. `count` and the README read it without LFS, and a reviewer
  reads its diff. The weekly check compares plain-content hashes, so it does not depend on the
  compressor writing the same bytes on every machine.
- **`.lfsconfig`:** sets `lfs.fetchexclude` to the benchmark fixture sets, so a default clone
  downloads none of them. `bellwether unpack --model` or `git lfs pull --include` fetches what a
  consumer asks for. Compressed corpus sets are not excluded.

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
  keep working. The organization's allowance is not known here; it is a question for the maintainer.

Why: plain files in git would put gigabytes, then tens of gigabytes, in every checkout. Compression
cuts LFS storage and transfer about 23 times. A clone fetches only the sets it asks for, and there
is no second repository to keep in step. A diff of thousands of fixture lines is not reviewable in
either form. Bulk sets are reviewed through `sets.toml`, by re-recording and comparing plain-content
hashes, and through the per-model table.

## Running vendor code

Some models need code from their own repositories: vendor encoders, custom tokenizers or configs.
The vendor-code oracle runs that code, pinned to the manifest's revision, inside a container that
holds only bellwether and the repository's files, in a step that does not hold the Hugging Face
token. The files are downloaded first, by a step that does.

## Counting

`bellwether count` prints the cases per model, kind and source (the `origin` dataset, or
hand-written) as a table or JSON, read from the `sets.toml` files. The README carries the table, and
`gaps` marks rows by it.

## CI

- **Per pull request:** ruff and pytest. The hand-written fixtures, and the benchmark sets the pull
  request changes, are validated in full; Git LFS fetches only those paths. The importers re-run,
  and the corpus must come out byte-identical; so must the list of models built from the
  registries alone.
- **Weekly:** every group is re-recorded and each set's plain-content sha256 is compared with
  `sets.toml`, which needs no download; one other member of each group is recorded and compared.
- **Credentials:** recording downloads tokenizer files, so CI uses a Hugging Face token stored as
  a repository secret, for rate limits and for gated repositories. The maintainer adds it; no token is ever
  asked for or written here.

## Delivery

Each step is its own pull request.

1. The BFCL importer, its corpus sets, and `count` (#30).
2. Tool-call arguments given to the template exactly as vLLM gives them, with the existing fixtures
   re-recorded byte-identical (#29).
3. The storage form: zstd sets in Git LFS, `sets.toml`, `unpack` and `.lfsconfig`.
4. Tier 1's Hugging Face references, as soon as the storage form is in: groups and manifests, its
   recorded sets, and the per-model table. They are what every consumer reads. vLLM's witnesses
   join the same lines when step 5 lands; until then a line without its witness says so, and counts
   as neither settled nor disputed. GLM-5.3-Flash's parse cases join with step 6.
5. vLLM as the second source, in four pull requests:
   1. What render and derender return: their requests and responses, whether render reports stop
      ids, how per-chunk text is obtained, and derender's reading of the 16 hand-written parse
      cases, the evidence #16 and #17 wait for.
   2. `record --oracle vllm` against the pinned image: render, and derender whole and per chunk.
   3. Witness results and `disputed` in the case schema (question 4).
   4. The comparison: fingerprints and the printed issues.
6. The end of a turn from both sources: GLM-5.3-Flash first, then the other templates the probe
   found.
7. The vendor-code oracle: Kimi-K3, then DeepSeek V3.2 and V4 and Mistral's `mistral-common`
   checkpoints.
8. `bellwether models` and the committed list.
9. Tiers 2 and 3 in batches, with the extra work in tier order: tokenizers that need `tiktoken` or
   custom code, gated models, and a reference for templates that drop the tool list.
10. The next sources, one importer per source in the order of Sources; #19's `tool_choice` cases
    are built from the BFCL import.
11. Second references from vLLM's example tool templates, and not applicable, in the steps of
    "Delivery of second references".

## Questions for the maintainer

1. Answered (2026-10-06): Git LFS goes ahead, and the maintainer raises the organization's allowance if the
   sets pass it.
2. `arguments_verbatim` on parse lines, a case-schema addition (#24).
3. The two scope assumptions: multimodal chat models are in, with text-only cases; embedding,
   reranking and classification models are out.
4. Witness results and `disputed` in the case schema: vLLM's prompt ids for render; its text, whole
   and per chunk, and its message for parse; and a disagreement's fingerprint.
5. The rendered prompt on parse lines (`prompt_ids` and `prompt_text`), a case-schema addition. A
   parser needs to know the state the output starts in: Qwen 3.8's generation prompt leaves the
   think block open, so its output begins `\n</think>`. Symphony may not depend on a tokenizer, so
   it cannot render the prompt itself; the round trip already renders it.
6. Per-token pieces for each render case's prompt ids (`input_pieces`), a case-schema addition. The
   recorded pieces cover output ids only, 8.6 million in the first full-width run; prompt ids would
   put the incremental decoder over 181 million. The reference must be Hugging Face's own
   incremental decode of those ids: joining the decoded pieces does not give back `reference.text`
   for a tokenizer whose normalizer changes the text before encoding.
7. Second references in the case schema: `second_references`, and `rejected` and
   `not_applicable` on a reference ("Proposed case-schema change"). It needs nothing from question
   4: a second reference is not a witness.
8. Templates vLLM ships that no page of its docs names, `tool_chat_template_hunyuan_a13b.jinja` and
   `tool_chat_template_phi4_mini.jinja` among them, the two that started #59. Under the rule as
   decided they make no second reference, and Hunyuan-A13B's line in the tool calling page says its
   chat template is already included in its Hugging Face files. Counting a file name would make
   bellwether choose the checkpoints of the family it names, and counting vLLM's tool-use tests,
   which start Hermes-3-Llama-3.1-8B with `hermes` and Llama-4-Scout with `llama4_json`, would take
   what the docs do not say. Should either count?
