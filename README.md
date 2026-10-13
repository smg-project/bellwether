# bellwether

> Given a request, what's expected tokens. Given a token, what's expected output text. With insane
> amount of data. Then smg uses it in many places. Tokenizer, detokenization, gRPC router, and
> symphony.
>
> The maintainer, 2026-10-06

Bellwether holds expected values for several consumers in smg: the tokenizer crate (encoding and
incremental decoding), the gRPC router's request path, and Symphony's parsers.
`docs/benchmark-sets.md` says how they are recorded and read.

Does SMG send the engine the same prompt tokens the model vendor's own code would, and does it
turn the engine's tokens back into the same response? Bellwether answers that without a GPU.

SMG renders chat messages and tools into token ids before the engine sees them, and detokenizes
and parses the engine's token stream into content, reasoning and tool calls. Both halves are CPU
work with a reference to compare against: the vendor's encoding code or chat template on Hugging
Face, and the Python frontends of vLLM and SGLang. Bellwether records what those references
produce, replays the same cases through a real SMG fronting a scripted mock engine, and
classifies every difference.

The bellwether is the sheep that wears the bell and leads the flock. Here it is the reference each
result is measured against.

## How it works

1. **gaps.** Enumerate the tool parsers, reasoning parsers, renderers and tokenizer modes vLLM and
   SGLang register, map them onto SMG's, and print the matrix. Every row missing an SMG column is a
   parser to write, and `record` hands the implementer fixtures before a line of Rust exists.
2. **record.** Run a request corpus through the references and the engines on CPU and commit the
   results as fixtures: prompt token ids for the render side, parsed responses at several chunk
   plans for the parse side, plus tokenize and incremental detokenize cases.
3. **verify.** Replay fixtures against SMG fronting its `mock-worker`, which captures the token ids
   SMG sent and streams back the recorded output tokens. Compare exactly, classify, apply waivers,
   fail only on SMG regressions.

## Who is right

Two sources of truth, equal in authority (the maintainer, 2026-10-06):

- **Hugging Face:** the checkpoint's own material at a pinned revision. That is the vendor's shipped
  encoder when there is one, else the chat template and tokenizer. For parsing it is the round trip:
  rendering the parsed message back through the template must reproduce the output.
- **vLLM:** at a pinned release, through its GPU-less render server.

When they agree, the case is settled. When they disagree, the case is `disputed`, the disagreement
is an issue, and nothing in bellwether is configured to make them agree. A line's `reference` is
Hugging Face's result and vLLM's is a `witness`. SGLang and engine runs on a GPU are witnesses too,
as evidence without authority. Where vLLM's docs name one of its example tool templates for a
checkpoint, a second reference rendered with that template sits beside the first; a difference
between the two is an issue, not a dispute, since they answer for two setups.
`docs/benchmark-sets.md` has the rules.

Every undisputed case gets a verdict: `match`, `engine_defect` (SMG agrees with the sources of
truth, a witnessing engine does not), `engines_split`, `policy` (malformed output, documented
fallback expected), or `regression` (SMG disagrees with them). A `disputed` case gets none; it waits
for its disagreement's issue. A case that has only its Hugging Face result so far is judged against
it, and becomes settled or disputed once vLLM's result arrives. Only `regression`, an
`engine_defect` without a waiver, and a stale waiver fail the gate. Waivers live in
`waivers/engine_defects.toml`, carry evidence and an upstream link, and expire when the engine
catches up.

## Vocabulary

| Term | Meaning |
|---|---|
| case | one request or one engine output, with everything needed to reproduce it |
| fixture | a case plus the recorded reference result and each witness's result |
| reference | Hugging Face's result: the checkpoint's own material at its pinned revision; one of the two sources of truth |
| second reference | the reference's oracle run with a tool chat template from vLLM's `examples/` that vLLM's docs name for the checkpoint, in place of its own template; recorded beside the reference, one per such template |
| witness | an engine's result; vLLM's, at the pinned release, is the other source of truth, and the rest are evidence, not authority |
| disputed | a case whose two sources of truth disagree; it carries the disagreement's fingerprint, and no parity is counted against it |
| verdict | the classification of one case after comparing SMG with reference and witnesses |
| waiver | a reviewed, expiring record explaining an `engine_defect`, with an upstream link |
| manifest | per-checkpoint file: revision, tier, oracle inputs, group, authority order, SMG and engine parser names |
| oracle inputs | every file the oracle reads for a checkpoint, each with its sha256 |
| checkpoint group | checkpoints with equal oracle inputs, recorded once under the slug of its primary |
| member | any checkpoint of a checkpoint group, its primary included; the other members' manifests name the group |
| primary | the member a checkpoint group is recorded by, under its own slug, once for all its members |
| chunk plan | how an output token stream is cut into engine chunks for a streaming replay |
| capture | the mock worker's record of each request SMG sends it, one JSON line per request |
| known difference | a case where SMG is known to differ from the reference, listed for `verify --known` with the outcome SMG gives on it, the reason and the issue |

## Status

`gaps` (M1, 2026-10-05), `record` for render and parse with the reference oracle (M2, M4), `import`,
`count`, `unpack`, and `verify` on render cases against a running SMG (the first half of M3) are
implemented. `report`, `record` with an engine oracle or for tokenize and detokenize, and `verify` for
the other kinds exist and exit with status 2 until their milestone lands:

| Milestone | Deliverable |
|---|---|
| M1 | `gaps` against the live registries and SMG (done) |
| M2 | render fixtures from the checkpoint template, in the order Symphony needs them: Qwen3-8B and DeepSeek-R1 (done), then DeepSeek-V4.1-Flash, GLM-5.3-Flash, MiniMax-M3, Kimi-K3; engine witnesses on Linux; fixture-driven tests in SMG |
| M3 | `mock-worker --script/--capture` in SMG; `verify` end to end on render cases (against a running SMG: done; starting SMG and the mock per model: next) |
| M4 | parse and detokenize fixtures with chunk plans, the round-trip oracle (Qwen3-8B done; DeepSeek-R1 needs a decision, issue #14), waivers |
| M5 | CI in both repositories; weekly record against engine nightlies; reports to `smg-project/artifacts` |
| M6 | coverage work from the gaps list |

## What bellwether holds

`uv run bellwether count --readme README.md` writes the two tables below from `fixtures/` and `corpus/`, and CI runs it with `--check`, so they say what main holds.

<!-- bellwether count --readme writes everything from here to the end marker: edit none of it by hand -->

### Models

Each checkpoint group is recorded once, by its primary, for every checkpoint that shares its tokenizer and template; the counts are each group's `sets.toml`.

| Group | Model | Checkpoints | Tier | Render cases | Parse cases | Refused | Sources |
|---|---|---:|---:|---:|---:|---:|---|
| [a.x-3.1-light](fixtures/a.x-3.1-light/sets.toml) | skt/A.X-3.1-Light @ 9b41bb24 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [a.x-4.0](fixtures/a.x-4.0/sets.toml) | skt/A.X-4.0 @ a3393fd6 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [a.x-4.0-light](fixtures/a.x-4.0-light/sets.toml) | skt/A.X-4.0-Light @ ba21c20e | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [academic-ds-9b](fixtures/academic-ds-9b/sets.toml) | ByteDance-Seed/academic-ds-9B @ 2a205727 | 1 | 3 | 531736 | 3114 | 332750 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [aceinstruct-1.5b](fixtures/aceinstruct-1.5b/sets.toml) | nvidia/AceInstruct-1.5B @ 1e3d0207 | 3 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [acemath-1.5b-instruct](fixtures/acemath-1.5b-instruct/sets.toml) | nvidia/AceMath-1.5B-Instruct @ 166818c3 | 3 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [acereason-nemotron-1.1-7b](fixtures/acereason-nemotron-1.1-7b/sets.toml) | nvidia/AceReason-Nemotron-1.1-7B @ 2be9ed2f | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [acereason-nemotron-7b](fixtures/acereason-nemotron-7b/sets.toml) | nvidia/AceReason-Nemotron-7B @ 7d385d30 | 4 | 3 | 531736 | 94830 | 329706 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [afm-4.5b](fixtures/afm-4.5b/sets.toml) | arcee-ai/AFM-4.5B @ e84d1971 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [afm-4.5b-preview](fixtures/afm-4.5b-preview/sets.toml) | arcee-ai/AFM-4.5B-Preview @ 5cf715ac | 1 | 3 | 531736 | 328643 | 111787 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [agentcpm-explore](fixtures/agentcpm-explore/sets.toml) | openbmb/AgentCPM-Explore @ f528bbb5 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [agentcpm-report](fixtures/agentcpm-report/sets.toml) | openbmb/AgentCPM-Report @ c6232d64 | 1 | 2 | 531736 | 328660 | 111770 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [aguvis-7b-720p](fixtures/aguvis-7b-720p/sets.toml) | xlangai/Aguvis-7B-720P @ 6dd54127 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ai21-jamba-reasoning-3b](fixtures/ai21-jamba-reasoning-3b/sets.toml) | ai21labs/AI21-Jamba-Reasoning-3B @ 75243704 | 1 | 3 | 531736 | 12885 | 11 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ai21-jamba2-3b](fixtures/ai21-jamba2-3b/sets.toml) | ai21labs/AI21-Jamba2-3B @ 525c6c8e | 2 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [androidgen-llama-3-70b](fixtures/androidgen-llama-3-70b/sets.toml) | zai-org/androidgen-llama-3-70b @ 1c0e09bd | 1 | 3 | 531736 | 328643 | 111787 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [apertus-70b-instruct-2509](fixtures/apertus-70b-instruct-2509/sets.toml) | swiss-ai/Apertus-70B-Instruct-2509 @ 0f276766 | 1 | 3 | 136946 | 137178 | 382850 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [apertus-8b-instruct-2509](fixtures/apertus-8b-instruct-2509/sets.toml) | swiss-ai/Apertus-8B-Instruct-2509 @ b946d404 | 1 | 2 | 136946 | 137178 | 382850 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [apertus-v1.1-4b-instruct](fixtures/apertus-v1.1-4b-instruct/sets.toml) | swiss-ai/Apertus-v1.1-4B-Instruct @ 0d004ae5 | 3 | 2 | 238488 | 184972 | 533009 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [arcee-blitz](fixtures/arcee-blitz/sets.toml) | arcee-ai/Arcee-Blitz @ d8f23453 | 1 | 3 | 238488 | 184972 | 533009 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [arcee-maestro-7b-preview](fixtures/arcee-maestro-7b-preview/sets.toml) | arcee-ai/Arcee-Maestro-7B-Preview @ dc3ee7b5 | 1 | 3 | 531736 | 94830 | 329706 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [arcee-supernova-v1](fixtures/arcee-supernova-v1/sets.toml) | arcee-ai/Arcee-SuperNova-v1 @ 7380333e | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [areal-boba-2-14b](fixtures/areal-boba-2-14b/sets.toml) | inclusionAI/AReaL-boba-2-14B @ 3642aaaa | 1 | 3 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [areal-boba-2-32b](fixtures/areal-boba-2-32b/sets.toml) | inclusionAI/AReaL-boba-2-32B @ 803ae052 | 1 | 3 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [areal-boba-2-8b](fixtures/areal-boba-2-8b/sets.toml) | inclusionAI/AReaL-boba-2-8B @ 88ba93ee | 3 | 3 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [areal-boba-rl-7b](fixtures/areal-boba-rl-7b/sets.toml) | inclusionAI/AReaL-boba-RL-7B @ 5040fc18 | 3 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [areal-boba-sft-32b](fixtures/areal-boba-sft-32b/sets.toml) | inclusionAI/AReaL-boba-SFT-32B @ 85f0f152 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [areal-sea-235b-a22b](fixtures/areal-sea-235b-a22b/sets.toml) | inclusionAI/AReaL-SEA-235B-A22B @ a186e783 | 2 | 2 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [areal-tau2-telecom-sft-30b](fixtures/areal-tau2-telecom-sft-30b/sets.toml) | inclusionAI/AReaL-tau2-telecom-sft-30B @ 41d6b078 | 3 | 2 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [areal-tau2-telecom-user-sft-30b](fixtures/areal-tau2-telecom-user-sft-30b/sets.toml) | inclusionAI/AReaL-tau2-telecom-user-sft-30B @ 5a61a868 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [arex-2](fixtures/arex-2/sets.toml) | BAAI/AREX-2 @ d4e3502f | 1 | 2 | 238478 | 474563 | 349811 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [arex-base](fixtures/arex-base/sets.toml) | BAAI/AREX-Base @ 970a3279 | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [arex-turbo](fixtures/arex-turbo/sets.toml) | BAAI/AREX-Turbo @ e455a798 | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [aria](fixtures/aria/sets.toml) | rhymes-ai/Aria @ 53a09c77 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [asearcher-local-7b](fixtures/asearcher-local-7b/sets.toml) | inclusionAI/ASearcher-Local-7B @ bd1b05b8 | 4 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [asearcher-web-qwq](fixtures/asearcher-web-qwq/sets.toml) | inclusionAI/ASearcher-Web-QwQ @ 6bf738b9 | 2 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [astabrief-8b](fixtures/astabrief-8b/sets.toml) | allenai/AstaBrief_8B @ 6a34f54d | 2 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [atria-dawn-preview](fixtures/atria-dawn-preview/sets.toml) | internlm/Atria-Dawn-Preview @ 311a9b8c | 2 | 2 | 238490 | 503273 | 364862 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [audio-flamingo-3-hf](fixtures/audio-flamingo-3-hf/sets.toml) | nvidia/audio-flamingo-3-hf @ 7d4bae64 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [autoglm-phone-9b](fixtures/autoglm-phone-9b/sets.toml) | zai-org/AutoGLM-Phone-9B @ 66a46ea2 | 1 | 2 | 531736 | 328664 | 111766 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [autoglm-phone-9b-multilingual](fixtures/autoglm-phone-9b-multilingual/sets.toml) | zai-org/AutoGLM-Phone-9B-Multilingual @ 832ab5e0 | 1 | 2 | 531736 | 328664 | 111766 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [baichuan-m3-235b](fixtures/baichuan-m3-235b/sets.toml) | baichuan-inc/Baichuan-M3-235B @ 03a1fd5e | 1 | 2 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [baichuanmed-ocr-7b](fixtures/baichuanmed-ocr-7b/sets.toml) | baichuan-inc/BaichuanMed-OCR-7B @ 826944c7 | 2 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [bar-2x7b-base](fixtures/bar-2x7b-base/sets.toml) | allenai/BAR-2x7B-Base @ 80987e28 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [bar-2x7b-code](fixtures/bar-2x7b-code/sets.toml) | allenai/BAR-2x7B-Code @ dc5c4003 | 5 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [bar-2x7b-math](fixtures/bar-2x7b-math/sets.toml) | allenai/BAR-2x7B-Math @ 772eb58b | 2 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [bar-7b](fixtures/bar-7b/sets.toml) | allenai/BAR-7B @ bee49db0 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [bee-8b-sft](fixtures/bee-8b-sft/sets.toml) | Open-Bee/Bee-8B-SFT @ a81758f4 | 3 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [bfs-prover-v1-7b](fixtures/bfs-prover-v1-7b/sets.toml) | ByteDance-Seed/BFS-Prover-V1-7B @ 750e3903 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [bitcpm-cann-3b-unquantized](fixtures/bitcpm-cann-3b-unquantized/sets.toml) | openbmb/BitCPM-CANN-3B-unquantized @ 6971d179 | 4 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [bytedance-seed-oss-36b-alternate-tokenizer](fixtures/bytedance-seed-oss-36b-alternate-tokenizer/sets.toml) | NousResearch/ByteDance-Seed-OSS-36B-Alternate-Tokenizer @ e82f3d5c | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [caller](fixtures/caller/sets.toml) | arcee-ai/Caller @ 2e71e25e | 1 | 3 | 531736 | 324160 | 142324 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [caprl-3b](fixtures/caprl-3b/sets.toml) | internlm/CapRL-3B @ f83373e8 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [caprl-eval-3b](fixtures/caprl-eval-3b/sets.toml) | internlm/CapRL-Eval-3B @ 484b8ca3 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [caprl-internvl3.5-8b](fixtures/caprl-internvl3.5-8b/sets.toml) | internlm/CapRL-InternVL3.5-8B @ e7ebc6db | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [caprl-qwen3vl-4b](fixtures/caprl-qwen3vl-4b/sets.toml) | internlm/CapRL-Qwen3VL-4B @ 1db1c1dd | 2 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [caprl-video-4b](fixtures/caprl-video-4b/sets.toml) | internlm/CapRL-Video-4B @ d932a039 | 2 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [cogalign-internvl2-5-mpo-4b](fixtures/cogalign-internvl2-5-mpo-4b/sets.toml) | Salesforce/cogalign-internvl2_5-mpo-4b @ 2860db15 | 2 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [cogalign-llava-ov-0-5b](fixtures/cogalign-llava-ov-0-5b/sets.toml) | Salesforce/cogalign-llava-ov-0_5b @ a7298047 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [command-a-plus-05-2026-bf16](fixtures/command-a-plus-05-2026-bf16/sets.toml) | CohereLabs/command-a-plus-05-2026-bf16 @ 5fb6fde5 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [contextpilot-14b](fixtures/contextpilot-14b/sets.toml) | tencent/ContextPilot-14B @ 8eafd835 | 8 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [cosmos-reason1-7b](fixtures/cosmos-reason1-7b/sets.toml) | nvidia/Cosmos-Reason1-7B @ 375e2400 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [cosmos3-edge](fixtures/cosmos3-edge/sets.toml) | nvidia/Cosmos3-Edge @ 344d602b | 1 | 2 | 238489 | 12829 | 278544 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [cosmos3-edge-policy-droid](fixtures/cosmos3-edge-policy-droid/sets.toml) | nvidia/Cosmos3-Edge-Policy-DROID @ a7c7288f | 1 | 2 | 238489 | 12829 | 278544 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [cosmos3-super](fixtures/cosmos3-super/sets.toml) | nvidia/Cosmos3-Super @ f543c562 | 7 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [crispers-7b-v1](fixtures/crispers-7b-v1/sets.toml) | thu-coai/Crispers-7B-v1 @ 9dfb2cea | 2 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [d1-3b](fixtures/d1-3b/sets.toml) | LiquidAI/d1-3B @ 051bcc46 | 1 | 2 | 243122 | 589657 | 273846 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deep-ignorance-unfiltered-instruct-test](fixtures/deep-ignorance-unfiltered-instruct-test/sets.toml) | EleutherAI/deep-ignorance-unfiltered-instruct-test @ 1e498ea5 | 5 | 2 | 531736 | 273678 | 166752 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deephermes-financial-fundamentals-prediction-specialist-atropos](fixtures/deephermes-financial-fundamentals-prediction-specialist-atropos/sets.toml) | NousResearch/DeepHermes-Financial-Fundamentals-Prediction-Specialist-Atropos @ 31aa97fb | 5 | 3 | 531736 | 137178 | 199366 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [deepseek-r1](fixtures/deepseek-r1/sets.toml) | deepseek-ai/DeepSeek-R1 @ 56d4cbbb | 2 | 3 | 493319 | 0 | 1 | glaive-v2, hand-written, swehero, tau2 |
| [deepseek-r1-bf16](fixtures/deepseek-r1-bf16/sets.toml) | arcee-ai/DeepSeek-R1-bf16 @ c5eb2865 | 1 | 3 | 531735 | 3110 | 332755 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deepseek-v3-0324](fixtures/deepseek-v3-0324/sets.toml) | deepseek-ai/DeepSeek-V3-0324 @ e9b33add | 2 | 3 | 531735 | 19299 | 328694 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deepseek-v3-0324-bf16](fixtures/deepseek-v3-0324-bf16/sets.toml) | arcee-ai/DeepSeek-V3-0324-bf16 @ 0fcb3b10 | 1 | 3 | 531735 | 3110 | 332755 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deepseek-v3.1](fixtures/deepseek-v3.1/sets.toml) | deepseek-ai/DeepSeek-V3.1 @ c0781d03 | 3 | 3 | 531735 | 19299 | 328694 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deepseek-v3.2-exp](fixtures/deepseek-v3.2-exp/sets.toml) | deepseek-ai/DeepSeek-V3.2-Exp @ 194c67e1 | 2 | 3 | 531735 | 184972 | 254532 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [deepseek-v4.1-flash](fixtures/deepseek-v4.1-flash/sets.toml) | deepseek-ai/DeepSeek-V4.1-Flash @ 2cba9e42 | 1 | 1 | 531735 | 589659 | 1 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [devstral-2-123b-instruct-2512](fixtures/devstral-2-123b-instruct-2512/sets.toml) | mistralai/Devstral-2-123B-Instruct-2512 @ 1613bf01 | 1 | 2 | 531730 | 570194 | 6591 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [dler-llama-nemotron-8b-merge-research](fixtures/dler-llama-nemotron-8b-merge-research/sets.toml) | nvidia/DLER-Llama-Nemotron-8B-Merge-Research @ 029e0043 | 1 | 3 | 471553 | 416850 | 176459 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [dler-r1-1.5b-research](fixtures/dler-r1-1.5b-research/sets.toml) | nvidia/DLER-R1-1.5B-Research @ d0523143 | 2 | 3 | 531736 | 94830 | 329706 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [dogr](fixtures/dogr/sets.toml) | tencent/DOGR @ ee374bd5 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [dots.mocr](fixtures/dots.mocr/sets.toml) | dots-studio/dots.mocr @ e539fbb5 | 2 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [dots3-note-prev](fixtures/dots3-note-prev/sets.toml) | dots-studio/dots3-note-prev @ 1e1e7b0c | 1 | 3 | 238490 | 589296 | 278839 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [dse-qwen-2.5-sota](fixtures/dse-qwen-2.5-sota/sets.toml) | lightonai/DSE_Qwen_2.5_SOTA @ 70fdc18b | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [e1-acereason-14b](fixtures/e1-acereason-14b/sets.toml) | Salesforce/E1-AceReason-14B @ 54537936 | 1 | 3 | 531736 | 94830 | 329706 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [e1-code-14b](fixtures/e1-code-14b/sets.toml) | Salesforce/E1-Code-14B @ 9eebf1fb | 1 | 3 | 531736 | 94830 | 329706 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [egm-4b-sft](fixtures/egm-4b-sft/sets.toml) | nvidia/EGM-4B-SFT @ 60cfbf34 | 2 | 2 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ernie-4.5-21b-a3b-thinking](fixtures/ernie-4.5-21b-a3b-thinking/sets.toml) | baidu/ERNIE-4.5-21B-A3B-Thinking @ 4341bb42 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ernie-image-aes](fixtures/ernie-image-aes/sets.toml) | baidu/ERNIE-Image-Aes @ 8a0bb5ea | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [evisrag-7b](fixtures/evisrag-7b/sets.toml) | openbmb/EVisRAG-7B @ a3a1f464 | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [exaone-4.5-33b](fixtures/exaone-4.5-33b/sets.toml) | LGAI-EXAONE/EXAONE-4.5-33B @ 570aa4b1 | 1 | 2 | 531735 | 503634 | 86026 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-e-3b-base-prequantized](fixtures/falcon-e-3b-base-prequantized/sets.toml) | tiiuae/Falcon-E-3B-Base-prequantized @ 5feb4d43 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [falcon-h1-tiny-90m-instruct](fixtures/falcon-h1-tiny-90m-instruct/sets.toml) | tiiuae/Falcon-H1-Tiny-90M-Instruct @ e6389502 | 2 | 2 | 531736 | 429360 | 140619 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-h1-tiny-multilingual-100m-instruct](fixtures/falcon-h1-tiny-multilingual-100m-instruct/sets.toml) | tiiuae/Falcon-H1-Tiny-Multilingual-100M-Instruct @ 295b1eb9 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-h1-tiny-r-0.6b](fixtures/falcon-h1-tiny-r-0.6b/sets.toml) | tiiuae/Falcon-H1-Tiny-R-0.6B @ ccf3d617 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-h1-tiny-r-0.6b-pre-grpo](fixtures/falcon-h1-tiny-r-0.6b-pre-grpo/sets.toml) | tiiuae/Falcon-H1-Tiny-R-0.6B-pre-GRPO @ 24ec8732 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-h1-tiny-r-90m](fixtures/falcon-h1-tiny-r-90m/sets.toml) | tiiuae/Falcon-H1-Tiny-R-90M @ 7385612b | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [falcon-h1-tiny-tool-calling-90m](fixtures/falcon-h1-tiny-tool-calling-90m/sets.toml) | tiiuae/Falcon-H1-Tiny-Tool-Calling-90M @ fc099d79 | 2 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [falcon-h1r-7b](fixtures/falcon-h1r-7b/sets.toml) | tiiuae/Falcon-H1R-7B @ a6f74bf1 | 1 | 2 | 531736 | 442119 | 140740 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [fara1.5-27b](fixtures/fara1.5-27b/sets.toml) | microsoft/Fara1.5-27B @ 299c8406 | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [fara1.5-9b](fixtures/fara1.5-9b/sets.toml) | microsoft/Fara1.5-9B @ 1a93677c | 2 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [flex-public-7b-1t](fixtures/flex-public-7b-1t/sets.toml) | allenai/Flex-public-7B-1T @ f78a73be | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [flexolmo-7x7b-1t-rt](fixtures/flexolmo-7x7b-1t-rt/sets.toml) | allenai/FlexOlmo-7x7B-1T-RT @ 193b1e91 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [fun-asr-nano-2512-vllm](fixtures/fun-asr-nano-2512-vllm/sets.toml) | allendou/Fun-ASR-Nano-2512-vllm @ e718b36e | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [fun-audio-chat-8b](fixtures/fun-audio-chat-8b/sets.toml) | FunAudioLLM/Fun-Audio-Chat-8B @ 7bf72dc7 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [functiongemma-270m-it](fixtures/functiongemma-270m-it/sets.toml) | google/functiongemma-270m-it @ 39eccb09 | 1 | 2 | 238483 | 246750 | 499217 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [game-rl-internvl3-8b](fixtures/game-rl-internvl3-8b/sets.toml) | OpenMOSS-Team/Game-RL-InternVL3-8B @ defc1e86 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [game-rl-llava-ov-7b](fixtures/game-rl-llava-ov-7b/sets.toml) | OpenMOSS-Team/Game-RL-LLaVA-OV-7B @ 464e45f0 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [game-rl-qwen2.5-vl-7b](fixtures/game-rl-qwen2.5-vl-7b/sets.toml) | OpenMOSS-Team/Game-RL-Qwen2.5-VL-7B @ 205b5934 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [gelab-zero-4b-preview](fixtures/gelab-zero-4b-preview/sets.toml) | stepfun-ai/GELab-Zero-4B-preview @ 838a527e | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [gemma-3-12b-it-qat-q4-0-unquantized](fixtures/gemma-3-12b-it-qat-q4-0-unquantized/sets.toml) | google/gemma-3-12b-it-qat-q4_0-unquantized @ 68f7ee4f | 3 | 3 | 238484 | 184969 | 533016 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [gemma-3-1b-it](fixtures/gemma-3-1b-it/sets.toml) | google/gemma-3-1b-it @ dcc83ea8 | 1 | 3 | 238484 | 184969 | 533016 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [gemma-3-4b-it](fixtures/gemma-3-4b-it/sets.toml) | google/gemma-3-4b-it @ 093f9f38 | 3 | 3 | 238484 | 184969 | 533016 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [gemma-4-12b-it](fixtures/gemma-4-12b-it/sets.toml) | google/gemma-4-12B-it @ 707f0a3b | 2 | 2 | 238484 | 143685 | 689711 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [gemma-4-12b-it-qat-q4-0-unquantized-assistant](fixtures/gemma-4-12b-it-qat-q4-0-unquantized-assistant/sets.toml) | google/gemma-4-12B-it-qat-q4_0-unquantized-assistant @ 18934064 | 1 | 2 | 238484 | 143685 | 689711 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [gemma-4-26b-a4b-it](fixtures/gemma-4-26b-a4b-it/sets.toml) | google/gemma-4-26B-A4B-it @ 4d7ae498 | 2 | 2 | 238484 | 143685 | 689711 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [gemma-4-31b-it-qat-q4-0-unquantized](fixtures/gemma-4-31b-it-qat-q4-0-unquantized/sets.toml) | google/gemma-4-31B-it-qat-q4_0-unquantized @ 1e4d8bee | 2 | 2 | 238484 | 143685 | 689711 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [gemma-4-31b-it-qat-q4-0-unquantized-assistant](fixtures/gemma-4-31b-it-qat-q4-0-unquantized-assistant/sets.toml) | google/gemma-4-31B-it-qat-q4_0-unquantized-assistant @ 96d4c8ca | 2 | 2 | 238484 | 143685 | 689711 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [gemma-4-e2b-it-qat-q4-0-unquantized](fixtures/gemma-4-e2b-it-qat-q4-0-unquantized/sets.toml) | google/gemma-4-E2B-it-qat-q4_0-unquantized @ 6befbaca | 2 | 2 | 238484 | 478241 | 389900 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [gemma-4-e4b-it](fixtures/gemma-4-e4b-it/sets.toml) | google/gemma-4-E4B-it @ ee0ef602 | 2 | 2 | 238484 | 478241 | 389900 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4.5v](fixtures/glm-4.5v/sets.toml) | zai-org/GLM-4.5V @ ed47433b | 1 | 3 | 238490 | 492389 | 364866 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4.6](fixtures/glm-4.6/sets.toml) | zai-org/GLM-4.6 @ be721948 | 3 | 2 | 238490 | 492389 | 364866 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4.6v](fixtures/glm-4.6v/sets.toml) | zai-org/GLM-4.6V @ 4e2d47eb | 1 | 2 | 238490 | 492389 | 364866 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4.6v-flash](fixtures/glm-4.6v-flash/sets.toml) | zai-org/GLM-4.6V-Flash @ 411bb4d7 | 1 | 2 | 238490 | 492389 | 364866 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4.7](fixtures/glm-4.7/sets.toml) | zai-org/GLM-4.7 @ 602d01ef | 1 | 2 | 238490 | 1947 | 278545 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4.7-flash](fixtures/glm-4.7-flash/sets.toml) | zai-org/GLM-4.7-Flash @ 7dd20894 | 1 | 2 | 238490 | 1947 | 278545 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-5](fixtures/glm-5/sets.toml) | zai-org/GLM-5 @ c183ef8c | 1 | 2 | 238490 | 1947 | 278545 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-5.1](fixtures/glm-5.1/sets.toml) | zai-org/GLM-5.1 @ 26e1bd6e | 1 | 2 | 238490 | 163746 | 677804 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-5.2](fixtures/glm-5.2/sets.toml) | zai-org/GLM-5.2 @ cf457fa7 | 1 | 2 | 238490 | 492389 | 364866 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-5.3](fixtures/glm-5.3/sets.toml) | zai-org/GLM-5.3 @ aca966e4 | 2 | 2 | 238490 | 503273 | 364862 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-5.3-flash](fixtures/glm-5.3-flash/sets.toml) | zai-org/GLM-5.3-Flash @ eb9eb208 | 2 | 1 | 238490 | 503273 | 364862 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-asr-nano-2512](fixtures/glm-asr-nano-2512/sets.toml) | zai-org/GLM-ASR-Nano-2512 @ 61ba4e0b | 1 | 2 | 531736 | 328643 | 111787 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-ocr](fixtures/glm-ocr/sets.toml) | zai-org/GLM-OCR @ 2e85a628 | 1 | 2 | 238490 | 328645 | 390261 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glyph](fixtures/glyph/sets.toml) | zai-org/Glyph @ 074a0727 | 1 | 2 | 238490 | 492389 | 364866 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-3.2-8b-instruct-preview](fixtures/granite-3.2-8b-instruct-preview/sets.toml) | ibm-granite/granite-3.2-8b-instruct-preview @ 191a764d | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-3.3-8b-math-prm-v2](fixtures/granite-3.3-8b-math-prm-v2/sets.toml) | ibm-granite/granite-3.3-8b-math-prm-v2 @ 301abdda | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-4.0-1b-speech](fixtures/granite-4.0-1b-speech/sets.toml) | ibm-granite/granite-4.0-1b-speech @ bd87ab86 | 2 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [granite-4.0-3b-vision](fixtures/granite-4.0-3b-vision/sets.toml) | ibm-granite/granite-4.0-3b-vision @ bf108f36 | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-4.0-h-3b-ar](fixtures/granite-4.0-h-3b-ar/sets.toml) | ibm-research/granite-4.0-h-3b-ar @ b32260f6 | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-4.1-3b](fixtures/granite-4.1-3b/sets.toml) | ibm-granite/granite-4.1-3b @ c0650403 | 2 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-guardian-4.1-8b](fixtures/granite-guardian-4.1-8b/sets.toml) | ibm-granite/granite-guardian-4.1-8b @ ab01ccca | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-speech-3.2-8b](fixtures/granite-speech-3.2-8b/sets.toml) | ibm-granite/granite-speech-3.2-8b @ d5835026 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-speech-4.1-2b-plus](fixtures/granite-speech-4.1-2b-plus/sets.toml) | ibm-granite/granite-speech-4.1-2b-plus @ 1454e6e1 | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-vision-3.1-2b-preview](fixtures/granite-vision-3.1-2b-preview/sets.toml) | ibm-granite/granite-vision-3.1-2b-preview @ bd1c154e | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-vision-3.2-2b](fixtures/granite-vision-3.2-2b/sets.toml) | ibm-granite/granite-vision-3.2-2b @ ec894191 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-vision-3.3-2b](fixtures/granite-vision-3.3-2b/sets.toml) | ibm-granite/granite-vision-3.3-2b @ bf66a824 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-vision-3.3-2b-chart2csv-preview](fixtures/granite-vision-3.3-2b-chart2csv-preview/sets.toml) | ibm-granite/granite-vision-3.3-2b-chart2csv-preview @ 298ed4d8 | 1 | 2 | 498897 | 328662 | 144607 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-vision-4.1-4b](fixtures/granite-vision-4.1-4b/sets.toml) | ibm-granite/granite-vision-4.1-4b @ 37d591f0 | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [graspmolmo](fixtures/graspmolmo/sets.toml) | allenai/GraspMolmo @ 9aa184d6 | 1 | 3 | 86316 | 0 | 238455 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm |
| [h2o-lightning-4b](fixtures/h2o-lightning-4b/sets.toml) | h2oai/h2o-lightning-4b @ 672dc01e | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [harc-llama-3.1-8b-instruct](fixtures/harc-llama-3.1-8b-instruct/sets.toml) | microsoft/HARC-Llama-3.1-8B-Instruct @ c6f25949 | 1 | 2 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [harc-qwen2.5-7b-instruct](fixtures/harc-qwen2.5-7b-instruct/sets.toml) | microsoft/HARC-Qwen2.5-7B-Instruct @ f650ffae | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hermes-3-llama-3.1-8b](fixtures/hermes-3-llama-3.1-8b/sets.toml) | NousResearch/Hermes-3-Llama-3.1-8B @ 896ea440 | 1 | 3 | 531736 | 324160 | 142324 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hermes-4-14b](fixtures/hermes-4-14b/sets.toml) | NousResearch/Hermes-4-14B @ d6ce765c | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hermes-4.3-36b](fixtures/hermes-4.3-36b/sets.toml) | NousResearch/Hermes-4.3-36B @ 3899db2b | 1 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hpd-parsing](fixtures/hpd-parsing/sets.toml) | PaddlePaddle/HPD-Parsing @ 91de8005 | 1 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [huatuogpt-3-27b](fixtures/huatuogpt-3-27b/sets.toml) | FreedomIntelligence/HuatuoGPT-3-27B @ e759ddae | 1 | 2 | 238478 | 474563 | 349811 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [huatuogpt-3-8b](fixtures/huatuogpt-3-8b/sets.toml) | FreedomIntelligence/HuatuoGPT-3-8B @ 507f63a1 | 1 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [huatuogpt-3-9b](fixtures/huatuogpt-3-9b/sets.toml) | FreedomIntelligence/HuatuoGPT-3-9B @ 6870b60e | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [huatuogpt-3-grader-8b](fixtures/huatuogpt-3-grader-8b/sets.toml) | FreedomIntelligence/HuatuoGPT-3-Grader-8B @ 8b17b10d | 1 | 2 | 531736 | 541389 | 6580 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hunyuan-7b-instruct](fixtures/hunyuan-7b-instruct/sets.toml) | tencent/Hunyuan-7B-Instruct @ 6fd6ecb0 | 1 | 3 | 531735 | 328662 | 111769 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hunyuan-a13b-instruct](fixtures/hunyuan-a13b-instruct/sets.toml) | tencent/Hunyuan-A13B-Instruct @ 290ddb9a | 1 | 3 | 531735 | 328662 | 111769 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hunyuan-mt-7b](fixtures/hunyuan-mt-7b/sets.toml) | tencent/Hunyuan-MT-7B @ 9305c783 | 2 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hunyuanocr](fixtures/hunyuanocr/sets.toml) | tencent/HunyuanOCR @ 47644ecc | 1 | 2 | 531736 | 328664 | 111766 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hy-mt2-1.8b](fixtures/hy-mt2-1.8b/sets.toml) | tencent/Hy-MT2-1.8B @ 9a341cd1 | 4 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [hy-mt2-30b-a3b](fixtures/hy-mt2-30b-a3b/sets.toml) | tencent/Hy-MT2-30B-A3B @ d3ead4db | 1 | 2 | 238490 | 569901 | 285354 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hy-mt2-7b](fixtures/hy-mt2-7b/sets.toml) | tencent/Hy-MT2-7B @ 9b0eb4e8 | 2 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hy3](fixtures/hy3/sets.toml) | tencent/Hy3 @ a960ebc3 | 1 | 2 | 238490 | 569901 | 285354 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hy3-preview-base](fixtures/hy3-preview-base/sets.toml) | tencent/Hy3-preview-Base @ 54a62bb0 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [hy4-preview](fixtures/hy4-preview/sets.toml) | tencent/Hy4-preview @ 705d81ee | 1 | 1 | 238489 | 589298 | 278838 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hyperclovax-seed-omni-8b](fixtures/hyperclovax-seed-omni-8b/sets.toml) | naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B @ f1240321 | 1 | 2 | 238490 | 582728 | 285407 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hyperclovax-seed-text-instruct-0.5b](fixtures/hyperclovax-seed-text-instruct-0.5b/sets.toml) | naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B @ 3da5046f | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [idefics3-8b-llama3](fixtures/idefics3-8b-llama3/sets.toml) | HuggingFaceM4/Idefics3-8B-Llama3 @ fddb4ff7 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [inkling](fixtures/inkling/sets.toml) | thinkingmachines/Inkling @ 828496ee | 2 | 3 | 238489 | 589659 | 278477 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [intent-aware-lfqa-llama3-8b-intent-explicit](fixtures/intent-aware-lfqa-llama3-8b-intent-explicit/sets.toml) | allenai/intent-aware-lfqa-llama3-8b-intent-explicit @ fdf98cd4 | 4 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [intent-aware-lfqa-qwen3-4b-intent-explicit](fixtures/intent-aware-lfqa-qwen3-4b-intent-explicit/sets.toml) | allenai/intent-aware-lfqa-qwen3-4b-intent-explicit @ 07bd65c9 | 8 | 2 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [intern-decision-0.8b](fixtures/intern-decision-0.8b/sets.toml) | internlm/Intern-Decision-0.8B @ 85a0cc5a | 2 | 2 | 238478 | 461722 | 349772 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [intern-decision-4b](fixtures/intern-decision-4b/sets.toml) | internlm/Intern-Decision-4B @ 0e5e6aa7 | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [internvl3-1b-instruct](fixtures/internvl3-1b-instruct/sets.toml) | OpenGVLab/InternVL3-1B-Instruct @ f91c6391 | 12 | 3 | 199970 | 145044 | 320255 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes |
| [internvl3-38b](fixtures/internvl3-38b/sets.toml) | OpenGVLab/InternVL3-38B @ b99bcba4 | 2 | 3 | 199970 | 145044 | 320255 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes |
| [iquest-q1](fixtures/iquest-q1/sets.toml) | IQuestLab/IQuest-Q1 @ 5c21b063 | 1 | 2 | 238490 | 503273 | 364862 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ising-calibration-1-35b-a3b](fixtures/ising-calibration-1-35b-a3b/sets.toml) | nvidia/Ising-Calibration-1-35B-A3B @ b07b7d22 | 1 | 2 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ising-calibration-1.5-31b-bf16](fixtures/ising-calibration-1.5-31b-bf16/sets.toml) | nvidia/Ising-Calibration-1.5-31B-BF16 @ b9b4f513 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [januscoder-14b](fixtures/januscoder-14b/sets.toml) | internlm/JanusCoder-14B @ 28487c47 | 2 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [jedi-3b-1080p](fixtures/jedi-3b-1080p/sets.toml) | xlangai/Jedi-3B-1080p @ 4aef56c6 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [jedi-7b-1080p](fixtures/jedi-7b-1080p/sets.toml) | xlangai/Jedi-7B-1080p @ 7ccf6e40 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [jet-nemotron-2b](fixtures/jet-nemotron-2b/sets.toml) | jet-ai/Jet-Nemotron-2B @ afd4ee28 | 2 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [k-exaone-236b-a23b](fixtures/k-exaone-236b-a23b/sets.toml) | LGAI-EXAONE/K-EXAONE-236B-A23B @ 61e6d578 | 1 | 2 | 531735 | 503634 | 86026 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [k2-horizon-36b](fixtures/k2-horizon-36b/sets.toml) | IFM/K2-Horizon-36B @ cca48b66 | 3 | 2 | 70766 | 12814 | 254087 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes |
| [k2-think](fixtures/k2-think/sets.toml) | IFM/K2-Think @ 9e218ac1 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [k2-think-v2](fixtures/k2-think-v2/sets.toml) | IFM/K2-Think-V2 @ f86e5d11 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [k2-v2-instruct](fixtures/k2-v2-instruct/sets.toml) | IFM/K2-V2-Instruct @ 6ef3e31c | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [kanana-1.5-15.7b-a3b-base](fixtures/kanana-1.5-15.7b-a3b-base/sets.toml) | kakaocorp/kanana-1.5-15.7b-a3b-base @ c9f2bbfb | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [kanana-2-30b-a3b-thinking-2601](fixtures/kanana-2-30b-a3b-thinking-2601/sets.toml) | kakaocorp/kanana-2-30b-a3b-thinking-2601 @ cdcbf8b8 | 2 | 2 | 531736 | 12885 | 11 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [kanana-2-3b-instruct](fixtures/kanana-2-3b-instruct/sets.toml) | kakaocorp/kanana-2-3b-instruct @ 6a5d7889 | 1 | 2 | 502928 | 461740 | 86231 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [kanana-safeguard-siren-8b](fixtures/kanana-safeguard-siren-8b/sets.toml) | kakaocorp/kanana-safeguard-siren-8b @ dc65aab3 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [kernelllm](fixtures/kernelllm/sets.toml) | facebook/KernelLLM @ 26d1191a | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [kimi-k3](fixtures/kimi-k3/sets.toml) | moonshotai/Kimi-K3 @ f831ab66 | 1 | 1 | 38439 | 54418 | 1 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [laguna-s-2.1](fixtures/laguna-s-2.1/sets.toml) | poolside/Laguna-S-2.1 @ 0f573140 | 1 | 2 | 238490 | 589298 | 278837 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [laguna-xs-2.1](fixtures/laguna-xs-2.1/sets.toml) | poolside/Laguna-XS-2.1 @ c5f36269 | 1 | 2 | 238490 | 576475 | 278780 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [laguna-xs.2](fixtures/laguna-xs.2/sets.toml) | poolside/Laguna-XS.2 @ a397bde0 | 1 | 3 | 238490 | 490452 | 364803 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-1.2b-instruct](fixtures/lfm2.5-1.2b-instruct/sets.toml) | LiquidAI/LFM2.5-1.2B-Instruct @ 0f604ada | 2 | 2 | 243122 | 589657 | 273846 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ling-3.0-flash](fixtures/ling-3.0-flash/sets.toml) | inclusionAI/Ling-3.0-flash @ ef06d91f | 2 | 3 | 238490 | 589298 | 278837 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ling-coder-lite](fixtures/ling-coder-lite/sets.toml) | inclusionAI/Ling-Coder-lite @ e83c3b87 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ling-lite-1.5-2507](fixtures/ling-lite-1.5-2507/sets.toml) | inclusionAI/Ling-lite-1.5-2507 @ 6656efdc | 2 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ling-lite-base-1.5](fixtures/ling-lite-base-1.5/sets.toml) | inclusionAI/Ling-lite-base-1.5 @ 69500a0d | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ling-mini-base-2.0](fixtures/ling-mini-base-2.0/sets.toml) | inclusionAI/Ling-mini-base-2.0 @ 1b53ca4c | 6 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [ling-plus](fixtures/ling-plus/sets.toml) | inclusionAI/Ling-plus @ a31ce668 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [llama-3-1-nemotron-ultra-253b-cpt-v1](fixtures/llama-3-1-nemotron-ultra-253b-cpt-v1/sets.toml) | nvidia/Llama-3_1-Nemotron-Ultra-253B-CPT-v1 @ 00e37b5d | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3-1-nemotron-ultra-253b-v1](fixtures/llama-3-1-nemotron-ultra-253b-v1/sets.toml) | nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 @ 5b47def5 | 1 | 3 | 531736 | 328643 | 111787 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3-3-nemotron-super-49b-genrm](fixtures/llama-3-3-nemotron-super-49b-genrm/sets.toml) | nvidia/Llama-3_3-Nemotron-Super-49B-GenRM @ 31609793 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [llama-3-3-nemotron-super-49b-genrm-multilingual](fixtures/llama-3-3-nemotron-super-49b-genrm-multilingual/sets.toml) | nvidia/Llama-3_3-Nemotron-Super-49B-GenRM-Multilingual @ 334fda42 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [llama-3-3-nemotron-super-49b-v1-5](fixtures/llama-3-3-nemotron-super-49b-v1-5/sets.toml) | nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 @ 420ba7d2 | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.1-nemotron-8b-ultralong-1m-instruct](fixtures/llama-3.1-nemotron-8b-ultralong-1m-instruct/sets.toml) | nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct @ fa3a10b9 | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.1-nemotron-8b-ultralong-2m-instruct](fixtures/llama-3.1-nemotron-8b-ultralong-2m-instruct/sets.toml) | nvidia/Llama-3.1-Nemotron-8B-UltraLong-2M-Instruct @ 243daed3 | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.1-nemotron-8b-ultralong-4m-instruct](fixtures/llama-3.1-nemotron-8b-ultralong-4m-instruct/sets.toml) | nvidia/Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct @ 02ae0431 | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.1-nemotron-nano-4b-v1.1](fixtures/llama-3.1-nemotron-nano-4b-v1.1/sets.toml) | nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1 @ d552708a | 1 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.1-nemotron-nano-8b-v1](fixtures/llama-3.1-nemotron-nano-8b-v1/sets.toml) | nvidia/Llama-3.1-Nemotron-Nano-8B-v1 @ 54641c16 | 1 | 3 | 471553 | 416850 | 176459 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.1-nemotron-nano-vl-8b-v1-fp4-qad](fixtures/llama-3.1-nemotron-nano-vl-8b-v1-fp4-qad/sets.toml) | nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1-FP4-QAD @ aefd101d | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.1-nemotron-safety-guard-8b-v3](fixtures/llama-3.1-nemotron-safety-guard-8b-v3/sets.toml) | nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3 @ 8fdc246b | 1 | 3 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-3.3-nemotron-70b-reward](fixtures/llama-3.3-nemotron-70b-reward/sets.toml) | nvidia/Llama-3.3-Nemotron-70B-Reward @ 2500cf14 | 3 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [llama-3.3-nemotron-70b-reward-principle](fixtures/llama-3.3-nemotron-70b-reward-principle/sets.toml) | nvidia/Llama-3.3-Nemotron-70B-Reward-Principle @ ac3e237f | 3 | 2 | 508776 | 458185 | 140114 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-xlam-2-8b-fc-r](fixtures/llama-xlam-2-8b-fc-r/sets.toml) | Salesforce/Llama-xLAM-2-8b-fc-r @ a0efe39a | 1 | 3 | 531736 | 483062 | 92717 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llava-1.5-7b-hf](fixtures/llava-1.5-7b-hf/sets.toml) | llava-hf/llava-1.5-7b-hf @ b234b804 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [medgemma-27b-it](fixtures/medgemma-27b-it/sets.toml) | google/medgemma-27b-it @ 2d3e00ea | 1 | 3 | 238484 | 184969 | 533016 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mimo-v2.5](fixtures/mimo-v2.5/sets.toml) | XiaomiMiMo/MiMo-V2.5 @ 63651580 | 2 | 2 | 238490 | 589278 | 278857 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm5-2b](fixtures/minicpm5-2b/sets.toml) | openbmb/MiniCPM5-2B @ f9740005 | 1 | 2 | 238490 | 589277 | 278858 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minimax-m2](fixtures/minimax-m2/sets.toml) | MiniMaxAI/MiniMax-M2 @ 757303d4 | 1 | 2 | 238490 | 12821 | 278551 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minimax-m2.5](fixtures/minimax-m2.5/sets.toml) | MiniMaxAI/MiniMax-M2.5 @ f710177d | 1 | 2 | 238490 | 12821 | 278551 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minimax-m2.7](fixtures/minimax-m2.7/sets.toml) | MiniMaxAI/MiniMax-M2.7 @ d494266a | 1 | 3 | 238490 | 12821 | 278551 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minimax-m3](fixtures/minimax-m3/sets.toml) | MiniMaxAI/MiniMax-M3 @ f0e1c1e0 | 1 | 1 | 238490 | 589302 | 278833 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ministral-3-14b-instruct-2512-bf16](fixtures/ministral-3-14b-instruct-2512-bf16/sets.toml) | mistralai/Ministral-3-14B-Instruct-2512-BF16 @ 3cea74c1 | 1 | 2 | 531730 | 570194 | 6591 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ministral-3-14b-reasoning-2512](fixtures/ministral-3-14b-reasoning-2512/sets.toml) | mistralai/Ministral-3-14B-Reasoning-2512 @ 51f9210f | 3 | 2 | 531730 | 583079 | 6586 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ministral-3-3b-instruct-2512](fixtures/ministral-3-3b-instruct-2512/sets.toml) | mistralai/Ministral-3-3B-Instruct-2512 @ b35d4dfe | 1 | 2 | 531730 | 570194 | 6591 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ministral-3-3b-instruct-2512-bf16](fixtures/ministral-3-3b-instruct-2512-bf16/sets.toml) | mistralai/Ministral-3-3B-Instruct-2512-BF16 @ b6d637be | 1 | 2 | 531730 | 570194 | 6591 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ministral-3-8b-instruct-2512-bf16](fixtures/ministral-3-8b-instruct-2512-bf16/sets.toml) | mistralai/Ministral-3-8B-Instruct-2512-BF16 @ f6fae979 | 1 | 2 | 531730 | 570194 | 6591 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mistral-7b-instruct-v0.2](fixtures/mistral-7b-instruct-v0.2/sets.toml) | mistralai/Mistral-7B-Instruct-v0.2 @ 63a8b081 | 2 | 3 | 238483 | 0 | 278483 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [mistral-7b-instruct-v0.3](fixtures/mistral-7b-instruct-v0.3/sets.toml) | mistralai/Mistral-7B-Instruct-v0.3 @ c170c708 | 1 | 3 | 238483 | 0 | 278483 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [mistral-medium-3.5-128b](fixtures/mistral-medium-3.5-128b/sets.toml) | mistralai/Mistral-Medium-3.5-128B @ 22b2b868 | 1 | 2 | 531729 | 583079 | 6587 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mistral-small-24b-instruct-2501](fixtures/mistral-small-24b-instruct-2501/sets.toml) | mistralai/Mistral-Small-24B-Instruct-2501 @ 9527884b | 1 | 3 | 238488 | 184972 | 533009 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mistral-small-4-119b-2603](fixtures/mistral-small-4-119b-2603/sets.toml) | mistralai/Mistral-Small-4-119B-2603 @ a11f36be | 1 | 2 | 531729 | 583079 | 6587 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mixtral-8x7b-instruct-v0.1](fixtures/mixtral-8x7b-instruct-v0.1/sets.toml) | mistralai/Mixtral-8x7B-Instruct-v0.1 @ eba92302 | 1 | 3 | 238483 | 0 | 278483 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [muse-glimmer-30b](fixtures/muse-glimmer-30b/sets.toml) | meta-models/Muse-Glimmer-30B @ a4e59da5 | 1 | 2 | 238490 | 494673 | 371462 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nanbeige4.2-3b](fixtures/nanbeige4.2-3b/sets.toml) | Nanbeige/Nanbeige4.2-3B @ b82e54bd | 1 | 2 | 238489 | 589298 | 278838 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nemotron-3-content-safety](fixtures/nemotron-3-content-safety/sets.toml) | nvidia/Nemotron-3-Content-Safety @ b47f6f6c | 1 | 2 | 238484 | 0 | 278482 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nemotron-3.5-content-safety](fixtures/nemotron-3.5-content-safety/sets.toml) | nvidia/Nemotron-3.5-Content-Safety @ 35645ed3 | 1 | 2 | 238484 | 0 | 278482 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nemotron-cascade-14b-thinking](fixtures/nemotron-cascade-14b-thinking/sets.toml) | nvidia/Nemotron-Cascade-14B-Thinking @ 8aa18a07 | 3 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nemotron-cascade-2-30b-a3b](fixtures/nemotron-cascade-2-30b-a3b/sets.toml) | nvidia/Nemotron-Cascade-2-30B-A3B @ 6327cdbc | 1 | 2 | 238490 | 12829 | 278543 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nemotron-orchestrator-8b](fixtures/nemotron-orchestrator-8b/sets.toml) | nvidia/Nemotron-Orchestrator-8B @ 26df4b9a | 2 | 2 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nemotron-research-goosereason-4b-instruct](fixtures/nemotron-research-goosereason-4b-instruct/sets.toml) | nvidia/Nemotron-Research-GooseReason-4B-Instruct @ 82197937 | 3 | 2 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nft-7b](fixtures/nft-7b/sets.toml) | nvidia/NFT-7B @ a7b4aec0 | 1 | 3 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nvidia-nemotron-3-nano-30b-a3b-bf16](fixtures/nvidia-nemotron-3-nano-30b-a3b-bf16/sets.toml) | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 @ bf77c317 | 1 | 2 | 238490 | 12829 | 278543 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-3-nano-4b-bf16](fixtures/nvidia-nemotron-3-nano-4b-bf16/sets.toml) | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 @ dfaf35de | 1 | 2 | 238490 | 12829 | 278543 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-3-super-120b-a12b-bf16](fixtures/nvidia-nemotron-3-super-120b-a12b-bf16/sets.toml) | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 @ 2dc98e2a | 3 | 2 | 238490 | 12829 | 278543 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-3-ultra-550b-a55b-bf16](fixtures/nvidia-nemotron-3-ultra-550b-a55b-bf16/sets.toml) | nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 @ 77df655d | 7 | 2 | 238490 | 12829 | 278543 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-3-ultra-550b-a55b-genrm](fixtures/nvidia-nemotron-3-ultra-550b-a55b-genrm/sets.toml) | nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM @ 03c50a66 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nvidia-nemotron-3.5-lightning-30b-a3b-bf16](fixtures/nvidia-nemotron-3.5-lightning-30b-a3b-bf16/sets.toml) | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 @ a9904d24 | 1 | 2 | 238490 | 12829 | 278543 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-labs-3-elastic-30b-a3b-bf16](fixtures/nvidia-nemotron-labs-3-elastic-30b-a3b-bf16/sets.toml) | nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16 @ 4bcc4326 | 1 | 2 | 238490 | 12829 | 278543 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-labs-3-puzzle-75b-a9b-bf16](fixtures/nvidia-nemotron-labs-3-puzzle-75b-a9b-bf16/sets.toml) | nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16 @ 8fe55468 | 1 | 2 | 238490 | 12829 | 278543 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nvidia-nemotron-parse-2.0](fixtures/nvidia-nemotron-parse-2.0/sets.toml) | nvidia/NVIDIA-Nemotron-Parse-2.0 @ b6742064 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nvidia-nemotron-parse-v1.1](fixtures/nvidia-nemotron-parse-v1.1/sets.toml) | nvidia/NVIDIA-Nemotron-Parse-v1.1 @ fd1e3c6b | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [nvidia-nemotronlabs-ai-for-media-sports-tennis](fixtures/nvidia-nemotronlabs-ai-for-media-sports-tennis/sets.toml) | nvidia/NVIDIA-NemotronLabs-AI-for-Media-Sports-Tennis @ af26673c | 1 | 2 | 238489 | 12825 | 278548 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [olmo-3-7b-instruct](fixtures/olmo-3-7b-instruct/sets.toml) | allenai/Olmo-3-7B-Instruct @ 6e5971d9 | 1 | 2 | 238490 | 570199 | 285056 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [opencodereasoning-nemotron-7b](fixtures/opencodereasoning-nemotron-7b/sets.toml) | nvidia/OpenCodeReasoning-Nemotron-7B @ 3baa2a3d | 7 | 3 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [openmath-nemotron-14b](fixtures/openmath-nemotron-14b/sets.toml) | nvidia/OpenMath-Nemotron-14B @ b7eeebb0 | 4 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [openmath-nemotron-14b-kaggle](fixtures/openmath-nemotron-14b-kaggle/sets.toml) | nvidia/OpenMath-Nemotron-14B-Kaggle @ 735d7fa0 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [openreasoning-nemotron-1.5b](fixtures/openreasoning-nemotron-1.5b/sets.toml) | nvidia/OpenReasoning-Nemotron-1.5B @ f7e9c457 | 3 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [openreasoning-nemotron-14b](fixtures/openreasoning-nemotron-14b/sets.toml) | nvidia/OpenReasoning-Nemotron-14B @ 9e2805b5 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [phi-4-mini-instruct](fixtures/phi-4-mini-instruct/sets.toml) | microsoft/Phi-4-mini-instruct @ cfbefacb | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [phi-4-multimodal-instruct](fixtures/phi-4-multimodal-instruct/sets.toml) | microsoft/Phi-4-multimodal-instruct @ 93f923e1 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen-agentworld-35b-a3b](fixtures/qwen-agentworld-35b-a3b/sets.toml) | Qwen/Qwen-AgentWorld-35B-A3B @ 60d2b043 | 1 | 1 | 238478 | 474563 | 349811 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen-drive-1.0-4b](fixtures/qwen-drive-1.0-4b/sets.toml) | Qwen/Qwen-Drive-1.0-4B @ 28484089 | 1 | 1 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen2.5-7b-instruct-1m](fixtures/qwen2.5-7b-instruct-1m/sets.toml) | Qwen/Qwen2.5-7B-Instruct-1M @ e28526f7 | 2 | 1 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen2.5-omni-7b](fixtures/qwen2.5-omni-7b/sets.toml) | Qwen/Qwen2.5-Omni-7B @ ae9e1690 | 2 | 1 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen2.5-vl-32b-instruct](fixtures/qwen2.5-vl-32b-instruct/sets.toml) | Qwen/Qwen2.5-VL-32B-Instruct @ 7cfb30d7 | 1 | 1 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen2.5-vl-7b-instruct](fixtures/qwen2.5-vl-7b-instruct/sets.toml) | Qwen/Qwen2.5-VL-7B-Instruct @ cc594898 | 5 | 1 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-30b-a3b](fixtures/qwen3-30b-a3b/sets.toml) | Qwen/Qwen3-30B-A3B @ ad44e777 | 2 | 1 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-30b-a3b-instruct-2507](fixtures/qwen3-30b-a3b-instruct-2507/sets.toml) | Qwen/Qwen3-30B-A3B-Instruct-2507 @ 0d7cf239 | 2 | 1 | 38440 | 41533 | 5 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-30b-a3b-thinking-2507](fixtures/qwen3-30b-a3b-thinking-2507/sets.toml) | Qwen/Qwen3-30B-A3B-Thinking-2507 @ 144afc2f | 3 | 1 | 38440 | 54418 | 0 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-4b-instruct-2507](fixtures/qwen3-4b-instruct-2507/sets.toml) | Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f | 1 | 1 | 38440 | 41533 | 5 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-4b-saferl](fixtures/qwen3-4b-saferl/sets.toml) | Qwen/Qwen3-4B-SafeRL @ 1b95ccb8 | 1 | 1 | 38440 | 54418 | 0 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-4b-thinking-2507](fixtures/qwen3-4b-thinking-2507/sets.toml) | Qwen/Qwen3-4B-Thinking-2507 @ 768f209d | 1 | 1 | 38440 | 54418 | 0 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-8b](fixtures/qwen3-8b/sets.toml) | Qwen/Qwen3-8B @ b968826d | 7 | 1 | 38440 | 54418 | 0 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-coder-30b-a3b-instruct](fixtures/qwen3-coder-30b-a3b-instruct/sets.toml) | Qwen/Qwen3-Coder-30B-A3B-Instruct @ b2cff646 | 2 | 1 | 238490 | 490451 | 364804 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-coder-next](fixtures/qwen3-coder-next/sets.toml) | Qwen/Qwen3-Coder-Next @ a7fbcb5c | 2 | 1 | 238490 | 490469 | 364786 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-nemotron-235b-a22b-genrm-2603](fixtures/qwen3-nemotron-235b-a22b-genrm-2603/sets.toml) | nvidia/Qwen3-Nemotron-235B-A22B-GenRM-2603 @ e65a7cb3 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [qwen3-next-80b-a3b-instruct](fixtures/qwen3-next-80b-a3b-instruct/sets.toml) | Qwen/Qwen3-Next-80B-A3B-Instruct @ 9c7f2fbe | 1 | 1 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-next-80b-a3b-thinking](fixtures/qwen3-next-80b-a3b-thinking/sets.toml) | Qwen/Qwen3-Next-80B-A3B-Thinking @ e502dd41 | 1 | 1 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-omni-30b-a3b-instruct](fixtures/qwen3-omni-30b-a3b-instruct/sets.toml) | Qwen/Qwen3-Omni-30B-A3B-Instruct @ 26291f79 | 1 | 1 | 38436 | 54418 | 4 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-omni-30b-a3b-thinking](fixtures/qwen3-omni-30b-a3b-thinking/sets.toml) | Qwen/Qwen3-Omni-30B-A3B-Thinking @ 2f443cfc | 1 | 1 | 36112 | 54418 | 92 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-vl-235b-a22b-thinking](fixtures/qwen3-vl-235b-a22b-thinking/sets.toml) | Qwen/Qwen3-VL-235B-A22B-Thinking @ 6664affd | 2 | 1 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-vl-30b-a3b-instruct](fixtures/qwen3-vl-30b-a3b-instruct/sets.toml) | Qwen/Qwen3-VL-30B-A3B-Instruct @ 9c4b90e1 | 2 | 1 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-vl-8b-instruct](fixtures/qwen3-vl-8b-instruct/sets.toml) | Qwen/Qwen3-VL-8B-Instruct @ 0c351dd0 | 10 | 1 | 531736 | 576772 | 7 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3-vl-8b-thinking](fixtures/qwen3-vl-8b-thinking/sets.toml) | Qwen/Qwen3-VL-8B-Thinking @ 92f3c4b4 | 4 | 1 | 531736 | 554276 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3.5-27b](fixtures/qwen3.5-27b/sets.toml) | Qwen/Qwen3.5-27B @ fc05daec | 1 | 1 | 27159 | 54179 | 10593 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3.5-2b](fixtures/qwen3.5-2b/sets.toml) | Qwen/Qwen3.5-2B @ 15852e8c | 3 | 1 | 238478 | 461722 | 349772 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3.5-35b-a3b](fixtures/qwen3.5-35b-a3b/sets.toml) | Qwen/Qwen3.5-35B-A3B @ 59d61f3c | 5 | 1 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3.5-9b](fixtures/qwen3.5-9b/sets.toml) | Qwen/Qwen3.5-9B @ c2022362 | 3 | 1 | 238478 | 474543 | 349831 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3.6-27b](fixtures/qwen3.6-27b/sets.toml) | Qwen/Qwen3.6-27B @ 6a9e13bd | 1 | 1 | 238478 | 474563 | 349811 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3.6-35b-a3b](fixtures/qwen3.6-35b-a3b/sets.toml) | Qwen/Qwen3.6-35B-A3B @ 995ad96e | 1 | 1 | 238478 | 474563 | 349811 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3.8-2.4t-a95b](fixtures/qwen3.8-2.4t-a95b/sets.toml) | Qwen/Qwen3.8-2.4T-A95B @ 207bd685 | 1 | 1 | 238477 | 474553 | 349822 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3.8-27b](fixtures/qwen3.8-27b/sets.toml) | Qwen/Qwen3.8-27B @ 1d4bf0f2 | 2 | 1 | 238478 | 474563 | 349811 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3.8-flash-next](fixtures/qwen3.8-flash-next/sets.toml) | Qwen/Qwen3.8-Flash-Next @ de4b8e4d | 1 | 1 | 238478 | 474563 | 349811 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [qwen3guard-gen-0.6b](fixtures/qwen3guard-gen-0.6b/sets.toml) | Qwen/Qwen3Guard-Gen-0.6B @ fada3b2f | 3 | 1 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [qwq-32b](fixtures/qwq-32b/sets.toml) | Qwen/QwQ-32B @ 976055f8 | 1 | 1 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [seed-oss-36b-instruct](fixtures/seed-oss-36b-instruct/sets.toml) | ByteDance-Seed/Seed-OSS-36B-Instruct @ 497f1dca | 2 | 2 | 236463 | 495977 | 373185 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [step-3.5-flash](fixtures/step-3.5-flash/sets.toml) | stepfun-ai/Step-3.5-Flash @ ab446a3d | 1 | 3 | 238490 | 54165 | 278713 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [step3](fixtures/step3/sets.toml) | stepfun-ai/step3 @ e2d67fc9 | 1 | 2 | 238490 | 0 | 278476 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [tinyllama-1.1b-chat-v1.0](fixtures/tinyllama-1.1b-chat-v1.0/sets.toml) | TinyLlama/TinyLlama-1.1B-Chat-v1.0 @ fe8a4ea1 | 1 | 3 | 531736 | 328662 | 111768 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [trinity-mini](fixtures/trinity-mini/sets.toml) | arcee-ai/Trinity-Mini @ cdb81e81 | 1 | 2 | 531736 | 0 | 0 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [webworld-32b](fixtures/webworld-32b/sets.toml) | Qwen/WebWorld-32B @ e7fceedc | 3 | 1 | 531736 | 583086 | 6573 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| all | | 517 | | 129862902 | 97829960 | 48243531 | |

625 more groups (953 checkpoints) have a manifest and nothing recorded yet.

### Corpus

The cases every group is recorded over, by the source they were imported from.

| Source | From | Licenses | Render sets | Render cases | Parse sets | Parse cases | Plain | Stored |
|---|---|---|---:|---:|---:|---:|---:|---|
| bfcl | pypi:bfcl-eval==2026.3.23 | Apache-2.0 | 17 | 4119 | 14 | 2735 | 37.0 MB | 37.0 MB, plain |
| glaive-v2 | hf:datasets/glaiveai/glaive-function-calling-v2@e7f4b645 | Apache-2.0 | 71 | 289494 | 71 | 306240 | 1.24 GB | 81.4 MB, zstd in Git LFS |
| gsm8k | github:openai/grade-school-math@3101c7d5 | MIT | 2 | 8792 | 4 | 17584 | 22.6 MB | 22.6 MB, plain |
| hand-written | written in this repository | | 1 | 24 | 1 | 16 | 0.0 MB | 0.0 MB, plain |
| hermes | hf:datasets/NousResearch/hermes-function-calling-v1@dae3e1d2 | Apache-2.0 | 3 | 20769 | 3 | 20773 | 99.6 MB | 5.8 MB, zstd in Git LFS |
| mgsm | github:google-research/url-nlp@3622039c | CC-BY-4.0 | 10 | 2500 | 12 | 2838 | 4.2 MB | 4.2 MB, plain |
| shapes | github:openai/grade-school-math@3101c7d5 and pypi:bfcl-eval==2026.3.23 | Apache-2.0, MIT | 0 | 0 | 6 | 6000 | 17.9 MB | 17.9 MB, plain |
| swebench | hf:datasets/SWE-bench/SWE-bench@c6fe717f, hf:datasets/SWE-bench/SWE-bench_Verified@78f471bf | 8 licenses | 4 | 2236 | 8 | 4472 | 45.8 MB | 45.8 MB, plain |
| swehero | hf:datasets/nvidia/SWE-Hero-openhands-trajectories@150bc119 | CC-BY-4.0 | 14 | 79627 | 14 | 101568 | 14.84 GB | 882.8 MB, zstd in Git LFS |
| tau2 | github:sierra-research/tau2-bench@4ce7c039 | MIT | 22 | 124175 | 22 | 127433 | 9.77 GB | 30.8 MB, zstd in Git LFS |
| all | | | 144 | 531736 | 155 | 589659 | 26.08 GB | 1.13 GB |

### Images

The images the references of every processor family are recorded over, by the dataset they were imported from; the files are in Git LFS.

| Dataset | From | Licenses | Images | Bytes |
|---|---|---|---:|---:|
| docvqa | hf:datasets/lmms-lab-encoder/DocVQA@539088ef | Apache-2.0 | 1285 | 832.7 MB |
| mmmu | hf:datasets/MMMU/MMMU@876ce5cb | Apache-2.0 | 959 | 340.6 MB |
| all | | | 2244 | 1.17 GB |

<!-- end of what bellwether count --readme writes -->

## Quick start

```bash
uv sync --extra dev
uv run bellwether --help
uv run pytest -q

# The coverage matrix: engine registries at a pinned commit or tag, SMG from a checkout.
uv run bellwether gaps --vllm-ref v0.30.0 --sglang-ref v0.5.20 --smg-src ../smg
uv run bellwether gaps --vllm-src ../vllm --sglang-src ../sglang --smg-src ../smg --format json --out gaps.json
```

`gaps` reads the registries from source with `ast` and regular expressions, never by importing an
engine: vLLM's four `name -> (module, class)` tables, SGLang's two name lists and class maps plus
its native renderer and tokenizer modules, and SMG's two factories, renderer enum and tokenizer
types. Names that differ across systems for one format are merged through
`src/bellwether/gaps/aliases.toml`, where every entry states its evidence; the matrix also lists
one implementation behind several rows as alias candidates, so the table is kept honest by what
the code says rather than by memory.

## Checkpoint groups and manifests

```bash
HF_HUB_OFFLINE=1 uv run bellwether manifests
```

Reads the list of checkpoints committed beside the manifests, `fixtures/models.tsv` (`--models` names another file):
one `model<TAB>revision<TAB>downloads<TAB>day<TAB>tier` row per checkpoint, with the Hub's downloads over the last 30
days and the day they were read. It computes each one's oracle inputs (the tokenizer files, the chat templates, and the
few fields of `config.json`, `generation_config.json` and `chat_template.json` that the oracles read) at the pinned
revision from the Hugging Face cache, groups the checkpoints whose inputs are equal, and writes every checkpoint's
`fixtures/<slug>/manifest.toml`. A group is recorded once, under its primary's slug: the recorded one if there is one,
else its most-downloaded member's. The others' manifests name the group and hold no fixtures. It prints each group with
its members and their tiers. Run again over the same list, it writes the same files, so a change to the list is
reviewed as a diff of the list and the manifests. `fixtures/README.md` describes both. `models.jsonl` (below) is
another list: what the engines' registries name, with no revision or downloads until the Hub is read.

## Recording render fixtures

```bash
uv run bellwether record --model Qwen/Qwen3-8B --kind render --oracle reference
```

`record` checks the manifest before it reads the corpus: it refuses any member of a group but its primary, naming the
primary to record instead, and a checkpoint whose oracle inputs at the pinned revision cannot be read or are not the
ones its manifest lists, naming the files that differ.

Finds the manifest whose `model` is the given id (`fixtures/qwen3-8b/manifest.toml`), runs every
case under `corpus/render/` through the checkpoint's own chat template at the pinned revision
(`transformers.apply_chat_template`, each message's content in the format vLLM selects for the
template: a one-item text part list for a template that loops over the content, else the string),
and writes `fixtures/qwen3-8b/render/<set>.jsonl`, or
`<set>.jsonl.zst` in Git LFS for an imported set (`corpus/README.md`), with each set's counts and
plain-content sha256 in `fixtures/qwen3-8b/sets.toml`. `--set NAME` records only the named sets. Per
case the file holds the request, the prompt token ids, the rendered text and the oracle versions. A
re-run replaces each reference, keeps the witnesses already recorded for an unchanged request, and
removes cases and sets the corpus no longer has. A case the template cannot render is reported on
stderr and left out, and the command exits 1. Engine witnesses (`--oracle vllm|sglang`) run on Linux
inside the engine's image and are the second half of M2.

## Recording parse fixtures (the round trip)

```bash
uv run bellwether record --model Qwen/Qwen3-8B --kind parse --oracle reference
```

Each case under `corpus/parse/` states the assistant message (content, reasoning, tool calls) the
output must parse to. The checkpoint's template renders it as the final assistant turn after the
generation prompt; the turn up to where generation stops, at the first of vLLM's stop ids, is the
output, its token ids the engine chunks a replay feeds, and the fixture carries the chunk plans (fixed
sizes, every two-way split for short outputs, thirty seeded random plans) and the stop id that ends
the output (`end_of_turn`). A template that does not extend the generation prompt when the turn is
appended (DeepSeek-R1 never renders `<think>`) cannot be this oracle for that case; the case is
reported and not recorded, and the run exits 1. Recording it is left to the manifest's next authority,
the engine witnesses, which nothing here invokes. `fixtures/README.md` has the rules.

The stop ids are the eos ids of the checkpoint's `generation_config.json` (of its `config.json` when it
ships none) together with the tokenizer's eos, so `record --kind parse` needs `generation_config.json`
cached at the manifest's revision, or known to be absent: offline, transformers would take a
file that is merely not cached for one the repository does not ship. The run stops with an error
naming the file and the command that fetches it,
`hf download <model> generation_config.json --revision <sha>`, which leaves the hub cache's
`.no_exist` marker when the repository has no such file.

The output's token ids must decode back to its text. When the tokenizer's own ids do not, because its
normalizer applies a Unicode normalization form (Qwen3-8B's NFC writes U+09DF, one code point in the
text, as two), the ids are built from the text as written, by a copy of the tokenizer whose normalizer
leaves out NFC, NFD, NFKC and NFKD and keeps every other step. The run prints how many cases of each
set took that path (#57); `sets.toml` does not record it. A case whose ids do not give back its text
either way is reported and not recorded.

## Recording with the vendor's code

```bash
uv run bellwether sandbox-record --model moonshotai/Kimi-K3 --kind render
uv run bellwether sandbox-record --model moonshotai/Kimi-K3 --kind parse
```

Some checkpoints ship their tokenizer as their own Python class, like Kimi-K3's
`tokenization_kimi.TikTokenTokenizer`. Such a checkpoint is recorded by the vendor-code oracle, which
runs that class only inside a container:
- with no network and a read-only root;
- as a user without privileges;
- holding only the files the manifest lists, each checked against its sha256 before anything runs.

It needs a Docker engine. `record --oracle vendor` refuses outside the container and names this
command. `docs/benchmark-sets.md`, "Running vendor code", has the steps. At benchmark scale the
`sandbox-record` workflow runs it a few sets at a time, with `--set` and a larger `--memory`, on the
`bellwether-sandbox` runners.

## The list of models

```bash
uv run bellwether models --registry-only           # the registries alone; writes the committed models.jsonl
uv run bellwether models --registry-only --check   # build it again and compare with models.jsonl, as CI does
uv run bellwether models                           # registries and the Hub; writes runs/models-<date>.jsonl
```

`models` builds the list of checkpoints to record: every generative model vLLM or SGLang supports,
except gpt-oss (the rule is in `docs/benchmark-sets.md`, "Which models"); the gpt-oss checkpoints the
registries name, by name or by vLLM's `GptOssForCausalLM`, are named on stderr as set aside. It reads
two sources:

- **The engines' registries at pinned commits.** vLLM's `tests/models/registry.py` at v0.31.0, read
  with `ast`: every checkpoint its text-generation and multimodal tables name, except under a
  ranking or classification head. SGLang's three "Text Generation" docs pages at 7d22b7a8: every id
  in a table's example column. SGLang's code at the same commit, read with `ast`: each module of
  `python/sglang/srt/models/` names the architectures it serves as `EntryClass` (251), and the
  multimodal processors name those they serve with images, audio or video. The registered
  architectures are vLLM's generative tables and those SGLang's code serves, leaving out pooling and
  speculative-decoding draft heads. The files come from a checkout given with `--vllm-src` and
  `--sglang-src` (read with `git show`, whatever the checkout has checked out) or from GitHub, are
  checked against their pinned sha256 on every use, and are cached by commit under
  `~/.cache/bellwether/registries`.
- **The Hugging Face Hub as it is today.** Every model of the organizations whose checkpoints the
  engines give as an architecture's example (vLLM's default checkpoint, the ids in SGLang's docs),
  or as one of vLLM's extras that is a real checkpoint rather than a tiny or random test model or a
  quantized copy (NousResearch's Hermes 3, mistral-community's Pixtral), whose `config.json` names a
  registered architecture and that ships a chat template, leaving out quantized and converted copies
  (GGUF, AWQ, GPTQ, MLX, ONNX, FP8, NVFP4, MXFP4, MXFP8, Int4, Int8 or bitsandbytes in the name, or
  the Hub's `base_model:quantized` tag), embedding, reranking and classification models, and
  checkpoints created before 2025 that no registry names. A token (`HF_TOKEN`) raises the Hub's rate
  limits; none is needed, and a rate-limited call waits and is made again. A model whose tokenizer or
  processor config cannot be read (gated, an error from the Hub, a file that is not JSON) is kept,
  with a status that says so. A checkpoint vLLM loads with another repository's tokenizer
  (moondream3-preview with starmie-v1) is judged on that repository's template. When the Hub gives no
  answer about a model at all, a checkpoint a registry names keeps its row with the error as its
  status, and a model or an organization only a listing would have added is left out and named on
  stderr.

The file's first line says what the list was built from: each engine's repository, ref and commit,
and `hub`, the day the Hub was read, or `null` without it. Tier 2 and downloads depend on that day;
nothing in a list built without the Hub depends on a date. One JSON line follows per checkpoint, and
one per registry entry that names none, ordered by tier and then within the tier:

| field | meaning |
|---|---|
| `model` | the Hugging Face id, as the Hub spells it; for a registry entry that names no checkpoint, the entry's name (vLLM's architecture, SGLang's model family) |
| `revision` | the Hub's sha when the list was built; for a checkpoint vLLM's registry pins (`refs/pr/17` for ERNIE-4.5-VL, a commit for HyperCLOVAX-SEED-Think-32B), the sha of that revision, and without the Hub the revision as vLLM writes it |
| `tier` | 1: the maintainer's priority models, in the order given; 2: created in the twelve months before the build; 3: the rest. Within tiers 2 and 3, by 30-day downloads. `null` where the Hub decides the tier and was not asked (`--registry-only`) or gave no answer; those rows come after tier 2, by id |
| `status` | `pending`, or why nothing can be recorded yet: `no-checkpoint-named` (the registry entry names no checkpoint), `gated` (also when a config answers 401 or 403), `needs-vendor-code` (vLLM loads it with `trust_remote_code`, so the oracle would need the vendor's code; known without the Hub too), `no-chat-template`, `processor-chat-template` (the template is only in the processor's files, `chat_template.json` or the `chat_template` in `processor_config.json`, which `AutoProcessor` and vLLM read but the oracle's `AutoTokenizer` does not; the oracle reading processor templates is the follow-up), `not-on-hub`; when a config could not be read, `invalid-tokenizer-config` or `invalid-processor-config` (not a JSON object), or `hub-error-` and the HTTP status or the error (`hub-error-503`, `hub-error-read-timeout`); `unchecked` without the Hub |
| `created`, `downloads` | the Hub's creation date and downloads over the last 30 days |
| `modality` | `multimodal` when one of its architectures is multimodal in either engine's code (vLLM's multimodal table, an SGLang multimodal processor), else `text`; its architectures are those vLLM lists it under and, with the Hub, those its config names. `null` when none is known: an id only SGLang's docs give, without the Hub |
| `sources` | the engines whose registries name the checkpoint, or `hub` |

Without the Hub the list depends on the pins alone, so it is committed: `models.jsonl` at the
repository's root, which CI builds again, with the registry files cached, and compares (`--check`),
as the importers check their sets. A list read from the Hub changes every day, its downloads and
their order with it, so it is that day's evidence, not a committed file: it goes to
`runs/models-<date>.jsonl` and is published to smg-project/artifacts, each row pinning the sha the
Hub gave that day.

## Verifying render fixtures against a running SMG

```bash
uv run bellwether verify --smg http://127.0.0.1:30000 --capture capture.jsonl --model Qwen/Qwen3-8B \
  --report runs/verify.json --junit runs/verify.xml
```

SMG runs in front of its `mock-worker`, started with `--capture capture.jsonl`, which creates the file
and appends each request SMG sends it as one JSON line; `verify` opens the file before its first
request. `verify` posts every render fixture's request to `/v1/chat/completions` with the manifest's
`model`, the fixture id as `rid`, `stream: false` and, unless the request sets a limit, `max_tokens: 1`;
none of these reach the chat template. SMG passes the `rid` to the engine as `request_id`, verbatim, or
in prefill-decode mode as `<rid>-<uuid>`, with a fresh UUID for each attempt; `verify` takes that suffix
off, so the capture lines written during the run are joined to the cases on it, and each case gets a
verdict. Every render set of the selected models is read whole, plain or compressed, before the first
request, and nothing in them is passed over: a set Git LFS has not fetched stops the run with the
command that fetches it, and so does a set `sets.toml` lists that the checkout lacks, a set that is not
zstd or is cut short, a line that is not a case with a request and reference ids, and a case id in two
of a model's sets. A model named with `--model` must have render cases; any other model without them is
named in the report. `--set NAME` verifies only the named render sets, and each must be some selected
model's. That first read keeps only the case ids; the cases are then sent, judged and written one at a
time, so a run's memory does not grow with its number of cases, and the JSON report and the JUnit XML
are put together when the run finishes.

| Verdict | Meaning |
|---|---|
| `match` | the `input_ids` SMG sent equal the reference's |
| `regression` | they do not; the report gives the first differing index, the ids around it on both sides, and whether the prompt text SMG sent equals the reference text (equal text points at tokenization, other text at rendering) |
| `rejected` | SMG refused the request: a 400 whose body is SMG's error object; its code and message are kept |
| `missing` | SMG answered 200, but no capture line written during the run carries the case's id: the file is not the one the engine behind this SMG writes, or the `rid` did not reach it |
| `measurement_failed` | SMG answered neither 200 nor its refusal (a redirect, a 404 for a model no worker serves, a 429 or 5xx, a proxy's page) and no capture line carries the case's id: nothing was measured |

Every capture line that carries a case's id is compared, whatever SMG answered after sending it, and the
answer is kept beside the verdict. A request can reach the engine more than once (a retry; under
prefill-decode, both legs), so a case matches only when each of its lines does. The report counts the
lines written during the run, those that joined a case and those that did not (another client's, a line
for no case), so a line no case takes is never dropped unseen. `missing` and `measurement_failed` are
about the setup, so no known difference excuses them. Before the first case, `verify` asks SMG's
`/v1/models` for the models it serves; a model with cases that SMG does not list stops the run, as does
an answer other than SMG's list (a 404 there usually means `--smg` ends in `/v1`). The report and the
messages give `--smg` without a user and password, should the URL carry them.

`--known PATH` lists SMG's known differences, one TOML table per fixture id stating the outcome SMG is
known to give, why, and the issue that tracks it; bellwether ships no such list:

```toml
["qwen3-8b/render/some-case"]
verdict = "rejected"   # or "regression"
code = "bad_request"   # a rejected entry's: SMG's error code, or 400 for its validation errors
reason = "what SMG does on this case, and why it is not fixed yet"
issue = "https://github.com/smg-project/smg/issues/NNN"
```

A listed case passes while it has exactly that outcome and fails on any other: a match, so the entry
goes as soon as SMG is fixed; another verdict or code; or an outcome about the setup. A listed id that
can name no case fails the run: one that is not a fixture id, one whose slug no manifest has, or a
render id of a verified model that a run of every set did not find. Ids of models, sets or kinds the run
does not verify are counted and listed in the report, so the list cannot go stale. The exit status is 0
when every case passes, 1 when one does not, and 2 when the run gives no verdict (no such model, a
manifest or a set verify cannot read, a capture file verify cannot open, no answer from SMG, a model it
does not serve) or stops before its last case. After the first request, a case left without a verdict
(no answer from SMG, a capture line verify cannot read, a set that cannot be read again) stops the
sending: the cases answered so far are judged and reported, and the report names the case the run
stopped at and every case it did not send. A case is judged against its reference alone for now: telling
an `engine_defect` or `engines_split` from a `regression` needs the engine witnesses, which come once
they are recorded.

## Layout

```
src/bellwether/      package: cli.py, one subpackage per command, schemas/case.schema.json (the fixture line format)
corpus/              request corpora, one JSON Lines file per set (see corpus/README.md)
fixtures/            per-model manifests and recorded cases (see fixtures/README.md)
docker/vendor/       the vendor-code oracle's base image (bellwether sandbox-record)
waivers/             engine_defects.toml
tests/
```

## Related

- [smg](https://github.com/smg-project/smg): the gateway under test; `crates/mock_worker` is the replay engine.
- [providers-verifier](https://github.com/smg-project/providers-verifier): record a vendor's API, replay against SMG. Its request catalogue is bellwether's first corpus.
- [deepseek-provider-verifier](https://github.com/smg-project/deepseek-provider-verifier).
- [artifacts](https://github.com/smg-project/artifacts): published evidence.

Apache-2.0.
