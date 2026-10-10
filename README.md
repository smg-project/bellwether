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
| [ai21-jamba2-3b](fixtures/ai21-jamba2-3b/sets.toml) | ai21labs/AI21-Jamba2-3B @ 525c6c8e | 2 | 2 | 38436 | 41533 | 9 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [apertus-8b-instruct-2509](fixtures/apertus-8b-instruct-2509/sets.toml) | swiss-ai/Apertus-8B-Instruct-2509 @ b946d404 | 1 | 2 | 14374 | 14607 | 32259 | bfcl, gsm8k, hand-written, hermes, mgsm |
| [deepseek-r1](fixtures/deepseek-r1/sets.toml) | deepseek-ai/DeepSeek-R1 @ 56d4cbbb | 2 | 3 | 23 | 0 | 1 | hand-written |
| [deepseek-v3-0324](fixtures/deepseek-v3-0324/sets.toml) | deepseek-ai/DeepSeek-V3-0324 @ e9b33add | 2 | 3 | 38439 | 18947 | 15694 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [deepseek-v3.1](fixtures/deepseek-v3.1/sets.toml) | deepseek-ai/DeepSeek-V3.1 @ c0781d03 | 3 | 3 | 38439 | 18947 | 15694 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [deepseek-v4.1-flash](fixtures/deepseek-v4.1-flash/sets.toml) | deepseek-ai/DeepSeek-V4.1-Flash @ 2cba9e42 | 1 | 1 | 38439 | 54418 | 1 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [dots3-note-prev](fixtures/dots3-note-prev/sets.toml) | dots-studio/dots3-note-prev @ 1e1e7b0c | 1 | 3 | 238490 | 589296 | 278839 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ernie-4.5-21b-a3b-thinking](fixtures/ernie-4.5-21b-a3b-thinking/sets.toml) | baidu/ERNIE-4.5-21B-A3B-Thinking @ 4341bb42 | 1 | 3 | 38440 | 0 | 0 | bfcl, gsm8k, hand-written, hermes, mgsm, swebench |
| [gemma-4-e4b-it](fixtures/gemma-4-e4b-it/sets.toml) | google/gemma-4-E4B-it @ ee0ef602 | 2 | 2 | 238484 | 478241 | 389900 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4.6](fixtures/glm-4.6/sets.toml) | zai-org/GLM-4.6 @ be721948 | 3 | 2 | 238490 | 492389 | 364866 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-4.7-flash](fixtures/glm-4.7-flash/sets.toml) | zai-org/GLM-4.7-Flash @ 7dd20894 | 1 | 2 | 238490 | 1947 | 278545 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [glm-5.3-flash](fixtures/glm-5.3-flash/sets.toml) | zai-org/GLM-5.3-Flash @ eb9eb208 | 2 | 1 | 238490 | 503273 | 364862 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [granite-4.1-3b](fixtures/granite-4.1-3b/sets.toml) | ibm-granite/granite-4.1-3b @ c0650403 | 2 | 2 | 38440 | 41533 | 5 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [hermes-4-14b](fixtures/hermes-4-14b/sets.toml) | NousResearch/Hermes-4-14B @ d6ce765c | 1 | 2 | 38440 | 41533 | 5 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [hunyuan-a13b-instruct](fixtures/hunyuan-a13b-instruct/sets.toml) | tencent/Hunyuan-A13B-Instruct @ 290ddb9a | 1 | 3 | 498896 | 328662 | 144608 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [hy4-preview](fixtures/hy4-preview/sets.toml) | tencent/Hy4-preview @ 705d81ee | 1 | 1 | 238489 | 589298 | 278838 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [inkling](fixtures/inkling/sets.toml) | thinkingmachines/Inkling @ 828496ee | 2 | 3 | 238489 | 589659 | 278477 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [iquest-q1](fixtures/iquest-q1/sets.toml) | IQuestLab/IQuest-Q1 @ 5c21b063 | 1 | 2 | 238490 | 503273 | 364862 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [k-exaone-236b-a23b](fixtures/k-exaone-236b-a23b/sets.toml) | LGAI-EXAONE/K-EXAONE-236B-A23B @ 61e6d578 | 1 | 2 | 38439 | 54416 | 3 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [k2-horizon-36b](fixtures/k2-horizon-36b/sets.toml) | IFM/K2-Horizon-36B @ cca48b66 | 3 | 2 | 122503 | 12817 | 394542 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [kimi-k3](fixtures/kimi-k3/sets.toml) | moonshotai/Kimi-K3 @ f831ab66 | 1 | 1 | 38439 | 54418 | 1 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [laguna-xs.2](fixtures/laguna-xs.2/sets.toml) | poolside/Laguna-XS.2 @ a397bde0 | 1 | 3 | 238490 | 490452 | 364803 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [lfm2.5-1.2b-instruct](fixtures/lfm2.5-1.2b-instruct/sets.toml) | LiquidAI/LFM2.5-1.2B-Instruct @ 0f604ada | 2 | 2 | 243122 | 589657 | 273846 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [ling-3.0-flash](fixtures/ling-3.0-flash/sets.toml) | inclusionAI/Ling-3.0-flash @ ef06d91f | 2 | 3 | 238490 | 589298 | 278837 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [llama-xlam-2-8b-fc-r](fixtures/llama-xlam-2-8b-fc-r/sets.toml) | Salesforce/Llama-xLAM-2-8b-fc-r @ a0efe39a | 1 | 3 | 38440 | 40529 | 9 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [llava-1.5-7b-hf](fixtures/llava-1.5-7b-hf/sets.toml) | llava-hf/llava-1.5-7b-hf @ b234b804 | 1 | 3 | 38439 | 0 | 1 | bfcl, gsm8k, hand-written, hermes, mgsm, swebench |
| [mimo-v2.5](fixtures/mimo-v2.5/sets.toml) | XiaomiMiMo/MiMo-V2.5 @ 63651580 | 2 | 2 | 238489 | 556400 | 311736 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minicpm5-2b](fixtures/minicpm5-2b/sets.toml) | openbmb/MiniCPM5-2B @ f9740005 | 1 | 2 | 238490 | 589277 | 278858 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minimax-m2](fixtures/minimax-m2/sets.toml) | MiniMaxAI/MiniMax-M2 @ 757303d4 | 1 | 2 | 238490 | 12821 | 278551 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minimax-m2.7](fixtures/minimax-m2.7/sets.toml) | MiniMaxAI/MiniMax-M2.7 @ d494266a | 1 | 3 | 238490 | 12821 | 278551 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [minimax-m3](fixtures/minimax-m3/sets.toml) | MiniMaxAI/MiniMax-M3 @ f0e1c1e0 | 1 | 1 | 238490 | 589302 | 278833 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [mistral-7b-instruct-v0.3](fixtures/mistral-7b-instruct-v0.3/sets.toml) | mistralai/Mistral-7B-Instruct-v0.3 @ c170c708 | 1 | 3 | 238483 | 0 | 278483 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, swebench, swehero, tau2 |
| [muse-glimmer-30b](fixtures/muse-glimmer-30b/sets.toml) | meta-models/Muse-Glimmer-30B @ a4e59da5 | 1 | 2 | 238490 | 494673 | 371462 | bfcl, glaive-v2, gsm8k, hand-written, hermes, mgsm, shapes, swebench, swehero, tau2 |
| [nanbeige4.2-3b](fixtures/nanbeige4.2-3b/sets.toml) | Nanbeige/Nanbeige4.2-3B @ b82e54bd | 1 | 2 | 27160 | 54195 | 10576 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [nvidia-nemotron-3-nano-30b-a3b-bf16](fixtures/nvidia-nemotron-3-nano-30b-a3b-bf16/sets.toml) | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 @ bf77c317 | 1 | 2 | 27161 | 12829 | 10419 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [olmo-3-7b-instruct](fixtures/olmo-3-7b-instruct/sets.toml) | allenai/Olmo-3-7B-Instruct @ 6e5971d9 | 1 | 2 | 27161 | 41533 | 10357 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [phi-4-mini-instruct](fixtures/phi-4-mini-instruct/sets.toml) | microsoft/Phi-4-mini-instruct @ cfbefacb | 1 | 3 | 38436 | 27865 | 7706 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [phi-4-multimodal-instruct](fixtures/phi-4-multimodal-instruct/sets.toml) | microsoft/Phi-4-multimodal-instruct @ 93f923e1 | 1 | 3 | 38436 | 27865 | 7706 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen-agentworld-35b-a3b](fixtures/qwen-agentworld-35b-a3b/sets.toml) | Qwen/Qwen-AgentWorld-35B-A3B @ 60d2b043 | 1 | 1 | 27159 | 54193 | 10579 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen-drive-1.0-4b](fixtures/qwen-drive-1.0-4b/sets.toml) | Qwen/Qwen-Drive-1.0-4B @ 28484089 | 1 | 1 | 27159 | 54179 | 10593 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen2.5-7b-instruct-1m](fixtures/qwen2.5-7b-instruct-1m/sets.toml) | Qwen/Qwen2.5-7B-Instruct-1M @ e28526f7 | 2 | 1 | 38440 | 41533 | 5 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen2.5-omni-7b](fixtures/qwen2.5-omni-7b/sets.toml) | Qwen/Qwen2.5-Omni-7B @ ae9e1690 | 2 | 1 | 38436 | 27865 | 7706 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen2.5-vl-32b-instruct](fixtures/qwen2.5-vl-32b-instruct/sets.toml) | Qwen/Qwen2.5-VL-32B-Instruct @ 7cfb30d7 | 1 | 1 | 38436 | 27865 | 7706 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen2.5-vl-7b-instruct](fixtures/qwen2.5-vl-7b-instruct/sets.toml) | Qwen/Qwen2.5-VL-7B-Instruct @ cc594898 | 5 | 1 | 38436 | 27865 | 7706 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-30b-a3b](fixtures/qwen3-30b-a3b/sets.toml) | Qwen/Qwen3-30B-A3B @ ad44e777 | 2 | 1 | 38440 | 54418 | 0 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-30b-a3b-instruct-2507](fixtures/qwen3-30b-a3b-instruct-2507/sets.toml) | Qwen/Qwen3-30B-A3B-Instruct-2507 @ 0d7cf239 | 2 | 1 | 38440 | 41533 | 5 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-30b-a3b-thinking-2507](fixtures/qwen3-30b-a3b-thinking-2507/sets.toml) | Qwen/Qwen3-30B-A3B-Thinking-2507 @ 144afc2f | 3 | 1 | 38440 | 54418 | 0 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-4b-instruct-2507](fixtures/qwen3-4b-instruct-2507/sets.toml) | Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f | 1 | 1 | 38440 | 41533 | 5 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-4b-saferl](fixtures/qwen3-4b-saferl/sets.toml) | Qwen/Qwen3-4B-SafeRL @ 1b95ccb8 | 1 | 1 | 38440 | 54418 | 0 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-4b-thinking-2507](fixtures/qwen3-4b-thinking-2507/sets.toml) | Qwen/Qwen3-4B-Thinking-2507 @ 768f209d | 1 | 1 | 38440 | 54418 | 0 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-8b](fixtures/qwen3-8b/sets.toml) | Qwen/Qwen3-8B @ b968826d | 7 | 1 | 38440 | 54418 | 0 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-coder-30b-a3b-instruct](fixtures/qwen3-coder-30b-a3b-instruct/sets.toml) | Qwen/Qwen3-Coder-30B-A3B-Instruct @ b2cff646 | 2 | 1 | 27161 | 41362 | 10528 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-coder-next](fixtures/qwen3-coder-next/sets.toml) | Qwen/Qwen3-Coder-Next @ a7fbcb5c | 2 | 1 | 27161 | 41374 | 10516 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-next-80b-a3b-instruct](fixtures/qwen3-next-80b-a3b-instruct/sets.toml) | Qwen/Qwen3-Next-80B-A3B-Instruct @ 9c7f2fbe | 1 | 1 | 38440 | 41533 | 5 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-next-80b-a3b-thinking](fixtures/qwen3-next-80b-a3b-thinking/sets.toml) | Qwen/Qwen3-Next-80B-A3B-Thinking @ e502dd41 | 1 | 1 | 38440 | 54418 | 0 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-omni-30b-a3b-instruct](fixtures/qwen3-omni-30b-a3b-instruct/sets.toml) | Qwen/Qwen3-Omni-30B-A3B-Instruct @ 26291f79 | 1 | 1 | 38436 | 54418 | 4 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-omni-30b-a3b-thinking](fixtures/qwen3-omni-30b-a3b-thinking/sets.toml) | Qwen/Qwen3-Omni-30B-A3B-Thinking @ 2f443cfc | 1 | 1 | 38436 | 54418 | 4 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-vl-235b-a22b-thinking](fixtures/qwen3-vl-235b-a22b-thinking/sets.toml) | Qwen/Qwen3-VL-235B-A22B-Thinking @ 6664affd | 2 | 1 | 38436 | 54418 | 4 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-vl-30b-a3b-instruct](fixtures/qwen3-vl-30b-a3b-instruct/sets.toml) | Qwen/Qwen3-VL-30B-A3B-Instruct @ 9c4b90e1 | 2 | 1 | 38436 | 41533 | 9 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-vl-8b-instruct](fixtures/qwen3-vl-8b-instruct/sets.toml) | Qwen/Qwen3-VL-8B-Instruct @ 0c351dd0 | 10 | 1 | 38436 | 41533 | 9 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3-vl-8b-thinking](fixtures/qwen3-vl-8b-thinking/sets.toml) | Qwen/Qwen3-VL-8B-Thinking @ 92f3c4b4 | 4 | 1 | 38436 | 54418 | 4 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3.5-27b](fixtures/qwen3.5-27b/sets.toml) | Qwen/Qwen3.5-27B @ fc05daec | 1 | 1 | 27159 | 54179 | 10593 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3.5-2b](fixtures/qwen3.5-2b/sets.toml) | Qwen/Qwen3.5-2B @ 15852e8c | 3 | 1 | 27159 | 41360 | 10532 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3.5-35b-a3b](fixtures/qwen3.5-35b-a3b/sets.toml) | Qwen/Qwen3.5-35B-A3B @ 59d61f3c | 5 | 1 | 27159 | 54179 | 10593 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3.5-9b](fixtures/qwen3.5-9b/sets.toml) | Qwen/Qwen3.5-9B @ c2022362 | 3 | 1 | 27159 | 54179 | 10593 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3.6-27b](fixtures/qwen3.6-27b/sets.toml) | Qwen/Qwen3.6-27B @ 6a9e13bd | 1 | 1 | 27159 | 54193 | 10579 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3.6-35b-a3b](fixtures/qwen3.6-35b-a3b/sets.toml) | Qwen/Qwen3.6-35B-A3B @ 995ad96e | 1 | 1 | 27159 | 54193 | 10579 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3.8-2.4t-a95b](fixtures/qwen3.8-2.4t-a95b/sets.toml) | Qwen/Qwen3.8-2.4T-A95B @ 207bd685 | 1 | 1 | 27158 | 54183 | 10590 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3.8-27b](fixtures/qwen3.8-27b/sets.toml) | Qwen/Qwen3.8-27B @ 1d4bf0f2 | 2 | 1 | 27159 | 54193 | 10579 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3.8-flash-next](fixtures/qwen3.8-flash-next/sets.toml) | Qwen/Qwen3.8-Flash-Next @ de4b8e4d | 1 | 1 | 27159 | 54193 | 10579 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [qwen3guard-gen-0.6b](fixtures/qwen3guard-gen-0.6b/sets.toml) | Qwen/Qwen3Guard-Gen-0.6B @ fada3b2f | 3 | 1 | 38436 | 0 | 4 | bfcl, gsm8k, hand-written, hermes, mgsm, swebench |
| [qwq-32b](fixtures/qwq-32b/sets.toml) | Qwen/QwQ-32B @ 976055f8 | 1 | 1 | 38436 | 0 | 4 | bfcl, gsm8k, hand-written, hermes, mgsm, swebench |
| [seed-oss-36b-instruct](fixtures/seed-oss-36b-instruct/sets.toml) | ByteDance-Seed/Seed-OSS-36B-Instruct @ 497f1dca | 2 | 2 | 27161 | 53178 | 10592 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [step-3.5-flash](fixtures/step-3.5-flash/sets.toml) | stepfun-ai/Step-3.5-Flash @ ab446a3d | 1 | 3 | 27161 | 54181 | 10589 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [step3](fixtures/step3/sets.toml) | stepfun-ai/step3 @ e2d67fc9 | 1 | 2 | 27161 | 0 | 10352 | bfcl, gsm8k, hand-written, hermes, mgsm, swebench |
| [tinyllama-1.1b-chat-v1.0](fixtures/tinyllama-1.1b-chat-v1.0/sets.toml) | TinyLlama/TinyLlama-1.1B-Chat-v1.0 @ fe8a4ea1 | 1 | 3 | 38436 | 27865 | 7706 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| [trinity-mini](fixtures/trinity-mini/sets.toml) | arcee-ai/Trinity-Mini @ cdb81e81 | 1 | 2 | 38440 | 0 | 0 | bfcl, gsm8k, hand-written, hermes, mgsm, swebench |
| [webworld-32b](fixtures/webworld-32b/sets.toml) | Qwen/WebWorld-32B @ e7fceedc | 3 | 1 | 38436 | 54418 | 4 | bfcl, gsm8k, hand-written, hermes, mgsm, shapes, swebench |
| all | | 140 | | 6871477 | 10291299 | 6442601 | |

826 more groups (1284 checkpoints) have a manifest and nothing recorded yet.

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
(`transformers.apply_chat_template`), and writes `fixtures/qwen3-8b/render/<set>.jsonl`, or
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
command. `docs/benchmark-sets.md`, "Running vendor code", has the steps.

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
