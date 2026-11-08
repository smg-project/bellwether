# The list of models

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
