# The coverage matrix

```bash
# Engine registries at a pinned commit or tag, SMG from a checkout.
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
