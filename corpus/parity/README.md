# Parity references

What a reference computes for fixed inputs, kept so that a consumer can check its own preprocessing against it. One
directory per model family: a README that states where the numbers come from, the reference files, and the generator
where one exists. `bellwether unpack` copies this directory into `parity/` of the tree consumers read, each file as
it is stored.

`minimax-m3`'s size rule was recorded here. Every other file came from `smg-project/smg`
(`crates/multimodal/tests/fixtures/golden/` and `crates/multimodal/scripts/`) at commit
c160608083dc6940ec654478ed7ce7b19c077c61, byte for byte: the reference files, so that the tests that read them read
the same bytes, and the generators (`generate_*.py`) as provenance. The generators are not reformatted
(`pyproject.toml` keeps ruff off them), and their default paths and usage lines still name that repository's layout.

A README states a file's source as the file states it. Where a file states nothing, the README says so.
