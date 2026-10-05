# Contributor instructions

These apply to people and to coding agents alike.

- Do not add `Co-Authored-By` trailers naming an AI tool to commits. Sign off every commit with
  your own identity (`git commit -s`).
- Branch names take the form `<type>/<topic>` (`feat/gaps-matrix`, `fix/schema-ids`).
- Before pushing: `uv run ruff check . && uv run ruff format --check . && uv run pytest -q`.
- Runtime behavior is test-first: write the failing test, then the code.
- Fixtures are evidence. They are produced by `bellwether record`, never edited by hand, and every
  change to a fixture is reviewed as a diff with its provenance (oracle versions, Hugging Face
  revision, vendor file sha).
- Engine oracles (vLLM, SGLang) never share one Python environment; run them in the engine's
  official image or in a dedicated venv.
- Nothing here needs a GPU. Do not add a step that does.
- Never commit credentials, `.env`, or `runs/` output. Published evidence goes to
  `smg-project/artifacts`.
- Plain wording: one name per concept (see the vocabulary table in `README.md`); no coinages.
