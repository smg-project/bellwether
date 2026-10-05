# Contributor instructions

These apply to people and to coding agents alike.

## Branches, commits and pull requests

- Never commit to `main`. Every change, including the smallest, goes through a pull request from
  a `<type>/<topic>` branch (`feat/gaps-matrix`, `fix/schema-ids`). `main` is protected: one
  approving review and a green `test` check are required, administrators included, and nothing is
  force-pushed. The only commit that ever landed without a pull request is the initial scaffold on
  the empty repository; it is not a precedent.
- Sign off every commit with your own identity (`git commit -s`). Do not add `Co-Authored-By`
  trailers naming an AI tool.
- One concern per pull request. The body names the fixture ids or the verdict report the change
  rests on.
- Before pushing: `uv run ruff check . && uv run ruff format --check . && uv run pytest -q`.
- Runtime behavior is test-first: write the failing test, then the code.

## Review and approval

- The author never approves their own pull request. Approval comes from a second account acting
  for the maintainers, and a pull request is merged by a person, never automatically.
- An approval is a written review, not a click. Before approving, the reviewer posts a comment
  stating, with links or ids, that: the fixture ids or the verdict report the pull request cites
  were run and are green; no protected path (the harness under `src/`, manifests, existing
  fixtures, `.github/`) changed in a pull request authored by an automated agent; every commit
  carries the sign-off and no AI trailer; and the `STATE.md` tracker the stream owner keeps
  outside this repository was updated. A pull request missing any of these gets "request
  changes", not a question in chat.
- Some changes wait for the project sponsor's own approval even when everything else is green:
  the authority order in a manifest, a waiver, the case schema, the repository's visibility, and
  anything that touches another contributor's open branch.

## Evidence and environments

- Fixtures are evidence. They are produced by `bellwether record`, never edited by hand, and every
  change to a fixture is reviewed as a diff with its provenance (oracle versions, Hugging Face
  revision, vendor file sha).
- Engine oracles (vLLM, SGLang) never share one Python environment; run them in the engine's
  official image or in a dedicated venv.
- Nothing here needs a GPU. Do not add a step that does.
- Never commit credentials, `.env`, or `runs/` output. Published evidence goes to
  `smg-project/artifacts`.
- Plain wording: one name per concept (see the vocabulary table in `README.md`); no coinages.
