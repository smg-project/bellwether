#!/usr/bin/env python3
"""gen_corpus.py: the generated share ("gen", tenant profile "coder") of the agentic session corpus v2.

Each session is one coding task on one synthetic Python repository: the harness system prompt shared by the whole tenant
(instructions + coding rules + the seven tool schemas: the only prefix two sessions share), a first user turn carrying the
task and the repository context (a tree listing, key file excerpts, the failing test's summary), N tool-call turns
(3/4/5/6/8, weights .20/.25/.25/.20/.10) whose recorded assistant output is a short reasoning, one sentence and ONE or TWO
tool calls answered by synthetic tool results (file reads with line numbers, directory listings, search hits, pytest output
with a traceback, edit acknowledgements, unified diffs), and a final report turn without tool calls. Context growth is
front-loaded: the early turns read whole files, the late turns edit, test and diff.

Every text of a session depends only on (seed, index) and the pinned tokenizer: the first and the last prompt of a session
are drawn log-uniform (6,000-15,000 and 15,000-70,000 tokens) and the synthetic texts are sized against the real tokenizer
(the assembled texts are encoded; no words-to-tokens guess) so that both land within 5 % of the plan. Text is plain English
and Python built from >= 40 task/repository themes with per-session identifiers; no random-vocabulary filler.

USAGE (standalone; the corpus builder imports build_session / HARNESS_SYSTEM / TOOLS / session_texts)
  gen_corpus.py --tokenizer DIR --seed 20261022 --n 900 [--jobs 16] [--out sessions.jsonl] [--index I] [--selftest]
  DIR holds tokenizer.json (run with python -I: the tokenizer files are downloaded data).
"""
import argparse, collections, difflib, hashlib, json, math, os, random, re, sys, time

TURN_CHOICE = ([3, 4, 5, 6, 8], [0.20, 0.25, 0.25, 0.20, 0.10])
FIRST_RANGE = (6000, 15000)
FINAL_RANGE = (15000, 70000)
TOOL_TURN_MAX_TOKENS = 1024
FINAL_TURN_MAX_TOKENS = 4096
VERSION = "gen-v2.1"

# ---- the seven tool schemas (Chat Completions function tools) --------------------------------------------------------------
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file from the repository and return its content with line numbers (one line per output line, the number, a tab, then the text). Paths are relative to the repository root. Without start_line/end_line the whole file is returned; the agent should read a file before editing it.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Repository-relative path of the file to read."}, "start_line": {"type": "integer", "description": "First line to return (1-based, inclusive). Omit to start at the top."}, "end_line": {"type": "integer", "description": "Last line to return (1-based, inclusive). Omit to read to the end."}}, "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List the entries of a directory: one entry per line with its kind (file or dir) and size in bytes. Hidden entries and __pycache__ directories are skipped.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Repository-relative directory path; '.' is the repository root."}, "recursive": {"type": "boolean", "description": "Descend into sub-directories (default false)."}}, "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "search_code", "description": "Search the repository for a regular expression and return the matching lines as path:line: text, grouped by file. Use it to find definitions, call sites and strings before reading whole files.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Regular expression (Python syntax) matched against each line."}, "path": {"type": "string", "description": "Directory to search (default: the repository root)."}, "glob": {"type": "string", "description": "Only files whose name matches this glob, for example '*.py'."}, "max_results": {"type": "integer", "description": "Stop after this many matching lines (default 200)."}}, "required": ["pattern"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run the project's test suite with pytest and return its output (the summary, and for failures the traceback in pytest's long format). Narrow the run with a path and a keyword expression while iterating; run the whole suite before reporting.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "A test file or directory (default: the configured test paths)."}, "keyword": {"type": "string", "description": "A pytest -k expression selecting tests by name."}, "verbose": {"type": "boolean", "description": "Print one line per test instead of the dot summary."}}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace one exact occurrence of old_text in the file with new_text. old_text must match the file byte for byte (including indentation) and must occur exactly once; include enough surrounding lines to make it unique. Returns the replaced line range.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Repository-relative path of the file to edit."}, "old_text": {"type": "string", "description": "The exact text to replace."}, "new_text": {"type": "string", "description": "The replacement text."}}, "required": ["path", "old_text", "new_text"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create a file or overwrite an existing one with the given content. Prefer edit_file for changes to existing files; use write_file for new files such as a new test module.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Repository-relative path of the file to write; parent directories are created."}, "content": {"type": "string", "description": "The complete file content."}}, "required": ["path", "content"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "git_diff", "description": "Return the unified diff of the working tree against HEAD (what the session changed so far), optionally restricted to one path. New files appear as additions against /dev/null.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Restrict the diff to this file or directory."}, "staged": {"type": "boolean", "description": "Show the index instead of the working tree (default false)."}}, "required": [], "additionalProperties": False}}},
]

# ---- the harness prompt: one text for the whole tenant, the shared prefix -------------------------------------------------
HARNESS_SYSTEM = """You are Forge, a coding agent that works inside a checked-out software repository on behalf of an engineer. You receive one task per session, carry it out with the tools described below, and finish with a written report. You have no network access, no interactive shell and no way to ask the engineer a question mid-task: when something is ambiguous, take the most conservative reading, state the assumption in your report, and keep the change small enough that it is easy to review and easy to revert.

## Environment

- The repository is mounted read-write at the working directory. Every path you pass to a tool is relative to the repository root; absolute paths and paths that leave the repository are rejected.
- Python 3.12 is installed together with the project's own dependencies. `pytest` is available through the run_tests tool. Nothing else is guaranteed: do not assume a database, a browser, a package index or a compiler.
- The working tree is clean at the start of the session and checked out on the default branch. Your edits stay uncommitted; git_diff shows exactly what the session changed. Nobody else edits the tree while you work.
- Tool results are returned verbatim and can be long. A whole file comes back with line numbers; read it once and keep what you need instead of reading it again. Test output is the raw pytest output.
- Time and randomness in the sandbox are real: tests that depend on wall-clock time or on hash ordering can be flaky, and a flaky test is a finding worth reporting, not something to retry until it passes.

## Workflow

Work in this order and do not skip steps because the task looks small.

1. Understand the task. Read the task text twice. Identify what is being asked (a behaviour change, a bug fix, a refactor, a new flag), what must not change (public names, file formats, existing tests), and how success will be judged (a named test, a described behaviour, a reviewer's reading of the diff).
2. Reproduce. When the task names a failing test or an error message, run that test first and read the traceback before you read any code. The traceback tells you which frames matter. When the task describes a behaviour without a test, write the smallest test that demonstrates the problem before you change anything.
3. Locate. Read the files the traceback and the task point at, then the modules they import and the tests that cover them. Use search_code to find every definition and call site of a name before you change it. Do not guess at the structure of a module you have not read.
4. Change. Make the smallest edit that fixes the cause rather than the symptom. Keep the style of the surrounding code. Do not reformat lines you are not changing, do not reorder imports, and do not rename things the task did not ask you to rename.
5. Verify. Run the narrowest relevant tests while you iterate, then the whole suite once before you report. Read the failures: a test that fails for an unrelated reason is reported as such, not silenced.
6. Report. Finish with the report described at the end of this prompt. The report is what the engineer reads; the diff is what they review.

## Using the tools

- read_file returns the file with line numbers. Line numbers are for your orientation; never include them in old_text or new_text.
- list_dir is for orientation in an unfamiliar tree. Prefer search_code over listing deep trees when you are looking for a name.
- search_code takes a Python regular expression. Escape dots and parentheses when you search for a literal call. Search for both the definition (`def name`, `class Name`) and the uses (`name(`) before changing a name.
- run_tests without arguments runs the configured test paths. While iterating, pass the test file and a -k keyword so the loop stays fast; run the full suite once at the end.
- edit_file replaces exactly one occurrence of old_text. Copy old_text from the read_file output without the line-number prefix and with the original indentation. If the text occurs more than once, widen it with the surrounding lines until it is unique. One logical change per edit call; several edits to one file are fine.
- write_file overwrites the whole file. Use it for new files (a new test module, a new fixture file). Do not use it to rewrite an existing file you could have edited; a full rewrite hides the change from the reviewer.
- git_diff shows the complete change of the session. Read your own diff before you report and remove anything that is not needed: a stray print, a debugging comment, an unrelated whitespace change.

Call one or two tools per turn. Reading two related files in one turn is efficient; editing and testing in the same turn is not, because the edit result may change what you want to test. Explain in one short sentence what you are about to do before each tool call so the transcript stays readable.

## Coding rules

- Match the conventions of the file you are in: naming, quoting, import grouping, the way errors are raised and the way logging is done. A consistent file beats a stylistically better one.
- Type hints on every new or changed signature; `from __future__ import annotations` is in use across the project, so write modern union syntax where the file already does.
- Docstrings on every new public function and class: one sentence saying what it does and, if it is not obvious, one saying what it raises or returns in the edge cases. Do not write docstrings that restate the signature.
- Raise the project's own exception types where they exist. Do not swallow exceptions with a bare except; do not convert a programming error into a return value.
- No new third-party dependencies. If the fix needs something the standard library cannot do, say so in the report and stop.
- Keep functions small. If a fix makes a function longer than about sixty lines, extract a helper with a clear name rather than adding another nesting level.
- Preserve behaviour you were not asked to change. A refactor must not change observable behaviour; a bug fix must not change behaviour for inputs that were already handled correctly. When a fix changes an output format, say so explicitly in the report.
- Comments explain why, not what. Remove comments that your change makes false. Do not leave TODO comments; put open points in the report instead.
- Logging: use the module logger (`log = logging.getLogger(__name__)`), log at info for state changes and at warning for recoverable problems, never log secrets or whole payloads, and prefer structured `extra=` fields over formatted messages where the module already does that.
- Concurrency: when you touch code that runs under a lock or in a thread pool, state in the report which invariant the lock protects and why your change keeps it. A check followed by an action on shared state must happen under the same lock.
- Dates and times: new code produces timezone-aware datetimes in UTC and compares only aware values. Never call `datetime.now()` or `time.time()` directly in code that has a clock parameter or a clock attribute available.
- Text: treat strings as sequences of characters, not bytes. Length limits on names and labels count characters; encode only at the boundary where bytes are needed.
- Resources: files, sockets and locks are released on every path, which in practice means a `with` statement or a `try/finally`. A resource acquired in a loop is released in that iteration.

## Testing rules

- Every bug fix comes with a test that fails before the fix and passes after it, unless such a test already exists and was the one that failed. Name tests after the behaviour (`test_expired_entry_is_not_returned`), not after the function.
- Tests must be deterministic. Inject clocks, seeds and identifiers; do not sleep to wait for a condition; do not depend on dictionary order of unordered inputs.
- Do not weaken an assertion to make a test pass. Do not mark a test as expected-to-fail or skip it unless the task says the test is wrong, and then explain why in the report.
- Keep tests close to the code they cover: a test for `src/pkg/module.py` lives in `tests/test_module.py` and uses the project's fixtures rather than copying their setup.
- When the suite takes long, run the touched test files first, but the final run before the report is always the whole suite.

## Things you must not do

- Do not delete or rename files, branches or tags unless the task asks for it.
- Do not change version numbers, changelogs, lock files or CI configuration unless the task asks for it.
- Do not edit generated files, vendored code or migration files that have already been applied.
- Do not read or copy files that look like credentials (`.env`, key material, token files). If a test needs a secret, use the fixture the project already has for that.
- Do not introduce network calls, background threads or global mutable state to make a test pass.
- Do not claim a test passed that you did not run in this session, and do not describe a verification you did not do.

## When things go wrong

- If the task is impossible as stated (the named function does not exist, the test cannot fail, the request contradicts a documented invariant), stop, do not edit anything, and report what you found with the evidence.
- If a change you made causes new failures you cannot explain within a few turns, revert it with edit_file (restore the original text you read) and report the attempt.
- If the test suite has pre-existing failures unrelated to the task, say so with the test names and continue; do not fix them unless the task asks.

## The final report

Finish the session with one message and no tool call. Use these sections, in this order, with these headings:

- Summary: two or three sentences: what was wrong or what was asked, what you changed, and the state of the tests.
- Root cause (for fixes) or Approach (for features and refactors): what the code did, why that was wrong or insufficient, and how you established it (the traceback, the test, the reading of the code). Be specific: name the function and the lines.
- Changes: one bullet per file with the path and what changed in it. Mention new files and new tests explicitly.
- Verification: the exact test commands you ran and their summary lines; if you ran the whole suite, say so. If anything is still failing, list it.
- Notes: assumptions you made, behaviour that changed on purpose, follow-ups you recommend but did not do, and anything the reviewer should look at closely. Omit the section if there is nothing to say.

Write the report in plain prose with short bullets; include a code block only when a snippet is necessary to understand the change. Do not paste whole files. Do not include tool output verbatim beyond the summary lines. Do not end with a question.

## Abbreviated example of a good session

Task: "test_due_date_is_timezone_aware fails since the parser change." Turn 1: run the named test; the traceback shows a TypeError comparing a naive and an aware datetime inside `parse_due_on`. Turn 2: read the parser module and the test. Turn 3: edit `parse_due_on` to attach UTC to the parsed value; the comparison in the caller is unchanged because it was already correct. Turn 4: run the test file, then the whole suite; both green. Turn 5: read the diff, confirm it touches one function and one import. Final message: the report with the summary, the root cause naming the line that produced the naive value, the single changed file, the two test commands with their summary lines, and a note that stored timestamps without an offset are now interpreted as UTC, which is the behaviour the project's documentation already promised.

Turn structure you are expected to follow: a brief sentence of intent, then the tool call(s); never a tool call without the sentence, never a sentence without a following action except in the final report."""


# ---- repository domains: the vocabulary every synthetic repository is written in --------------------------------------------
# entity: snake, camel, plural, id (field, sample prefix), status field, timestamp field, amount field, name field, extra fields
# (name, type, sample literal). The modules list names the package's modules; the first is the one most sessions read first.
def _e(snake, camel, plural, id_, status, ts, amount, name, extra):
    return {"snake": snake, "camel": camel, "plural": plural, "id": id_, "status": status, "ts": ts, "amount": amount, "name": name, "extra": extra}


DOMAINS = {
    "billing": {
        "pkgs": ["ledgerline", "invoicekit", "tallybox", "billflow", "duesheet"],
        "topic": "invoicing and payment reconciliation",
        "blurb": "a library and command-line tool that issues invoices, records payments and reconciles both against a ledger",
        "entities": [
            _e("invoice", "Invoice", "invoices", ("number", "INV-"), "status", "issued_at", "total_cents", "customer_name",
               [("currency", "str", "'EUR'"), ("due_on", "date", "date(2026, 4, 30)"), ("line_count", "int", "3")]),
            _e("payment", "Payment", "payments", ("payment_id", "PAY-"), "state", "received_at", "amount_cents", "payer",
               [("method", "str", "'card'"), ("reference", "str", "'ref-7731'")]),
            _e("customer", "Customer", "customers", ("customer_id", "CUS-"), "tier", "created_at", "credit_limit_cents", "display_name",
               [("email", "str", "'billing@example.org'"), ("country", "str", "'NL'")]),
            _e("refund", "Refund", "refunds", ("refund_id", "RF-"), "state", "requested_at", "amount_cents", "reason",
               [("invoice_number", "str", "'INV-2041'"), ("approved", "bool", "False")]),
        ],
        "statuses": ["draft", "issued", "paid", "void", "overdue"],
        "verbs": [("issue", "issued"), ("settle", "settled"), ("void", "voided"), ("reconcile", "reconciled"), ("reopen", "reopened")],
        "errors": ["InvoiceStateError", "CurrencyMismatch", "LedgerImbalance", "UnknownCustomer"],
        "services": [("the payment gateway", "gateway"), ("the ledger service", "ledger"), ("the tax table", "taxes")],
        "modules": ["invoices", "payments", "customers", "ledger", "tax", "statements", "currency", "exports", "reconcile"],
    },
    "jobs": {
        "pkgs": ["queueline", "taskforge", "runwell", "dispatchly", "cronbox"],
        "topic": "background job scheduling",
        "blurb": "a job scheduler that leases work to a bounded pool of workers, retries failures and records every attempt",
        "entities": [
            _e("job", "Job", "jobs", ("job_id", "job-"), "state", "enqueued_at", "priority", "name",
               [("attempts", "int", "0"), ("max_attempts", "int", "3"), ("payload", "dict", "{'path': 'reports/q1.csv'}")]),
            _e("worker", "Worker", "workers", ("worker_id", "w-"), "state", "started_at", "capacity", "hostname",
               [("heartbeat_at", "datetime", "datetime(2026, 3, 14, 10, 0, tzinfo=timezone.utc)"), ("tags", "list[str]", "['batch']")]),
            _e("queue", "Queue", "queues", ("queue_name", "q-"), "mode", "created_at", "depth_limit", "label",
               [("visibility_s", "int", "30"), ("dead_letter", "str", "'q-dead'")]),
            _e("lease", "Lease", "leases", ("lease_id", "lease-"), "state", "granted_at", "ttl_s", "owner",
               [("job_id", "str", "'job-1182'"), ("renewals", "int", "0")]),
        ],
        "statuses": ["queued", "leased", "running", "done", "failed", "dead"],
        "verbs": [("enqueue", "enqueued"), ("lease", "leased"), ("complete", "completed"), ("retry", "retried"), ("cancel", "cancelled")],
        "errors": ["JobStateError", "LeaseExpired", "QueueFull", "UnknownWorker"],
        "services": [("the job store", "store"), ("the heartbeat monitor", "monitor"), ("the metrics sink", "metrics")],
        "modules": ["jobs", "workers", "queues", "leases", "scheduler", "retry", "heartbeat", "storage", "cli"],
    },
    "config": {
        "pkgs": ["settingsmith", "confmerge", "layerconf", "profilekit", "knobfile"],
        "topic": "layered application configuration",
        "blurb": "a configuration loader that merges files, profiles and environment overrides into one validated settings object",
        "entities": [
            _e("profile", "Profile", "profiles", ("profile_name", "prof-"), "state", "loaded_at", "precedence", "label",
               [("inherits", "str", "'base'"), ("path", "str", "'config/prod.toml'")]),
            _e("section", "Section", "sections", ("section_key", "sec-"), "kind", "parsed_at", "entry_count", "title",
               [("required", "bool", "True"), ("schema_ref", "str", "'schemas/database.json'")]),
            _e("override", "Override", "overrides", ("override_id", "ovr-"), "origin", "applied_at", "rank", "key",
               [("value", "str", "'postgres://db.internal/app'"), ("env_var", "str", "'APP_DATABASE_URL'")]),
            _e("source", "Source", "sources", ("source_id", "src-"), "kind", "read_at", "byte_size", "path",
               [("fmt", "str", "'toml'"), ("optional", "bool", "False")]),
        ],
        "statuses": ["pending", "parsed", "merged", "validated", "rejected"],
        "verbs": [("load", "loaded"), ("merge", "merged"), ("validate", "validated"), ("apply", "applied"), ("freeze", "frozen")],
        "errors": ["ConfigError", "MissingSection", "TypeMismatch", "UnknownProfile"],
        "services": [("the secrets backend", "secrets"), ("the schema registry", "schemas"), ("the environment", "env")],
        "modules": ["loader", "profiles", "sections", "overrides", "schema", "sources", "render", "defaults", "cli"],
    },
    "httpclient": {
        "pkgs": ["wirecall", "fetchline", "httpkit", "courierpy", "relayreq"],
        "topic": "an HTTP client with retries and connection pooling",
        "blurb": "an HTTP client library with a connection pool, retry policies, request signing and response caching",
        "entities": [
            _e("request", "Request", "requests", ("request_id", "req-"), "state", "sent_at", "timeout_s", "url",
               [("method", "str", "'GET'"), ("headers", "dict", "{'accept': 'application/json'}"), ("body_bytes", "int", "0")]),
            _e("response", "Response", "responses", ("response_id", "resp-"), "outcome", "received_at", "status_code", "url",
               [("elapsed_ms", "float", "41.5"), ("from_cache", "bool", "False")]),
            _e("endpoint", "Endpoint", "endpoints", ("endpoint_key", "ep-"), "health", "checked_at", "rate_limit", "base_url",
               [("region", "str", "'eu-west'"), ("weight", "int", "10")]),
            _e("retry_policy", "RetryPolicy", "retry_policies", ("policy_name", "rp-"), "mode", "created_at", "max_attempts", "label",
               [("base_delay_s", "float", "0.2"), ("max_delay_s", "float", "5.0"), ("retry_on", "list[str]", "['502', '503']")]),
        ],
        "statuses": ["pending", "sent", "succeeded", "failed", "retrying"],
        "verbs": [("send", "sent"), ("retry", "retried"), ("sign", "signed"), ("cancel", "cancelled"), ("cache", "cached")],
        "errors": ["TransientError", "ClientError", "PoolExhausted", "SignatureInvalid"],
        "services": [("the upstream API", "upstream"), ("the connection pool", "pool"), ("the response cache", "cache")],
        "modules": ["client", "pool", "retry", "signing", "responses", "endpoints", "cache", "transport", "cli"],
    },
    "cache": {
        "pkgs": ["memolite", "cachebox", "shardcache", "keepwarm", "ttlstore"],
        "topic": "an in-process cache with sharding and expiry",
        "blurb": "an in-process cache with sharded storage, time-to-live expiry, size-based eviction and snapshotting to disk",
        "entities": [
            _e("entry", "Entry", "entries", ("key", "k:"), "state", "stored_at", "size_bytes", "namespace",
               [("expires_at_ms", "int", "1773482400000"), ("hits", "int", "0")]),
            _e("shard", "Shard", "shards", ("shard_id", "shard-"), "state", "opened_at", "capacity_bytes", "label",
               [("entry_count", "int", "0"), ("lock_wait_ms", "float", "0.0")]),
            _e("namespace", "Namespace", "namespaces", ("namespace_id", "ns-"), "policy", "created_at", "default_ttl_s", "name",
               [("max_entries", "int", "10000"), ("evict", "str", "'lru'")]),
            _e("snapshot", "Snapshot", "snapshots", ("snapshot_id", "snap-"), "state", "taken_at", "byte_size", "path",
               [("entry_count", "int", "1200"), ("compressed", "bool", "True")]),
        ],
        "statuses": ["fresh", "stale", "expired", "evicted", "pinned"],
        "verbs": [("store", "stored"), ("evict", "evicted"), ("refresh", "refreshed"), ("pin", "pinned"), ("snapshot", "snapshotted")],
        "errors": ["CacheError", "EntryTooLarge", "ShardLocked", "UnknownNamespace"],
        "services": [("the backing store", "backing"), ("the snapshot writer", "snapshots"), ("the metrics sink", "metrics")],
        "modules": ["entries", "shards", "namespaces", "expiry", "eviction", "snapshot", "keys", "stats", "cli"],
    },
    "parser": {
        "pkgs": ["markline", "lexkit", "parsewell", "spanparse", "docgrammar"],
        "topic": "parsing a lightweight markup language",
        "blurb": "a tokenizer and parser for a lightweight markup language that produces a document tree with source spans",
        "entities": [
            _e("token", "Token", "tokens", ("token_id", "tok-"), "kind", "scanned_at", "width", "text",
               [("line", "int", "12"), ("column", "int", "4")]),
            _e("node", "Node", "nodes", ("node_id", "n-"), "kind", "built_at", "child_count", "label",
               [("depth", "int", "2"), ("attrs", "dict", "{'class': 'note'}")]),
            _e("span", "Span", "spans", ("span_id", "sp-"), "kind", "created_at", "length", "source_name",
               [("start", "int", "118"), ("end", "int", "131")]),
            _e("document", "Document", "documents", ("doc_id", "doc-"), "state", "parsed_at", "byte_size", "title",
               [("encoding", "str", "'utf-8'"), ("line_count", "int", "240")]),
        ],
        "statuses": ["raw", "scanned", "parsed", "validated", "rejected"],
        "verbs": [("scan", "scanned"), ("parse", "parsed"), ("render", "rendered"), ("normalize", "normalized"), ("validate", "validated")],
        "errors": ["ParseError", "UnexpectedToken", "UnterminatedBlock", "EncodingError"],
        "services": [("the source loader", "sources"), ("the renderer", "renderer"), ("the grammar table", "grammar")],
        "modules": ["lexer", "parser", "nodes", "spans", "render", "grammar", "blocks", "sources", "cli"],
    },
    "storage": {
        "pkgs": ["rowbin", "tablet", "recordkeep", "storeline", "shelfdb"],
        "topic": "an embedded record store",
        "blurb": "an embedded record store with tables, secondary indexes, migrations and a small query layer",
        "entities": [
            _e("record", "Record", "records", ("record_id", "rec-"), "state", "written_at", "version", "table_name",
               [("fields", "dict", "{'name': 'alpha', 'qty': 4}"), ("tombstone", "bool", "False")]),
            _e("table", "Table", "tables", ("table_name", "tbl-"), "state", "created_at", "row_count", "label",
               [("primary_key", "str", "'id'"), ("indexes", "list[str]", "['by_name']")]),
            _e("migration", "Migration", "migrations", ("migration_id", "m-"), "state", "applied_at", "sequence", "description",
               [("checksum", "str", "'9f2c1e'"), ("reversible", "bool", "True")]),
            _e("index_spec", "IndexSpec", "index_specs", ("index_name", "idx-"), "state", "built_at", "cardinality", "column",
               [("unique", "bool", "False"), ("table_name", "str", "'orders'")]),
        ],
        "statuses": ["pending", "applied", "active", "dropped", "corrupt"],
        "verbs": [("insert", "inserted"), ("migrate", "migrated"), ("compact", "compacted"), ("reindex", "reindexed"), ("drop", "dropped")],
        "errors": ["StorageError", "MigrationConflict", "DuplicateKey", "UnknownTable"],
        "services": [("the write-ahead log", "wal"), ("the index builder", "indexer"), ("the file lock", "lockfile")],
        "modules": ["records", "tables", "migrations", "indexes", "query", "wal", "compaction", "schema", "cli"],
    },
    "pipeline": {
        "pkgs": ["flowbatch", "stagerunner", "etlkit", "conveyorpy", "batchline"],
        "topic": "batch data pipelines",
        "blurb": "a batch pipeline runner that moves records through stages with checkpoints, retries and pluggable sinks",
        "entities": [
            _e("batch", "Batch", "batches", ("batch_id", "b-"), "state", "started_at", "record_count", "source_name",
               [("partition", "str", "'2026-03-14'"), ("bytes_in", "int", "48213")]),
            _e("stage", "Stage", "stages", ("stage_name", "st-"), "state", "configured_at", "parallelism", "label",
               [("depends_on", "list[str]", "['extract']"), ("timeout_s", "int", "600")]),
            _e("checkpoint", "Checkpoint", "checkpoints", ("checkpoint_id", "cp-"), "state", "written_at", "offset", "path",
               [("batch_id", "str", "'b-4410'"), ("stage_name", "str", "'transform'")]),
            _e("sink", "Sink", "sinks", ("sink_name", "sink-"), "mode", "opened_at", "flush_every", "target",
               [("fmt", "str", "'parquet'"), ("append", "bool", "True")]),
        ],
        "statuses": ["pending", "running", "checkpointed", "completed", "failed"],
        "verbs": [("extract", "extracted"), ("transform", "transformed"), ("load", "loaded"), ("checkpoint", "checkpointed"), ("replay", "replayed")],
        "errors": ["PipelineError", "StageTimeout", "CheckpointMismatch", "SinkUnavailable"],
        "services": [("the object store", "objects"), ("the catalog", "catalog"), ("the scheduler", "scheduler")],
        "modules": ["batches", "stages", "checkpoints", "sinks", "runner", "sources", "transforms", "manifest", "cli"],
    },
    "metrics": {
        "pkgs": ["gaugekit", "meterline", "tallymetrics", "scrapebox", "histolite"],
        "topic": "application metrics and export",
        "blurb": "a metrics library with counters, gauges and histograms, label validation and exporters for text and JSON formats",
        "entities": [
            _e("counter", "Counter", "counters", ("metric_name", "ctr-"), "state", "created_at", "value", "help_text",
               [("labels", "dict", "{'route': '/items'}"), ("unit", "str", "'requests'")]),
            _e("gauge", "Gauge", "gauges", ("metric_name", "g-"), "state", "updated_at", "value", "help_text",
               [("labels", "dict", "{'pool': 'main'}"), ("min_value", "float", "0.0")]),
            _e("histogram", "Histogram", "histograms", ("metric_name", "h-"), "state", "created_at", "sample_count", "help_text",
               [("buckets", "list[float]", "[0.005, 0.05, 0.5, 5.0]"), ("sum_value", "float", "12.75")]),
            _e("exporter", "Exporter", "exporters", ("exporter_id", "exp-"), "state", "started_at", "interval_s", "target",
               [("fmt", "str", "'text'"), ("include_timestamps", "bool", "False")]),
        ],
        "statuses": ["registered", "active", "stale", "retired", "invalid"],
        "verbs": [("register", "registered"), ("observe", "observed"), ("export", "exported"), ("reset", "reset"), ("retire", "retired")],
        "errors": ["MetricError", "LabelMismatch", "DuplicateMetric", "ExportFailed"],
        "services": [("the scrape endpoint", "scrape"), ("the push gateway", "push"), ("the registry", "registry")],
        "modules": ["counters", "gauges", "histograms", "registry", "labels", "exporters", "textfmt", "jsonfmt", "cli"],
    },
    "auth": {
        "pkgs": ["gatekit", "sessionly", "grantline", "scopeguard", "keyringpy"],
        "topic": "authentication and authorization",
        "blurb": "an authentication library that issues sessions and tokens, checks scopes and records grants with an audit trail",
        "entities": [
            _e("user", "User", "users", ("user_id", "u-"), "state", "created_at", "login_count", "email",
               [("roles", "list[str]", "['reader']"), ("locked", "bool", "False")]),
            _e("session", "Session", "sessions", ("session_id", "s-"), "state", "started_at", "idle_timeout_s", "user_id",
               [("ip_hash", "str", "'3b1f'"), ("remember", "bool", "False")]),
            _e("token", "Token", "tokens", ("token_id", "t-"), "state", "issued_at", "ttl_s", "subject",
               [("scopes", "list[str]", "['read:items']"), ("audience", "str", "'api'")]),
            _e("grant", "Grant", "grants", ("grant_id", "g-"), "state", "granted_at", "rank", "scope",
               [("granted_by", "str", "'u-100'"), ("expires_on", "date", "date(2026, 12, 31)")]),
        ],
        "statuses": ["pending", "active", "expired", "revoked", "locked"],
        "verbs": [("issue", "issued"), ("revoke", "revoked"), ("refresh", "refreshed"), ("verify", "verified"), ("lock", "locked")],
        "errors": ["AuthError", "TokenExpired", "ScopeDenied", "UnknownUser"],
        "services": [("the identity provider", "idp"), ("the key store", "keys"), ("the audit log", "audit")],
        "modules": ["users", "sessions", "tokens", "grants", "scopes", "audit", "keys", "policy", "cli"],
    },
    "filesync": {
        "pkgs": ["chunkmirror", "syncwell", "manifestio", "copyline", "driftsync"],
        "topic": "file synchronization between directories",
        "blurb": "a file synchronization tool that splits files into chunks, compares manifests and transfers only the differences",
        "entities": [
            _e("file_entry", "FileEntry", "file_entries", ("path", "docs/"), "state", "modified_at", "size_bytes", "relative_path",
               [("digest", "str", "'sha256:4e1a'"), ("mode", "int", "0o644")]),
            _e("chunk", "Chunk", "chunks", ("chunk_id", "c-"), "state", "hashed_at", "length", "digest",
               [("offset", "int", "0"), ("file_path", "str", "'docs/guide.md'")]),
            _e("manifest", "Manifest", "manifests", ("manifest_id", "man-"), "state", "generated_at", "entry_count", "root",
               [("version", "int", "2"), ("generator", "str", "'cli'")]),
            _e("transfer", "Transfer", "transfers", ("transfer_id", "tr-"), "state", "started_at", "bytes_total", "destination",
               [("bytes_done", "int", "0"), ("resume", "bool", "True")]),
        ],
        "statuses": ["pending", "hashing", "transferring", "verified", "failed"],
        "verbs": [("scan", "scanned"), ("hash", "hashed"), ("transfer", "transferred"), ("verify", "verified"), ("prune", "pruned")],
        "errors": ["SyncError", "DigestMismatch", "TransferInterrupted", "UnknownManifest"],
        "services": [("the remote endpoint", "remote"), ("the chunk store", "chunks"), ("the progress reporter", "progress")],
        "modules": ["entries", "chunks", "manifest", "transfer", "digest", "planner", "walker", "resume", "cli"],
    },
    "inventory": {
        "pkgs": ["stockline", "binkeeper", "shelfcount", "reservelot", "warekit"],
        "topic": "inventory and stock reservations",
        "blurb": "an inventory service that tracks stock per warehouse, reserves units for orders and reconciles counts",
        "entities": [
            _e("product", "Product", "products", ("sku", "SKU-"), "state", "created_at", "unit_price_cents", "title",
               [("weight_g", "int", "250"), ("category", "str", "'tools'")]),
            _e("stock_level", "StockLevel", "stock_levels", ("level_id", "lvl-"), "state", "counted_at", "on_hand", "sku",
               [("reserved", "int", "0"), ("warehouse_code", "str", "'WH-1'")]),
            _e("warehouse", "Warehouse", "warehouses", ("code", "WH-"), "state", "opened_at", "capacity_units", "name",
               [("region", "str", "'north'"), ("accepts_returns", "bool", "True")]),
            _e("reservation", "Reservation", "reservations", ("reservation_id", "rsv-"), "state", "placed_at", "quantity", "order_ref",
               [("sku", "str", "'SKU-7731'"), ("expires_at", "datetime", "datetime(2026, 3, 15, 9, 0, tzinfo=timezone.utc)")]),
        ],
        "statuses": ["pending", "reserved", "committed", "released", "backordered"],
        "verbs": [("reserve", "reserved"), ("commit", "committed"), ("release", "released"), ("count", "counted"), ("restock", "restocked")],
        "errors": ["InventoryError", "InsufficientStock", "UnknownSku", "ReservationExpired"],
        "services": [("the order service", "orders"), ("the warehouse feed", "feed"), ("the pricing table", "pricing")],
        "modules": ["products", "stock", "warehouses", "reservations", "counts", "pricing", "feeds", "reports", "cli"],
    },
}


# ---- per-session naming ------------------------------------------------------------------------------------------------------
# synonym tables: each session picks one word per role, so two sessions of one domain do not share function names
NAMING = {
    "validate": ["validate", "check", "ensure_valid", "verify"],
    "load": ["load", "read", "import"],
    "save": ["save", "write", "dump"],
    "find": ["find", "lookup", "get", "locate"],
    "format": ["format", "render", "describe"],
    "total": ["total", "sum", "aggregate"],
    "summarize": ["summarize", "tally", "count"],
    "list": ["list", "page", "browse"],
    "fixture": ["sample", "example", "fake", "stub"],
    "store": ["Repository", "Store", "Registry", "Book"],
    "now": ["_now", "_utcnow", "_clock_now"],
}
PY_VERSIONS = ["3.11", "3.12", "3.12", "3.13"]
PEOPLE = ["Maya", "Jonas", "Priya", "Tomasz", "Aiko", "Lucas", "Noor", "Elena", "Rafael", "Ines", "Kwame", "Sofia"]


def sid_hash(*parts):
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def camel(s):
    return "".join(w.capitalize() for w in s.split("_"))


def pick_distinct(rnd, seq, k):
    seq = list(seq)
    rnd.shuffle(seq)
    return seq[:k]


class Ctx:
    """Everything the generators need to know about one session's repository: domain vocabulary, chosen names, the random stream."""

    def __init__(self, rnd, domain_key, index):
        self.rnd = rnd
        self.index = index
        self.d = DOMAINS[domain_key]
        self.domain_key = domain_key
        self.pkg = rnd.choice(self.d["pkgs"])
        self.py = rnd.choice(PY_VERSIONS)
        self.names = {k: rnd.choice(v) for k, v in NAMING.items()}
        self.ents = list(self.d["entities"])
        rnd.shuffle(self.ents)
        self.main = self.ents[0]
        mods = list(self.d["modules"])
        cli = "cli" if "cli" in mods else None
        mods = [m for m in mods if m != "cli"]
        rnd.shuffle(mods)
        self.modules = mods[:6] + (["cli"] if cli or True else [])
        # the module each entity lives in: entity i -> modules[i]
        self.ent_module = {e["snake"]: self.modules[i] for i, e in enumerate(self.ents)}
        self.errors = list(self.d["errors"])
        self.base_error = self.errors[0]
        self.services = list(self.d["services"])
        self.author = rnd.choice(PEOPLE)
        self.year = 2026
        self.docvar = rnd.randrange(3)  # docstring phrasing variant of this session
        self.nums = rnd.randrange(100, 900)  # sample number base of this session

    def num(self, k=0):
        return self.nums * 7 + k * 13 + 41

    def lit(self, ent, field):
        """A sample literal for a field of an entity (tests, docstrings)."""
        fid, pref = ent["id"]
        if field == fid:
            return f"'{pref}{self.num(1) % 9000 + 1000}'"
        if field == ent["status"]:
            return f"'{self.d['statuses'][1]}'"
        if field == ent["ts"]:
            return "datetime(2026, 3, 14, 10, 22, tzinfo=timezone.utc)"
        if field == ent["amount"]:
            return str(self.num(2) % 50000 + 100)
        if field == ent["name"]:
            return f"'{ent['camel'].lower()} {self.num(3) % 97}'"
        for n, t, s in ent["extra"]:
            if n == field:
                return s
        return "None"

    def fields(self, ent):
        """[(field, annotation, sample)] of the entity's dataclass, in declaration order."""
        fid, _ = ent["id"]
        out = [(fid, "str"), (ent["name"], "str"), (ent["status"], "str"), (ent["ts"], "datetime"), (ent["amount"], "int")]
        out += [(n, t) for n, t, _ in ent["extra"]]
        return [(n, t, self.lit(ent, n)) for n, t in out]

    def module_of(self, ent):
        return self.ent_module.get(ent["snake"], self.modules[0])

    def src(self, module):
        return f"src/{self.pkg}/{module}.py"

    def test_path(self, module):
        return f"tests/test_{module}.py"

    def verb(self, i=0):
        return self.d["verbs"][i % len(self.d["verbs"])]

    def statuses(self):
        return self.d["statuses"]


# ---- docstring phrasing -----------------------------------------------------------------------------------------------------
def doc_sentence(ctx, kind, ent, **kw):
    """One of three phrasings per kind, fixed per session (ctx.docvar), so sessions differ in prose while staying natural."""
    e, E, P = ent["snake"], ent["camel"], ent["plural"]
    v = kw.get("verb", ("update", "updated"))
    topic = ctx.d["topic"]
    table = {
        "validate": [f"Raise :class:`{ctx.base_error}` when the {e} violates one of the invariants documented in ``docs/{ctx.module_of(ent)}.md``.",
                     f"Check the {e} before it enters the store; the rules mirror the ones the {topic} documentation lists.",
                     f"Validate a {e} record and raise :class:`{ctx.base_error}` with a readable message on the first violation."],
        "to_dict": [f"Return the JSON-serialisable form of the {e} as the export and the CLI print it.",
                    f"Convert the {e} into a plain dictionary; timestamps become ISO 8601 strings.",
                    f"Serialise the {e} for the on-disk format (version 2, see ``docs/format.md``)."],
        "from_dict": [f"Build a {E} from its dictionary form, tolerating the fields older exports did not write.",
                      f"Inverse of the dictionary conversion; unknown keys are ignored so old files keep loading.",
                      f"Construct a {E} from a decoded JSON object, filling defaults for optional fields."],
        "load": [f"Read {P} from a JSON-lines file; malformed lines are skipped with a warning.",
                 f"Load every {e} stored in ``path`` (one JSON object per line) in file order.",
                 f"Return the {P} found in ``path``; an empty or missing file yields an empty list."],
        "save": [f"Write {P} to ``path`` as JSON lines, one {e} per line, replacing the file atomically.",
                 f"Persist the given {P}; the file is written to a temporary name and renamed into place.",
                 f"Store {P} on disk in the format :func:`load_{P}` reads."],
        "find": [f"Return the {e} with the given identifier or ``None`` when there is no such {e}.",
                 f"Look a {e} up by its identifier; the scan is linear because the lists are small.",
                 f"Find one {e} by id; callers that need many lookups should build an index first."],
        "by_status": [f"Return the {P} whose {ent['status']} equals ``{ent['status']}``, preserving the input order.",
                      f"Filter {P} by {ent['status']}; the comparison is exact and case-sensitive.",
                      f"Select the {P} in a given {ent['status']}."],
        "total": [f"Sum the ``{ent['amount']}`` of the given {P}.",
                  f"Return the total ``{ent['amount']}`` over {P}; an empty iterable totals zero.",
                  f"Add up ``{ent['amount']}`` across {P} without touching the records."],
        "transition": [f"Mark the {e} as {v[1]} and return the updated copy; the original is not modified.",
                       f"Apply the ``{v[0]}`` transition to a {e}, enforcing the allowed source states.",
                       f"Return a copy of the {e} after it has been {v[1]}, or raise when the transition is not allowed."],
        "format": [f"Render a one-line human-readable description of the {e} for logs and the CLI.",
                   f"Return the {e} as the text the ``show`` command prints.",
                   f"Format the {e} for display; the layout is fixed-width so tables align."],
        "sort": [f"Return the {P} ordered by ``{ent['ts']}`` and then by identifier for a stable output.",
                 f"Sort {P} oldest first; ties are broken by the identifier.",
                 f"Order {P} the way the reports expect them."],
        "group": [f"Group {P} by ``{kw.get('field', ent['status'])}``, keeping the first-seen order of the keys.",
                  f"Return a mapping from ``{kw.get('field', ent['status'])}`` to the {P} with that value.",
                  f"Bucket {P} by ``{kw.get('field', ent['status'])}``."],
        "iter_since": [f"Yield the {P} whose ``{ent['ts']}`` is at or after ``since`` in chronological order.",
                       f"Iterate over {P} newer than ``since``; ``since`` must be timezone-aware.",
                       f"Walk the {P} from ``since`` onwards without materialising the whole list."],
        "summarize": [f"Return per-{ent['status']} counts and the total ``{ent['amount']}`` of the {P}.",
                      f"Produce the small summary dictionary the dashboard reads for {P}.",
                      f"Count {P} by {ent['status']} and total their ``{ent['amount']}``."],
        "repository": [f"In-memory collection of {P} keyed by identifier; the file-backed store has the same interface.",
                       f"A small keyed collection of {P} used by the tests and by the CLI's dry-run mode.",
                       f"Holds {P} by identifier and enforces uniqueness on insert."],
        "parse_line": [f"Parse one line of the legacy CSV export into a {E}.",
                       f"Build a {E} from a line of the comma-separated import format; whitespace around fields is ignored.",
                       f"Decode a CSV line of the v1 export format into a {E}."],
        "merge": [f"Apply a partial update to a {e}; only the keys present in ``changes`` are touched.",
                  f"Return the {e} with ``changes`` applied, rejecting keys that are not fields.",
                  f"Merge a dictionary of changes into a {e} and return the new copy."],
        "cmd": [f"Entry point of the ``{v[0]}`` sub-command; returns the process exit code.",
                f"Handle ``{ctx.pkg} {v[0]}``: parse the arguments, do the work, print the outcome.",
                f"CLI handler for ``{v[0]}``; prints one line per affected {e}."],
        "client": [f"Thin client for {kw.get('service', 'the service')}; retries are the caller's responsibility.",
                   f"Talks to {kw.get('service', 'the service')} over its JSON interface.",
                   f"Client for {kw.get('service', 'the service')} with a fixed timeout per call."],
        "module": [f"{E} handling for the {topic} package: construction, validation, persistence and the state transitions.",
                   f"Everything about {P}: the record type, its invariants, the transitions and the file format.",
                   f"The {e} model and the helpers the rest of the package uses to read, change and persist {P}."],
    }
    return table[kind][ctx.docvar % 3]


# ---- code shapes: each returns one top-level unit of a module ---------------------------------------------------------------
def sh_dataclass(ctx, ent):
    f = ctx.fields(ent)
    lines = ["@dataclass(frozen=True)", f"class {ent['camel']}:", f'    """One {ent["snake"].replace("_", " ")} as the package stores it; instances are immutable, use :func:`dataclasses.replace`."""', ""]
    lines += [f"    {n}: {t}" for n, t, _ in f]
    st = ctx.statuses()
    lines += ["", "    @property", f"    def is_{st[1]}(self) -> bool:", f"        return self.{ent['status']} == \"{st[1]}\"", "",
              f"    def with_{ent['status']}(self, {ent['status']}: str) -> \"{ent['camel']}\":", f"        if {ent['status']} not in {ent['status'].upper()}_VALUES:",
              f"            raise {ctx.base_error}(f\"unknown {ent['status']} {{{ent['status']}!r}} for {ent['snake']} {{self.{ent['id'][0]}}}\")",
              f"        return replace(self, {ent['status']}={ent['status']})"]
    return "\n".join(lines)


def sh_constants(ctx, ent):
    st = ctx.statuses()
    lines = [f"{ent['status'].upper()}_VALUES = frozenset({{{', '.join(repr(s) for s in st)}}})",
             f"TERMINAL_{ent['status'].upper()} = frozenset({{{', '.join(repr(s) for s in st[-2:])}}})",
             f"DEFAULT_PAGE_SIZE = {ctx.rnd.choice([20, 25, 50, 100])}",
             f"MAX_{ent['name'].upper()}_CHARS = {ctx.rnd.choice([64, 80, 120, 200])}"]
    return "\n".join(lines)


def sh_validate(ctx, ent):
    n = ctx.names["validate"]
    e, E = ent["snake"], ent["camel"]
    lines = [f"def {n}_{e}({e}: {E}) -> None:", f'    """{doc_sentence(ctx, "validate", ent)}"""',
             f"    if not {e}.{ent['id'][0]}.startswith(\"{ent['id'][1]}\"):",
             f"        raise {ctx.base_error}(f\"{e} id {{{e}.{ent['id'][0]}!r}} does not carry the {ent['id'][1]} prefix\")",
             f"    if not {e}.{ent['name']}.strip():", f"        raise {ctx.base_error}(f\"{e} {{{e}.{ent['id'][0]}}} has an empty {ent['name']}\")",
             f"    if len({e}.{ent['name']}) > MAX_{ent['name'].upper()}_CHARS:",
             f"        raise {ctx.base_error}(f\"{e} {{{e}.{ent['id'][0]}}}: {ent['name']} longer than {{MAX_{ent['name'].upper()}_CHARS}} characters\")",
             f"    if {e}.{ent['amount']} < 0:", f"        raise {ctx.base_error}(f\"{e} {{{e}.{ent['id'][0]}}} has a negative {ent['amount']}\")",
             f"    if {e}.{ent['status']} not in {ent['status'].upper()}_VALUES:",
             f"        raise {ctx.base_error}(f\"{e} {{{e}.{ent['id'][0]}}} has unknown {ent['status']} {{{e}.{ent['status']}!r}}\")",
             f"    if {e}.{ent['ts']}.tzinfo is None:", f"        raise {ctx.base_error}(f\"{e} {{{e}.{ent['id'][0]}}}: {ent['ts']} must be timezone-aware\")"]
    return "\n".join(lines)


def sh_to_dict(ctx, ent):
    e = ent["snake"]
    lines = [f"def {e}_to_dict({e}: {ent['camel']}) -> dict[str, object]:", f'    """{doc_sentence(ctx, "to_dict", ent)}"""',
             f"    data = asdict({e})", f"    data[\"{ent['ts']}\"] = {e}.{ent['ts']}.isoformat()"]
    for n, t, _ in ent["extra"]:
        if t == "date":
            lines.append(f"    data[\"{n}\"] = {e}.{n}.isoformat()")
        elif t == "datetime":
            lines.append(f"    data[\"{n}\"] = {e}.{n}.isoformat()")
    lines += ["    data[\"schema\"] = 2", "    return data"]
    return "\n".join(lines)


def sh_from_dict(ctx, ent):
    e, E = ent["snake"], ent["camel"]
    f = ctx.fields(ent)
    lines = [f"def {e}_from_dict(data: dict[str, object]) -> {E}:", f'    """{doc_sentence(ctx, "from_dict", ent)}"""',
             f"    missing = [k for k in ({', '.join(repr(n) for n, _, _ in f[:5])}) if k not in data]", "    if missing:",
             f"        raise {ctx.base_error}(f\"{e} record is missing {{', '.join(missing)}}\")", f"    return {E}("]
    for n, t, s in f:
        if t == "datetime":
            lines.append(f"        {n}=_parse_ts(str(data[\"{n}\"])),")
        elif t == "date":
            lines.append(f"        {n}=date.fromisoformat(str(data.get(\"{n}\", \"{s[5:9] if s.startswith('date(') else '2026'}-01-01\"))),")
        elif t == "int":
            lines.append(f"        {n}=int(data.get(\"{n}\", 0)),")
        elif t == "float":
            lines.append(f"        {n}=float(data.get(\"{n}\", 0.0)),")
        elif t == "bool":
            lines.append(f"        {n}=bool(data.get(\"{n}\", False)),")
        elif t.startswith("list"):
            lines.append(f"        {n}=list(data.get(\"{n}\") or []),")
        elif t == "dict":
            lines.append(f"        {n}=dict(data.get(\"{n}\") or {{}}),")
        else:
            lines.append(f"        {n}=str(data.get(\"{n}\", \"\")),")
    lines.append("    )")
    return "\n".join(lines)


def sh_parse_ts(ctx, ent):
    return "\n".join(["def _parse_ts(raw: str) -> datetime:", '    """Parse an ISO 8601 timestamp; values without an offset are taken as UTC, as the export format promises."""',
                      "    value = datetime.fromisoformat(raw.replace(\"Z\", \"+00:00\"))", "    if value.tzinfo is None:",
                      "        value = value.replace(tzinfo=timezone.utc)", "    return value"])


def sh_load(ctx, ent):
    n = ctx.names["load"]
    e, P = ent["snake"], ent["plural"]
    lines = [f"def {n}_{P}(path: Path) -> list[{ent['camel']}]:", f'    """{doc_sentence(ctx, "load", ent)}"""', "    if not path.exists():",
             f"        log.info(\"no {P} file at %s\", path)", "        return []", f"    out: list[{ent['camel']}] = []",
             "    with path.open(encoding=\"utf-8\") as fh:", "        for lineno, line in enumerate(fh, start=1):", "            line = line.strip()",
             "            if not line or line.startswith(\"#\"):", "                continue", "            try:",
             f"                out.append({e}_from_dict(json.loads(line)))", f"            except (ValueError, {ctx.base_error}) as exc:",
             f"                log.warning(\"skipping {e} on line %d of %s: %s\", lineno, path, exc)", "    return out"]
    return "\n".join(lines)


def sh_save(ctx, ent):
    n = ctx.names["save"]
    e, P = ent["snake"], ent["plural"]
    lines = [f"def {n}_{P}(path: Path, {P}: Iterable[{ent['camel']}]) -> int:", f'    """{doc_sentence(ctx, "save", ent)}"""',
             "    tmp = path.with_suffix(path.suffix + \".tmp\")", "    count = 0", "    with tmp.open(\"w\", encoding=\"utf-8\") as fh:",
             f"        for {e} in {P}:", f"            fh.write(json.dumps({e}_to_dict({e}), sort_keys=True))", "            fh.write(\"\\n\")",
             "            count += 1", "    tmp.replace(path)", f"    log.info(\"wrote %d {P} to %s\", count, path)", "    return count"]
    return "\n".join(lines)


def sh_find(ctx, ent):
    n = ctx.names["find"]
    e, P, fid = ent["snake"], ent["plural"], ent["id"][0]
    lines = [f"def {n}_{e}({P}: Iterable[{ent['camel']}], {fid}: str) -> {ent['camel']} | None:", f'    """{doc_sentence(ctx, "find", ent)}"""',
             f"    for {e} in {P}:", f"        if {e}.{fid} == {fid}:", f"            return {e}", "    return None"]
    return "\n".join(lines)


def sh_by_status(ctx, ent):
    e, P, s = ent["snake"], ent["plural"], ent["status"]
    lines = [f"def {P}_with_{s}({P}: Iterable[{ent['camel']}], {s}: str) -> list[{ent['camel']}]:", f'    """{doc_sentence(ctx, "by_status", ent)}"""',
             f"    if {s} not in {s.upper()}_VALUES:", f"        raise {ctx.base_error}(f\"unknown {s} {{{s}!r}}\")",
             f"    return [{e} for {e} in {P} if {e}.{s} == {s}]"]
    return "\n".join(lines)


def sh_total(ctx, ent):
    n = ctx.names["total"]
    e, P, a = ent["snake"], ent["plural"], ent["amount"]
    lines = [f"def {n}_{a}({P}: Iterable[{ent['camel']}]) -> int:", f'    """{doc_sentence(ctx, "total", ent)}"""',
             f"    return sum({e}.{a} for {e} in {P})"]
    return "\n".join(lines)


def sh_transition(ctx, ent, vi):
    v, past = ctx.verb(vi)
    e, E, s = ent["snake"], ent["camel"], ent["status"]
    st = ctx.statuses()
    src = st[max(0, (vi % len(st)) - 1)]
    dst = st[vi % len(st)]
    lines = [f"def {v}_{e}({e}: {E}, *, at: datetime | None = None) -> {E}:", f'    """{doc_sentence(ctx, "transition", ent, verb=(v, past))}"""',
             f"    if {e}.{s} != \"{src}\":", f"        raise {ctx.base_error}(f\"cannot {v} {e} {{{e}.{ent['id'][0]}}} in {s} {{{e}.{s}!r}}\")",
             f"    when = at or {ctx.names['now']}()", "    if when.tzinfo is None:", f"        raise {ctx.base_error}(\"{v} needs a timezone-aware timestamp\")",
             f"    updated = replace({e}, {s}=\"{dst}\", {ent['ts']}=when)",
             f"    log.info(\"{past} {e}\", extra={{\"{e}_id\": {e}.{ent['id'][0]}, \"{s}\": updated.{s}}})", "    return updated"]
    return "\n".join(lines)


def sh_format(ctx, ent):
    n = ctx.names["format"]
    e = ent["snake"]
    lines = [f"def {n}_{e}({e}: {ent['camel']}) -> str:", f'    """{doc_sentence(ctx, "format", ent)}"""',
             f"    when = {e}.{ent['ts']}.strftime(\"%Y-%m-%d %H:%M\")",
             f"    return f\"{{{e}.{ent['id'][0]}:<14}} {{{e}.{ent['status']}:<10}} {{{e}.{ent['amount']}:>10}}  {{when}}  {{{e}.{ent['name']}}}\""]
    return "\n".join(lines)


def sh_sort(ctx, ent):
    e, P = ent["snake"], ent["plural"]
    lines = [f"def sorted_{P}({P}: Iterable[{ent['camel']}], *, newest_first: bool = False) -> list[{ent['camel']}]:", f'    """{doc_sentence(ctx, "sort", ent)}"""',
             f"    ordered = sorted({P}, key=lambda {e}: ({e}.{ent['ts']}, {e}.{ent['id'][0]}))", "    if newest_first:", "        ordered.reverse()", "    return ordered"]
    return "\n".join(lines)


def sh_group(ctx, ent, field=None):
    e, P = ent["snake"], ent["plural"]
    field = field or ent["status"]
    lines = [f"def group_{P}_by_{field}({P}: Iterable[{ent['camel']}]) -> dict[str, list[{ent['camel']}]]:", f'    """{doc_sentence(ctx, "group", ent, field=field)}"""',
             f"    groups: dict[str, list[{ent['camel']}]] = {{}}", f"    for {e} in {P}:", f"        groups.setdefault(str({e}.{field}), []).append({e})", "    return groups"]
    return "\n".join(lines)


def sh_iter_since(ctx, ent):
    e, P = ent["snake"], ent["plural"]
    lines = [f"def iter_{P}_since({P}: Iterable[{ent['camel']}], since: datetime) -> Iterator[{ent['camel']}]:", f'    """{doc_sentence(ctx, "iter_since", ent)}"""',
             "    if since.tzinfo is None:", f"        raise {ctx.base_error}(\"since must be timezone-aware\")",
             f"    for {e} in sorted_{P}({P}):", f"        if {e}.{ent['ts']} >= since:", f"            yield {e}"]
    return "\n".join(lines)


def sh_summarize(ctx, ent):
    n = ctx.names["summarize"]
    e, P = ent["snake"], ent["plural"]
    lines = [f"def {n}_{P}({P}: Iterable[{ent['camel']}]) -> dict[str, int]:", f'    """{doc_sentence(ctx, "summarize", ent)}"""',
             "    summary: dict[str, int] = {\"count\": 0, \"" + ent["amount"] + "\": 0}", f"    for {e} in {P}:", "        summary[\"count\"] += 1",
             f"        summary[\"{ent['amount']}\"] += {e}.{ent['amount']}", f"        key = f\"{ent['status']}:{{{e}.{ent['status']}}}\"",
             "        summary[key] = summary.get(key, 0) + 1", "    return summary"]
    return "\n".join(lines)


def sh_repository(ctx, ent):
    e, E, P, fid = ent["snake"], ent["camel"], ent["plural"], ent["id"][0]
    R = E + ctx.names["store"]
    lines = [f"class {R}:", f'    """{doc_sentence(ctx, "repository", ent)}"""', "", "    def __init__(self) -> None:", f"        self._items: dict[str, {E}] = {{}}", "",
             "    def __len__(self) -> int:", "        return len(self._items)", "", f"    def add(self, {e}: {E}) -> None:", f"        if {e}.{fid} in self._items:",
             f"            raise {ctx.base_error}(f\"duplicate {e} {{{e}.{fid}}}\")", f"        self._items[{e}.{fid}] = {e}", "",
             f"    def get(self, {fid}: str) -> {E}:", "        try:", f"            return self._items[{fid}]", "        except KeyError:",
             f"            raise {ctx.base_error}(f\"no {e} {{{fid}}}\") from None", "", f"    def remove(self, {fid}: str) -> {E}:",
             f"        {e} = self.get({fid})", f"        del self._items[{fid}]", f"        return {e}", "", f"    def all(self) -> list[{E}]:",
             "        return list(self._items.values())"]
    return "\n".join(lines)


def sh_client(ctx, si):
    svc, short = ctx.services[si % len(ctx.services)]
    C = camel(short) + "Client"
    ent = ctx.main
    e = ent["snake"]
    lines = [f"class {C}:", f'    """{doc_sentence(ctx, "client", ent, service=svc)}"""', "", f"    def __init__(self, base_url: str, *, timeout_s: float = {ctx.rnd.choice([2.0, 5.0, 10.0])}) -> None:",
             "        self.base_url = base_url.rstrip(\"/\")", "        self.timeout_s = timeout_s", "        self._transport = Transport(timeout_s=timeout_s)", "",
             f"    def fetch_{e}(self, {ent['id'][0]}: str) -> dict[str, object]:", f"        response = self._transport.get(f\"{{self.base_url}}/{ent['plural']}/{{{ent['id'][0]}}}\")",
             "        if response.status_code == 404:", f"            raise {ctx.errors[-1]}({ent['id'][0]})", "        response.raise_for_status()", "        return response.json()", "",
             f"    def push_{e}(self, payload: dict[str, object]) -> str:", f"        response = self._transport.post(f\"{{self.base_url}}/{ent['plural']}\", json=payload)",
             "        response.raise_for_status()", f"        return str(response.json()[\"{ent['id'][0]}\"])"]
    return "\n".join(lines)


def sh_retry_helper(ctx):
    lines = ["def _call_with_retry(fn: Callable[[], T], *, attempts: int = 3, base_delay_s: float = 0.2, sleep: Callable[[float], None] = time.sleep) -> T:",
             '    """Call ``fn`` up to ``attempts`` times on :class:`' + ctx.errors[0] + '`, sleeping base_delay_s * 2**n between attempts."""', "    last: Exception | None = None",
             "    for n in range(attempts):", "        try:", "            return fn()", f"        except {ctx.errors[0]} as exc:", "            last = exc",
             "            if n + 1 < attempts:", "                sleep(base_delay_s * (2 ** n))", "    assert last is not None", "    raise last"]
    return "\n".join(lines)


def sh_parse_line(ctx, ent):
    e, E = ent["snake"], ent["camel"]
    f = ctx.fields(ent)[:5]
    lines = [f"def parse_{e}_line(line: str) -> {E}:", f'    """{doc_sentence(ctx, "parse_line", ent)}"""', "    parts = [p.strip() for p in line.rstrip(\"\\n\").split(\",\")]",
             f"    if len(parts) < {len(f)}:", f"        raise {ctx.base_error}(f\"expected {len(f)} fields, got {{len(parts)}}: {{line!r}}\")",
             f"    {', '.join(n for n, _, _ in f)} = parts[:{len(f)}]", f"    return {E}("]
    for n, t, s in f:
        if t == "datetime":
            lines.append(f"        {n}=_parse_ts({n}),")
        elif t == "int":
            lines.append(f"        {n}=int({n}),")
        else:
            lines.append(f"        {n}={n},")
    for n, t, s in ent["extra"]:
        lines.append(f"        {n}={s},")
    lines.append("    )")
    return "\n".join(lines)


def sh_merge(ctx, ent):
    e, E = ent["snake"], ent["camel"]
    lines = [f"def apply_changes({e}: {E}, changes: dict[str, object]) -> {E}:", f'    """{doc_sentence(ctx, "merge", ent)}"""',
             f"    allowed = {{f.name for f in fields({E})}}", "    unknown = sorted(set(changes) - allowed)", "    if unknown:",
             f"        raise {ctx.base_error}(f\"unknown {e} fields: {{', '.join(unknown)}}\")", f"    return replace({e}, **changes)"]
    return "\n".join(lines)


def sh_cmd(ctx, ent, vi):
    v, past = ctx.verb(vi)
    e, P = ent["snake"], ent["plural"]
    lines = [f"def cmd_{v}(args: argparse.Namespace) -> int:", f'    """{doc_sentence(ctx, "cmd", ent, verb=(v, past))}"""',
             f"    {P} = {ctx.names['load']}_{P}(Path(args.file))", f"    target = {ctx.names['find']}_{e}({P}, args.{ent['id'][0]})", "    if target is None:",
             f"        print(f\"no {e} {{args.{ent['id'][0]}}}\", file=sys.stderr)", "        return 2", "    try:", f"        updated = {v}_{e}(target)",
             f"    except {ctx.base_error} as exc:", "        print(str(exc), file=sys.stderr)", "        return 1",
             f"    {ctx.names['save']}_{P}(Path(args.file), [updated if x.{ent['id'][0]} == updated.{ent['id'][0]} else x for x in {P}])",
             f"    print({ctx.names['format']}_{e}(updated))", "    return 0"]
    return "\n".join(lines)


def sh_build_parser(ctx):
    ent = ctx.main
    lines = ["def build_parser() -> argparse.ArgumentParser:", f'    """The ``{ctx.pkg}`` command line: one sub-command per transition plus ``show`` and ``export``."""',
             f"    parser = argparse.ArgumentParser(prog=\"{ctx.pkg}\", description=\"{ctx.d['blurb'][0].upper() + ctx.d['blurb'][1:]}.\")",
             f"    parser.add_argument(\"--file\", default=\"{ent['plural']}.jsonl\", help=\"the {ent['plural']} file to operate on\")",
             "    parser.add_argument(\"--verbose\", \"-v\", action=\"count\", default=0, help=\"more logging; repeat for debug output\")",
             "    sub = parser.add_subparsers(dest=\"command\", required=True)"]
    for v, _ in ctx.d["verbs"][:4]:
        lines += [f"    p_{v} = sub.add_parser(\"{v}\", help=\"{v} one {ent['snake'].replace('_', ' ')}\")", f"    p_{v}.add_argument(\"{ent['id'][0]}\")", f"    p_{v}.set_defaults(func=cmd_{v})"]
    lines += ["    p_show = sub.add_parser(\"show\", help=\"print the file as a table\")", f"    p_show.add_argument(\"--{ent['status']}\", default=None, help=\"only this {ent['status']}\")",
              "    p_show.set_defaults(func=cmd_show)", "    return parser"]
    return "\n".join(lines)


def sh_cmd_show(ctx, ent):
    e, P = ent["snake"], ent["plural"]
    lines = ["def cmd_show(args: argparse.Namespace) -> int:", f'    """Print the {P} of the file as a fixed-width table, optionally filtered by {ent["status"]}."""',
             f"    {P} = {ctx.names['load']}_{P}(Path(args.file))", f"    if args.{ent['status']}:", f"        {P} = {P}_with_{ent['status']}({P}, args.{ent['status']})",
             f"    for {e} in sorted_{P}({P}):", f"        print({ctx.names['format']}_{e}({e}))", f"    print(f\"{{len({P})}} {P}\")", "    return 0"]
    return "\n".join(lines)


def sh_now(ctx):
    return "\n".join([f"def {ctx.names['now']}() -> datetime:", '    """The package clock: timezone-aware UTC; tests patch this name instead of the datetime module."""', "    return datetime.now(timezone.utc)"])


def sh_exceptions(ctx):
    lines = [f"class {ctx.errors[0]}(Exception):", f'    """Base class of every error the {ctx.pkg} package raises on purpose."""']
    for err in ctx.errors[1:]:
        lines += ["", "", f"class {err}({ctx.errors[0]}):", f'    """{err[0] + re.sub(r"([A-Z])", lambda m: " " + m.group(1).lower(), err[1:])}: see the module docstring for when it is raised."""']
    return "\n".join(lines)


def sh_transport(ctx):
    lines = ["class Transport:", '    """Minimal HTTP transport around the standard library; the tests replace it with a recording fake."""', "",
             "    def __init__(self, *, timeout_s: float) -> None:", "        self.timeout_s = timeout_s", "", "    def get(self, url: str) -> \"HttpResponse\":",
             "        return self._do(\"GET\", url, None)", "", "    def post(self, url: str, *, json: dict[str, object]) -> \"HttpResponse\":",
             "        return self._do(\"POST\", url, json)", "", "    def _do(self, method: str, url: str, payload: dict[str, object] | None) -> \"HttpResponse\":",
             "        body = None if payload is None else json.dumps(payload).encode()", "        req = urllib.request.Request(url, data=body, method=method, headers={\"content-type\": \"application/json\"})",
             "        try:", "            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:", "                return HttpResponse(resp.status, resp.read())",
             "        except urllib.error.HTTPError as exc:", "            return HttpResponse(exc.code, exc.read())"]
    return "\n".join(lines)


MODULE_SHAPES = ["dataclass", "constants", "validate", "to_dict", "from_dict", "load", "save", "find", "by_status", "total", "transition", "format",
                 "sort", "group", "iter_since", "summarize", "repository", "client", "parse_line", "merge", "cmd", "cmd_show"]


def render_shape(ctx, shape, ent, k):
    if shape == "dataclass":
        return sh_dataclass(ctx, ent)
    if shape == "constants":
        return sh_constants(ctx, ent)
    if shape == "validate":
        return sh_validate(ctx, ent)
    if shape == "to_dict":
        return sh_to_dict(ctx, ent)
    if shape == "from_dict":
        return sh_from_dict(ctx, ent)
    if shape == "load":
        return sh_load(ctx, ent)
    if shape == "save":
        return sh_save(ctx, ent)
    if shape == "find":
        return sh_find(ctx, ent)
    if shape == "by_status":
        return sh_by_status(ctx, ent)
    if shape == "total":
        return sh_total(ctx, ent)
    if shape == "transition":
        return sh_transition(ctx, ent, k)
    if shape == "format":
        return sh_format(ctx, ent)
    if shape == "sort":
        return sh_sort(ctx, ent)
    if shape == "group":
        return sh_group(ctx, ent, [ent["status"], ent["name"]][k % 2])
    if shape == "iter_since":
        return sh_iter_since(ctx, ent)
    if shape == "summarize":
        return sh_summarize(ctx, ent)
    if shape == "repository":
        return sh_repository(ctx, ent)
    if shape == "client":
        return sh_client(ctx, k)
    if shape == "parse_line":
        return sh_parse_line(ctx, ent)
    if shape == "merge":
        return sh_merge(ctx, ent)
    if shape == "cmd":
        return sh_cmd(ctx, ent, k)
    if shape == "cmd_show":
        return sh_cmd_show(ctx, ent)
    raise KeyError(shape)


def module_header(ctx, module, ent, extra_imports=()):
    """Module docstring + imports + logger; the first unit of every source module."""
    doc = doc_sentence(ctx, "module", ent)
    second = {0: f"Nothing here talks to {ctx.services[0][0]} directly; the clients live in ``{ctx.modules[4]}.py``.",
              1: f"The functions are pure except for the loaders and the logger; the CLI in ``cli.py`` composes them.",
              2: f"See ``docs/{module}.md`` for the state diagram and the file format this module reads and writes."}[ctx.docvar % 3]
    lines = [f'"""{doc}', "", second, '"""', "from __future__ import annotations", "", "import argparse", "import json", "import logging", "import os", "import re", "import sys", "import threading", "import time", "import warnings",
             "from dataclasses import asdict, dataclass, fields, replace", "from datetime import date, datetime, timezone", "from pathlib import Path",
             "from typing import Callable, Iterable, Iterator, TypeVar", ""]
    lines += list(extra_imports)
    lines += [f"from .errors import {', '.join(ctx.errors)}", "", "log = logging.getLogger(__name__)", "T = TypeVar(\"T\")"]
    return "\n".join(lines)


def module_pool(ctx, module, ent, min_units=12, est_tokens=0):
    """The ordered units of a module after its header: a session-specific sequence of shapes over the module's entity and its
    neighbours; long files cycle through the shapes with other entities. The bug unit of the task is inserted by the caller."""
    rnd = random.Random(sid_hash(ctx.index, "pool", module))
    first = ["constants", "dataclass", "validate", "to_dict", "from_dict", "load", "save"]
    rest = [s for s in MODULE_SHAPES if s not in first]
    units = []
    for s in first:
        units.append(render_shape(ctx, s, ent, 0))
    if ent is ctx.main or True:
        units.insert(3, sh_parse_ts(ctx, ent))
    k = 0
    others = [e for e in ctx.ents if e is not ent]
    order = list(rest)
    rnd.shuffle(order)
    need = max(min_units, int(est_tokens / 110) + 4)
    pass_no = 0
    while len(units) < need:
        for s in order:
            if len(units) >= need:
                break
            if pass_no == 0:
                target_ent = ent
            else:
                target_ent = others[(k + pass_no) % len(others)]
            if s in ("cmd_show", "client") and pass_no > 1:
                continue
            units.append(render_shape(ctx, s, target_ent, k))
            k += 1
        pass_no += 1
        if pass_no == 1:
            units.append(sh_now(ctx))
            units.append(sh_retry_helper(ctx))
        rnd.shuffle(order)
    return units


def tail_fillers(ctx, ent):
    """Small natural units for the end of a module, used to close a token gap: a registry, a helper, a main block."""
    e, P = ent["snake"], ent["plural"]
    v = ctx.d["verbs"]
    return [
        "\n".join([f"_TRANSITIONS = {{", *[f"    \"{a}\": {a}_{e}," for a, _ in v[:3]], "}"]),
        "\n".join(["def _chunked(items: list[T], size: int) -> Iterator[list[T]]:", '    """Yield ``items`` in lists of at most ``size`` elements."""', "    for i in range(0, len(items), size):", "        yield items[i:i + size]"]),
        "\n".join(["def _slug(text: str) -> str:", '    """Lower-case, dash-separated form of a label for file names."""', "    return re.sub(r\"[^a-z0-9]+\", \"-\", text.lower()).strip(\"-\")"]),
        "\n".join([f"__all__ = [{', '.join(repr(x) for x in [ent['camel'], ctx.names['load'] + '_' + P, ctx.names['save'] + '_' + P, ctx.names['validate'] + '_' + e])}]"]),
        "\n".join(["if __name__ == \"__main__\":", "    logging.basicConfig(level=logging.INFO)", "    sys.exit(build_parser().parse_args().func(build_parser().parse_args()))"]),
        "\n".join([f"def _describe_{ent['status']}({ent['status']}: str) -> str:", f'    """Human wording of a {e} {ent["status"]} for messages."""', f"    return {ent['status']}.replace(\"_\", \" \")"]),
        "\n".join([f"DEFAULT_{e.upper()}_FILE = Path(\"{P}.jsonl\")"]),
        "\n".join(["def _coerce_int(value: object, *, default: int = 0) -> int:", '    """Best-effort integer conversion for values read from old exports."""', "    try:", "        return int(value)  # type: ignore[arg-type]", "    except (TypeError, ValueError):", "        return default"]),
    ]


# ---- test shapes -------------------------------------------------------------------------------------------------------------
def test_header(ctx, module, ent, names):
    lines = [f'"""Tests for ``{ctx.pkg}.{module}``: construction, validation, persistence and the transitions of {ent["plural"]}."""', "from __future__ import annotations", "",
             "import json", "import threading", "import time", "from dataclasses import replace", "from datetime import date, datetime, timedelta, timezone", "from pathlib import Path", "", "import pytest", "",
             f"from {ctx.pkg}.errors import {', '.join(ctx.errors)}", f"from {ctx.pkg}.{module} import (", *[f"    {n}," for n in names], ")", "",
             "UTC = timezone.utc", "T0 = datetime(2026, 3, 14, 10, 22, tzinfo=UTC)"]
    return "\n".join(lines)


def ts_fixture(ctx, ent):
    fx = ctx.names["fixture"]
    e, E = ent["snake"], ent["camel"]
    f = ctx.fields(ent)
    lines = ["@pytest.fixture", f"def {fx}_{e}() -> {E}:", f"    return {E}("]
    for n, t, s in f:
        lines.append(f"        {n}={'T0' if t == 'datetime' and n == ent['ts'] else s},")
    lines.append("    )")
    return "\n".join(lines)


def ts_fixture_list(ctx, ent):
    fx = ctx.names["fixture"]
    e, E, P = ent["snake"], ent["camel"], ent["plural"]
    fid, pref = ent["id"]
    lines = ["@pytest.fixture", f"def {fx}_{P}({fx}_{e}: {E}) -> list[{E}]:", f'    """Twelve {P} with distinct ids and ascending timestamps."""', "    return [",
             f"        replace({fx}_{e}, {fid}=f\"{pref}{{n:04d}}\", {ent['ts']}=T0 + timedelta(minutes=n), {ent['amount']}={fx}_{e}.{ent['amount']} + n)", "        for n in range(1, 13)", "    ]"]
    return "\n".join(lines)


def test_units(ctx, ent, module):
    """The test functions of a module's test file, in a session-specific order; the failing test of the task is inserted by the caller."""
    rnd = random.Random(sid_hash(ctx.index, "tests", module))
    fx, nv, nl, ns, nf, nt, nfo = (ctx.names[k] for k in ("fixture", "validate", "load", "save", "find", "total", "format"))
    e, E, P, s, a, fid = ent["snake"], ent["camel"], ent["plural"], ent["status"], ent["amount"], ent["id"][0]
    st = ctx.statuses()
    v0, past0 = ctx.verb(0)
    units = [
        "\n".join([f"def test_{nv}_{e}_accepts_the_fixture({fx}_{e}):", f"    {nv}_{e}({fx}_{e})"]),
        "\n".join([f"def test_{nv}_{e}_rejects_empty_{ent['name']}({fx}_{e}):", f"    broken = replace({fx}_{e}, {ent['name']}=\"  \")", f"    with pytest.raises({ctx.base_error}, match=\"empty {ent['name']}\"):", f"        {nv}_{e}(broken)"]),
        "\n".join([f"def test_{nv}_{e}_rejects_negative_{a}({fx}_{e}):", f"    with pytest.raises({ctx.base_error}, match=\"negative\"):", f"        {nv}_{e}(replace({fx}_{e}, {a}=-1))"]),
        "\n".join([f"def test_{nv}_{e}_rejects_naive_{ent['ts']}({fx}_{e}):", f"    naive = replace({fx}_{e}, {ent['ts']}=T0.replace(tzinfo=None))", f"    with pytest.raises({ctx.base_error}, match=\"timezone-aware\"):", f"        {nv}_{e}(naive)"]),
        "\n".join([f"def test_{e}_dict_round_trip({fx}_{e}):", f"    data = {e}_to_dict({fx}_{e})", "    assert data[\"schema\"] == 2", f"    assert {e}_from_dict(data) == {fx}_{e}"]),
        "\n".join([f"def test_{e}_from_dict_reports_missing_keys():", f"    with pytest.raises({ctx.base_error}, match=\"missing\"):", f"        {e}_from_dict({{\"{fid}\": \"x\"}})"]),
        "\n".join([f"def test_{nl}_{P}_skips_blank_and_comment_lines(tmp_path: Path, {fx}_{e}):", f"    path = tmp_path / \"{P}.jsonl\"",
                   f"    path.write_text(\"# header\\n\\n\" + json.dumps({e}_to_dict({fx}_{e})) + \"\\n\", encoding=\"utf-8\")", f"    assert {nl}_{P}(path) == [{fx}_{e}]"]),
        "\n".join([f"def test_{nl}_{P}_missing_file_is_empty(tmp_path: Path):", f"    assert {nl}_{P}(tmp_path / \"absent.jsonl\") == []"]),
        "\n".join([f"def test_{ns}_then_{nl}_preserves_order(tmp_path: Path, {fx}_{P}):", f"    path = tmp_path / \"{P}.jsonl\"", f"    assert {ns}_{P}(path, {fx}_{P}) == 12", f"    assert {nl}_{P}(path) == {fx}_{P}",
                   "    assert not path.with_suffix(\".jsonl.tmp\").exists()"]),
        "\n".join([f"def test_{nf}_{e}_returns_none_for_unknown_id({fx}_{P}):", f"    assert {nf}_{e}({fx}_{P}, \"missing\") is None", f"    assert {nf}_{e}({fx}_{P}, {fx}_{P}[3].{fid}) is {fx}_{P}[3]"]),
        "\n".join([f"def test_{P}_with_{s}_rejects_unknown_value({fx}_{P}):", f"    with pytest.raises({ctx.base_error}):", f"        {P}_with_{s}({fx}_{P}, \"nonsense\")"]),
        "\n".join([f"def test_{nt}_{a}_of_empty_is_zero():", f"    assert {nt}_{a}([]) == 0"]),
        "\n".join([f"def test_{nt}_{a}_adds_every_{e}({fx}_{P}):", f"    assert {nt}_{a}({fx}_{P}) == sum(x.{a} for x in {fx}_{P})"]),
        "\n".join([f"def test_{v0}_{e}_moves_to_{st[0]}_only_from_the_allowed_state({fx}_{e}):", f"    ready = replace({fx}_{e}, {s}=\"{st[0]}\")",
                   f"    with pytest.raises({ctx.base_error}, match=\"cannot {v0}\"):", f"        {v0}_{e}(replace(ready, {s}=\"{st[-1]}\"))"]),
        "\n".join([f"def test_{v0}_{e}_stamps_the_given_time({fx}_{e}):", f"    ready = replace({fx}_{e}, {s}=\"{st[0]}\")", f"    done = {v0}_{e}(ready, at=T0 + timedelta(hours=1))",
                   f"    assert done.{ent['ts']} == T0 + timedelta(hours=1)", f"    assert ready.{ent['ts']} == T0"]),
        "\n".join([f"def test_{v0}_{e}_rejects_naive_timestamp({fx}_{e}):", f"    ready = replace({fx}_{e}, {s}=\"{st[0]}\")", f"    with pytest.raises({ctx.base_error}, match=\"timezone-aware\"):",
                   f"        {v0}_{e}(ready, at=datetime(2026, 1, 1))"]),
        "\n".join([f"def test_{nfo}_{e}_mentions_id_and_{ent['name']}({fx}_{e}):", f"    text = {nfo}_{e}({fx}_{e})", f"    assert {fx}_{e}.{fid} in text", f"    assert {fx}_{e}.{ent['name']} in text", "    assert \"2026-03-14\" in text"]),
        "\n".join([f"def test_sorted_{P}_is_stable_on_equal_timestamps({fx}_{P}):", f"    same = [replace(x, {ent['ts']}=T0) for x in reversed({fx}_{P})]", f"    assert [x.{fid} for x in sorted_{P}(same)] == sorted(x.{fid} for x in same)"]),
        "\n".join([f"def test_group_{P}_by_{s}_keeps_first_seen_order({fx}_{P}):", f"    mixed = [replace(x, {s}=[\"{st[0]}\", \"{st[1]}\", \"{st[2]}\"][i % 3]) for i, x in enumerate({fx}_{P})]",
                   f"    assert list(group_{P}_by_{s}(mixed)) == [\"{st[0]}\", \"{st[1]}\", \"{st[2]}\"]"]),
        "\n".join([f"def test_iter_{P}_since_requires_aware_datetime({fx}_{P}):", f"    with pytest.raises({ctx.base_error}):", f"        list(iter_{P}_since({fx}_{P}, datetime(2026, 1, 1)))"]),
        "\n".join([f"def test_iter_{P}_since_is_inclusive({fx}_{P}):", f"    got = list(iter_{P}_since({fx}_{P}, T0 + timedelta(minutes=6)))", f"    assert [x.{fid} for x in got] == [x.{fid} for x in {fx}_{P}[5:]]"]),
        "\n".join([f"def test_{ctx.names['summarize']}_{P}_counts_per_{s}({fx}_{P}):", f"    summary = {ctx.names['summarize']}_{P}({fx}_{P})", "    assert summary[\"count\"] == 12",
                   f"    assert summary[\"{s}:{st[1]}\"] == 12", f"    assert summary[\"{a}\"] == {nt}_{a}({fx}_{P})"]),
        "\n".join([f"def test_{E.lower()}{ctx.names['store'].lower()}_rejects_duplicates({fx}_{e}):", f"    repo = {E}{ctx.names['store']}()", f"    repo.add({fx}_{e})", f"    with pytest.raises({ctx.base_error}, match=\"duplicate\"):", f"        repo.add({fx}_{e})", "    assert len(repo) == 1"]),
        "\n".join([f"def test_{E.lower()}{ctx.names['store'].lower()}_remove_returns_the_item({fx}_{e}):", f"    repo = {E}{ctx.names['store']}()", f"    repo.add({fx}_{e})", f"    assert repo.remove({fx}_{e}.{fid}) is {fx}_{e}",
                   f"    with pytest.raises({ctx.base_error}):", f"        repo.get({fx}_{e}.{fid})"]),
        "\n".join([f"def test_parse_{e}_line_rejects_short_lines():", f"    with pytest.raises({ctx.base_error}, match=\"expected\"):", f"        parse_{e}_line(\"a,b\")"]),
        "\n".join([f"def test_apply_changes_rejects_unknown_fields({fx}_{e}):", f"    with pytest.raises({ctx.base_error}, match=\"unknown {e} fields\"):", f"        apply_changes({fx}_{e}, {{\"colour\": \"red\"}})"]),
        "\n".join([f"def test_apply_changes_returns_a_new_{e}({fx}_{e}):", f"    changed = apply_changes({fx}_{e}, {{\"{a}\": {fx}_{e}.{a} + 5}})", f"    assert changed.{a} == {fx}_{e}.{a} + 5", f"    assert {fx}_{e}.{a} != changed.{a}"]),
    ]
    rnd.shuffle(units)
    return units


def test_pool(ctx, ent, module):
    """The module's tests after the entity's fixtures and the task's test: the entity's own tests first, then, per other entity,
    its fixtures and the tests of the helpers the module carries for it."""
    units = test_units(ctx, ent, module)
    for other in ctx.ents:
        if other is not ent:
            units += [ts_fixture(ctx, other), ts_fixture_list(ctx, other)] + test_units(ctx, other, module + ":" + other["snake"])
    return units


# ---- prose: README, CHANGELOG, pyproject --------------------------------------------------------------------------------------
def readme_text(ctx):
    d = ctx.d
    ent = ctx.main
    v = d["verbs"]
    paras = [f"# {ctx.pkg}", "", f"{ctx.pkg} is {d['blurb']}. It is a small, dependency-free package: everything is standard library, the data lives in JSON-lines files (or in memory for the tests), and the command line is a thin layer over the library functions.",
             "", "## Installation", "", "```", "pip install -e '.[dev]'", "```", "", f"Python {ctx.py} or newer. The development extras pull in pytest only.", "", "## Quick start", "", "```python",
             f"from pathlib import Path", f"from {ctx.pkg}.{ctx.module_of(ent)} import {ctx.names['load']}_{ent['plural']}, {v[0][0]}_{ent['snake']}, {ctx.names['save']}_{ent['plural']}", "",
             f"{ent['plural']} = {ctx.names['load']}_{ent['plural']}(Path(\"{ent['plural']}.jsonl\"))", f"updated = [{v[0][0]}_{ent['snake']}(x) for x in {ent['plural']} if x.{ent['status']} == \"{d['statuses'][0]}\"]",
             f"{ctx.names['save']}_{ent['plural']}(Path(\"{ent['plural']}.jsonl\"), updated)", "```", "", "## Command line", "",
             f"`{ctx.pkg} <command> [--file FILE] <{ent['id'][0]}>` where the commands are {', '.join('`' + x[0] + '`' for x in v[:4])} and `show`. Every command reads the file, applies one transition and writes the file back atomically; the exit code is 0 on success, 1 when the transition is not allowed and 2 when the {ent['snake'].replace('_', ' ')} does not exist.",
             "", "## Design notes", "",
             f"- Records are frozen dataclasses. A transition returns a new record; nothing is mutated in place, which keeps the loaders and the CLI trivially safe to retry.",
             f"- Timestamps are timezone-aware UTC everywhere. Files written by version 1 of the format carried naive timestamps; the loader interprets them as UTC, which is what the writers of that era meant.",
             f"- Errors raised on purpose derive from `{ctx.base_error}`. Anything else escaping the package is a bug.",
             f"- The package does not talk to {d['services'][0][0]} by itself; the client in `{ctx.modules[4]}.py` is the only place with I/O other than the files.",
             "", "## Development", "", "```", "pytest -q", "ruff check src tests", "```", "",
             f"Tests use the fixtures in `tests/conftest.py`; please add a test for every behaviour change and keep the suite deterministic (inject clocks, never sleep). The maintainers are {ctx.author} and the {d['topic']} working group; issues go to the tracker linked in `pyproject.toml`."]
    return "\n".join(paras)


def changelog_text(ctx):
    ent = ctx.main
    lines = ["# Changelog", "", "## Unreleased", "", f"- Nothing yet.", "", f"## 0.{ctx.num(4) % 9 + 3}.0 ({ctx.year}-0{ctx.num(5) % 8 + 1}-1{ctx.num(6) % 9})", "",
             f"- `{ctx.names['load']}_{ent['plural']}` skips comment lines and logs the line number of malformed records instead of raising.",
             f"- The `show` command gained `--{ent['status']}`.", f"- Dropped Python 3.9 and 3.10.", "", f"## 0.{ctx.num(4) % 9 + 2}.0", "",
             f"- Timestamps are timezone-aware; naive values in old files are read as UTC.", f"- `{ctx.base_error}` is the base class of all package errors.",
             "", f"## 0.{ctx.num(4) % 9 + 1}.0", "", "- First tagged release."]
    return "\n".join(lines)


def pyproject_text(ctx):
    return "\n".join(["[build-system]", "requires = [\"setuptools>=68\"]", "build-backend = \"setuptools.build_meta\"", "", "[project]", f"name = \"{ctx.pkg}\"",
                      f"version = \"0.{ctx.num(4) % 9 + 3}.0\"", f"description = \"{ctx.d['blurb'][0].upper() + ctx.d['blurb'][1:]}\"", f"requires-python = \">={ctx.py}\"", "dependencies = []", "",
                      "[project.optional-dependencies]", "dev = [\"pytest>=8\", \"ruff>=0.5\"]", "", "[project.scripts]", f"{ctx.pkg} = \"{ctx.pkg}.cli:main\"", "", "[tool.pytest.ini_options]",
                      "testpaths = [\"tests\"]", "addopts = \"-ra\"", "", "[tool.ruff]", "line-length = 120", "target-version = \"py312\""])


# ---- task types: the bug (buggy and fixed unit), the failing test, the traceback facts, the ask and the report material ------------
# Each returns a dict: ask, module (where the unit lives), buggy, fixed (the unit's two versions; None for tasks without a module bug),
# unit_name, test (the test unit, inserted into the module's test file), test_name, failure {kind, test_marker, e_lines, exc, frame},
# extra (per-file edits beyond the bug unit: [(path, old, new)] resolved after the files exist), new_file (path, content) | None,
# search (regex the agent searches for), report {root_cause: [...], changes: [...], notes: [...]}, kind (fix|feature|refactor|perf).
def _unit_edit(buggy, fixed):
    """old_text/new_text of the minimal differing block of two versions of a unit, with one line of context on each side."""
    a, b = buggy.split("\n"), fixed.split("\n")
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    j = 0
    while j < min(len(a), len(b)) - i and a[-1 - j] == b[-1 - j]:
        j += 1
    lo = max(0, i - 1)
    old = a[lo:len(a) - j + 1] if j else a[lo:]
    new = b[lo:len(b) - j + 1] if j else b[lo:]
    return "\n".join(old), "\n".join(new)


def task_pagination_offset(ctx):
    ent = ctx.main
    e, E, P, fid = ent["snake"], ent["camel"], ent["plural"], ent["id"][0]
    n = ctx.names["list"]
    fx = ctx.names["fixture"]
    head = [f"def {n}_{P}({P}: list[{E}], *, page: int = 1, size: int = DEFAULT_PAGE_SIZE) -> list[{E}]:",
            f'    """Return one page of {P}; pages are numbered from 1 as the CLI and the HTTP layer expose them."""',
            "    if page < 1 or size < 1:", f"        raise {ctx.base_error}(f\"page and size must be positive, got page={{page}} size={{size}}\")"]
    buggy = "\n".join(head + ["    start = (page - 1) * (size + 1)", f"    return {P}[start:start + size]"])
    fixed = "\n".join(head + ["    start = (page - 1) * size", f"    return {P}[start:start + size]"])
    test = "\n".join([f"def test_{n}_{P}_page_two_starts_after_the_first_page({fx}_{P}):", f"    first = {n}_{P}({fx}_{P}, page=1, size=5)", f"    second = {n}_{P}({fx}_{P}, page=2, size=5)",
                      f"    assert [x.{fid} for x in first] == [x.{fid} for x in {fx}_{P}[:5]]", f"    assert second[0].{fid} == {fx}_{P}[5].{fid}", "    assert len(second) == 5"])
    pref = ent["id"][1]
    return {"kind": "fix", "ask": f"`test_{n}_{P}_page_two_starts_after_the_first_page` in `tests/test_{ctx.module_of(ent)}.py` fails since the pagination helper was added to `{ctx.module_of(ent)}.py`. The CLI and the HTTP layer number pages from 1, and the first page comes back right, but every later page starts one record late, so one record per page boundary is never shown. Please fix the helper rather than the callers; the page numbering is part of the documented interface.",
            "module": ctx.module_of(ent), "buggy": buggy, "fixed": fixed, "unit_name": f"{n}_{P}", "test": test, "test_name": f"test_{n}_{P}_page_two_starts_after_the_first_page",
            "failure": {"kind": "assert", "test_marker": 4, "e_lines": [f"AssertionError: assert '{pref}0007' == '{pref}0006'", f"  - {pref}0006", f"  + {pref}0007"], "exc": "AssertionError", "frame": None},
            "search": f"{n}_{P}\\(", "new_file": None,
            "report": {"root_cause": [f"`{n}_{P}` computed the slice start as `(page - 1) * (size + 1)`: the stride between pages was one larger than the page size, so page 1 (start 0) was right and every later page started one record per page too late, dropping one record at each boundary.",
                                      "The first assertion of the test passed because the error is zero for page 1; the second page in the test starts at record 6 instead of 5, which is what the traceback shows."],
                       "changes": [f"`{ctx.src(ctx.module_of(ent))}`: `{n}_{P}` uses `(page - 1) * size` as the slice start; the argument check is unchanged."],
                       "notes": ["No caller passed page=0; I searched for the call sites and all of them use the 1-based numbering the docstring promises.", "The HTTP layer's own page arithmetic was not touched."]}}


def task_wrong_exception(ctx):
    ent = ctx.main
    e, E, P, fid = ent["snake"], ent["camel"], ent["plural"], ent["id"][0]
    nf, fx = ctx.names["find"], ctx.names["fixture"]
    err = ctx.errors[-1]
    head = [f"def {nf}_{e}_or_raise({P}: Iterable[{E}], {fid}: str) -> {E}:", f'    """Like :func:`{nf}_{e}` but raise :class:`{err}` instead of returning ``None``."""',
            f"    found = {nf}_{e}({P}, {fid})", "    if found is None:"]
    buggy = "\n".join(head + [f"        raise KeyError({fid})", "    return found"])
    fixed = "\n".join(head + [f"        raise {err}(f\"no {e} with {fid} {{{fid}!r}}\")", "    return found"])
    test = "\n".join([f"def test_{nf}_{e}_or_raise_raises_the_package_error({fx}_{P}):", f"    with pytest.raises({err}, match=\"missing\"):", f"        {nf}_{e}_or_raise({fx}_{P}, \"missing\")"])
    return {"kind": "fix", "ask": f"The CLI prints a bare Python traceback when a {e.replace('_', ' ')} id does not exist, instead of the one-line message the other commands print. `test_{nf}_{e}_or_raise_raises_the_package_error` reproduces it. All errors the package raises on purpose must derive from `{ctx.base_error}` (see README, design notes); please make `{nf}_{e}_or_raise` follow that rule and keep the message useful.",
            "module": ctx.module_of(ent), "buggy": buggy, "fixed": fixed, "unit_name": f"{nf}_{e}_or_raise", "test": test, "test_name": f"test_{nf}_{e}_or_raise_raises_the_package_error",
            "failure": {"kind": "raise", "test_marker": 2, "e_lines": ["KeyError: 'missing'"], "exc": "KeyError", "frame": {"marker": 4, "e_lines": ["KeyError: 'missing'"]}},
            "search": "KeyError", "new_file": None,
            "report": {"root_cause": [f"`{nf}_{e}_or_raise` raised a builtin `KeyError` when the lookup failed. The CLI handlers catch `{ctx.base_error}` and print its message; a `KeyError` escapes them and the user sees a traceback.",
                                      f"The traceback of the failing test shows the `KeyError('missing')` leaving the function; `pytest.raises({err})` does not catch it."],
                       "changes": [f"`{ctx.src(ctx.module_of(ent))}`: `{nf}_{e}_or_raise` raises `{err}` with the id in the message."],
                       "notes": [f"`{err}` already derives from `{ctx.base_error}` in `errors.py`, so the CLI handlers need no change.", "I searched for other `raise KeyError` sites in the package; there are none."]}}


def task_check_then_act_race(ctx):
    ent = ctx.main
    e, E = ent["snake"], ent["camel"]
    D = f"{E}Dispatcher"
    err = ctx.errors[2] if len(ctx.errors) > 2 else ctx.base_error
    common_head = [f"class {D}:", f'    """Bounded admission for in-flight {ent["plural"]}: at most ``limit`` are active at once; ``release`` frees a slot."""', "",
                   "    def __init__(self, limit: int) -> None:", "        if limit < 1:", f"            raise {ctx.base_error}(\"limit must be positive\")", "        self._limit = limit", "        self._active = 0",
                   "        self._lock = threading.Lock()", "        self.peak = 0", ""]
    tail = ["", "    def release(self) -> None:", "        with self._lock:", "            if self._active == 0:", f"                raise {ctx.base_error}(\"release without a matching acquire\")", "            self._active -= 1", "",
            "    @property", "    def active(self) -> int:", "        with self._lock:", "            return self._active"]
    buggy = "\n".join(common_head + ["    def acquire(self) -> None:", "        if self._active >= self._limit:", f"            raise {err}(f\"{D}: {{self._limit}} {ent['plural']} already in flight\")",
                                     "        with self._lock:", "            self._active += 1", "            self.peak = max(self.peak, self._active)"] + tail)
    fixed = "\n".join(common_head + ["    def acquire(self) -> None:", "        with self._lock:", "            if self._active >= self._limit:", f"                raise {err}(f\"{D}: {{self._limit}} {ent['plural']} already in flight\")",
                                     "            self._active += 1", "            self.peak = max(self.peak, self._active)"] + tail)
    test = "\n".join([f"def test_{D.lower()}_never_exceeds_its_limit_under_contention():", f"    dispatcher = {D}(limit=4)", "    start = threading.Barrier(16)", "    rejected = []", "",
                      "    def worker() -> None:", "        start.wait()", "        try:", "            dispatcher.acquire()", f"        except {err}:", "            rejected.append(1)", "            return", "        time.sleep(0.01)", "        dispatcher.release()", "",
                      "    threads = [threading.Thread(target=worker) for _ in range(16)]", "    for t in threads:", "        t.start()", "    for t in threads:", "        t.join()", "    assert dispatcher.peak <= 4", "    assert len(rejected) + 4 >= 16 - dispatcher.peak"])
    return {"kind": "fix", "ask": f"`test_{D.lower()}_never_exceeds_its_limit_under_contention` fails about one run in three on the CI runners and never on a laptop. The dispatcher is supposed to admit at most `limit` {ent['plural']} at a time. Please find the actual race rather than adding a retry to the test, and explain in the report which invariant the lock protects.",
            "module": ctx.module_of(ent), "buggy": buggy, "fixed": fixed, "unit_name": D, "test": test, "test_name": f"test_{D.lower()}_never_exceeds_its_limit_under_contention",
            "failure": {"kind": "assert", "test_marker": 20, "e_lines": ["assert 6 <= 4", " +  where 6 = <" + ctx.pkg + "." + ctx.module_of(ent) + "." + D + " object at 0x7f3c1a2b4e90>.peak"], "exc": "AssertionError", "frame": None},
            "search": "_lock", "new_file": None,
            "report": {"root_cause": [f"`{D}.acquire` read `_active` and compared it with the limit before taking the lock, then incremented under the lock. Between the check and the increment any number of threads could pass the check with the same stale count, so under contention the active count overshoots the limit; the test's 16 threads released at a barrier show peaks of 5 or 6.",
                                      "The lock protects the invariant `0 <= _active <= _limit` together with `peak`; it only does so if the check and the mutation happen inside the same critical section."],
                       "changes": [f"`{ctx.src(ctx.module_of(ent))}`: `{D}.acquire` performs the limit check inside the `with self._lock:` block; the exception type and message are unchanged."],
                       "notes": ["The test is sound as written: it fails deterministically with the race present when the threads are released together, and the barrier makes that likely.", "`release` already held the lock for both the check and the decrement."]}}


def task_rename_function(ctx):
    ent = ctx.main
    e, E, P = ent["snake"], ent["camel"], ent["plural"]
    old = f"{ctx.names['load']}_{P}"
    new = f"read_{P}_file" if ctx.names["load"] != "read" else f"load_{P}_from_file"
    return {"kind": "refactor", "ask": f"Rename `{old}` to `{new}` in the package: the definition and every caller inside `src/`. The name has confused two new contributors because it takes a path, not an open file. Keep `{old}` as a deprecated alias that emits a `DeprecationWarning` naming the new function, add a test for the alias, and do not change behaviour; the existing tests may keep calling the alias until it is removed next release.",
            "module": ctx.module_of(ent), "buggy": None, "fixed": None, "unit_name": old, "test": None, "test_name": None, "failure": None, "rename": (old, new),
            "search": f"\\b{old}\\b", "new_file": None,
            "report": {"root_cause": [f"`{old}` takes a `Path` and reads a JSON-lines file; the new name says so. The definition lives in `{ctx.src(ctx.module_of(ent))}`, the callers are the CLI handlers, the tests and the README's quick start."],
                       "changes": [f"`{ctx.src(ctx.module_of(ent))}`: the function is defined as `{new}`; `{old}` remains directly above it as a thin wrapper that warns with `DeprecationWarning` and delegates.",
                                   f"`{ctx.src('cli')}`: the import and both call sites use `{new}`.",
                                   f"`{ctx.test_path(ctx.module_of(ent))}`: a new test checks that the alias warns and delegates; the existing tests keep calling the alias on purpose, as the task allows."],
                       "notes": ["Behaviour is unchanged: the alias forwards its only argument.", "When the alias is removed, the existing tests that call it need the one-word rename; I left them so the diff stays about the package."]}}


def task_add_cli_flag(ctx):
    ent = ctx.main
    e, E, P, fid = ent["snake"], ent["camel"], ent["plural"], ent["id"][0]
    v, past = ctx.verb(0)
    return {"kind": "feature", "ask": f"Add a `--dry-run` flag to the transition commands of the `{ctx.pkg}` CLI (`{', '.join(x[0] for x in ctx.d['verbs'][:4])}`). With the flag the command must do everything except write the file back, print the would-be result the same way, and exit 0. Add a test for `{v}` under `tests/test_cli.py` (that file does not exist yet) using the existing fixtures. Do not change the library functions.",
            "module": "cli", "buggy": None, "fixed": None, "unit_name": f"cmd_{v}", "test": None, "test_name": f"test_cli_{v}_dry_run_does_not_write", "failure": None, "flag": "dry_run",
            "search": "add_argument\\(", "new_file": "tests/test_cli.py",
            "report": {"root_cause": [f"The four transition commands share one shape in `{ctx.src('cli')}`: load the file, find the {e.replace('_', ' ')}, apply the transition, save, print. A dry run skips the save step only, so the flag belongs on the shared argument parser and the handlers check `args.dry_run` before saving."],
                       "changes": [f"`{ctx.src('cli')}`: `build_parser` adds `--dry-run` (store_true) to the top-level parser so every sub-command accepts it; the transition handlers skip `{ctx.names['save']}_{P}` when it is set and print `(dry run)` after the formatted record.",
                                   f"`tests/test_cli.py` (new): `test_cli_{v}_dry_run_does_not_write` runs `main([\"--file\", str(path), \"--dry-run\", \"{v}\", <id>])` against a file written from the fixtures and asserts the file is byte-identical afterwards and the exit code is 0; a second test checks the normal run still writes."],
                       "notes": ["The flag is top-level, before the sub-command, which is how `--file` and `--verbose` already work; a per-sub-command flag would have duplicated the same three lines four times.", "The library functions are unchanged as asked."]}}


def task_config_env_override(ctx):
    ent = ctx.main
    e, E, P = ent["snake"], ent["camel"], ent["plural"]
    envvar = f"{ctx.pkg.upper()}_{P.upper()}_FILE"
    head = [f"def default_{P}_file() -> Path:", f'    """The file the CLI operates on when ``--file`` is not given."""']
    buggy = "\n".join(head + [f"    return Path(\"{P}.jsonl\")"])
    fixed = "\n".join(head + ["", f"    Set ``{envvar}`` to point the CLI at another file; an empty value is ignored.", '    """', f"    override = os.environ.get(\"{envvar}\", \"\").strip()", "    if override:", "        return Path(override).expanduser()", f"    return Path(\"{P}.jsonl\")"])
    fixed = fixed.replace(f'    """The file the CLI operates on when ``--file`` is not given."""\n', f'    """The file the CLI operates on when ``--file`` is not given.\n')
    fx = ctx.names["fixture"]
    test = "\n".join([f"def test_default_{P}_file_honours_the_environment(monkeypatch, tmp_path: Path):", f"    monkeypatch.setenv(\"{envvar}\", str(tmp_path / \"other.jsonl\"))",
                      f"    assert default_{P}_file() == tmp_path / \"other.jsonl\"", f"    monkeypatch.setenv(\"{envvar}\", \"  \")", f"    assert default_{P}_file() == Path(\"{P}.jsonl\")"])
    return {"kind": "feature", "ask": f"Operators run the `{ctx.pkg}` CLI from systemd units and cannot pass `--file` on every invocation. Make the default file overridable with the environment variable `{envvar}` (an empty or blank value means the built-in default), document it in the README's command-line section, and cover it with a test. `--file` on the command line must still win over the variable.",
            "module": ctx.module_of(ent), "buggy": buggy, "fixed": fixed, "unit_name": f"default_{P}_file", "test": test, "test_name": f"test_default_{P}_file_honours_the_environment",
            "failure": {"kind": "assert", "test_marker": 2, "e_lines": [f"AssertionError: assert PosixPath('{P}.jsonl') == PosixPath('/tmp/pytest-of-ci/pytest-12/test_default0/other.jsonl')", f" +  where PosixPath('{P}.jsonl') = default_{P}_file()"], "exc": "AssertionError", "frame": None},
            "search": f"default_{P}_file|{P}\\.jsonl", "new_file": None, "envvar": envvar,
            "report": {"root_cause": [f"The default file name was a literal in `default_{P}_file` and the parser's `--file` default pointed at the same helper, so one place decides the default and the environment lookup belongs there."],
                       "changes": [f"`{ctx.src(ctx.module_of(ent))}`: `default_{P}_file` reads `{envvar}`, ignores blank values and expands `~`; the docstring documents the variable.",
                                   f"`{ctx.test_path(ctx.module_of(ent))}`: `test_default_{P}_file_honours_the_environment` covers the set, blank and unset cases with `monkeypatch`.",
                                   "`README.md`: one paragraph in the command-line section."],
                       "notes": [f"`--file` still wins: argparse only uses the default when the option is absent, and the default is computed when the parser is built, which happens at process start.", "I did not add a `--file` precedence test because `test_cli.py` does not exist yet; say if you want it."]}}


def task_resource_leak(ctx):
    ent = ctx.main
    e, E, P = ent["snake"], ent["camel"], ent["plural"]
    n, fx = ctx.names["load"], ctx.names["fixture"]
    head = [f"def {n}_{P}(path: Path) -> list[{E}]:", f'    """{doc_sentence(ctx, "load", ent)}"""', "    if not path.exists():", f"        log.info(\"no {P} file at %s\", path)", "        return []", f"    out: list[{E}] = []"]
    buggy = "\n".join(head + ["    fh = path.open(encoding=\"utf-8\")", "    for lineno, line in enumerate(fh, start=1):", "        line = line.strip()", "        if not line or line.startswith(\"#\"):", "            continue",
                              f"        out.append({e}_from_dict(json.loads(line)))", "    fh.close()", "    return out"])
    fixed = "\n".join(head + ["    with path.open(encoding=\"utf-8\") as fh:", "        for lineno, line in enumerate(fh, start=1):", "            line = line.strip()", "            if not line or line.startswith(\"#\"):", "                continue",
                              "            try:", f"                out.append({e}_from_dict(json.loads(line)))", f"            except (ValueError, {ctx.base_error}) as exc:",
                              f"                log.warning(\"skipping {e} on line %d of %s: %s\", lineno, path, exc)", "    return out"])
    test = "\n".join([f"def test_{n}_{P}_closes_the_file_on_a_malformed_line(tmp_path: Path, monkeypatch):", f"    path = tmp_path / \"{P}.jsonl\"", "    path.write_text(\"{not json}\\n\", encoding=\"utf-8\")", "    handles = []",
                      "    real_open = Path.open", "", "    def tracking_open(self, *args, **kwargs):", "        fh = real_open(self, *args, **kwargs)", "        handles.append(fh)", "        return fh", "",
                      "    monkeypatch.setattr(Path, \"open\", tracking_open)", f"    assert {n}_{P}(path) == []", "    assert handles and all(fh.closed for fh in handles)"])
    return {"kind": "fix", "ask": f"A long-running importer that calls `{n}_{P}` in a loop runs out of file descriptors after a few hours on hosts where the input occasionally contains a malformed line. `test_{n}_{P}_closes_the_file_on_a_malformed_line` is a new test that reproduces it (it currently fails because the malformed line raises). Make the loader release the file on every path and skip malformed lines with a warning, as the changelog already claims it does.",
            "module": ctx.module_of(ent), "buggy": buggy, "fixed": fixed, "unit_name": f"{n}_{P}", "test": test, "test_name": f"test_{n}_{P}_closes_the_file_on_a_malformed_line", "skip_shapes": ["load"],
            "failure": {"kind": "raise", "test_marker": 12, "e_lines": ["json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"], "exc": "JSONDecodeError",
                        "frame": {"marker": 11, "e_lines": ["json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"]}},
            "search": "\\.open\\(", "new_file": None,
            "report": {"root_cause": [f"`{n}_{P}` opened the file with `path.open(...)` and closed it with an explicit `fh.close()` after the loop. A malformed line raised `JSONDecodeError` out of the loop, the close was never reached and the handle stayed open until garbage collection, which in the importer's loop is late enough to exhaust the descriptor table.",
                                      "The changelog says malformed records are skipped with a warning; the code did not do that either."],
                       "changes": [f"`{ctx.src(ctx.module_of(ent))}`: `{n}_{P}` uses a `with` block for the file and wraps the per-line decode in `try/except (ValueError, {ctx.base_error})`, logging the line number and skipping the record."],
                       "notes": ["`JSONDecodeError` is a `ValueError` subclass, so the except clause covers it without importing the json exception type.", "The new test tracks handles through `Path.open`; it passes now and fails on the previous code with the decode error."]}}


def task_stale_seconds(ctx):
    ent = ctx.main
    e, E, ts = ent["snake"], ent["camel"], ent["ts"]
    fx = ctx.names["fixture"]
    head = [f"def {e}_is_stale({e}: {E}, *, now: datetime, max_age_s: int = 86400) -> bool:", f'    """True when the {e}\'s ``{ts}`` is older than ``max_age_s`` seconds at ``now``."""',
            "    if now.tzinfo is None:", f"        raise {ctx.base_error}(\"now must be timezone-aware\")"]
    buggy = "\n".join(head + [f"    age = (now - {e}.{ts}).seconds", "    return age > max_age_s"])
    fixed = "\n".join(head + [f"    age = (now - {e}.{ts}).total_seconds()", "    return age > max_age_s"])
    test = "\n".join([f"def test_{e}_older_than_a_day_is_stale({fx}_{e}):", f"    assert not {e}_is_stale({fx}_{e}, now=T0 + timedelta(hours=23))", f"    assert {e}_is_stale({fx}_{e}, now=T0 + timedelta(days=2))",
                      f"    assert {e}_is_stale({fx}_{e}, now=T0 + timedelta(days=30, seconds=5))"])
    return {"kind": "fix", "ask": f"The nightly cleanup never considers {ent['plural'].replace('_', ' ')} older than a day stale, although `{e}_is_stale` is called with the default `max_age_s`. `test_{e}_older_than_a_day_is_stale` fails on the second assertion. Find out why and fix the function; the signature is used by the cleanup job and must stay.",
            "module": ctx.module_of(ent), "buggy": buggy, "fixed": fixed, "unit_name": f"{e}_is_stale", "test": test, "test_name": f"test_{e}_older_than_a_day_is_stale",
            "failure": {"kind": "assert", "test_marker": 2, "e_lines": ["assert False", f" +  where False = {e}_is_stale({E}({ent['id'][0]}={ctx.lit(ent, ent['id'][0])}, ...), now=datetime.datetime(2026, 3, 16, 10, 22, tzinfo=datetime.timezone.utc))"], "exc": "AssertionError", "frame": None},
            "search": "\\.seconds\\b", "new_file": None,
            "report": {"root_cause": [f"`{e}_is_stale` read `timedelta.seconds`, which is only the seconds component of the difference (0-86399) and ignores the days. Any age of a whole number of days reads as nearly zero, so nothing older than a day was ever stale and the comparison with the default of 86,400 seconds could never be true.",
                                      "`timedelta.total_seconds()` is the value the comparison needs."],
                       "changes": [f"`{ctx.src(ctx.module_of(ent))}`: `{e}_is_stale` uses `total_seconds()`."],
                       "notes": ["I searched the package for other uses of `.seconds` on a timedelta; there are none.", "The third assertion of the test (30 days plus 5 seconds) now holds as well; it would have been false before for the same reason."]}}


def task_retry_policy(ctx):
    ent = ctx.main
    e, E, fid = ent["snake"], ent["camel"], ent["id"][0]
    fx = ctx.names["fixture"]
    head = [f"def fetch_{e}_with_retry(client: object, {fid}: str, *, attempts: int = 3, base_delay_s: float = 0.2, sleep: Callable[[float], None] = time.sleep) -> dict[str, object]:",
            f'    """Fetch a {e} from {ctx.services[0][0]}, retrying transient failures with exponential backoff."""', "    last: Exception | None = None", "    for n in range(attempts):", "        try:",
            f"            return client.fetch_{e}({fid})"]
    buggy = "\n".join(head + ["        except Exception as exc:", "            last = exc", "            sleep(base_delay_s)", "    assert last is not None", "    raise last"])
    fixed = "\n".join(head + ["        except (OSError, TimeoutError) as exc:", "            last = exc", "            if n + 1 < attempts:", "                sleep(base_delay_s * (2 ** n))", "    assert last is not None", "    raise last"])
    test = "\n".join([f"def test_fetch_{e}_does_not_retry_on_package_errors():", "    calls = []", "", "    class Failing:", f"        def fetch_{e}(self, {fid}):", "            calls.append(" + fid + ")",
                      f"            raise {ctx.errors[-1]}({fid})", "", "    slept = []", f"    with pytest.raises({ctx.errors[-1]}):", f"        fetch_{e}_with_retry(Failing(), \"x\", sleep=slept.append)", "    assert calls == [\"x\"]", "    assert slept == []"])
    return {"kind": "fix", "ask": f"`fetch_{e}_with_retry` retries three times on every exception, including `{ctx.errors[-1]}` and other errors of ours that will never succeed on retry, and it sleeps the same 200 ms each time. That triples the latency of every bad request to {ctx.services[0][0]} and the on-call noticed it in the logs. Retry only transient transport errors (`OSError`, `TimeoutError`), back off exponentially, and do not sleep after the last attempt. `test_fetch_{e}_does_not_retry_on_package_errors` is the failing test.",
            "module": ctx.module_of(ent), "buggy": buggy, "fixed": fixed, "unit_name": f"fetch_{e}_with_retry", "test": test, "test_name": f"test_fetch_{e}_does_not_retry_on_package_errors",
            "failure": {"kind": "assert", "test_marker": 11, "e_lines": ["AssertionError: assert ['x', 'x', 'x'] == ['x']", "  Left contains 2 more items, first extra item: 'x'", "  Use -v to get more diff"], "exc": "AssertionError", "frame": None},
            "search": "except Exception", "new_file": None,
            "report": {"root_cause": [f"The except clause of `fetch_{e}_with_retry` caught `Exception`, so the package's own errors (which signal a bad request, not a bad network) were retried like timeouts; the sleep was a constant and ran after the final failure too.",
                                      "The failing test shows the three calls and the three sleeps."],
                       "changes": [f"`{ctx.src(ctx.module_of(ent))}`: `fetch_{e}_with_retry` catches `(OSError, TimeoutError)` only, sleeps `base_delay_s * 2**n` between attempts and not after the last one."],
                       "notes": ["Callers that relied on the old behaviour to retry `ValueError` from a half-written response would now see it immediately; I found no such caller.", "A jitter term would be a reasonable follow-up for fleets of clients; the task did not ask for it."]}}


def task_unicode_truncation(ctx):
    ent = ctx.main
    e, E, nm = ent["snake"], ent["camel"], ent["name"]
    fx = ctx.names["fixture"]
    head = [f"def truncate_{nm}(text: str, limit: int = MAX_{nm.upper()}_CHARS) -> str:", f'    """Shorten a {nm.replace("_", " ")} to at most ``limit`` characters, marking the cut with an ellipsis."""', "    if limit < 2:",
            f"        raise {ctx.base_error}(\"limit must be at least 2\")"]
    buggy = "\n".join(head + ["    raw = text.encode(\"utf-8\")", "    if len(raw) <= limit:", "        return text", "    return raw[: limit - 1].decode(\"utf-8\") + \"…\""])
    fixed = "\n".join(head + ["    if len(text) <= limit:", "        return text", "    return text[: limit - 1] + \"…\""])
    test = "\n".join([f"def test_truncate_{nm}_counts_characters_not_bytes():", f"    assert truncate_{nm}(\"Zoë Ångström\", limit=20) == \"Zoë Ångström\"", f"    assert truncate_{nm}(\"Zoë Ångström\", limit=5) == \"Zoë …\"", f"    assert len(truncate_{nm}(\"ünïcödé\" * 4, limit=10)) == 10"])
    return {"kind": "fix", "ask": f"`truncate_{nm}` raises `UnicodeDecodeError` for some non-ASCII names and shortens others too aggressively; the limit is documented as a number of characters. `test_truncate_{nm}_counts_characters_not_bytes` fails. Please fix the function so that the limit counts characters and the output never breaks a character; the ellipsis convention stays.",
            "module": ctx.module_of(ent), "buggy": buggy, "fixed": fixed, "unit_name": f"truncate_{nm}", "test": test, "test_name": f"test_truncate_{nm}_counts_characters_not_bytes",
            "failure": {"kind": "raise", "test_marker": 2, "e_lines": ["UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 3: unexpected end of data"], "exc": "UnicodeDecodeError",
                        "frame": {"marker": 7, "e_lines": ["UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 3: unexpected end of data"]}},
            "search": "\\.encode\\(", "new_file": None,
            "report": {"root_cause": [f"`truncate_{nm}` encoded the text to UTF-8 and sliced the bytes. A limit that lands inside a multi-byte sequence leaves an incomplete sequence, which `decode` rejects; and because the length check counted bytes, names with accented characters were cut although they were within the character limit.",
                                      "Slicing the string itself counts code points, which is what the docstring promises."],
                       "changes": [f"`{ctx.src(ctx.module_of(ent))}`: `truncate_{nm}` compares and slices characters; the ellipsis is still appended within the limit."],
                       "notes": ["Grapheme clusters (combining marks) can still be split by a code-point slice; that is the documented behaviour and matches the rest of the package, so I left it.", "No other function in the package slices encoded bytes."]}}


def task_timezone_naive(ctx):
    ent = ctx.main
    e, E, ts = ent["snake"], ent["camel"], ent["ts"]
    fx = ctx.names["fixture"]
    head = [f"def parse_{ts}(raw: str) -> datetime:", f'    """Parse the ``{ts}`` column of the legacy CSV export (``YYYY-MM-DD HH:MM``, always UTC)."""', "    raw = raw.strip()"]
    buggy = "\n".join(head + ["    try:", "        return datetime.strptime(raw, \"%Y-%m-%d %H:%M\")", "    except ValueError as exc:", f"        raise {ctx.base_error}(f\"bad {ts} {{raw!r}}\") from exc"])
    fixed = "\n".join(head + ["    try:", "        parsed = datetime.strptime(raw, \"%Y-%m-%d %H:%M\")", "    except ValueError as exc:", f"        raise {ctx.base_error}(f\"bad {ts} {{raw!r}}\") from exc", "    return parsed.replace(tzinfo=timezone.utc)"])
    test = "\n".join([f"def test_parse_{ts}_returns_an_aware_utc_datetime():", f"    parsed = parse_{ts}(\"2026-03-14 10:22\")", "    assert parsed.tzinfo is not None", "    assert parsed == T0", f"    assert parse_{ts}(\" 2026-03-14 10:22 \") == T0"])
    return {"kind": "fix", "ask": f"Importing the legacy CSV export crashes with `TypeError: can't compare offset-naive and offset-aware datetimes` as soon as a record is sorted or checked for staleness. `test_parse_{ts}_returns_an_aware_utc_datetime` pins the expected behaviour: the export is always UTC (README, design notes) and the parser must return aware values. Fix `parse_{ts}`; do not change the comparisons in the callers, they are right.",
            "module": ctx.module_of(ent), "buggy": buggy, "fixed": fixed, "unit_name": f"parse_{ts}", "test": test, "test_name": f"test_parse_{ts}_returns_an_aware_utc_datetime",
            "failure": {"kind": "assert", "test_marker": 2, "e_lines": ["assert None is not None", " +  where None = datetime.datetime(2026, 3, 14, 10, 22).tzinfo"], "exc": "AssertionError", "frame": None},
            "search": "strptime|tzinfo", "new_file": None,
            "report": {"root_cause": [f"`datetime.strptime` returns a naive value and `parse_{ts}` returned it as-is. The callers compare and sort with the aware timestamps the rest of the package produces, hence the `TypeError`; the test asserts `tzinfo is not None` and failed on that line.",
                                      "The export format is documented as UTC, so attaching `timezone.utc` is the correct interpretation rather than a guess."],
                       "changes": [f"`{ctx.src(ctx.module_of(ent))}`: `parse_{ts}` attaches UTC to the parsed value; the error path is unchanged."],
                       "notes": ["Stored timestamps without an offset are now interpreted as UTC, which is what the README already promised.", "The JSON loader's `_parse_ts` already did the same; the two could share one helper, left as a follow-up."]}}


def task_quadratic_match(ctx):
    ent = ctx.main
    other = ctx.ents[1]
    e, E, P, fid = ent["snake"], ent["camel"], ent["plural"], ent["id"][0]
    oe, OE, OP, ofid = other["snake"], other["camel"], other["plural"], other["id"][0]
    fx = ctx.names["fixture"]
    head = [f"def match_{P}_to_{OP}({P}: list[{E}], {OP}: list[{OE}]) -> dict[str, {OE} | None]:", f'    """Map every {e} id to the {oe} with the same ``{ent["name"]}``, or ``None`` when there is no such {oe}."""']
    buggy = "\n".join(head + [f"    result: dict[str, {OE} | None] = {{}}", f"    for {e} in {P}:", f"        matches = [{oe} for {oe} in {OP} if {oe}.{other['name']} == {e}.{ent['name']}]",
                              f"        result[{e}.{fid}] = matches[0] if matches else None", "    return result"])
    fixed = "\n".join(head + [f"    by_name: dict[str, {OE}] = {{}}", f"    for {oe} in {OP}:", f"        by_name.setdefault({oe}.{other['name']}, {oe})", f"    return {{{e}.{fid}: by_name.get({e}.{ent['name']}) for {e} in {P}}}"])
    test = "\n".join([f"def test_match_{P}_to_{OP}_is_linear_for_ten_thousand({fx}_{e}, {fx}_{oe}):", f"    {P} = [replace({fx}_{e}, {fid}=f\"{ent['id'][1]}{{n}}\", {ent['name']}=f\"name {{n}}\") for n in range(10_000)]",
                      f"    {OP} = [replace({fx}_{oe}, {ofid}=f\"{other['id'][1]}{{n}}\", {other['name']}=f\"name {{n}}\") for n in range(10_000)]",
                      "    started = time.perf_counter()", f"    matched = match_{P}_to_{OP}({P}, {OP})", "    elapsed = time.perf_counter() - started", f"    assert matched[\"{ent['id'][1]}42\"].{ofid} == \"{other['id'][1]}42\"", "    assert elapsed < 0.5"])
    return {"kind": "perf", "ask": f"The reconciliation report takes over a minute on the production file sizes (about forty thousand {P.replace('_', ' ')} against as many {OP.replace('_', ' ')}). Profiling points at `match_{P}_to_{OP}`. `test_match_{P}_to_{OP}_is_linear_for_ten_thousand` encodes the budget (half a second for ten thousand). Make the function linear without changing its result, including the rule that the first {oe.replace('_', ' ')} with a given {ent['name'].replace('_', ' ')} wins.",
            "module": ctx.module_of(ent), "buggy": buggy, "fixed": fixed, "unit_name": f"match_{P}_to_{OP}", "test": test, "test_name": f"test_match_{P}_to_{OP}_is_linear_for_ten_thousand",
            "failure": {"kind": "assert", "test_marker": 7, "e_lines": ["assert 2.8437 < 0.5"], "exc": "AssertionError", "frame": None},
            "search": f"match_{P}_to_{OP}", "new_file": None,
            "report": {"root_cause": [f"`match_{P}_to_{OP}` rebuilt a list comprehension over all {OP} for every {e}: O(n * m) comparisons, 100 million for ten thousand of each, 2.8 s in the test and minutes at production sizes.",
                                      f"A dictionary from `{other['name']}` to the first {oe} with that name, built once, gives the same answer in one pass over each list."],
                       "changes": [f"`{ctx.src(ctx.module_of(ent))}`: `match_{P}_to_{OP}` builds the index with `setdefault` (first occurrence wins, as before) and looks each {e} up in it."],
                       "notes": ["The result dictionary has the same keys in the same order and the same values; the test's first assertion checks a representative pair.", "Run time of the test on this sandbox: 0.03 s against 2.84 s before."]}}


def task_deprecate_parameter(ctx):
    ent = ctx.main
    e, E, s = ent["snake"], ent["camel"], ent["status"]
    v, past = ctx.verb(1)
    st = ctx.statuses()
    fx = ctx.names["fixture"]
    head = [f"def {v}_{e}_again({e}: {E}, *, allow_terminal: bool = False, force: bool | None = None) -> {E}:",
            f'    """Re-run the ``{v}`` transition on a {e} that already went through it; terminal states need ``allow_terminal``.']
    body_common = [f"    if {e}.{s} in TERMINAL_{s.upper()} and not allow_terminal:", f"        raise {ctx.base_error}(f\"{e} {{{e}.{ent['id'][0]}}} is {{{e}.{s}}}; pass allow_terminal=True to {v} it again\")",
                   f"    return replace({e}, {s}=\"{st[1]}\", {ent['ts']}={ctx.names['now']}())"]
    buggy = "\n".join(head + ['    """', "    if force is not None:", "        allow_terminal = force"] + body_common)
    fixed = "\n".join(head + ["", "    ``force`` is the old name of ``allow_terminal``; passing it warns and will stop working in the next release.", '    """', "    if force is not None:",
                              f"        warnings.warn(\"{v}_{e}_again(force=...) is deprecated; use allow_terminal\", DeprecationWarning, stacklevel=2)", "        allow_terminal = force"] + body_common)
    test = "\n".join([f"def test_{v}_{e}_again_force_parameter_warns({fx}_{e}):", f"    terminal = replace({fx}_{e}, {s}=\"{st[-1]}\")", "    with pytest.warns(DeprecationWarning, match=\"allow_terminal\"):",
                      f"        {v}_{e}_again(terminal, force=True)", f"    with pytest.raises({ctx.base_error}):", f"        {v}_{e}_again(terminal)"])
    return {"kind": "feature", "ask": f"`{v}_{e}_again` grew a second keyword, `allow_terminal`, last release but the old `force` keyword was kept silently. Deprecate `force`: passing it must emit a `DeprecationWarning` naming the replacement, keep working for this release, and be documented in the docstring. `test_{v}_{e}_again_force_parameter_warns` fails with DID NOT WARN. Do not remove the parameter yet.",
            "module": ctx.module_of(ent), "buggy": buggy, "fixed": fixed, "unit_name": f"{v}_{e}_again", "test": test, "test_name": f"test_{v}_{e}_again_force_parameter_warns",
            "failure": {"kind": "failed", "test_marker": 2, "e_lines": ["Failed: DID NOT WARN. No warnings of type (<class 'DeprecationWarning'>,) were emitted.", " Emitted warnings: []."], "exc": "Failed", "frame": None},
            "search": "force", "new_file": None,
            "report": {"root_cause": [f"`{v}_{e}_again` mapped `force` onto `allow_terminal` without any notice, so callers had no reason to migrate and the parameter could never be removed."],
                       "changes": [f"`{ctx.src(ctx.module_of(ent))}`: `{v}_{e}_again` warns with `DeprecationWarning` (stacklevel 2, so the caller's line is reported) when `force` is passed and documents the deprecation."],
                       "notes": ["`force=False` also warns, on purpose: the point is that the keyword itself is going away.", "The removal is a one-line follow-up once the warning has shipped in a release."]}}


def task_missing_validation(ctx):
    ent = ctx.main
    e, E, a, fid = ent["snake"], ent["camel"], ent["amount"], ent["id"][0]
    err = ctx.errors[1] if len(ctx.errors) > 1 else ctx.base_error
    fx = ctx.names["fixture"]
    head = [f"def adjust_{a}({e}: {E}, delta: int) -> {E}:", f'    """Return the {e} with ``{a}`` changed by ``delta``; the result must not go negative."""']
    buggy = "\n".join(head + [f"    return replace({e}, {a}={e}.{a} + delta)"])
    fixed = "\n".join(head + [f"    new_value = {e}.{a} + delta", "    if new_value < 0:", f"        raise {err}(f\"{a} of {e} {{{e}.{fid}}} would become {{new_value}}\")", f"    return replace({e}, {a}=new_value)"])
    test = "\n".join([f"def test_adjust_{a}_rejects_a_negative_result({fx}_{e}):", f"    with pytest.raises({err}, match=\"would become\"):", f"        adjust_{a}({fx}_{e}, -({fx}_{e}.{a} + 1))", f"    assert adjust_{a}({fx}_{e}, -{fx}_{e}.{a}).{a} == 0"])
    return {"kind": "fix", "ask": f"`adjust_{a}` lets ``{a}`` go negative, and a negative value then fails much later in `{ctx.names['validate']}_{e}` with a message that does not say where it came from. `test_adjust_{a}_rejects_a_negative_result` fails with DID NOT RAISE. Reject the adjustment at the source with `{err}` and a message that includes the id and the would-be value; zero stays allowed.",
            "module": ctx.module_of(ent), "buggy": buggy, "fixed": fixed, "unit_name": f"adjust_{a}", "test": test, "test_name": f"test_adjust_{a}_rejects_a_negative_result",
            "failure": {"kind": "failed", "test_marker": 1, "e_lines": [f"Failed: DID NOT RAISE <class '{ctx.pkg}.errors.{err}'>"], "exc": "Failed", "frame": None},
            "search": f"adjust_{a}", "new_file": None,
            "report": {"root_cause": [f"`adjust_{a}` applied the delta without a check; the invariant that ``{a}`` is non-negative was only enforced by `{ctx.names['validate']}_{e}`, which runs at save time, long after the caller that produced the bad value has returned."],
                       "changes": [f"`{ctx.src(ctx.module_of(ent))}`: `adjust_{a}` computes the new value first and raises `{err}` with the id and the value when it is negative."],
                       "notes": ["Zero is allowed, matching the validator.", f"`{err}` derives from `{ctx.base_error}`, so existing `except {ctx.base_error}` handlers in the CLI keep working."]}}


def task_structured_logging(ctx):
    ent = ctx.main
    e, E, s, fid = ent["snake"], ent["camel"], ent["status"], ent["id"][0]
    v, past = ctx.verb(2)
    st = ctx.statuses()
    fx = ctx.names["fixture"]
    head = [f"def record_{v}({e}: {E}, *, actor: str) -> {E}:", f'    """Apply the ``{v}`` transition on behalf of ``actor`` and log it so the audit pipeline can index the event."""',
            f"    updated = replace({e}, {s}=\"{st[2]}\", {ent['ts']}={ctx.names['now']}())"]
    buggy = "\n".join(head + [f"    log.info(f\"{past} {e} {{{e}.{fid}}} by {{actor}} ({{{e}.{s}}} -> {{updated.{s}}})\")", "    return updated"])
    fixed = "\n".join(head + [f"    log.info(\"{past} {e}\", extra={{\"{e}_id\": {e}.{fid}, \"actor\": actor, \"from_{s}\": {e}.{s}, \"to_{s}\": updated.{s}}})", "    return updated"])
    test = "\n".join([f"def test_record_{v}_logs_the_id_as_a_field(caplog, {fx}_{e}):", f"    with caplog.at_level(\"INFO\", logger=\"{ctx.pkg}.{ctx.module_of(ent)}\"):", f"        record_{v}({fx}_{e}, actor=\"ops\")",
                      "    record = caplog.records[-1]", f"    assert getattr(record, \"{e}_id\", None) == {fx}_{e}.{fid}", "    assert getattr(record, \"actor\", None) == \"ops\"", f"    assert record.getMessage() == \"{past} {e}\""])
    return {"kind": "feature", "ask": f"The audit pipeline indexes log records by their `extra` fields and cannot parse the free-text message `record_{v}` writes. Switch that call to structured logging: a fixed message and the {e.replace('_', ' ')} id, the actor and both states as fields, the way the other transitions in the module log. `test_record_{v}_logs_the_id_as_a_field` is the failing test; do not change what the function returns.",
            "module": ctx.module_of(ent), "buggy": buggy, "fixed": fixed, "unit_name": f"record_{v}", "test": test, "test_name": f"test_record_{v}_logs_the_id_as_a_field",
            "failure": {"kind": "assert", "test_marker": 4, "e_lines": [f"AssertionError: assert None == {ctx.lit(ent, fid)}", f" +  where None = getattr(<LogRecord: {ctx.pkg}.{ctx.module_of(ent)}, 20, {ctx.src(ctx.module_of(ent))}, 0, \"{past} {e} ...\">, '{e}_id', None)"], "exc": "AssertionError", "frame": None},
            "search": "log\\.info\\(", "new_file": None,
            "report": {"root_cause": [f"`record_{v}` formatted the id, the actor and the states into the message string. Log records carry those only as text, and the audit pipeline reads record attributes, not the message."],
                       "changes": [f"`{ctx.src(ctx.module_of(ent))}`: `record_{v}` logs the fixed message `\"{past} {e}\"` with `extra={{...}}` carrying `{e}_id`, `actor`, `from_{s}` and `to_{s}`, matching the other transitions in the module."],
                       "notes": ["The message text changed on purpose (it is now constant); dashboards that grep the old sentence must switch to the fields.", "The return value and the state transition are unchanged."]}}


def task_flaky_clock(ctx):
    ent = ctx.main
    e, E, ts = ent["snake"], ent["camel"], ent["ts"]
    now = ctx.names["now"]
    fx = ctx.names["fixture"]
    head = [f"def {e}_age_s({e}: {E}) -> float:", f'    """Seconds since the {e}\'s ``{ts}`` according to the package clock."""']
    buggy = "\n".join(head + [f"    return (datetime.now(timezone.utc) - {e}.{ts}).total_seconds()"])
    fixed = "\n".join(head + [f"    return ({now}() - {e}.{ts}).total_seconds()"])
    test = "\n".join([f"def test_{e}_age_uses_the_package_clock(monkeypatch, {fx}_{e}):", f"    monkeypatch.setattr(\"{ctx.pkg}.{ctx.module_of(ent)}.{now}\", lambda: T0 + timedelta(hours=1))", f"    assert {e}_age_s({fx}_{e}) == 3600.0"])
    return {"kind": "fix", "ask": f"`test_{e}_age_uses_the_package_clock` is flaky in CI (it fails on roughly every tenth run with a value that is off by a fraction of a second or by hours, depending on the runner). The test patches the package clock `{now}` like every other time-dependent test in the suite. Make the test deterministic by fixing the code it exercises; do not loosen the assertion or retry.",
            "module": ctx.module_of(ent), "buggy": buggy, "fixed": fixed, "unit_name": f"{e}_age_s", "test": test, "test_name": f"test_{e}_age_uses_the_package_clock",
            "failure": {"kind": "assert", "test_marker": 2, "e_lines": ["assert 19526041.372819 == 3600.0", f" +  where 19526041.372819 = {e}_age_s({E}({ent['id'][0]}={ctx.lit(ent, ent['id'][0])}, ...))"], "exc": "AssertionError", "frame": None},
            "search": "datetime\\.now\\(|time\\.time\\(", "new_file": None,
            "report": {"root_cause": [f"`{e}_age_s` called `datetime.now(timezone.utc)` directly instead of the module clock `{now}`. The test patches `{now}`, which this function never consulted, so the result depended on the real wall clock: the difference between the fixture's timestamp and the moment the test ran.",
                                      "The failing value in the traceback is that real age in seconds."],
                       "changes": [f"`{ctx.src(ctx.module_of(ent))}`: `{e}_age_s` reads `{now}()`."],
                       "notes": ["I searched for other direct `datetime.now(` and `time.time(` calls in the package; the only remaining one is inside `" + now + "` itself, which is the intended place.", "The test is unchanged."]}}


TASK_BUILDERS = {
    "pagination_offset": task_pagination_offset, "wrong_exception": task_wrong_exception, "check_then_act_race": task_check_then_act_race,
    "rename_function": task_rename_function, "add_cli_flag": task_add_cli_flag, "config_env_override": task_config_env_override,
    "resource_leak": task_resource_leak, "stale_seconds": task_stale_seconds, "retry_policy": task_retry_policy, "unicode_truncation": task_unicode_truncation,
    "timezone_naive": task_timezone_naive, "quadratic_match": task_quadratic_match, "deprecate_parameter": task_deprecate_parameter,
    "missing_validation": task_missing_validation, "structured_logging": task_structured_logging, "flaky_clock": task_flaky_clock,
}
# themes = (task type, domain) pairs; every task type applies to every domain's vocabulary, so 16 x 12 = 192 themes (>= 40 required)
THEMES = [(t, d) for t in TASK_BUILDERS for d in DOMAINS]


# ---- token counting ------------------------------------------------------------------------------------------------------------
class Sizer:
    """Counts tokens with the pinned tokenizer.json (HF tokenizers); the only measurement the generator trusts."""

    def __init__(self, tok_dir):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
        self.calls = 0
        self.tokens = 0

    def count(self, text):
        self.calls += 1
        n = len(self.tk.encode(text, add_special_tokens=False).ids)
        self.tokens += n
        return n

    def count_many(self, texts):
        self.calls += 1
        out = [len(e.ids) for e in self.tk.encode_batch(list(texts), add_special_tokens=False)]
        self.tokens += sum(out)
        return out


def session_texts(s):
    """Per turn k: the texts added to the request of turn k (prefix sums give the prompt of each turn) and the recorded output.
    Identical to the corpus builder's function (template-free counting)."""
    added = [[s.get("system") or "", json.dumps(s.get("tools") or [])]]
    outputs = []
    for i, t in enumerate(s["turns"]):
        chunk = [] if i == 0 else [outputs[-1] if outputs else ""]
        if i > 0:
            prev = s["turns"][i - 1]
            chunk.extend(r.get("content") or "" for r in prev.get("tool_results") or [])
        if t.get("user"):
            chunk.append(t["user"])
        if t.get("response_format"):
            chunk.append(json.dumps(t["response_format"]))
        if i == 0:
            added[0].extend(chunk)
        else:
            added.append(chunk)
        ra = t.get("recorded_assistant") or {}
        out = (ra.get("content") or "") + (ra.get("reasoning") or "") + json.dumps([e.get("arguments") for e in t.get("expected_tool_calls") or []])
        outputs.append(out)
    return added, outputs


def prompt_tokens(sz, s):
    added, outputs = session_texts(s)
    flat, idx = [], []
    for k, chunk in enumerate(added):
        for t in chunk:
            flat.append(t); idx.append(k)
    counts = sz.count_many(flat)
    per = [0] * len(added)
    for k, c in zip(idx, counts):
        per[k] += c
    run, prompts = 0, []
    for c in per:
        run += c
        prompts.append(run)
    outs = sz.count_many(outputs)
    return prompts, outs


def numbered(text, start=1):
    return "".join(f"{start + i:6d}\t{line}\n" for i, line in enumerate(text.split("\n")))


# ---- the repository files of a session ------------------------------------------------------------------------------------------
class SourceFile:
    """An ordered pool of units; pack(budget) chooses the prefix (plus small tail fillers) whose numbered form has about budget tokens."""

    def __init__(self, path, header, units, fillers, sep="\n\n\n", code=True):
        self.path, self.header, self.units, self.fillers, self.sep, self.code = path, header, list(units), list(fillers), sep, code
        self.text = None  # the packed content (set by pack or fix)
        self.packed_units = None
        self.min_units = 3

    def render(self, n_units, fillers=()):
        parts = [self.header] + self.units[:n_units] + list(fillers)
        return self.sep.join(parts) + "\n"

    def fix(self, n_units=None):
        self.packed_units = len(self.units) if n_units is None else n_units
        self.text = self.render(self.packed_units)
        return self.text

    def pack(self, sz, budget):
        """Numbered-token budget -> packed text; returns the measured numbered token count."""
        if getattr(self, "_sizes", None) is None:
            blocks, line = [], 1
            head_lines = self.header.count("\n") + 1
            blocks.append(numbered(self.header, line)); line += head_lines
            for u in self.units:
                block = numbered("\n\n" + u, line)  # two separator lines then the unit
                blocks.append(block); line += u.count("\n") + 3
            self._sizes = sz.count_many(blocks)
            self._line_after = line
        sizes = self._sizes
        line = self._line_after
        cum, n = sizes[0], 0
        while n < len(self.units) and cum + sizes[n + 1] <= budget:
            cum += sizes[n + 1]; n += 1
        n = max(n, min(self.min_units, len(self.units)))
        if n > len(self.units):
            n = len(self.units)
        gap = budget - sum(sizes[:n + 1])
        chosen = []
        if gap > 12 and self.fillers:
            fsizes = sz.count_many([numbered("\n\n" + f, line) for f in self.fillers])
            best, best_sum = (), 0
            items = list(zip(fsizes, range(len(self.fillers))))
            for mask in range(1, 1 << min(len(items), 8)):
                tot, sel = 0, []
                for b in range(min(len(items), 8)):
                    if mask >> b & 1:
                        tot += items[b][0]; sel.append(items[b][1])
                if tot <= gap and tot > best_sum:
                    best_sum, best = tot, tuple(sel)
            chosen = [self.fillers[i] for i in sorted(best)]
        self.packed_units = n
        self.text = self.render(n, chosen)
        return sz.count(numbered(self.text[:-1]))

    def lines(self):
        return self.text.split("\n")

    def line_of(self, needle):
        """1-based line number of the first line containing needle, or None."""
        for i, l in enumerate(self.lines(), start=1):
            if needle in l:
                return i
        return None


def build_repo(ctx, task):
    """All files of the session's repository as SourceFile pools (unpacked). Returns (files dict path->SourceFile, tree list)."""
    files = {}
    mod_names = ctx.modules
    main_mod = ctx.module_of(ctx.main)
    for i, mod in enumerate(mod_names):
        if mod == "cli":
            continue
        ent = ctx.ents[i % len(ctx.ents)]
        extra = []
        if task.get("unit_name") and task["module"] == mod and task["buggy"] and "match_" in task["unit_name"]:
            other = ctx.ents[1]
            extra.append(f"from .{ctx.module_of(other)} import {other['camel']}")
        header = module_header(ctx, mod, ent, extra)
        if i == 4:
            extra += ["import urllib.error", "import urllib.request"]
        if i < 4:
            units = module_pool(ctx, mod, ent, est_tokens=42000 if i < 2 else 30000)
            if task.get("skip_shapes") and task["module"] == mod:
                units = [u for u in units if not u.startswith(f"def {ctx.names['load']}_{ent['plural']}(")]
        else:
            units = support_pool(ctx, mod, i)
        if task.get("buggy") and task["module"] == mod:
            units.insert(min(2, len(units)), task["buggy"])
        files[ctx.src(mod)] = SourceFile(ctx.src(mod), header, units, tail_fillers(ctx, ent))
        if task.get("rename") and task["module"] == mod:
            files[ctx.src(mod)].min_units = 8
    files[ctx.src("cli")] = cli_file(ctx, task)
    files[ctx.src("cli")].min_units = 8  # the shared handler, the four commands, show, build_parser and main are always present
    files[f"src/{ctx.pkg}/errors.py"] = SourceFile(f"src/{ctx.pkg}/errors.py", f'"""Exception types of {ctx.pkg}; every error raised on purpose derives from {ctx.base_error}."""', [sh_exceptions(ctx)], [])
    files[f"src/{ctx.pkg}/__init__.py"] = SourceFile(f"src/{ctx.pkg}/__init__.py", f'"""{ctx.pkg}: {ctx.d["blurb"]}."""\n\n__version__ = "0.{ctx.num(4) % 9 + 3}.0"', [], [])
    for i, mod in enumerate(mod_names):
        if mod == "cli" or i >= 4:
            continue
        ent = ctx.ents[i % len(ctx.ents)]
        tunits = test_pool(ctx, ent, mod)
        fixtures = [ts_fixture(ctx, ent), ts_fixture_list(ctx, ent)]
        if task.get("test") and task["module"] == mod:
            if "match_" in (task.get("unit_name") or ""):
                fixtures.append(ts_fixture(ctx, ctx.ents[1]))
                tunits = [u for u in tunits if not u.startswith(f"@pytest.fixture\ndef {ctx.names['fixture']}_{ctx.ents[1]['snake']}(")]
            tunits.insert(0, task["test"])
        modfile = files[ctx.src(mod)]
        defined = re.findall(r"^(?:def|class) ([A-Za-z_][A-Za-z0-9_]*)", modfile.header + "\n" + "\n".join(modfile.units), re.M)
        body = "\n".join(fixtures + tunits)
        names = [n for n in dict.fromkeys(defined) if not n.startswith("_") and re.search(r"\b" + re.escape(n) + r"\b", body)]
        if "match_" in (task.get("unit_name") or "") and mod == main_mod:
            names.append(ctx.ents[1]["camel"])
        header = test_header(ctx, mod, ent, names or [ent["camel"]])
        files[ctx.test_path(mod)] = SourceFile(ctx.test_path(mod), header, fixtures + tunits, [])
    files["tests/conftest.py"] = SourceFile("tests/conftest.py", '"""Shared fixtures: a frozen clock and a temporary data directory."""\nfrom __future__ import annotations\n\nfrom datetime import datetime, timezone\nfrom pathlib import Path\n\nimport pytest\n\n\n@pytest.fixture\ndef frozen_now(monkeypatch):\n    """Pin the package clock to a fixed instant for the duration of a test."""\n    instant = datetime(2026, 3, 14, 10, 22, tzinfo=timezone.utc)\n    monkeypatch.setattr("' + ctx.pkg + "." + main_mod + "." + ctx.names["now"] + '", lambda: instant)\n    return instant\n\n\n@pytest.fixture\ndef data_dir(tmp_path: Path) -> Path:\n    d = tmp_path / "data"\n    d.mkdir()\n    return d', [], [])
    files["README.md"] = SourceFile("README.md", readme_text(ctx), [], [], code=False)
    files["CHANGELOG.md"] = SourceFile("CHANGELOG.md", changelog_text(ctx), [], [], code=False)
    files["pyproject.toml"] = SourceFile("pyproject.toml", pyproject_text(ctx), [], [], code=False)
    tree = sorted(["./CHANGELOG.md", "./README.md", "./pyproject.toml", "./docs/format.md"] + [f"./docs/{m}.md" for m in mod_names if m != "cli"] + [f"./{p}" for p in files if p.startswith(("src/", "tests/"))])
    return files, tree


def support_pool(ctx, mod, i):
    """Modules 5 and 6 of a repository: the service clients and the reporting helpers (no second definition of the record types)."""
    rnd = random.Random(sid_hash(ctx.index, "support", mod))
    units = []
    if i == 4:
        units += [sh_transport(ctx), sh_client(ctx, 0), sh_client(ctx, 1), sh_retry_helper(ctx), sh_now(ctx)]
        shapes = ["find", "by_status", "format", "merge", "parse_line", "iter_since"]
    else:
        units += [sh_now(ctx)]
        shapes = ["format", "group", "summarize", "sort", "total", "iter_since", "by_status", "find"]
    k = 0
    for rep in range(12):
        for s in shapes:
            ent = ctx.ents[(k + rep) % len(ctx.ents)]
            units.append(render_shape(ctx, s, ent, k)); k += 1
        rnd.shuffle(shapes)
    return units


def cli_file(ctx, task):
    ent = ctx.main
    e, P = ent["snake"], ent["plural"]
    mod = ctx.module_of(ent)
    names = [f"{ctx.names['load']}_{P}", f"{ctx.names['save']}_{P}", f"{ctx.names['find']}_{e}", f"{ctx.names['format']}_{e}", f"sorted_{P}", f"{P}_with_{ent['status']}"] + [f"{v}_{e}" for v, _ in ctx.d["verbs"][:4]]
    header = "\n".join([f'"""Command line of {ctx.pkg}: thin handlers over the library functions. Every command loads the file, applies one transition and writes the file back."""',
                        "from __future__ import annotations", "", "import argparse", "import logging", "import sys", "from datetime import datetime, timezone", "from pathlib import Path", "", f"from .errors import {ctx.base_error}",
                        f"from .{mod} import (", *[f"    {n}," for n in names], ")", "", "log = logging.getLogger(__name__)"])
    run_tr = "\n".join([f"def _run_transition(args: argparse.Namespace, transition) -> int:", f'    """Shared body of the transition commands: load, find, apply, save, print."""',
                        f"    {P} = {ctx.names['load']}_{P}(Path(args.file))", f"    target = {ctx.names['find']}_{e}({P}, args.{ent['id'][0]})", "    if target is None:",
                        f"        print(f\"no {e} {{args.{ent['id'][0]}}}\", file=sys.stderr)", "        return 2", "    try:", "        updated = transition(target)", f"    except {ctx.base_error} as exc:",
                        "        print(str(exc), file=sys.stderr)", "        return 1", f"    {ctx.names['save']}_{P}(Path(args.file), [updated if x.{ent['id'][0]} == updated.{ent['id'][0]} else x for x in {P}])",
                        f"    print({ctx.names['format']}_{e}(updated))", "    return 0"])
    cmds = ["\n".join([f"def cmd_{v}(args: argparse.Namespace) -> int:", f'    """``{ctx.pkg} {v}``: {v} one {e.replace("_", " ")} and write the file back."""', f"    return _run_transition(args, {v}_{e})"]) for v, _ in ctx.d["verbs"][:4]]
    main_fn = "\n".join(["def main(argv: list[str] | None = None) -> int:", '    """Console entry point."""', "    parser = build_parser()", "    args = parser.parse_args(argv)",
                         "    logging.basicConfig(level=logging.WARNING - 10 * min(args.verbose, 2), format=\"%(levelname)s %(name)s: %(message)s\")", "    return int(args.func(args))"])
    units = [run_tr] + cmds + [sh_cmd_show(ctx, ent), sh_build_parser(ctx), main_fn]
    helpers = ["\n".join(["def _parse_when(raw: str) -> datetime:", '    """Parse the ``--at`` option: an ISO 8601 instant, naive values taken as UTC."""', "    value = datetime.fromisoformat(raw)",
                          "    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)"]),
               "\n".join(["def _print_table(rows: list[str]) -> None:", '    """Print rows with a trailing count line, the format every listing command shares."""', "    for row in rows:", "        print(row)", "    print(f\"{len(rows)} rows\")"]),
               "\n".join(["def _confirm(prompt: str, *, assume_yes: bool) -> bool:", '    """Ask on stderr unless ``--yes`` was given; a non-interactive stdin counts as no."""', "    if assume_yes:", "        return True",
                          "    if not sys.stdin.isatty():", "        return False", "    print(prompt + \" [y/N] \", end=\"\", file=sys.stderr, flush=True)", "    return sys.stdin.readline().strip().lower() in (\"y\", \"yes\")"])]
    for other in ctx.ents[1:]:
        for k in range(2):
            units.append(sh_cmd(ctx, other, k + 1))
        units.append(sh_cmd_show(ctx, other).replace("def cmd_show(", f"def cmd_show_{other['plural']}("))
    units += helpers
    return SourceFile(ctx.src("cli"), header, units, [])


# ---- tool results -----------------------------------------------------------------------------------------------------------------
def pytest_header(ctx, collected, extra=""):
    return ["============================= test session starts ==============================", f"platform linux -- Python {ctx.py}.{ctx.num(7) % 7 + 1}, pytest-8.3.{ctx.num(8) % 5}, pluggy-1.5.0",
            f"rootdir: /workspace/{ctx.pkg}", "configfile: pyproject.toml", f"collected {collected} item{'s' if collected != 1 else ''}{extra}", ""]


def pytest_dots(files):
    """[(path, n, fail_index|None)] -> the progress lines of pytest's default output (wrapped at 80 columns)."""
    total = sum(n for _, n, _ in files) or 1
    done, lines = 0, []
    for path, n, fi in files:
        marks = "".join("F" if i == fi else "." for i in range(n))
        prefix = f"{path} "
        width = 72 - len(prefix)
        first, rest = marks[:width], marks[width:]
        done_here = done + len(first)
        lines.append(f"{prefix}{first}".ljust(72) + f"[{int(done_here * 100 / total):3d}%]")
        while rest:
            chunk, rest = rest[:72], rest[72:]
            done_here += len(chunk)
            lines.append(chunk.ljust(72) + f"[{int(done_here * 100 / total):3d}%]")
        done += n
    return lines


def banner(text, ch="="):
    pad = max(0, 80 - len(text) - 2)
    return f"{ch * (pad // 2)} {text} {ch * (pad - pad // 2)}"


def failure_block(ctx, task, test_file, mod_file):
    f = task["failure"]
    tname = task["test_name"]
    test_src = task["test"].split("\n")
    tline = test_file.line_of(f"def {tname}(") or 1
    args_m = re.match(r"def \w+\((.*)\):", test_src[0])
    params = [p.strip().split(":")[0] for p in (args_m.group(1).split(",") if args_m else []) if p.strip()]
    lines = [banner(tname, "_"), ""]
    fx = ctx.names["fixture"]
    ent = ctx.main
    shown = []
    for p in params:
        if p.startswith(fx + "_") and p.endswith(ent["plural"]):
            shown.append(f"{p} = [{ent['camel']}({ent['id'][0]}='{ent['id'][1]}0001', {ent['name']}={ctx.lit(ent, ent['name'])}, ...), ...]")
        elif p.startswith(fx + "_"):
            shown.append(f"{p} = {ent['camel']}({ent['id'][0]}={ctx.lit(ent, ent['id'][0])}, {ent['name']}={ctx.lit(ent, ent['name'])}, {ent['status']}={ctx.lit(ent, ent['status'])}, ...)")
        elif p == "monkeypatch":
            shown.append("monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f3c1a1f0d10>")
        elif p == "tmp_path":
            shown.append(f"tmp_path = PosixPath('/tmp/pytest-of-ci/pytest-12/{tname[:30]}0')")
        elif p == "caplog":
            shown.append("caplog = <_pytest.logging.LogCaptureFixture object at 0x7f3c1a2d0e50>")
    if shown:
        lines += shown + [""]
    marker = f["test_marker"]
    for i, l in enumerate(test_src[: marker + 1]):
        lines.append(("> " if i == marker else "  ") + "  " + l)
    if f.get("frame"):
        lines += ["", f"{test_file.path}:{tline + marker}: ", "_ " * 40, ""]
        unit_src = task["buggy"].split("\n")
        mline = mod_file.line_of(unit_src[0]) or 1
        fm = f["frame"]["marker"]
        sig = re.match(r"def \w+\((.*?)\)", unit_src[0])
        if sig:
            argnames = [a.strip().split(":")[0].split("=")[0].strip("* ") for a in sig.group(1).split(",") if a.strip() and not a.strip().startswith("*")]
            lines.append(", ".join(f"{a} = ..." for a in argnames[:3]).replace(" = ...", " = <value>") if argnames else "")
            lines.append("")
        for i, l in enumerate(unit_src[: fm + 1]):
            lines.append(("> " if i == fm else "  ") + "  " + l)
        lines += [f"E       {x}" if j == 0 else f"E       {x}" for j, x in enumerate(f["frame"]["e_lines"])]
        lines += ["", f"{mod_file.path}:{mline + fm}: {f['exc']}"]
    else:
        lines += [f"E       {x}" for x in f["e_lines"]]
        lines += ["", f"{test_file.path}:{tline + marker}: {f['exc']}"]
    return lines


def run_tests_output(ctx, task, files, *, narrow, failing, seed):
    """pytest output: narrow = the task's test file (with -k the failing test when failing), else the whole suite."""
    rnd = random.Random(seed)
    test_files = sorted(p for p in files if p.startswith("tests/test_"))
    counts = {}
    for p in test_files:
        sf = files[p]
        counts[p] = (sf.text or sf.fix()).count("\ndef test_")
    tpath = ctx.test_path(task["module"]) if task["module"] != "cli" else ctx.test_path(ctx.module_of(ctx.main))
    tfile = files.get(tpath)
    mod_file = files.get(ctx.src(task["module"]))
    if narrow and failing:
        n = counts.get(tpath, 20)
        out = pytest_header(ctx, n, f" / {n - 1} deselected / 1 selected") + pytest_dots([(tpath, 1, 0)]) + ["", banner("FAILURES")]
        out += failure_block(ctx, task, tfile, mod_file)
        out += [banner("short test summary info"), f"FAILED {tpath}::{task['test_name']} - {task['failure']['exc']}: {task['failure']['e_lines'][0][:90]}".rstrip(),
                banner(f"1 failed, {n - 1} deselected in 0.{rnd.randrange(12, 60):02d}s")]
        return "\n".join(out)
    if narrow:
        n = counts.get(tpath, 20)
        out = pytest_header(ctx, n) + pytest_dots([(tpath, n, None)]) + ["", banner(f"{n} passed in 0.{rnd.randrange(20, 95):02d}s")]
        return "\n".join(out)
    total = sum(counts.values())
    if failing:
        fail_idx = (tfile.text or "").count("\ndef test_", 0, (tfile.text or "").find(f"def {task['test_name']}(")) if tfile else 0
        dots = [(p, counts[p], fail_idx if p == tpath else None) for p in test_files]
        out = pytest_header(ctx, total) + pytest_dots(dots) + ["", banner("FAILURES")] + failure_block(ctx, task, tfile, mod_file)
        out += [banner("short test summary info"), f"FAILED {tpath}::{task['test_name']} - {task['failure']['exc']}: {task['failure']['e_lines'][0][:90]}".rstrip(),
                banner(f"1 failed, {total - 1} passed in {rnd.randrange(1, 4)}.{rnd.randrange(10, 99)}s")]
        return "\n".join(out)
    out = pytest_header(ctx, total) + pytest_dots([(p, counts[p], None) for p in test_files]) + ["", banner(f"{total} passed in {rnd.randrange(1, 4)}.{rnd.randrange(10, 99)}s")]
    return "\n".join(out)


def search_output(files, pattern, max_results=60):
    rx = re.compile(pattern)
    out, n = [], 0
    for path in sorted(files):
        sf = files[path]
        text = sf.text or sf.fix()
        hits = [(i, l) for i, l in enumerate(text.split("\n"), start=1) if rx.search(l)]
        if not hits:
            continue
        out.append(f"== {path} ({len(hits)} match{'es' if len(hits) != 1 else ''})")
        for i, l in hits:
            if n >= max_results:
                break
            out.append(f"{i}: {l}")
            n += 1
        if n >= max_results:
            out.append(f"... stopped after {max_results} matching lines; narrow the pattern or raise max_results")
            break
    if not out:
        return "no matches"
    return "\n".join(out)


def list_output(files, tree, path):
    out = []
    prefix = "" if path == "." else path.rstrip("/") + "/"
    seen_dirs = set()
    for entry in tree:
        rel = entry[2:]
        if not rel.startswith(prefix):
            continue
        rest = rel[len(prefix):]
        if "/" in rest:
            d = rest.split("/")[0]
            if d not in seen_dirs:
                seen_dirs.add(d)
                out.append(f"dir          {d}/")
        else:
            sf = files.get(rel)
            size = len((sf.text or sf.fix()).encode()) if sf else 1200 + (sid_hash(rel) % 2400)
            out.append(f"file {size:>8}  {rest}")
    return "\n".join(sorted(out, key=lambda l: (not l.startswith("dir"), l)))


def git_diff_output(edits, new_files):
    """edits: [(path, before_text, after_text)]; new_files: [(path, content)]."""
    out = []
    for path, before, after in edits:
        if before == after:
            continue
        h1, h2 = hashlib.sha1(before.encode()).hexdigest()[:7], hashlib.sha1(after.encode()).hexdigest()[:7]
        out += [f"diff --git a/{path} b/{path}", f"index {h1}..{h2} 100644"]
        out += list(difflib.unified_diff(before.split("\n"), after.split("\n"), fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="", n=3))
    for path, content in new_files:
        lines = content.rstrip("\n").split("\n")
        h2 = hashlib.sha1(content.encode()).hexdigest()[:7]
        out += [f"diff --git a/{path} b/{path}", "new file mode 100644", f"index 0000000..{h2}", "--- /dev/null", f"+++ b/{path}", f"@@ -0,0 +1,{len(lines)} @@"] + [f"+{l}" for l in lines]
    return "\n".join(out) if out else "(no changes)"


def edit_ack(path, before, old, new):
    start = before[: before.index(old)].count("\n") + 1
    n_old, n_new = old.count("\n") + 1, new.count("\n") + 1
    after = before.replace(old, new, 1)
    snippet_lines = after.split("\n")[max(0, start - 2): start - 1 + n_new + 1]
    snippet = "".join(f"{max(1, start - 1) + i:6d}\t{l}\n" for i, l in enumerate(snippet_lines))
    return f"Edited {path}: replaced lines {start}-{start + n_old - 1} ({n_old} line{'s' if n_old != 1 else ''}) with {n_new} line{'s' if n_new != 1 else ''}. The file now reads around the change:\n{snippet}"


# ---- the change set of a task ----------------------------------------------------------------------------------------------------
def change_calls(ctx, task, files):
    """[(kind, path, old, new | content)] in the order the agent applies them; resolved against the packed file texts."""
    calls = []
    ent = ctx.main
    e, P = ent["snake"], ent["plural"]
    if task.get("buggy"):
        old, new = _unit_edit(task["buggy"], task["fixed"])
        calls.append(("edit", ctx.src(task["module"]), old, new))
    if task["kind"] == "refactor" and task.get("rename"):
        old_n, new_n = task["rename"]
        mod = ctx.src(task["module"])
        text = files[mod].text
        m = re.search(rf"^def {re.escape(old_n)}\((.*)\) -> (.*):\n(    \"\"\".*\"\"\")$", text, re.M)
        if m:
            sig_args, ret, doc = m.group(1), m.group(2), m.group(3)
            new = "\n".join([f"def {old_n}({sig_args}) -> {ret}:", f'    """Deprecated alias of :func:`{new_n}`; it will be removed in the next release."""',
                             f"    warnings.warn(\"{old_n} is deprecated; use {new_n}\", DeprecationWarning, stacklevel=2)", f"    return {new_n}(path)", "", "",
                             f"def {new_n}({sig_args}) -> {ret}:", doc])
            calls.append(("edit", mod, m.group(0), new))
        cli = ctx.src("cli")
        clines = files[cli].text.split("\n")
        if f"    {old_n}," in clines:
            calls.append(("edit", cli, f"    {old_n},", f"    {new_n},"))
        for i, ln in enumerate(clines):
            if f" = {old_n}(" in ln:
                calls.append(("edit", cli, clines[i - 1] + "\n" + ln, clines[i - 1] + "\n" + ln.replace(f" = {old_n}(", f" = {new_n}(")))
        tpath = ctx.test_path(task["module"])
        tlines = files[tpath].text.rstrip("\n").split("\n")
        for ctxn in (2, 3, 4):
            anchor = "\n".join(tlines[-ctxn:])
            if files[tpath].text.count(anchor) == 1:
                break
        calls.append(("edit", tpath, anchor, anchor + "\n\n\ndef test_" + old_n + "_is_a_deprecated_alias(tmp_path: Path):\n    with pytest.warns(DeprecationWarning, match=\"" + new_n + "\"):\n        assert " + old_n + "(tmp_path / \"absent.jsonl\") == []"))
    if task["kind"] == "feature" and task.get("flag") == "dry_run":
        cli = ctx.src("cli")
        text = files[cli].text
        verbose_line = next(l for l in text.split("\n") if "--verbose" in l)
        calls.append(("edit", cli, verbose_line, verbose_line + "\n    parser.add_argument(\"--dry-run\", action=\"store_true\", help=\"apply the transition and print the result without writing the file\")"))
        save_line = next(l for l in text.split("\n") if l.startswith(f"    {ctx.names['save']}_{P}(Path(args.file)"))
        calls.append(("edit", cli, save_line + f"\n    print({ctx.names['format']}_{e}(updated))",
                      "    if args.dry_run:\n" + f"        print({ctx.names['format']}_{e}(updated) + \"  (dry run)\")\n        return 0\n" + save_line + f"\n    print({ctx.names['format']}_{e}(updated))"))
        v = ctx.verb(0)[0]
        fx = ctx.names["fixture"]
        st = ctx.statuses()
        content = "\n".join([f'"""Tests of the ``{ctx.pkg}`` command line."""', "from __future__ import annotations", "", "from dataclasses import replace", "from pathlib import Path", "",
                             f"from {ctx.pkg}.cli import main", f"from {ctx.pkg}.{ctx.module_of(ent)} import {ctx.names['load']}_{P}, {ctx.names['save']}_{P}", "", "",
                             f"def _write(path: Path, {P}) -> bytes:", f"    {ctx.names['save']}_{P}(path, {P})", "    return path.read_bytes()", "", "",
                             f"def test_cli_{v}_dry_run_does_not_write(tmp_path: Path, {fx}_{P}, capsys):", f"    path = tmp_path / \"{P}.jsonl\"",
                             f"    ready = [replace(x, {ent['status']}=\"{st[0]}\") for x in {fx}_{P}]", "    before = _write(path, ready)",
                             f"    assert main([\"--file\", str(path), \"--dry-run\", \"{v}\", ready[0].{ent['id'][0]}]) == 0", "    assert path.read_bytes() == before",
                             "    assert \"(dry run)\" in capsys.readouterr().out", "", "",
                             f"def test_cli_{v}_writes_without_the_flag(tmp_path: Path, {fx}_{P}):", f"    path = tmp_path / \"{P}.jsonl\"",
                             f"    ready = [replace(x, {ent['status']}=\"{st[0]}\") for x in {fx}_{P}]", "    before = _write(path, ready)",
                             f"    assert main([\"--file\", str(path), \"{v}\", ready[0].{ent['id'][0]}]) == 0", "    assert path.read_bytes() != before",
                             f"    assert {ctx.names['load']}_{P}(path)[0].{ent['status']} == \"{st[1]}\"", ""])
        calls.append(("write", "tests/test_cli.py", None, content))
    if task.get("envvar"):
        readme = files["README.md"].text
        anchor = next(l for l in readme.split("\n") if l.startswith("## Design notes"))
        calls.append(("edit", "README.md", anchor, f"The default file can also be set with the environment variable `{task['envvar']}`; `--file` on the command line wins over it, and an empty value means the built-in default.\n\n{anchor}"))
    return calls


# ---- assistant turns ----------------------------------------------------------------------------------------------------------------
def assistant_text(ctx, task, step_kind, calls, rnd):
    """(reasoning, content) for a tool turn, from the step kind and the calls it makes."""
    ent = ctx.main
    paths = [c[1].get("path") for c in calls if isinstance(c[1], dict) and c[1].get("path")]
    tn = task.get("test_name")
    r = {
        "repro": [f"The task names {tn}. The harness rules say to run the failing test first and read the traceback before touching any code, so I will run it narrowed to that test and read the test file in the same turn.",
                  f"Before I read code I want the traceback of {tn}; it tells me which frames matter. I will run the single test and open the test module alongside it."],
        "baseline": ["The task is a change, not a failing test, so I want a green baseline before I edit anything; I will run the suite narrowed to the module's test file and read that file.",
                     "First a baseline run of the module's tests so I know the suite is green before my change, and the test file itself so I can follow the project's fixtures."],
        "read": [f"The next step is to read {' and '.join('`' + p + '`' for p in paths)} in full; I need the surrounding code and the project's conventions before I change a line.",
                 f"I will read {' and '.join('`' + p + '`' for p in paths)} once, completely, and keep what I need rather than reading them twice."],
        "search": [f"Before changing anything I want every definition and use of the pattern `{task['search']}` across the package, and I will read the related module the task does not mention.",
                   f"A search for `{task['search']}` tells me whether the problem exists elsewhere; I will also read one more related module."],
        "list": [f"A quick orientation in `src/{ctx.pkg}` to confirm the module layout, plus one more module that the earlier reads referenced.",
                 f"I will list the package directory to confirm which modules exist and read the one the CLI imports from."],
        "change": ([f"I still want `{paths[0]}` for the remaining context, and I have the cause, so I will read it and apply the edit in the same turn, copied against the exact text I read.",
                    f"One more read of `{paths[0]}` to confirm nothing else depends on the old behaviour, then the edit itself with the exact text from the read output."] if calls and calls[0][0] == "read_file" else
                   [f"I have the cause. The smallest change is in {' and '.join(sorted(set('`' + p + '`' for p in paths)))}; I will apply it with exact text from the read output and keep the surrounding style.",
                    f"Now the edit itself, copied against the exact text I read, one logical change per call."]),
        "verify": ["With the change in place I will run the tests and read the diff before I report, as the rules require.",
                   "Now the verification: the test run and the diff of the session so far."],
        "verify_narrow": ["First the narrow run of the touched test file; the whole suite comes next.", "A narrow run first, to see the originally failing test pass, then the full suite."],
    }[step_kind]
    c = {
        "repro": ["Let me run the failing test and read the test file.", f"I will reproduce {tn} first and look at the test."],
        "baseline": ["Let me run the module's tests for a baseline and read the test file.", "Baseline test run first, and the test module."],
        "read": [f"Reading {' and '.join('`' + p + '`' for p in paths)}.", f"Let me read {' and '.join('`' + p + '`' for p in paths)} in full."],
        "search": [f"Searching for `{task['search']}` and reading the related module.", "Let me find every use and read the neighbouring module."],
        "list": ["Listing the package directory and reading one more module.", "Let me check the layout and read the module the CLI uses."],
        "change": ([f"Reading `{paths[0]}` and applying the fix.", f"Let me read `{paths[0]}` and apply the change."] if calls and calls[0][0] == "read_file" else ["Applying the fix." if task["kind"] == "fix" else "Making the change.", "Let me apply the change."]),
        "verify": ["Running the tests and checking the diff.", "Let me run the tests and read the diff."],
        "verify_narrow": ["Running the touched test file first.", "Let me run the narrow test file."],
    }[step_kind]
    i = rnd.randrange(2)
    return r[i], c[i]


# ---- the final report -------------------------------------------------------------------------------------------------------------
def final_report(ctx, task, target_tokens, sz, calls, files_read, diff_text, test_summary_lines, failing_excerpt=""):
    rep = task["report"]
    ent = ctx.main
    head = {"fix": "Root cause", "perf": "Root cause", "feature": "Approach", "refactor": "Approach"}[task["kind"]]
    summary = {"fix": f"{task['test_name']} failed because of a defect in `{task['unit_name']}` in `{ctx.src(task['module'])}`. I fixed the function with a {'one' if len(calls) == 1 else 'small'}-line change, the test passes and the full suite is green.",
               "perf": f"`{task['unit_name']}` was quadratic in the input sizes. I replaced the inner scan with a one-time index; the result is identical and the budget test passes.",
               "feature": f"I implemented the requested change in {' and '.join(sorted(set('`' + c[1] + '`' for c in calls)))}, added the test the task asked for, and the full suite is green.",
               "refactor": f"The rename is done across the definition, the callers and the tests, with a deprecated alias in place; behaviour is unchanged and the suite is green."}[task["kind"]]
    sections = [f"## Summary\n\n{summary}", f"## {head}\n\n" + "\n\n".join(rep["root_cause"]), "## Changes\n\n" + "\n".join("- " + c for c in rep["changes"]),
                "## Verification\n\n" + "\n".join("- " + l for l in test_summary_lines)]
    text = "\n\n".join(sections)
    optional = []
    if rep.get("notes"):
        optional.append("## Notes\n\n" + "\n".join("- " + n for n in rep["notes"]))
    optional.append("## What I read and did not change\n\n" + "\n".join(f"- `{p}`: {d}" for p, d in files_read))
    if diff_text and diff_text != "(no changes)":
        optional.append("## Patch\n\n```diff\n" + diff_text + "\n```")
    if task.get("buggy"):
        optional.append("## Before and after\n\nThe unit as it was:\n\n```python\n" + task["buggy"] + "\n```\n\nAnd as it is now:\n\n```python\n" + task["fixed"] + "\n```")
    if failing_excerpt:
        optional.append("## The failing run, for the record\n\n```\n" + failing_excerpt + "\n```")
    optional.append("## Reviewer checklist\n\n" + "\n".join("- " + x for x in [
        f"The change is confined to {' and '.join(sorted(set('`' + c[1] + '`' for c in calls)))}; nothing else in the tree differs from HEAD.",
        "No new dependency, no reformatting of untouched lines, no change to the public names other than the ones the task asked for.",
        f"The tests I ran are listed above with their summary lines; the full suite ran once at the end of the session.",
        "Assumptions are listed in the notes; if one of them is wrong the change is small enough to revert with a single edit."]))
    n = sz.count(text)
    for part in optional:
        if n >= target_tokens:
            break
        text = text + "\n\n" + part
        n = sz.count(text)
    return text


# ---- the session -------------------------------------------------------------------------------------------------------------------
def plan_session(index, seed, first_range=FIRST_RANGE, final_range=FINAL_RANGE):
    rnd = random.Random(f"gen-v2:{seed}:{index}")
    n_tool = rnd.choices(*TURN_CHOICE)[0]
    first = int(round(math.exp(rnd.uniform(math.log(first_range[0]), math.log(first_range[1])))))
    final = int(round(math.exp(rnd.uniform(math.log(final_range[0]), math.log(final_range[1])))))
    final = max(final, first + n_tool * 1500)
    out_final = int(round(math.exp(rnd.uniform(math.log(300), math.log(3000)))))
    excluded = {3: ("rename_function", "add_cli_flag"), 4: ("rename_function",)}.get(n_tool, ())
    themes = [(t, d) for t, d in THEMES if t not in excluded]
    task_key, domain_key = rnd.choice(themes)
    return {"rnd": rnd, "n_tool": n_tool, "first": first, "final": final, "out_final": out_final, "task_key": task_key, "domain_key": domain_key}


def build_session(index, seed, sz, first_range=FIRST_RANGE, final_range=FINAL_RANGE, id_prefix="gen"):
    """One generated session (pre-finalize record) and its calibration facts."""
    plan = plan_session(index, seed, first_range, final_range)
    rnd = plan["rnd"]
    ctx = Ctx(rnd, plan["domain_key"], index)
    task = TASK_BUILDERS[plan["task_key"]](ctx)
    N = plan["n_tool"]
    ent = ctx.main
    sid = f"{id_prefix}-{index:03d}"
    # ---- the turn script: front turns (2 calls each), change turns, verify turns
    fix_like = task["kind"] in ("fix", "perf") or (task["kind"] == "feature" and task.get("test"))
    files, tree = build_repo(ctx, task)
    # provisional packing so the change calls can be resolved (they only need the module/cli/test/README texts)
    for sf in files.values():
        sf.fix(min(len(sf.units), 24))
    changes = change_calls(ctx, task, files)
    n_change_turns = max(1, math.ceil(len(changes) / 2))
    n_verify = 2 if N == 8 else 1
    n_front = N - n_change_turns - n_verify
    assert n_front >= 1, (index, N, len(changes))
    test_path = ctx.test_path(task["module"]) if task["module"] != "cli" else ctx.test_path(ctx.module_of(ent))
    bug_path = ctx.src(task["module"])
    read_order = [bug_path, test_path] if task["module"] != "cli" else [ctx.src(ctx.module_of(ent)), bug_path, test_path]
    for m in ctx.modules:
        p = ctx.src(m)
        if p not in read_order:
            read_order.append(p)
    for m in ctx.modules[:4]:
        p = ctx.test_path(m)
        if p not in read_order:
            read_order.append(p)
    reads_iter = iter(read_order)
    turns_spec = []  # list of (kind, [call specs]) ; call spec = (tool, args, result_kind, payload)
    first_kind = ("repro" if fix_like else "baseline") if N >= 4 else "read"
    for t in range(n_front):
        if t == 0:
            if first_kind == "read":
                turns_spec.append(("read", [("read_file", next(reads_iter)), ("read_file", next(reads_iter))]))
            else:
                turns_spec.append((first_kind, [("run_tests", "narrow"), ("read_file", next(reads_iter))]))
        elif t == 1:
            turns_spec.append(("read", [("read_file", next(reads_iter)), ("read_file", next(reads_iter))]))
        elif t == 2:
            turns_spec.append(("search", [("search_code", task["search"]), ("read_file", next(reads_iter))]))
        elif t == 3:
            turns_spec.append(("list", [("list_dir", f"src/{ctx.pkg}"), ("read_file", next(reads_iter))]))
        else:
            turns_spec.append(("read", [("read_file", next(reads_iter)), ("read_file", next(reads_iter))]))
    ch = list(enumerate(changes))
    for t in range(n_change_turns):
        batch = ch[:2]; ch = ch[2:]
        calls = [("edit_file" if c[0] == "edit" else "write_file", ci) for ci, c in batch]
        if len(calls) == 1:
            calls.insert(0, ("read_file", next(reads_iter)))
        turns_spec.append(("change", calls))
    if n_verify == 2:
        turns_spec.append(("verify_narrow", [("run_tests", "narrow_pass")]))
    turns_spec.append(("verify", [("run_tests", "full_pass"), ("git_diff", None)]))
    assert len(turns_spec) == N
    read_paths = [a for _, calls in turns_spec for tool, a in calls if tool == "read_file"]
    # ---- fixed-size pieces: assistant outputs and the natural (non-read) results; measured once with provisional texts
    def render_all(read_budgets, user_budget):
        # pack the read files to their budgets, fix the rest at a modest size
        for p, sf in files.items():
            if p not in read_paths:
                sf.fix(min(len(sf.units), 24))
        residual = 0
        for p in read_paths:
            got = files[p].pack(sz, max(200, read_budgets[p] + residual))
            residual = read_budgets[p] + residual - got
        # the change set against the packed texts; the file states before/after
        chg = change_calls(ctx, task, files)
        assert len(chg) == len(changes), (sid, len(chg), len(changes))
        before = {p: files[p].text for p in files}
        original = dict(before)
        after = dict(before)
        new_files = []
        for kind, path, old, new in chg:
            if kind == "edit":
                assert after[path].count(old) == 1, (sid, path, old[:60], after[path].count(old))
                after[path] = after[path].replace(old, new, 1)
            else:
                new_files.append((path, new))
        turns = []
        files_read = []
        call_no = 0
        for k, (kind, calls) in enumerate(turns_spec, start=1):
            ecalls, results = [], []
            state_before = dict(before)
            for j, (tool, arg) in enumerate(calls):
                cid = f"call_{id_prefix}_{index}_{k}_{j}"
                if tool == "read_file":
                    args = {"path": arg}
                    content = numbered(files[arg].text[:-1]) if files[arg].text.endswith("\n") else numbered(files[arg].text)
                    doc = re.search(r'"""(.+?)[\n"]', files[arg].text)
                    files_read.append((arg, (doc.group(1).strip().rstrip(".") if doc else "read for context") + (" (no change needed)" if arg not in {c[1] for c in chg} else " (changed)")))
                elif tool == "run_tests":
                    if arg == "narrow":
                        args = {"path": test_path, "keyword": task["test_name"]} if fix_like else {"path": test_path}
                        for p in before:
                            files[p].text = before[p]
                        content = run_tests_output(ctx, task, files, narrow=True, failing=fix_like, seed=sid_hash(sid, "t", k))
                    elif arg == "narrow_pass":
                        args = {"path": test_path}
                        for p in before:
                            files[p].text = after.get(p, before[p])
                        if new_files:
                            for np_, c in new_files:
                                files[np_] = SourceFile(np_, c.rstrip("\n"), [], []); files[np_].fix()
                        content = run_tests_output(ctx, task, files, narrow=True, failing=False, seed=sid_hash(sid, "t", k))
                    else:
                        args = {}
                        for p in before:
                            files[p].text = after.get(p, before[p])
                        if new_files:
                            for np_, c in new_files:
                                files[np_] = SourceFile(np_, c.rstrip("\n"), [], []); files[np_].fix()
                        content = run_tests_output(ctx, task, files, narrow=False, failing=False, seed=sid_hash(sid, "t", k))
                elif tool == "search_code":
                    args = {"pattern": arg, "path": "src", "glob": "*.py"} if "test" not in arg else {"pattern": arg}
                    content = search_output({p: f for p, f in files.items() if p.startswith("src/")}, arg)
                elif tool == "list_dir":
                    args = {"path": arg}
                    content = list_output(files, tree, arg)
                elif tool == "edit_file":
                    _, path, old, new = chg[arg]
                    args = {"path": path, "old_text": old, "new_text": new}
                    content = edit_ack(path, state_before[path], old, new)
                    state_before[path] = state_before[path].replace(old, new, 1)
                    before[path] = state_before[path]
                elif tool == "write_file":
                    _, path, _, new = chg[arg]
                    args = {"path": path, "content": new}
                    content = f"Wrote {path} ({new.count(chr(10))} lines, {len(new.encode()):,} bytes)."
                elif tool == "git_diff":
                    args = {}
                    content = git_diff_output([(p, original[p], after[p]) for p in sorted(after)], new_files)
                ecalls.append({"id": cid, "name": tool, "arguments": json.dumps(args, ensure_ascii=False)})
                results.append({"tool_call_id": cid, "name": tool, "content": content})
                call_no += 1
            reasoning, text = assistant_text(ctx, task, kind, [(c["name"], json.loads(c["arguments"])) for c in ecalls], random.Random(sid_hash(sid, "a", k)))
            turns.append({"index": k, "user": None, "expected_tool_calls": ecalls, "tool_results": results,
                          "recorded_assistant": {"content": text, "reasoning": reasoning, "tool_calls": ecalls}, "max_tokens": TOOL_TURN_MAX_TOKENS, "response_format": None})
        for np_, _ in new_files:
            files.pop(np_, None)
        for p in files:
            files[p].text = original[p]
        # the final turn
        diff_text = turns[-1]["tool_results"][-1]["content"]
        summary_lines = []
        for t in turns:
            for c, r in zip(t["expected_tool_calls"], t["tool_results"]):
                if c["name"] == "run_tests":
                    a = json.loads(c["arguments"])
                    cmd = "pytest " + " ".join(x for x in [a.get("path", ""), ("-k " + a["keyword"]) if a.get("keyword") else ""] if x) or "pytest"
                    summary_lines.append(f"`{cmd.strip()}`: {r['content'].split(chr(10))[-1].strip('= ')}")
        failing_excerpt = ""
        for t in turns:
            for c, r in zip(t["expected_tool_calls"], t["tool_results"]):
                if c["name"] == "run_tests" and "FAILURES" in r["content"]:
                    body = r["content"].split("\n")
                    start = next(i for i, l in enumerate(body) if "FAILURES" in l)
                    failing_excerpt = "\n".join(body[start + 1: start + 40])
        report = final_report(ctx, task, plan["out_final"], sz, chg, files_read, diff_text, summary_lines, failing_excerpt)
        turns.append({"index": N + 1, "user": None, "expected_tool_calls": [], "tool_results": [], "recorded_assistant": {"content": report, "reasoning": random.Random(sid_hash(sid, "fr")).choice([
            "Everything the report needs is in the transcript: the cause, the change, the two test runs and the diff. I will write it in the required sections and keep it to what I actually did.",
            "The tests are green and the diff is confined to the intended files. Time to write the report in the five sections, naming the function and the lines."]), "tool_calls": []},
                      "max_tokens": FINAL_TURN_MAX_TOKENS, "response_format": None})
        # the first user turn: task + repository context, fitted to user_budget
        n_tests = (files[test_path].text or "").count("\ndef test_")
        user = user_turn(ctx, task, files, tree, sz, user_budget, fix_like, test_path, bug_path, n_tests)
        turns[0]["user"] = user
        return turns, chg, new_files

    files_before_session = None
    # budgets: measure the natural pieces first with a provisional render, then allocate the growth to the reads
    read_budgets = {p: 3000 for p in read_paths}
    for f in files.values():
        f.fix(min(len(f.units), 24))
    sess = {"id": sid, "source": "gen", "tenant_profile": "coder", "task": f"{plan['task_key']} on {ctx.pkg} ({plan['domain_key']})", "origin": f"gen_corpus {VERSION} seed {seed} index {index}",
            "system": HARNESS_SYSTEM, "tools": TOOLS, "turns": [], "max_tokens": FINAL_TURN_MAX_TOKENS, "tool_results_source": "generated"}
    planned = None
    achieved = None
    user_budget = 2000
    for attempt in range(4):
        turns, chg, new_files = render_all(read_budgets, user_budget)
        sess["turns"] = turns
        prompts, outs = prompt_tokens(sz, sess)
        # per-turn plan from the measured natural pieces: reads get the growth, front-loaded by turn
        if attempt == 0:
            # tokens of everything except the read results, per turn
            read_tok = {}
            for t in turns:
                for c, r in zip(t["expected_tool_calls"], t["tool_results"]):
                    if c["name"] == "read_file":
                        read_tok[json.loads(c["arguments"])["path"]] = sz.count(r["content"])
            other_growth = (prompts[-1] - prompts[0]) - sum(read_tok.values())
            growth_for_reads = max(len(read_paths) * 350, plan["final"] - plan["first"] - other_growth)
            # weights: by turn position (0.68 decay), 55/45 inside a turn
            w = {}
            for k, (kind, calls) in enumerate(turns_spec):
                rp = [a for tool, a in calls if tool == "read_file"]
                for j, p in enumerate(rp):
                    w[p] = (0.68 ** k) * (0.55 if (len(rp) == 2 and j == 0) else (0.45 if len(rp) == 2 else 0.8))
            tot = sum(w.values())
            read_budgets = {p: int(growth_for_reads * w[p] / tot) for p in read_paths}
            user_budget = plan["first"] - (prompts[0] - sz.count(turns[0]["user"]))
            continue
        dev_first = prompts[0] - plan["first"]
        dev_last = prompts[-1] - plan["final"]
        planned = prompts_plan(plan, turns, read_budgets, prompts)
        achieved = prompts
        if abs(dev_first) <= max(25, 0.004 * plan["first"]) and abs(dev_last) <= max(40, 0.004 * plan["final"]):
            break
        user_budget -= dev_first
        spare = [p for p in read_paths if files[p].packed_units is not None and files[p].packed_units < len(files[p].units)]
        last_read = (spare or read_paths)[-1]
        read_budgets[last_read] = max(300, read_budgets[last_read] - (dev_last - dev_first))
    for t, p in zip(sess["turns"], planned):
        t["planned_prompt_tokens"] = int(p)
    for t in sess["turns"]:
        t["planned_final_prompt_tokens"] = plan["final"]
    calib = {"id": sid, "task": plan["task_key"], "domain": plan["domain_key"], "n_tool_turns": N, "planned_first": plan["first"], "planned_final": plan["final"], "achieved_first": achieved[0],
             "achieved_final": achieved[-1], "planned_out_final": plan["out_final"], "achieved_out_final": outs[-1], "attempts": attempt + 1, "changes": len(chg), "reads": len(read_paths)}
    return sess, calib


def prompts_plan(plan, turns, read_budgets, measured):
    """The planned prompt per turn: first and final are the drawn targets; the turns between follow the measured growth shape."""
    first, final = plan["first"], plan["final"]
    span = (measured[-1] - measured[0]) or 1
    return [first + (final - first) * (m - measured[0]) / span for m in measured]


def user_turn_fixed_part(ctx, task, tree, fix_like, test_path, n_tests=20):
    lines = [task["ask"], "", f"Repository: `{ctx.pkg}` ({ctx.d['blurb']}). Python {ctx.py}, standard library only; the tests run with pytest from the repository root.", "",
             "Tree (`find . -type f -not -path './.git/*' | sort`):", "```"] + tree + ["```", ""]
    if fix_like and task.get("failure"):
        lines += [f"Failing test (from `pytest {test_path} -k {task['test_name']} -q`):", "```", f"FAILED {test_path}::{task['test_name']} - {task['failure']['exc']}: {task['failure']['e_lines'][0][:100]}".rstrip(), f"1 failed, {max(0, n_tests - 1)} deselected in 0.{ctx.num(9) % 60 + 11}s", "```", ""]
    lines += ["Key files (excerpts):", ""]
    return "\n".join(lines)


def api_doc(ctx, sf, module):
    """docs/<module>.md: the API reference page generated from the module's definitions and docstrings (the whole pool)."""
    out = [f"# `{ctx.pkg}.{module}`", ""]
    doc = re.search(r'"""(.+?)"""', sf.header, re.S)
    if doc:
        out += [doc.group(1).strip().split("\n")[0], ""]
    out += ["## Reference", ""]
    for u in sf.units:
        first = u.split("\n")[0]
        if first.startswith("@"):
            first = u.split("\n")[1]
        m = re.match(r"(def|class) ([A-Za-z_][A-Za-z0-9_]*)(\(.*?\))?( -> .+?)?:$", first)
        if not m:
            continue
        kind, name, sig, ret = m.groups()
        d = re.search(r'"""(.+?)"""', u, re.S)
        text = d.group(1).strip().split("\n")[0] if d else ""
        out += [f"### `{name}{sig or ''}{ret or ''}`" if kind == "def" else f"### `class {name}`", "", text, ""]
        if kind == "class":
            methods = re.findall(r"^    def ([a-z_][a-z0-9_]*)\((.*?)\)", u, re.M)
            if methods:
                out += ["Methods: " + ", ".join(f"`{mn}({ma})`" for mn, ma in methods if not mn.startswith("__")), ""]
    return "\n".join(out).rstrip() + "\n"


def format_doc(ctx):
    ent = ctx.main
    f = ctx.fields(ent)
    lines = [f"# File format (version 2)", "", f"{ctx.pkg} stores {ent['plural']} as JSON lines: one UTF-8 encoded JSON object per line, no surrounding array, a trailing newline after the last record. Blank lines and lines starting with `#` are ignored by the loader so files can carry a header comment.", "",
             "## Keys", "", "| key | type | meaning |", "|---|---|---|"]
    for n, t, sample in f:
        meaning = {ent["id"][0]: f"the identifier, always prefixed `{ent['id'][1]}`", ent["name"]: "free text, at most the configured number of characters", ent["status"]: "one of the documented states", ent["ts"]: "ISO 8601 with an offset; naive values in old files are UTC", ent["amount"]: "a non-negative integer"}.get(n, f"see `{ent['camel']}`; example `{sample}`")
        lines.append(f"| `{n}` | {t} | {meaning} |")
    lines += ["| `schema` | int | always 2; version 1 files lack the key and are read with defaults |", "", "## Writing", "",
              "Writers produce the keys in sorted order and replace the file atomically (write to `<name>.tmp`, then rename). Readers must accept keys in any order and ignore unknown keys, which is how the format stays forward compatible across one release.", "",
              "## States", "", " -> ".join(f"`{s}`" for s in ctx.statuses()) + f"; the last two are terminal. The transitions are the functions named after the verbs in `{ctx.module_of(ent)}.py`; each one checks the source state and stamps `{ent['ts']}`.", ""]
    return "\n".join(lines)


def user_turn(ctx, task, files, tree, sz, budget, fix_like, test_path, bug_path, n_tests=20):
    """The first user turn: the task text, the repository context, then file excerpts cut to the token budget. The module and the
    test excerpts are prefixes of the files the agent reads later; the docs pages (generated references) take the rest."""
    fixed = user_turn_fixed_part(ctx, task, tree, fix_like, test_path, n_tests)
    mod_name = task["module"]
    other_mods = [m for m in ctx.modules if m not in (mod_name, "cli")]
    sources = [("README.md", 0.15, "", files["README.md"].text), (bug_path, 0.35, "python", files[bug_path].text), (test_path, 0.20, "python", files[test_path].text),
               (f"docs/{mod_name}.md", 1.0, "", api_doc(ctx, files[bug_path], mod_name) if mod_name != "cli" else format_doc(ctx)), ("docs/format.md", 1.0, "", format_doc(ctx))]
    for m in other_mods[:3]:
        sources.append((f"docs/{m}.md", 1.0, "", api_doc(ctx, files[ctx.src(m)], m)))
    total = budget - sz.count(fixed)
    remaining = total
    parts = []
    for path, share, lang, text in sources:
        if remaining < 80:
            break
        lines = text.rstrip("\n").split("\n")
        sizes = sz.count_many([l + "\n" for l in lines])
        allowed = min(remaining, int(total * share)) - 22
        cum, n = 0, 0
        while n < len(lines) and cum + sizes[n] <= allowed:
            cum += sizes[n]; n += 1
        if n < 4:
            continue
        head = f"### {path}" + (f" (lines 1-{n} of {len(lines)})" if n < len(lines) else "")
        parts.append(head + "\n```" + lang + "\n" + "\n".join(lines[:n]) + "\n```")
        remaining -= cum + 22
    return fixed + "\n\n".join(parts)


# ---- CLI -------------------------------------------------------------------------------------------------------------------------
_SZ = None


def _worker_init(tok_dir):
    global _SZ
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _SZ = Sizer(tok_dir)


def _worker(args):
    index, seed, first_range, final_range = args
    sess, calib = build_session(index, seed, _SZ, first_range, final_range)
    return index, sess, calib


def generate(n, seed, tok_dir, jobs=8, first_range=FIRST_RANGE, final_range=FINAL_RANGE, indices=None):
    """All sessions 0..n-1 (or the given indices), in index order; (sessions, calibrations). Every index is checked in; a worker
    result that did not arrive is rebuilt in the parent process (the build is deterministic per index)."""
    idx = list(indices) if indices is not None else list(range(n))
    work = [(i, seed, first_range, final_range) for i in idx]
    got = {}
    if jobs > 1 and len(work) > 2:
        import multiprocessing as mp
        t0 = time.time()
        with mp.get_context("fork").Pool(jobs, initializer=_worker_init, initargs=(tok_dir,), maxtasksperchild=40) as pool:
            for r in pool.imap_unordered(_worker, work, chunksize=1):
                if r is None:
                    continue
                got[r[0]] = (r[1], r[2])
                if len(got) % 100 == 0:
                    print(f"gen_corpus: {len(got)}/{len(work)} sessions in {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    missing = [w for w in work if w[0] not in got]
    if missing:
        if jobs > 1 and len(work) > 2:
            print(f"gen_corpus: {len(missing)} session(s) not returned by the workers; rebuilding in-process: {[w[0] for w in missing][:20]}", file=sys.stderr, flush=True)
        _worker_init(tok_dir)
        for w in missing:
            i, sess, calib = _worker(w)
            got[i] = (sess, calib)
    sessions = [got[i][0] for i in idx]
    calibs = [got[i][1] for i in idx]
    return sessions, calibs


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, math.ceil(p * len(v)) - 1))] if v else None


def calibration_summary(calibs):
    df = [(c["achieved_first"] - c["planned_first"]) / c["planned_first"] for c in calibs]
    dl = [(c["achieved_final"] - c["planned_final"]) / c["planned_final"] for c in calibs]
    out = {"sessions": len(calibs),
           "first": {"planned_p50": pct([c["planned_first"] for c in calibs], .5), "achieved_p50": pct([c["achieved_first"] for c in calibs], .5), "dev_abs_p50": pct([abs(x) for x in df], .5), "dev_abs_p90": pct([abs(x) for x in df], .9), "dev_abs_max": max(abs(x) for x in df), "within_5pct": sum(abs(x) <= 0.05 for x in df)},
           "final": {"planned_p50": pct([c["planned_final"] for c in calibs], .5), "achieved_p50": pct([c["achieved_final"] for c in calibs], .5), "dev_abs_p50": pct([abs(x) for x in dl], .5), "dev_abs_p90": pct([abs(x) for x in dl], .9), "dev_abs_max": max(abs(x) for x in dl), "within_5pct": sum(abs(x) <= 0.05 for x in dl)},
           "out_final": {"planned_p50": pct([c["planned_out_final"] for c in calibs], .5), "achieved_p50": pct([c["achieved_out_final"] for c in calibs], .5), "achieved_min": min(c["achieved_out_final"] for c in calibs), "achieved_max": max(c["achieved_out_final"] for c in calibs)},
           "attempts": dict(collections.Counter(c["attempts"] for c in calibs)), "turns": dict(collections.Counter(c["n_tool_turns"] + 1 for c in calibs)),
           "tasks": dict(collections.Counter(c["task"] for c in calibs)), "domains": dict(collections.Counter(c["domain"] for c in calibs))}
    return out


def selftest(tok_dir):
    import ast
    sz = Sizer(tok_dir)
    seen_text = set()
    for i in range(6):
        s, c = build_session(i, 1, sz)
        assert s["turns"][0]["user"] and s["turns"][-1]["expected_tool_calls"] == [] and s["turns"][-1]["recorded_assistant"]["content"]
        for t in s["turns"][:-1]:
            assert 1 <= len(t["expected_tool_calls"]) <= 2, (i, t["index"])
            assert len(t["tool_results"]) == len(t["expected_tool_calls"])
            for e in t["expected_tool_calls"]:
                a = json.loads(e["arguments"])
                assert e["name"] in {x["function"]["name"] for x in TOOLS}
                if e["name"] == "read_file":
                    body = next(r["content"] for r in t["tool_results"] if r["tool_call_id"] == e["id"])
                    if a["path"].endswith(".py"):
                        src = "\n".join(l.split("\t", 1)[1] if "\t" in l else l for l in body.rstrip("\n").split("\n"))
                        ast.parse(src)  # every Python file the agent reads is valid Python
        key = s["turns"][1]["tool_results"][0]["content"][:400]
        assert key not in seen_text
        seen_text.add(key)
        assert abs(c["achieved_first"] - c["planned_first"]) <= 0.05 * c["planned_first"], c
        assert abs(c["achieved_final"] - c["planned_final"]) <= 0.05 * c["planned_final"], c
        print(f"selftest session {i}: {c}", file=sys.stderr)
    s1, _ = build_session(3, 1, sz)
    s2, _ = build_session(3, 1, sz)
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True), "not deterministic"
    print(f"SELFTEST OK themes={len(THEMES)} shared_prefix_tokens={sz.count(HARNESS_SYSTEM) + sz.count(json.dumps(TOOLS))}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="directory holding tokenizer.json")
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--index", type=int, default=None, help="build only this index and print it")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.tokenizer)
        return
    t0 = time.time()
    if a.index is not None:
        sessions, calibs = generate(1, a.seed, a.tokenizer, 1, indices=[a.index])
        print(json.dumps(sessions[0], ensure_ascii=False) if not a.out else "", end="")
    else:
        sessions, calibs = generate(a.n, a.seed, a.tokenizer, a.jobs)
    if a.out:
        with open(a.out, "w") as f:
            for s in sessions:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(json.dumps(calibration_summary(calibs), indent=1), file=sys.stderr)
    print(f"generated {len(sessions)} sessions in {time.time() - t0:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
