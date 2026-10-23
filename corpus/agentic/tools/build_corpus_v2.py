#!/usr/bin/env python3
"""build_corpus_v2.py: the agentic session corpus v2 of plans/benchmark-2.md section 1.1 (lane bench-kit-1), for the driver's
`sessions` tenant kind. An adapted copy of corpus v1's build_corpus.py: the same session record, the same finalize rules
(salt point, json_schema share, thinking share, scrub, invariants), the same token-count method and manifest tables.

USAGE (run with python -I: the BFCL files and the tokenizer files are downloaded data; scripts take every path as an argument)
  build_corpus_v2.py --out DIR --bfcl BFCL_DIR --tokenizers deepseek-v4.1-flash=TOK_DIR [--gen 900] [--bfcl-per-category 40]
                     [--seed 20261022] [--json-schema-share 0.2] [--thinking-share 0.5] [--image-share 0.15] [--jobs 16] [--selftest]
  Sources and tenants:
    gen  tenant "coder": tools/gen_corpus.py (this directory): one coding task per session on a synthetic repository, one harness
                         prompt + 7 tool schemas for the whole tenant, 3/4/5/6/8 tool-call turns + 1 final (weights .2/.25/.25/.2/.1),
                         first prompt 6-15k and last prompt 15-70k tokens (log-uniform, calibrated on the real tokenizer).
    bfcl tenant "bfcl":  BFCL v4 multi-turn entries (base, miss_func, miss_param, long_context; 40 per category): the questions verbatim
                         as user turns, the involved classes' function docs as tools, the ground-truth call sequence as the expected
                         calls, tool results synthesized from the docs' response schemas (the accuracy slice).
    cli  tenant "cli":   ABSENT in v2 (the 181 recorded coding-CLI sessions of v1 were lost at the host move; re-recording pending).
  Every session gets a salt point ("{{SALT}}" in the first user turn), 20 % of the non-bfcl sessions end under response_format
  json_schema (the f3 schema), 50 % carry thinking=true, and 15 % of the gen sessions carry an image slot reference on the first
  user turn ("image": {"slot", "class", ...}; the image bytes are attached by the driver from the baked image set when present).
  Output: DIR/corpus.jsonl, DIR/manifest.json (sources and pins, per-session per-turn token counts, calibration, shares),
  DIR/MANIFEST.md (the summary tables). Run the campaign's leak scanner over DIR and this directory afterwards.

Session record (one JSON object per line of corpus.jsonl; unchanged from v1):
  {"id", "source": "gen|bfcl", "tenant_profile": "coder|bfcl", "task", "origin", "system", "tools": [chat-completions tools],
   "turns": [{"index", "user" (text or null), "expected_tool_calls": [{"id","name","arguments": JSON string}],
              "tool_results": [{"tool_call_id","name","content"}], "recorded_assistant": {"content","reasoning","tool_calls"} | null,
              "max_tokens", "response_format" | null, "planned_prompt_tokens" (gen)}],
   "max_tokens", "thinking": bool, "json_schema": bool, "tool_results_source": "generated|synthetic", "licence", "salt_points",
   "image": {"slot","class","mpixels_range","file","note"} (15 % of gen only)}
"""
import argparse, ast, collections, hashlib, importlib.util, json, math, os, random, re, sys, time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = "v2"
FAMILIES = {  # family -> (repo, bellwether-pinned revision, licence name); only the families given with --tokenizers are counted
    "minimax-m3": ("MiniMaxAI/MiniMax-M3", "f0e1c1e04d40177e4673a22097036854f536e9c0", "MiniMax Community License"),
    "deepseek-v4.1-flash": ("deepseek-ai/DeepSeek-V4.1-Flash", "2cba9e42aa026125f3ed06c6d98c1db82f7ca027", "MIT"),
    "kimi-k3": ("moonshotai/Kimi-K3", "f831ab66814297da540d832a5235f8e904f29d06", "Modified MIT (Kimi)"),
    "qwen3.6-35b-a3b": ("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0", "Apache-2.0"),
    "glm-5.3-flash": ("zai-org/GLM-5.3-Flash", "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a", "MIT"),
    "gemma-4-26b-a4b-it": ("google/gemma-4-26B-A4B-it", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52", "Gemma Terms of Use"),
}
PRIMARY_FAMILY = "deepseek-v4.1-flash"
BFCL_COMMIT = "26ad99ed327fc47fc7c8b91b43378c5709125a06"
JSON_SCHEMA_F3 = {"name": "trip_plan", "schema": {"type": "object", "properties": {
    "destination": {"type": "string"}, "days": {"type": "integer"}, "budget_usd": {"type": "number"},
    "activities": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 5},
    "packing": {"type": "object", "properties": {"essentials": {"type": "array", "items": {"type": "string"}}, "optional": {"type": "array", "items": {"type": "string"}}}, "required": ["essentials"]}},
    "required": ["destination", "days", "budget_usd", "activities", "packing"], "additionalProperties": False}, "strict": True}
JSON_SCHEMA_REPORT = {"name": "agent_report", "schema": {"type": "object", "properties": {
    "summary": {"type": "string"}, "root_cause": {"type": "string"},
    "files_changed": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 12},
    "tests_run": {"type": "array", "items": {"type": "string"}}, "verification_passed": {"type": "boolean"}, "patch": {"type": "string"}},
    "required": ["summary", "root_cause", "files_changed", "tests_run", "verification_passed", "patch"], "additionalProperties": False}, "strict": True}
JSON_FINAL_USER = "Return your final answer as one JSON object that matches the response schema, nothing else."
BFCL_SYSTEM = ("You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, "
               "you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it "
               "out. If the given question lacks the parameters required by the function, also point it out. You should only return the "
               "function calls in your response.\n\nAt each turn, you should try your best to complete the tasks requested by the user within "
               "the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. "
               "Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.")
BFCL_CLASS_DOC = {"GorillaFileSystem": "gorilla_file_system", "MathAPI": "math_api", "MessageAPI": "message_api", "TwitterAPI": "posting_api",
                  "TicketAPI": "ticket_api", "TradingBot": "trading_bot", "TravelAPI": "travel_booking", "VehicleControlAPI": "vehicle_control",
                  "WebSearchAPI": "web_search", "MemoryKV": "memory_kv", "MemoryRecSum": "memory_rec_sum", "MemoryVector": "memory_vector"}
BFCL_TYPE = {"dict": "object", "float": "number", "tuple": "array", "integer": "integer", "string": "string", "boolean": "boolean", "array": "array", "any": None}
LICENCE = {
    "gen": "Synthetic: generated by the kit's gen_corpus.py (the project's licence); no third-party text.",
    "bfcl": "Berkeley Function Calling Leaderboard v4 multi-turn data, github.com/ShishirPatil/gorilla (Apache-2.0; data files at commit "
            "26ad99ed of 2025-11-07): questions and function documentation verbatim, the ground truth as the expected calls; tool results "
            "synthesized from the documented response schemas (the executable backing classes are not run).",
    "tokenizers": "Family tokenizer files at the bellwether-pinned revisions (see families), used to count tokens only; not redistributed.",
}
V1_NUMBERS = {"sessions": 641, "by_source": {"cli": 181, "gen": 300, "bfcl": 160}, "json_schema_share": 0.139, "thinking_share": 0.509,
              "gen_first_p50": 11400, "gen_first_range": [6700, 15900], "gen_last_p50": 36400, "gen_last_range": [15200, 70300],
              "bfcl_first_p50": 3800, "cli_first_p50": 4600, "cli_last_p50": 5800, "corpus_bytes": 55600000, "gen_tool_schemas": 7}
# internal markers scrubbed from every text (generic patterns only): home directories, IPv6 addresses
SCRUB = [(re.compile(r"/home/[A-Za-z0-9_.-]+"), "/workspace"), (re.compile(r"\b(?:[0-9a-f]{1,4}:){4,7}[0-9a-f]{1,4}\b", re.I), "[addr]")]
SCRUBBED = collections.Counter()


def scrub(text):
    if not isinstance(text, str):
        return text
    for rx, rep in SCRUB:
        text, n = rx.subn(rep, text)
        if n:
            SCRUBBED[rep] += n
    return text


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gen_module():
    spec = importlib.util.spec_from_file_location("gen_corpus", os.path.join(HERE, "gen_corpus.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_corpus"] = mod  # importable by name in the generator's forked worker processes
    spec.loader.exec_module(mod)
    return mod


# ---- tokenizers ----------------------------------------------------------------------------------------------------------
class Counter:
    """family -> a callable counting the tokens of a text; only the families whose directory is given are loaded, the others are absent."""

    def __init__(self, spec):
        self.enc, self.where, self.sha = {}, {}, {}
        dirs = {}
        for part in [x for x in (spec or "").split(",") if x.strip()]:
            k, v = part.split("=", 1)
            dirs[k.strip()] = v.strip()
        for fam in FAMILIES:
            d = dirs.get(fam)
            if not d:
                print(f"tokenizer {fam}: absent", file=sys.stderr)
                continue
            p = os.path.join(d, "tokenizer.json")
            if not os.path.exists(p):
                print(f"tokenizer {fam}: absent (no tokenizer.json under the given directory)", file=sys.stderr)
                continue
            from tokenizers import Tokenizer
            self.enc[fam] = Tokenizer.from_file(p)
            self.where[fam] = os.path.basename(d.rstrip("/")) + "/tokenizer.json"
            self.sha[fam] = sha256_file(p)
            print(f"tokenizer {fam}: loaded ({self.where[fam]}, sha256 {self.sha[fam][:12]})", file=sys.stderr)

    def count_many(self, fam, texts):
        return [len(e.ids) for e in self.enc[fam].encode_batch(texts, add_special_tokens=False)]


def session_texts(s):
    """Per turn k: the texts added to the request of turn k (prefix sums give the prompt of each turn) and the recorded output."""
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


# ---- source 1: the generator ------------------------------------------------------------------------------------------------
def convert_gen(n, seed, tok_dir, jobs):
    gen = load_gen_module()
    sessions, calibs = gen.generate(n, seed, tok_dir, jobs)
    for s in sessions:
        s["licence_note"] = None
    stats = {"sessions": len(sessions), "turns": sum(len(s["turns"]) for s in sessions), "generator": gen.VERSION, "themes": len(gen.THEMES),
             "tool_schemas": len(gen.TOOLS), "turn_choice": gen.TURN_CHOICE, "first_range": gen.FIRST_RANGE, "final_range": gen.FINAL_RANGE,
             "calibration": gen.calibration_summary(calibs) if calibs else None,
             "calibration_per_session": {c["id"]: {k: v for k, v in c.items() if k != "id"} for c in calibs}}
    return sessions, stats


# ---- source 2: BFCL v4 multi-turn ------------------------------------------------------------------------------------------
def bfcl_schema(p):
    """BFCL parameter doc -> JSON schema (dict -> object, float -> number, tuple -> array, any -> untyped)."""
    if not isinstance(p, dict):
        return p
    out = {}
    for k, v in p.items():
        if k == "type" and isinstance(v, str):
            t = BFCL_TYPE.get(v, v)
            if t:
                out["type"] = t
        elif k in ("properties",):
            out[k] = {kk: bfcl_schema(vv) for kk, vv in v.items()}
        elif k == "items":
            out[k] = bfcl_schema(v)
        elif k in ("required", "description", "enum", "default"):
            out[k] = v
        else:
            out[k] = bfcl_schema(v) if isinstance(v, dict) else v
    if out.get("type") == "array" and "items" not in out:
        out["items"] = {"type": "string"}
    return out


def parse_call(s):
    """'cd(folder=\\'document\\')' -> ('cd', {'folder': 'document'}) with literal values; falls back to the raw string."""
    try:
        node = ast.parse(s.strip(), mode="eval").body
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            kw = {}
            for k in node.keywords:
                try:
                    kw[k.arg] = ast.literal_eval(k.value)
                except Exception:
                    kw[k.arg] = ast.unparse(k.value)
            return name, kw
    except Exception:
        pass
    return s.split("(")[0], {"_raw": s}


def synth_result(doc):
    resp = (doc or {}).get("response") or {}
    props = resp.get("properties") or {}
    if not props:
        return {"status": "ok"}
    out = {}
    for k, v in props.items():
        t = (v or {}).get("type", "string")
        out[k] = {"integer": 1, "float": 1.0, "number": 1.0, "boolean": True, "array": [], "dict": {}, "object": {}}.get(t, f"{k} ok")
    return out


def bfcl_pins(d):
    """The source pin of the BFCL directory: the commit and every file's sha256 from SOURCE.txt, verified against the files read."""
    pins = {"commit": BFCL_COMMIT, "source_txt": None, "files": {}, "verified": None}
    src = os.path.join(d, "SOURCE.txt")
    listed = {}
    if os.path.exists(src):
        text = open(src).read()
        m = re.search(r"commit ([0-9a-f]{40})", text)
        pins["source_txt"] = {"commit": m.group(1) if m else None, "fetched": (re.search(r"fetched: (\S+)", text) or [None, None])[1]}
        for line in text.split("\n"):
            m = re.match(r"^([0-9a-f]{64})\s+\./(.+)$", line.strip())
            if m:
                listed[m.group(2)] = m.group(1)
    ok = True
    for rel in sorted(listed):
        p = os.path.join(d, rel)
        actual = sha256_file(p) if os.path.exists(p) else None
        pins["files"][rel] = {"sha256": actual, "listed": listed[rel], "match": actual == listed[rel]}
        ok = ok and actual == listed[rel]
    pins["verified"] = ok if listed else None
    return pins


def convert_bfcl(d, per_cat, rnd):
    docs = {}
    for cls, fn in BFCL_CLASS_DOC.items():
        p = f"{d}/func_doc/{fn}.json"
        if os.path.exists(p):
            docs[cls] = [json.loads(l) for l in open(p) if l.strip()]
    sessions, stats = [], collections.Counter()
    for cat in ("base", "miss_func", "miss_param", "long_context"):
        p = f"{d}/BFCL_v4_multi_turn_{cat}.json"
        if not os.path.exists(p):
            stats[f"missing_{cat}"] += 1
            continue
        entries = [json.loads(l) for l in open(p) if l.strip()]
        gt = {}
        pa = f"{d}/possible_answer/BFCL_v4_multi_turn_{cat}.json"
        if os.path.exists(pa):
            for l in open(pa):
                if l.strip():
                    x = json.loads(l)
                    gt[x["id"]] = x.get("ground_truth") or []
        idx = list(range(len(entries)))
        rnd.shuffle(idx)
        taken = 0
        for i in idx:
            e = entries[i]
            if taken >= per_cat:
                break
            classes = e.get("involved_classes") or []
            excluded = set(e.get("excluded_function") or [])
            tools, byname = [], {}
            for cls in classes:
                for fdoc in docs.get(cls, []):
                    if fdoc["name"] in excluded or fdoc["name"] in byname:
                        continue
                    byname[fdoc["name"]] = fdoc
                    tools.append({"type": "function", "function": {"name": fdoc["name"], "description": fdoc.get("description", ""), "parameters": bfcl_schema(fdoc.get("parameters") or {"type": "dict", "properties": {}})}})
            if not tools:
                stats["no_tools"] += 1
                continue
            truth = gt.get(e["id"], [])
            turns = []
            k = 0
            for qi, q in enumerate(e.get("question") or []):
                user = "\n".join(m.get("content", "") for m in q if m.get("role") == "user").strip()
                if not user:
                    continue  # miss_func: the turn that only adds the held-out function
                calls = []
                for j, cs in enumerate(truth[qi] if qi < len(truth) else []):
                    name, kw = parse_call(cs)
                    calls.append({"id": f"call_bfcl_{k + 1}_{j}", "name": name, "arguments": json.dumps(kw)})
                results = [{"tool_call_id": c["id"], "name": c["name"], "content": json.dumps(synth_result(byname.get(c["name"])))} for c in calls]
                turns.append({"index": k + 1, "user": user, "expected_tool_calls": calls, "tool_results": results, "recorded_assistant": None,
                              "max_tokens": 1024, "response_format": None})
                k += 1
            if not turns:
                stats["no_turns"] += 1
                continue
            sessions.append({"id": f"bfcl-{len(sessions):03d}", "source": "bfcl", "tenant_profile": "bfcl", "task": e["id"], "origin": e["id"],
                             "system": BFCL_SYSTEM, "tools": tools, "turns": turns, "max_tokens": 1024, "tool_results_source": "synthetic",
                             "bfcl_category": cat, "bfcl_classes": classes})
            taken += 1
            stats["sessions"] += 1
            stats["turns"] += len(turns)
            stats[f"cat_{cat}"] += 1
    return sessions, dict(stats)


# ---- assembly --------------------------------------------------------------------------------------------------------------
def finalize(sessions, rnd, json_share, thinking_share, schema):
    for s in sessions:
        s.pop("licence_note", None)
        # the salt point: the first user turn starts with a per-play nonce line
        t0 = s["turns"][0]
        t0["user"] = f"Session nonce: {{{{SALT}}}}/{s['id']}\n" + (t0.get("user") or "")
        s["thinking"] = rnd.random() < thinking_share
        s["json_schema"] = rnd.random() < json_share and s["source"] != "bfcl"
        if s["json_schema"]:
            last = s["turns"][-1]
            last["response_format"] = {"type": "json_schema", "json_schema": schema}
            last["user"] = ((last.get("user") or "").rstrip() + "\n" + JSON_FINAL_USER).strip()
        s["licence"] = LICENCE[s["source"]]
        s["salt_points"] = 1
        # scrub internal markers in every text
        s["system"] = scrub(s["system"])
        for t in s["turns"]:
            if t.get("user"):
                t["user"] = scrub(t["user"])
            for r in t.get("tool_results") or []:
                r["content"] = scrub(r.get("content") or "")
            ra = t.get("recorded_assistant")
            if ra:
                ra["content"] = scrub(ra.get("content") or "")
                ra["reasoning"] = scrub(ra.get("reasoning") or "")
                for c in ra.get("tool_calls") or []:
                    c["arguments"] = scrub(c.get("arguments") or "{}")
            for c in t.get("expected_tool_calls") or []:
                c["arguments"] = scrub(c.get("arguments") or "{}")
    return sessions


def assign_image_slots(sessions, seed, share):
    """15 % (share) of the gen sessions carry an image slot on the first user turn; classes alternate screenshot/document."""
    gen = [s for s in sessions if s["source"] == "gen"]
    k = int(round(share * len(gen)))
    rnd = random.Random(f"{seed}:images")
    chosen = sorted(rnd.sample(range(len(gen)), k)) if k else []
    slots = []
    for n, i in enumerate(chosen):
        s = gen[i]
        cls = "screenshot" if n % 2 == 0 else "document"
        s["image"] = {"slot": f"img-{n}", "class": cls, "mpixels_range": [0.2, 2.0], "file": None,
                      "note": "data URL attached by the driver from the baked image set when present; text only otherwise"}
        slots.append({"slot": f"img-{n}", "session": s["id"], "class": cls, "turn": 1})
    return slots


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, math.ceil(p * len(v)) - 1))] if v else None


def dist(vals):
    return {"n": len(vals), "min": min(vals), "p50": pct(vals, .5), "p90": pct(vals, .9), "max": max(vals)} if vals else None


def public_argv(argv):
    """The build command line for the manifest: every path-like argument reduced to its last component."""
    out = []
    for a in argv:
        if "=" in a and "/" in a:
            k, v = a.split("=", 1)
            out.append(k + "=" + os.path.basename(v.rstrip("/")))
        elif "/" in a:
            out.append(os.path.basename(a.rstrip("/")) or a)
        else:
            out.append(a)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen", type=int, default=900, help="generator sessions")
    ap.add_argument("--bfcl", required=True, help="BFCL v4 multi-turn directory (BFCL_v4_multi_turn_<cat>.json, possible_answer/, func_doc/, SOURCE.txt)")
    ap.add_argument("--bfcl-per-category", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--tokenizers", required=True, help="fam=/dir,... (deepseek-v4.1-flash=<dir with tokenizer.json>); other families print 'absent'")
    ap.add_argument("--jobs", type=int, default=16, help="generator worker processes")
    ap.add_argument("--json-schema-share", type=float, default=0.2)
    ap.add_argument("--thinking-share", type=float, default=0.5)
    ap.add_argument("--image-share", type=float, default=0.15, help="share of gen sessions with an image slot on the first user turn")
    ap.add_argument("--json-schema", choices=["f3", "report"], default="f3")
    ap.add_argument("--selftest", action="store_true", help="the generator's self-test, then a tiny corpus (2 gen + 1 per BFCL category) + invariant checks")
    a = ap.parse_args()
    tok_dirs = dict(x.split("=", 1) for x in a.tokenizers.split(",") if "=" in x)
    tok_dir = tok_dirs.get(PRIMARY_FAMILY)
    if not tok_dir or not os.path.exists(os.path.join(tok_dir, "tokenizer.json")):
        sys.exit(f"--tokenizers must give {PRIMARY_FAMILY}=<dir with tokenizer.json> (the generator calibrates on it)")
    if a.selftest:
        load_gen_module().selftest(tok_dir)
        a.gen, a.bfcl_per_category = 2, 1
    rnd = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    src_stats = {}
    sessions = []
    if a.gen > 0:
        tg = time.time()
        s, st = convert_gen(a.gen, a.seed, tok_dir, a.jobs)
        st["seconds"] = round(time.time() - tg, 1)
        sessions += s; src_stats["gen"] = st
        print(f"gen: {len(s)} sessions in {st['seconds']} s", file=sys.stderr)
    if a.bfcl:
        s, st = convert_bfcl(a.bfcl, a.bfcl_per_category, rnd)
        sessions += s; src_stats["bfcl"] = st
    src_stats["cli"] = {"status": "absent", "reason": "recordings lost at the host move; re-recording brief pending", "v1_numbers": {"sessions": 181, "first_p50": V1_NUMBERS["cli_first_p50"], "last_p50": V1_NUMBERS["cli_last_p50"], "share_by_count": 0.282}}
    schema = JSON_SCHEMA_F3 if a.json_schema == "f3" else JSON_SCHEMA_REPORT
    sessions = finalize(sessions, rnd, a.json_schema_share, a.thinking_share, schema)
    image_slots = assign_image_slots(sessions, a.seed, a.image_share)
    ids = [s["id"] for s in sessions]
    assert len(ids) == len(set(ids)), "duplicate session ids"
    # token counts per family: count every text piece once, prefix-sum per turn
    counter = Counter(a.tokenizers)
    per_session = {}
    fams = list(counter.enc)
    pieces, index = [], []
    for s in sessions:
        added, outputs = session_texts(s)
        for k, chunk in enumerate(added):
            for text in chunk:
                index.append((s["id"], "in", k)); pieces.append(text)
        for k, text in enumerate(outputs):
            index.append((s["id"], "out", k)); pieces.append(text)
    counts = {}
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    for fam in fams:
        tc0 = time.time()
        counts[fam] = counter.count_many(fam, pieces)
        print(f"tokens counted: {fam} {len(pieces)} pieces, {sum(counts[fam])} tokens in {time.time() - tc0:.1f} s", file=sys.stderr)
    chars = [len(p) for p in pieces]
    for fam in fams + ["chars/4"]:
        vals = counts[fam] if fam in counts else [c / 4.0 for c in chars]
        acc = collections.defaultdict(lambda: collections.defaultdict(float))
        for (sid, kind, k), v in zip(index, vals):
            acc[(sid, kind)][k] += v
        for s in sessions:
            nturn = len(s["turns"])
            ins = acc[(s["id"], "in")]
            prompts, run = [], 0.0
            for k in range(nturn):
                run += ins.get(k, 0.0)
                prompts.append(int(round(run)))
            outs = [int(round(acc[(s["id"], "out")].get(k, 0.0))) for k in range(nturn)]
            per_session.setdefault(s["id"], {})[fam] = {"prompt": prompts, "out_recorded": outs, "first": prompts[0], "last": prompts[-1], "sum": sum(prompts)}
    with open(f"{a.out}/corpus.jsonl", "w") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    # calibration of the generator against the plan, on the final (finalized) texts
    calib = None
    if PRIMARY_FAMILY in fams and any(s["source"] == "gen" for s in sessions):
        df, dl = [], []
        for s in sessions:
            if s["source"] != "gen":
                continue
            pf, pl = s["turns"][0]["planned_prompt_tokens"], s["turns"][-1]["planned_prompt_tokens"]
            got = per_session[s["id"]][PRIMARY_FAMILY]
            df.append((got["first"] - pf) / pf); dl.append((got["last"] - pl) / pl)
        calib = {"family": PRIMARY_FAMILY, "sessions": len(df), "note": "after finalize (nonce line, json_schema response_format on 20 % of the last turns)",
                 "first": {"dev_p50": pct([abs(x) for x in df], .5), "dev_p90": pct([abs(x) for x in df], .9), "dev_max": max(abs(x) for x in df), "within_5pct": sum(abs(x) <= 0.05 for x in df)},
                 "last": {"dev_p50": pct([abs(x) for x in dl], .5), "dev_p90": pct([abs(x) for x in dl], .9), "dev_max": max(abs(x) for x in dl), "within_5pct": sum(abs(x) <= 0.05 for x in dl)}}
    n_non_bfcl = sum(1 for s in sessions if s["source"] != "bfcl")
    man = {"version": VERSION, "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": a.seed, "sessions": len(sessions),
           "build_command": public_argv(sys.argv), "by_source": collections.Counter(s["source"] for s in sessions), "by_tenant": collections.Counter(s["tenant_profile"] for s in sessions),
           "json_schema_sessions": sum(1 for s in sessions if s["json_schema"]), "thinking_sessions": sum(1 for s in sessions if s["thinking"]),
           "shares": {"json_schema": round(sum(1 for s in sessions if s["json_schema"]) / len(sessions), 4), "json_schema_of_non_bfcl": round(sum(1 for s in sessions if s["json_schema"]) / max(1, n_non_bfcl), 4),
                      "thinking": round(sum(1 for s in sessions if s["thinking"]) / len(sessions), 4), "image_of_gen": round(len(image_slots) / max(1, sum(1 for s in sessions if s["source"] == "gen")), 4),
                      "v1": {"json_schema": V1_NUMBERS["json_schema_share"], "thinking": V1_NUMBERS["thinking_share"]}},
           "json_schema": a.json_schema, "turns": dist([len(s["turns"]) for s in sessions]), "turns_by_source": {src: dict(collections.Counter(len(s["turns"]) for s in sessions if s["source"] == src)) for src in ("gen", "bfcl")},
           "tool_schemas_per_session": dist([len(s["tools"]) for s in sessions]), "tool_schemas_by_source": {src: dist([len(s["tools"]) for s in sessions if s["source"] == src]) for src in ("gen", "bfcl")},
           "image_slots": image_slots, "source_stats": src_stats, "scrubbed": dict(SCRUBBED), "licences": LICENCE,
           "sources": {"bfcl": bfcl_pins(a.bfcl) if a.bfcl else None,
                       "tokenizer": {fam: {"repo": FAMILIES[fam][0], "revision": FAMILIES[fam][1], "licence": FAMILIES[fam][2], "file": counter.where.get(fam), "sha256": counter.sha.get(fam)} for fam in fams},
                       "generator": {"file": "gen_corpus.py", "sha256": sha256_file(os.path.join(HERE, "gen_corpus.py")), "version": src_stats.get("gen", {}).get("generator")},
                       "builder": {"file": "build_corpus_v2.py", "sha256": sha256_file(os.path.abspath(__file__))}},
           "tokenizers": {fam: {"repo": FAMILIES[fam][0], "revision": FAMILIES[fam][1], "licence": FAMILIES[fam][2], "file": counter.where.get(fam), "status": "counted" if fam in fams else "absent"} for fam in FAMILIES},
           "token_count_method": "template-free: system + tools JSON + every message text (content, reasoning, tool-call JSON, tool results) the request of the turn carries; the engines' usage.prompt_tokens at play time is the rendered count",
           "families_counted": fams, "primary_family": PRIMARY_FAMILY, "gen_calibration_after_finalize": calib, "v1_numbers": V1_NUMBERS, "per_family": {}, "per_session": per_session,
           "corpus_bytes": os.path.getsize(f"{a.out}/corpus.jsonl")}
    md = [f"# Agentic corpus {VERSION} ({man['built']}, seed {a.seed}): {len(sessions)} sessions", "",
          f"by source: {dict(man['by_source'])} (cli: absent, see below); by tenant: {dict(man['by_tenant'])}; json_schema sessions {man['json_schema_sessions']} ({man['shares']['json_schema']:.1%} of all, {man['shares']['json_schema_of_non_bfcl']:.1%} of non-bfcl; v1 13.9 %); thinking sessions {man['thinking_sessions']} ({man['shares']['thinking']:.1%}; v1 50.9 %); image slots {len(image_slots)} ({man['shares']['image_of_gen']:.1%} of gen); corpus {man['corpus_bytes'] / 1e6:.1f} MB",
          f"turns per session: {man['turns']}; by source: {man['turns_by_source']}; tool schemas per session: {man['tool_schemas_by_source']}; scrubbed markers: {dict(SCRUBBED)}", "",
          "| family | source | first prompt p50 (min-max) | last prompt p50 (min-max) | sum over turns p50 | recorded output per turn p50 |", "|---|---|---|---|---|---|"]
    for fam in fams + ["chars/4"]:
        man["per_family"][fam] = {}
        for src in ("gen", "bfcl", "all"):
            ss = [s for s in sessions if src == "all" or s["source"] == src]
            if not ss:
                continue
            first = [per_session[s["id"]][fam]["first"] for s in ss]; last = [per_session[s["id"]][fam]["last"] for s in ss]
            tot = [per_session[s["id"]][fam]["sum"] for s in ss]; outs = [o for s in ss for o in per_session[s["id"]][fam]["out_recorded"] if o]
            man["per_family"][fam][src] = {"first": dist(first), "last": dist(last), "sum": dist(tot), "out_recorded": dist(outs)}
            md.append(f"| {fam} | {src} | {pct(first, .5)} ({min(first)}-{max(first)}) | {pct(last, .5)} ({min(last)}-{max(last)}) | {pct(tot, .5)} | {pct(outs, .5) if outs else '-'} |")
    md += ["", "v1 for comparison (641 sessions, V4.1 Flash counts): gen first prompt p50 11.4k (6.7-15.9k), last 36.4k (15.2-70.3k); bfcl 3.8k; cli 4.6k -> 5.8k (cli absent in v2).", "",
           "## The cli tenant", "", f"- status: absent; reason: {src_stats['cli']['reason']}; v1 carried 181 recorded coding-CLI sessions (28.2 % by count, first prompt p50 4.6k, last 5.8k).", "",
           "## Sources, pins and licences", ""] + [f"- {k}: {v}" for k, v in LICENCE.items()] + \
          [f"- bfcl pin: gorilla commit {BFCL_COMMIT}; files verified against SOURCE.txt: {man['sources']['bfcl']['verified'] if man['sources']['bfcl'] else None}"] + \
          [f"- tokenizer {fam}: {FAMILIES[fam][0]} @ {FAMILIES[fam][1][:12]} ({FAMILIES[fam][2]}) file {man['sources']['tokenizer'][fam]['file']} sha256 {man['sources']['tokenizer'][fam]['sha256']}" for fam in fams] + \
          [f"- tokenizers absent: {[f for f in FAMILIES if f not in fams]}", f"- generator: gen_corpus.py sha256 {man['sources']['generator']['sha256']}", "",
           "## Generator calibration (gen, V4.1 Flash, after finalize)", "", "```", json.dumps(calib, indent=1), "```", "",
           "## Image slots", "", f"{len(image_slots)} gen sessions carry an image slot on the first user turn (classes alternate screenshot/document, 0.2-2.0 Mpixel; the file is attached by the driver when the baked image set is present): " + ", ".join(x["slot"] + "=" + x["session"] for x in image_slots[:12]) + (" ..." if len(image_slots) > 12 else ""), "",
           "## Source statistics", "", "```", json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "calibration_per_session"} if isinstance(v, dict) else v) for k, v in src_stats.items()}, indent=1), "```"]
    json.dump(man, open(f"{a.out}/manifest.json", "w"), indent=1, default=lambda o: dict(o) if isinstance(o, collections.Counter) else str(o))
    open(f"{a.out}/MANIFEST.md", "w").write("\n".join(md) + "\n")
    # invariants
    for s in sessions:
        for t in s["turns"]:
            assert len(t["tool_results"]) == len(t["expected_tool_calls"]), (s["id"], t["index"])
            for c in t["expected_tool_calls"]:
                json.loads(c["arguments"])
            if s["source"] == "gen" and t is not s["turns"][-1]:
                assert 1 <= len(t["expected_tool_calls"]) <= 2, (s["id"], t["index"])
        assert "{{SALT}}" in s["turns"][0]["user"]
        assert s["turns"][0]["user"].count("{{SALT}}") == 1
        if s.get("image"):
            assert s["source"] == "gen" and s["image"]["slot"].startswith("img-")
        for fam in fams:
            p = per_session[s["id"]][fam]["prompt"]
            assert all(x <= y for x, y in zip(p, p[1:])), (s["id"], fam, p)
    print(f"CORPUS {VERSION} sessions={len(sessions)} {dict(man['by_source'])} json_schema={man['json_schema_sessions']} thinking={man['thinking_sessions']} images={len(image_slots)} "
          f"bytes={man['corpus_bytes']} families={fams} scrubbed={dict(SCRUBBED)} in {time.time() - t0:.1f} s -> {os.path.basename(a.out.rstrip('/'))}", file=sys.stderr)
    if a.selftest:
        print("SELFTEST OK", file=sys.stderr)


if __name__ == "__main__":
    main()
