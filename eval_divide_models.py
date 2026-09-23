# -*- coding: utf-8 -*-
"""
Divide-model benchmark for the KEDKG bridge pipeline: measures, for every
candidate model of an OpenAI-compatible endpoint, WHICH one decomposes
multi-hop questions BEST (quality) and FASTEST (efficiency).

Fidelity: every request replicates run_llm_divide of dynamic_exp_bridge3.py
verbatim -- same template (prompts/divide_api_bridge3.txt), same system
message, same sampling parameters (temperature=0, stop=["\\nQuestion:"],
zero frequency/presence penalties) -- so a model that scores well here will
score the same way inside the experiment.

Calling style follows test/test_api_conn.py: pure `requests` (no openai SDK),
constants at the top, informational console output.

QUALITY metrics (ground truth = the dataset's gold single_hops):
  hop-match  parsed #sub-questions == #gold hops (len(single_hops)).
             Under-splitting was the single largest Hop-Acc failure bucket in
             the 322-case bridge3 run (79/118), so this is the PRIMARY metric.
  under/over split rates (under-splitting is the known failure mode).
  format     structural rules the runtime parsing relies on: >=1 sub-question,
             no <think> block, not truncated by max_tokens, first sub-question
             self-contained (no [ENT]), every later sub-question carries [ENT].
  gold-sim   mean token-F1 of each generated sub-question vs its best-matching
             gold hop question -- are the sub-questions on-topic at all.
  gold-cov   fraction of gold hops matched with token-F1 >= 0.3 by at least
             one generated sub-question -- did the model cover every hop.
EFFICIENCY metrics: latency per request, completion tokens (cost proxy),
tokens/sec throughput. All requests are SEQUENTIAL so latencies are
comparable across models.

Composite score (weights are constants below -- tweak freely):
  score = success_rate * (W_HOP*hop_match + W_FMT*format + W_SPEED*speed)
  speed = fastest_model_avg_latency / this_model_avg_latency  (capped at 1)

Usage (from the test/ directory, like the main scripts):
  python eval_divide_models.py                        # all candidates, 15 questions
  python eval_divide_models.py --questions 5          # quick pass
  python eval_divide_models.py --only qwen3.6-chat,glm-5.2
  python eval_divide_models.py --resume               # continue an interrupted run
Output: console ranking + JSON report (ALL raw replies included, for offline
inspection) in ../output/divide_model_eval/eval_<dataset>.json. The report is
saved after EVERY model, so an interrupted long run loses at most one model's
work; --resume skips models already marked complete in it.
"""

import os
import sys
import re
import json
import time
import random
import argparse
import collections

import requests

# ---------------------------------------------------------------------------
# configuration: fill these in (same style as test_api_conn.py)
# ---------------------------------------------------------------------------
BASE_URL = "https://api.llm.ustc.edu.cn"   # server root; /v1 appended automatically
API_KEY = "sk-TVcESF80StWSNz7m874Keg"      # <-- put your key here
VERIFY_SSL = True                          # False only if the campus CA is untrusted

DATASET = "MQuAKE-CF-3k"        # MQuAKE-T or MQuAKE-CF-3k (questions are sampled from it)
N_QUESTIONS = 50           # questions per model (stratified by gold hop count)
SEED = 42                   # sampling seed -> reproducible question set
TIMEOUT = 300               # seconds per request (non-streaming returns only when
                            # generation FINISHES; reasoning models can be slow)
MAX_TOKENS = 1024           # safety cap against runaway loops (bridge3 itself sends
                            # no cap; a real decomposition needs ~100 tokens, so a
                            # 'length' truncation here means the reply is broken)
RETRY_ATTEMPTS = 2          # attempts per question on transport error / empty content
RETRY_SLEEP = 5             # seconds between attempts
DIVIDE_PROMPT_FILE = "divide_api_bridge3.txt"   # the template bridge3 actually uses

# composite-score weights (see module docstring)
W_HOP, W_FMT, W_SPEED = 0.55, 0.30, 0.15
GOLD_F1_COVER = 0.3         # token-F1 threshold for the gold-cov metric

# candidate models to benchmark (the endpoint's full list, in evaluation order)
CANDIDATE_MODELS = [
#     "qwen3.5",
#     "qwen-chat",
    "qwen3.6-chat",
#     "qwen-reasoner",
#     "qwen3.5-thinking",
#     "qwen3.6-reasoner",
    "claude-sonnet-4-6",
#     "qwen3.5-non-thinking",
#     "smart/default",
#     "smart/reasoning",
    "glm-5.2",
    "glm-5.3-flash",
#     "k3",
#     "deepseek-v4-flash-ascend1",
    "deepseek-v4-flash-ascend",
    "qwen3.8-chat",
    "deepseek-v4-pro",
#     "qwen3.8-reasoner",
]

# VERBATIM system message of run_llm_divide in dynamic_exp_bridge3.py -- keep
# the two copies in sync, this benchmark must send byte-identical requests.
DIVIDE_SYSTEM_MESSAGE = "You are a question decomposition assistant. You decompose a multi-hop question into single-hop sub-questions, one per line, using [ENT] placeholders for entities resolved by previous sub-questions. Each sub-question must match exactly one knowledge-graph fact: never merge two facts into one, and never skip an intermediate lookup. When unsure whether a line covers one or two facts, split it into two lines -- under-splitting is worse than over-splitting, and typical questions need 2 to 4 sub-questions. If the asked relation plausibly belongs not to the named entity but to a related one (a song has no manager -- its performer does), first resolve that intermediate entity with its own sub-question. You output only the sub-questions, with no numbering and no extra text."

# same stop sequence run_llm_divide sends (prevents the model from continuing
# into hallucinated new "Question:" examples after answering)
STOP_SEQUENCE = ["\nQuestion:"]

# thinking-block markers (same detection as test_api_conn.py): thinking models
# wrap their reasoning in these tags and the line-based parsing of the
# experiment then reads the reasoning as sub-questions
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


# ---------------------------------------------------------------------------
# endpoint plumbing (style of test_api_conn.py)
# ---------------------------------------------------------------------------
def api_url(path):
    root = BASE_URL.rstrip("/")
    if not root.endswith("/v1"):
        root += "/v1"
    return root + path


def headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = "Bearer " + API_KEY
    return h


def diagnose(resp):
    code = resp.status_code
    if code in (401, 403):
        return "auth failed: check API_KEY (and this key's access to the model)"
    if code == 404:
        return "not found: wrong URL path or unknown MODEL id"
    if code == 429:
        return "rate limited: retry later"
    if code >= 500:
        return "server-side error"
    return "unexpected status code"


def list_endpoint_models():
    """GET /v1/models -- informational pre-check (missing candidates are still tried)."""
    try:
        resp = requests.get(api_url("/models"), headers=headers(),
                            timeout=60, verify=VERIFY_SSL)
    except requests.exceptions.RequestException as e:
        print("WARN  could not list endpoint models: {}".format(e))
        return None
    if resp.status_code != 200:
        print("WARN  GET /models -> HTTP {} ({}); continuing anyway".format(
            resp.status_code, diagnose(resp)))
        return None
    try:
        return [m.get("id") for m in resp.json().get("data", [])]
    except ValueError:
        return None


def divide_request(model, user_prompt, timeout, minimal_payload=False):
    """One divide request, mirroring run_llm_divide of dynamic_exp_bridge3.py.

    minimal_payload=True drops stop / frequency_penalty / presence_penalty:
    sticky fallback for models whose serving stack 400-rejects them (the
    gateway 400-rejects unknown body params). Returns
    dict(ok, content, notes, finish_reason, usage, elapsed, error, status).
    """
    messages = [
        {"role": "system", "content": DIVIDE_SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]
    payload = {"model": model, "messages": messages, "temperature": 0}
    if MAX_TOKENS:
        payload["max_tokens"] = MAX_TOKENS
    if not minimal_payload:
        payload["stop"] = STOP_SEQUENCE
        payload["frequency_penalty"] = 0
        payload["presence_penalty"] = 0
    t0 = time.time()
    try:
        resp = requests.post(api_url("/chat/completions"), headers=headers(),
                             json=payload, timeout=timeout, verify=VERIFY_SSL)
    except requests.exceptions.SSLError as e:
        return {"ok": False, "error": "SSL error: {} (campus CA? try VERIFY_SSL=False)".format(e),
                "elapsed": time.time() - t0}
    except requests.exceptions.ConnectionError as e:
        return {"ok": False, "error": "cannot connect: {}".format(e),
                "elapsed": time.time() - t0}
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "read timeout after {}s".format(timeout),
                "elapsed": time.time() - t0}
    elapsed = time.time() - t0
    if resp.status_code != 200:
        return {"ok": False, "status": resp.status_code,
                "error": "HTTP {} - {}".format(resp.status_code, diagnose(resp)),
                "body": resp.text[:300], "elapsed": elapsed}
    try:
        data = resp.json()
        choice = data["choices"][0]
    except (ValueError, KeyError, IndexError) as e:
        return {"ok": False, "status": 200, "error": "malformed response: {}".format(e),
                "body": resp.text[:300], "elapsed": elapsed}
    msg = choice.get("message", {}) or {}
    content = msg.get("content")
    notes = []
    if content is None:
        # content=null: reasoning models may expose text as reasoning_content,
        # or the reply may be genuinely empty (bridge3's empty-completion guard
        # treats the latter as a transport error and retries -- same here)
        reasoning = msg.get("reasoning_content")
        if reasoning:
            content = reasoning
            notes.append("content null; used reasoning_content instead")
        else:
            content = ""
            notes.append("content null and reasoning_content empty")
    return {"ok": True, "content": content, "notes": notes,
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage", {}), "elapsed": elapsed}


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def load_divide_template(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in [os.path.join(here, "prompts", filename),
                 os.path.join(here, "..", "prompts", filename),
                 os.path.join(here, "..", "..", "prompts", filename)]:
        if os.path.isfile(cand):
            with open(cand, "r", encoding="utf-8") as f:
                print("using divide template: {}".format(cand))
                return f.read()
    sys.exit("ERROR: prompts/{} not found (run from the project root or test/)".format(filename))


def parse_sub_questions(output):
    # VERBATIM copy of the sub-question normalisation block of
    # dynamic_exp_bridge3.py (strip list numbering and a possible
    # 'Subquestion:' echo, drop empty lines) -- the benchmark must judge
    # exactly what the experiment's retrievers would receive
    sub_questions = []
    for s in (output or "").split('\n'):
        s = re.sub(r'(?i)^subquestions?\s*:?\s*', '', s.strip())
        s = re.sub(r'^\s*(?:\d+[\.\)]\s*|[-\*]\s*)', '', s).strip()
        if s:
            sub_questions.append(s)
    return sub_questions


def _tokens(s):
    return re.findall(r'[a-z0-9]+', (s or '').lower())


def token_f1(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    common = collections.Counter(ta) & collections.Counter(tb)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    return 2.0 * overlap / (len(ta) + len(tb))


def score_reply(result, gold_hops, gold_questions):
    """Quality metrics for one divide reply, judged exactly like the runtime."""
    content = result.get("content") or ""
    subs = parse_sub_questions(content)
    n = len(subs)
    think = (THINK_OPEN in content) or (THINK_CLOSE in content)
    truncated = result.get("finish_reason") == "length"
    first_ok = n >= 1 and "[ENT]" not in subs[0]
    later_ok = all("[ENT]" in s for s in subs[1:])
    fmt = (n >= 1) and (not think) and (not truncated) and first_ok and later_ok
    gold_sim = gold_cov = None
    if gold_questions and n:
        sims = [max(token_f1(s, g) for g in gold_questions) for s in subs]
        gold_sim = sum(sims) / len(sims)
        cov = [max((token_f1(s, g) for s in subs), default=0.0) for g in gold_questions]
        gold_cov = sum(1 for c in cov if c >= GOLD_F1_COVER) / len(gold_questions)
    return {
        "n_subq": n,
        "hop_match": n == gold_hops,
        "under": n < gold_hops,
        "over": n > gold_hops,
        "format_ok": fmt,
        "think_block": think,
        "truncated": truncated,
        "ent_any": any("[ENT]" in s for s in subs),
        "first_ok": first_ok,
        "later_ok": later_ok,
        "gold_sim": round(gold_sim, 3) if gold_sim is not None else None,
        "gold_cov": round(gold_cov, 3) if gold_cov is not None else None,
        "sub_questions": subs,
    }


def load_eval_questions(dataset, n, seed):
    """Sample n questions (first paraphrase per case), stratified by gold hop
    count so 3-hop and 4-hop cases are not drowned out by the 2-hop majority
    (MQuAKE-T: 1421 / 445 / 2 cases for 2/3/4 hops)."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here,  "datasets", dataset + ".json")
    if not os.path.isfile(path):
        sys.exit("ERROR: dataset not found: {}".format(path))
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_hops = collections.defaultdict(list)
    for d in data:
        hops = len(d.get("single_hops") or [])
        if hops >= 2 and d.get("questions"):
            by_hops[hops].append(d)
    if not by_hops:
        sys.exit("ERROR: no usable cases in {}".format(path))
    rng = random.Random(seed)
    buckets = sorted(by_hops)
    shuffled = {}
    for h in buckets:
        shuffled[h] = by_hops[h][:]
        rng.shuffle(shuffled[h])
    per = max(1, n // len(buckets))
    picked = []
    for h in buckets:
        picked += shuffled[h][:per]
    if len(picked) < n:                       # top up (e.g. only two 4-hop cases)
        rest = []
        for h in buckets:
            rest += shuffled[h][per:]
        rng.shuffle(rest)
        picked += rest[: n - len(picked)]
    out = []
    for d in picked[:n]:
        out.append({
            "question": d["questions"][0],
            "gold_hops": len(d["single_hops"]),
            "gold_questions": [h.get("question", "") for h in d["single_hops"]],
        })
    dist = collections.Counter(q["gold_hops"] for q in out)
    print("sampled {} questions from {} (hop distribution: {})".format(
        len(out), dataset, dict(sorted(dist.items()))))
    return out


def eval_model(model, questions, template, timeout):
    """Run every question through one model; returns the per-question records."""
    results = []
    minimal_mode = False    # sticky: once stop/penalties get 400-rejected, drop them
    for qi, qd in enumerate(questions):
        prompt = template.replace("<<<<QUESTION>>>>", qd["question"])
        rec = {"question": qd["question"], "gold_hops": qd["gold_hops"]}
        r = None
        err = None
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            r = divide_request(model, prompt, timeout, minimal_payload=minimal_mode)
            if r["ok"] and str(r.get("content") or "").strip():
                break
            if r.get("status") == 400 and not minimal_mode:
                # this model's stack rejects stop/penalties -> minimal payload
                minimal_mode = True
                r = divide_request(model, prompt, timeout, minimal_payload=True)
                if r["ok"] and str(r.get("content") or "").strip():
                    r["notes"].append("minimal payload (stop/penalties rejected)")
                    break
            err = r.get("error") or "empty content"
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_SLEEP)
        if r is None or not r["ok"] or not str(r.get("content") or "").strip():
            rec.update({"ok": False, "error": err or "unknown",
                        "elapsed": round(r["elapsed"], 2) if r and r.get("elapsed") else None,
                        "body": r.get("body")})
            results.append(rec)
            print("    [{}/{}] FAIL {}".format(qi + 1, len(questions), err))
            continue
        rec.update({"ok": True,
                    "elapsed": round(r["elapsed"], 2),
                    "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
                    "finish_reason": r.get("finish_reason"),
                    "notes": r.get("notes", []),
                    "raw": r.get("content")})
        rec.update(score_reply(r, qd["gold_hops"], qd["gold_questions"]))
        results.append(rec)
        print("    [{}/{}] {:6.1f}s  subq={} (gold {})  hop_match={}  format={}".format(
            qi + 1, len(questions), r["elapsed"], rec["n_subq"], qd["gold_hops"],
            "Y" if rec["hop_match"] else "N", "Y" if rec["format_ok"] else "N"))
    return results


def aggregate(model, results):
    n_total = len(results)
    oks = [r for r in results if r.get("ok")]
    s = {"model": model, "n_total": n_total, "n_ok": len(oks),
         "success_rate": (len(oks) / n_total) if n_total else 0.0}
    if not oks:
        s["failed"] = True
        s["error_sample"] = next((r.get("error") for r in results if r.get("error")), "")
        return s

    def mean(key):
        vals = [r[key] for r in oks if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    lat = [r["elapsed"] for r in oks if r.get("elapsed") is not None]
    toks = [r["completion_tokens"] for r in oks if r.get("completion_tokens") is not None]
    s.update({
        "failed": False,
        "hop_match_rate": mean("hop_match"),
        "under_rate": mean("under"),
        "over_rate": mean("over"),
        "format_rate": mean("format_ok"),
        "think_rate": mean("think_block"),
        "trunc_rate": mean("truncated"),
        "gold_sim": mean("gold_sim"),
        "gold_cov": mean("gold_cov"),
        "avg_latency": (sum(lat) / len(lat)) if lat else None,
        "avg_completion_tokens": (sum(toks) / len(toks)) if toks else None,
        "tokens_per_sec": (sum(toks) / sum(lat)) if (toks and lat and sum(lat) > 0) else None,
    })
    return s


def f2(v, na="-"):
    return format(v, ".2f") if isinstance(v, (int, float)) else na


def print_model_summary(s, sample_rec=None):
    if s.get("failed"):
        print("  summary: FAILED all {} request(s) -- {}".format(
            s["n_total"], s.get("error_sample", "")))
        return
    print("  summary: ok {}/{} | hop-match {} | under {} | format {} | gold-cov {} | "
          "gold-sim {} | {}s avg | {} tok avg | {} tok/s".format(
              s["n_ok"], s["n_total"], f2(s["hop_match_rate"]), f2(s["under_rate"]),
              f2(s["format_rate"]), f2(s["gold_cov"]), f2(s["gold_sim"]),
              f2(s["avg_latency"]), f2(s["avg_completion_tokens"]), f2(s["tokens_per_sec"])))
    if sample_rec:
        print("  sample reply (gold {} hops):".format(sample_rec["gold_hops"]))
        for sq in sample_rec.get("sub_questions", [])[:6]:
            print("    " + sq[:110])


def save_report(path, blob):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)     # atomic: a crash never truncates the report


# ---------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    global BASE_URL, API_KEY, TIMEOUT, MAX_TOKENS
    p = argparse.ArgumentParser(
        description="Benchmark divide models on the REAL bridge3 decomposition request "
                    "(quality: hop-match vs gold hops, format, gold coverage; "
                    "efficiency: latency / tokens / throughput).")
    p.add_argument("--dataset", type=str, default=DATASET, choices=["MQuAKE-T", "MQuAKE-CF-3k"])
    p.add_argument("--questions", type=int, default=N_QUESTIONS,
                   help="questions per model, stratified by gold hop count")
    p.add_argument("--models", type=str, default="",
                   help="comma-separated model list (default: CANDIDATE_MODELS)")
    p.add_argument("--only", type=str, default="",
                   help="comma-separated filter on the candidate list")
    p.add_argument("--timeout", type=int, default=TIMEOUT)
    p.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--base_url", type=str, default=BASE_URL)
    p.add_argument("--api_key", type=str, default=API_KEY)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=str, default="",
                   help="report path (default: ../output/divide_model_eval/eval_<dataset>.json)")
    p.add_argument("--resume", action="store_true",
                   help="skip models already complete in the existing report")
    args = p.parse_args()

    BASE_URL, API_KEY, TIMEOUT, MAX_TOKENS = args.base_url, args.api_key, args.timeout, args.max_tokens

    if not API_KEY:
        print("WARNING: API_KEY is empty -- every request will fail with 401.")

    models = [m.strip() for m in args.models.split(",")] if args.models else list(CANDIDATE_MODELS)
    if args.only:
        keep = {m.strip() for m in args.only.split(",")}
        models = [m for m in models if m in keep]
    seen = set()
    models = [m for m in models if m and not (m in seen or seen.add(m))]
    if not models:
        sys.exit("ERROR: empty model list")

    template = load_divide_template(DIVIDE_PROMPT_FILE)
    questions = load_eval_questions(args.dataset, args.questions, args.seed)

    here = os.path.dirname(os.path.abspath(__file__))
    report_path = args.out or os.path.join(here, "..", "output", "divide_model_eval",
                                           "eval_{}.json".format(args.dataset))

    report = {
        "meta": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": args.dataset,
            "n_questions": len(questions),
            "seed": args.seed,
            "prompt_file": DIVIDE_PROMPT_FILE,
            "base_url": BASE_URL,
            "timeout": TIMEOUT,
            "max_tokens": MAX_TOKENS,
            "weights": {"hop": W_HOP, "format": W_FMT, "speed": W_SPEED},
            "note": "requests replicate run_llm_divide of dynamic_exp_bridge3.py verbatim "
                    "(same template, system message, temperature=0, stop sequence)",
            "eval_questions": questions,
        },
        "models": {},
    }

    if args.resume and os.path.isfile(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            if old.get("meta", {}).get("dataset") == args.dataset:
                report["models"] = old.get("models", {})
                done = [m for m, r in report["models"].items() if r.get("complete")]
                print("[resume] loaded report with {} complete model(s): {}".format(
                    len(done), ", ".join(done)))
                if old["meta"].get("n_questions") != len(questions):
                    print("[resume] NOTE: question count differs from the saved report "
                          "({} vs {}); metrics of resumed models are not comparable".format(
                              old["meta"].get("n_questions"), len(questions)))
            else:
                print("[resume] saved report is for another dataset; starting fresh")
        except (ValueError, OSError) as e:
            print("[resume] could not read {}: {}; starting fresh".format(report_path, e))

    endpoint_models = list_endpoint_models()
    if endpoint_models is not None:
        missing = [m for m in models if m not in endpoint_models]
        print("endpoint serves {} model(s); {} candidate(s) not in the list (will still be "
              "tried): {}".format(len(endpoint_models), len(missing),
                                  ", ".join(missing) if missing else "none"))

    print("=" * 78)
    print("benchmarking {} model(s) x {} question(s) | dataset {} | template {}".format(
        len(models), len(questions), args.dataset, DIVIDE_PROMPT_FILE))
    print("report: {}".format(os.path.abspath(report_path)))
    print("=" * 78)

    interrupted = False
    try:
        for model in models:
            if args.resume and report["models"].get(model, {}).get("complete"):
                print("--- {}: already complete, skipping (--resume)".format(model))
                continue
            print("--- {} ---".format(model))
            t0 = time.time()
            results = eval_model(model, questions, template, TIMEOUT)
            s = aggregate(model, results)
            print_model_summary(s, next((r for r in results if r.get("ok")), None))
            print("  wall time: {:.1f}s".format(time.time() - t0))
            report["models"][model] = {"summary": s, "results": results, "complete": True}
            report["meta"]["generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_report(report_path, report)     # incremental: crash loses one model max
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupted] saving partial report ...")
        save_report(report_path, report)

    # ---- ranking ----------------------------------------------------------
    summaries = [r["summary"] for r in report["models"].values() if r.get("complete")]
    if not summaries:
        sys.exit("ERROR: no model produced results" + (" (interrupted)" if interrupted else ""))

    lat_vals = [s["avg_latency"] for s in summaries if s.get("avg_latency")]
    best_lat = min(lat_vals) if lat_vals else None
    for s in summaries:
        if s.get("failed") or s.get("hop_match_rate") is None:
            s["speed"] = None
            s["score"] = 0.0
            continue
        speed = 1.0
        if best_lat and s.get("avg_latency"):
            speed = min(1.0, best_lat / s["avg_latency"])
        s["speed"] = round(speed, 3)
        s["score"] = round(s["success_rate"] * (W_HOP * s["hop_match_rate"]
                                                + W_FMT * (s["format_rate"] or 0.0)
                                                + W_SPEED * speed), 4)

    print("=" * 78)
    print("RANKING  (score = success * ({}*hop-match + {}*format + {}*speed); "
          "speed = fastest_avg_latency / own_avg_latency)".format(W_HOP, W_FMT, W_SPEED))
    print("{:<30s} {:>7s} {:>6s} {:>6s} {:>6s} {:>6s} {:>7s} {:>7s} {:>7s}".format(
        "model", "ok", "hop", "under", "fmt", "cov", "avg-s", "tok/s", "score"))
    print("-" * 90)
    ranked = sorted(summaries, key=lambda x: -x["score"])
    for i, s in enumerate(ranked, 1):
        if s.get("failed"):
            print("{:<30s} {:>7s} {:>48s}".format(
                "{}. {}".format(i, s["model"]), "{}/{}".format(s["n_ok"], s["n_total"]),
                "FAILED: " + (s.get("error_sample") or "")[:40]))
            continue
        print("{:<30s} {:>7s} {:>6s} {:>6s} {:>6s} {:>6s} {:>7s} {:>7s} {:>7s}".format(
            "{}. {}".format(i, s["model"]),
            "{}/{}".format(s["n_ok"], s["n_total"]),
            f2(s["hop_match_rate"]), f2(s["under_rate"]), f2(s["format_rate"]),
            f2(s["gold_cov"]), f2(s["avg_latency"]), f2(s["tokens_per_sec"]),
            f2(s["score"])))

    usable = [s for s in ranked if not s.get("failed")]
    print("=" * 78)
    print("RECOMMENDATION (top 3 by score; prefer non-thinking models -- "
          "<think> blocks break the line-based parsing):")
    for i, s in enumerate(usable[:3], 1):
        extra = []
        if s.get("think_rate"):
            extra.append("thinking output in {} of replies".format(
                f2(s["think_rate"])))
        if s.get("under_rate") and s["under_rate"] > 0.2:
            extra.append("under-splits {} of questions".format(f2(s["under_rate"])))
        print("  {}. {}  (score {}){}".format(i, s["model"], f2(s["score"]),
                                              "  [!] " + "; ".join(extra) if extra else ""))
    if usable:
        best = usable[0]["model"]
        print("  next steps:")
        print('    DIVIDE_MODEL = "{}"          # in dynamic_exp_bridge3.py / bridge4.py'.format(best))
        print("    python build_divide_cache.py --dataset {} --model {}   # precompute the cache".format(
            args.dataset, best))
    if interrupted:
        print("\nNOTE: run was interrupted; re-run with --resume to finish the remaining models.")
    save_report(report_path, report)
    print("full report (all raw replies) saved to: {}".format(os.path.abspath(report_path)))
    sys.exit(0 if usable else 1)     # like test_api_conn.py: non-zero if nothing usable


if __name__ == "__main__":
    main()
