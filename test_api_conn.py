# -*- coding: utf-8 -*-
"""
Connectivity test for an OpenAI-compatible LLM endpoint (e.g. the USTC campus
service https://api.llm.ustc.edu.cn/). Pure `requests`, no openai SDK needed,
so it runs on any server with a plain python + requests environment.

Edit the constants below, then:  python test_api_conn.py

What it checks:
  1. GET  /v1/models            -> auth works, prints every available model id
                                    (confirm the exact model name here)
  2. POST /v1/chat/completions  -> basic round trip ("ping"), capped output
  3. POST /v1/chat/completions  -> a decomposition request built from the REAL
                                    prompts/divide_api.txt few-shot template (same
                                    prompt dynamic_exp.py sends in experiments),
                                    reports latency / tokens / finish_reason /
                                    [ENT] format compliance
  4. thinking-control probe     -> tries common ways to disable thinking
                                    (enable_thinking / chat_template_kwargs),
                                    reports which one this gateway accepts
  5. model comparison           -> runs the real divide prompt on every model in
                                    CANDIDATE_MODELS and prints latency / token
                                    usage / [ENT] format compliance side by side,
                                    so you can pick the decomposition model by
                                    evidence instead of guessing
Exit code 0 = tests 1-3 passed (tests 4-5 are informational).

Note: place this script either in the project root or in test/ -- it locates
prompts/divide_api.txt automatically and falls back to a minimal prompt (with a
warning) if the file is not found.
"""

import os
import sys
import time

import requests

# ---------------------------------------------------------------------------
# configuration: fill these in
# ---------------------------------------------------------------------------
BASE_URL = "https://api.llm.ustc.edu.cn"   # server root; /v1 is appended automatically
API_KEY = "sk-TVcESF80StWSNz7m874Keg"                               # <-- put your key here (leave empty to test without)
MODEL = "deepseek-v4-flash-ascend"         # <-- must exactly match an id printed by test 1
TIMEOUT = 300                              # seconds per request. Non-streaming chat returns
                                           # only when generation FINISHES, so a thinking
                                           # model on a loaded shared service can legitimately
                                           # need minutes.
MAX_TOKENS = 1024                          # cap on generated tokens; prevents runaway
                                           # repetition loops (temperature=0) from hanging
# candidates evaluated by test 5 for the decomposition role: fast non-thinking
# chat models only; add/remove ids from test 1's list as you like
CANDIDATE_MODELS = [
    "qwen3.6-chat",              # 现任选型（对照组）
    "deepseek-v4-flash-ascend",  # 已知格式风险（对照组）
    "qwen3.8-chat",              # 重测：旧 prompt 下超时过
    "qwen3.8-reasoner",          # 预期：思考痕迹 + 高延迟
    "claude-sonnet-4-6",         # 预期：共享服务排队
    "glm-5.2",                   # glm-chat 上次 500，换新版重试
]
COMPARISON_TIMEOUT = 180                   # per-candidate cap in test 5
# set to False only if the school CA cert is not trusted on your server
VERIFY_SSL = True

TEST_QUESTION = "What is the employer of the spouse of Ann Druyan?"
MINIMAL_DIVIDE_PROMPT = "Question: " + TEST_QUESTION + "\nSubquestion:"


def load_divide_prompt():
    """Return the real few-shot divide prompt with the test question substituted,
    exactly like dynamic_exp.py does (divide_prompt.replace("<<<<QUESTION>>>>", q)).
    Searches common locations so the script works from the project root or test/."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in [os.path.join(here, "prompts", "divide_api.txt"),
                 os.path.join(here, "..", "prompts", "divide_api.txt"),
                 os.path.join(here, "..", "..", "prompts", "divide_api.txt")]:
        if os.path.isfile(cand):
            with open(cand, "r", encoding="utf-8") as f:
                template = f.read()
            print("using real divide template: {}".format(cand))
            return template.replace("<<<<QUESTION>>>>", TEST_QUESTION)
    print("WARNING: prompts/divide_api.txt not found; falling back to a minimal prompt.")
    print("         Format results will then be pessimistic (no few-shot examples).")
    return MINIMAL_DIVIDE_PROMPT


THINK_CONTROL_ATTEMPTS = [
    ("enable_thinking=false", {"enable_thinking": False}),
    ("chat_template_kwargs.enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}}),
]


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
        return "auth failed: check API_KEY (and whether the key has access to this model)"
    if code == 404:
        return "not found: wrong URL path or wrong MODEL name (see the ids printed by test 1)"
    if code == 429:
        return "rate limited: too many requests, retry later"
    if code >= 500:
        return "server-side error, check the service status"
    return "unexpected status code"


def chat(messages, temperature=0.0, max_tokens=MAX_TOKENS, extra=None, timeout=TIMEOUT, model=None):
    """Returns dict(ok, status, content, finish_reason, elapsed, usage, error).
    Handles content=null responses (some reasoning models put text in
    reasoning_content, or return an empty content) without crashing."""
    payload = {"model": model or MODEL, "messages": messages, "temperature": temperature}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if extra:
        payload.update(extra)
    t0 = time.time()
    try:
        resp = requests.post(api_url("/chat/completions"), headers=headers(),
                             json=payload, timeout=timeout, verify=VERIFY_SSL)
    except requests.exceptions.SSLError as e:
        return {"ok": False, "error": "SSL error: {} (campus CA not trusted? try VERIFY_SSL = False)".format(e), "elapsed": time.time() - t0}
    except requests.exceptions.ConnectionError as e:
        return {"ok": False, "error": "cannot connect: {} (check network / VPN)".format(e), "elapsed": time.time() - t0}
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "read timeout after {}s: generation did not finish in time".format(timeout), "elapsed": time.time() - t0}
    elapsed = time.time() - t0
    if resp.status_code != 200:
        return {"ok": False, "status": resp.status_code,
                "error": "HTTP {} - {}".format(resp.status_code, diagnose(resp)),
                "body": resp.text[:500], "elapsed": elapsed}
    try:
        data = resp.json()
        choice = data["choices"][0]
    except (ValueError, KeyError, IndexError) as e:
        return {"ok": False, "status": 200, "error": "malformed response: {}".format(e),
                "body": resp.text[:500], "elapsed": elapsed}
    msg = choice.get("message", {}) or {}
    content = msg.get("content")
    notes = []
    if content is None:
        # content=null: reasoning models may expose the text as reasoning_content,
        # or the reply may be genuinely empty
        reasoning = msg.get("reasoning_content")
        if reasoning:
            content = reasoning
            notes.append("content was null; using reasoning_content instead")
        else:
            content = ""
            notes.append("content was null and reasoning_content empty (empty reply?)")
    return {"ok": True, "content": content, "notes": notes,
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage", {}), "elapsed": elapsed}


def describe_result(r):
    """One-line summary used by tests 3-5."""
    bits = ["{:.1f}s".format(r["elapsed"])]
    if r.get("usage"):
        u = r["usage"]
        bits.append("prompt {} / completion {} tokens".format(u.get("prompt_tokens", "?"), u.get("completion_tokens", "?")))
    if r.get("finish_reason"):
        bits.append("finish_reason={}".format(r["finish_reason"]))
    content = r.get("content") or ""
    if "<think>" in content:
        bits.append("REPLY CONTAINS <think> BLOCK (thinking mode is ON)")
    if r.get("finish_reason") == "length":
        bits.append("output TRUNCATED by max_tokens (no room left for the actual answer)")
    bits.extend(r.get("notes", []))
    return ", ".join(bits)


def print_divide_reply(content):
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    for line in lines:
        print("       " + line)
    has_ent = "[ENT]" in content
    if len(lines) == 2 and has_ent:
        verdict = "format OK (2 sub-questions, [ENT] used)"
    else:
        problems = []
        if len(lines) != 2:
            problems.append("{} line(s) (expected 2)".format(len(lines)))
        if not has_ent:
            problems.append("[ENT] MISSING")
        verdict = "format problem: " + "; ".join(problems)
    print("       -> {}".format(verdict))
    print("NOTE  dynamic_exp.py parses this with output.split('\\n')")
    return len(lines) == 2, has_ent


def test_list_models():
    print("=" * 60)
    print("[test 1] GET {}".format(api_url("/models")))
    try:
        resp = requests.get(api_url("/models"), headers=headers(),
                            timeout=TIMEOUT, verify=VERIFY_SSL)
    except requests.exceptions.RequestException as e:
        print("FAIL  request error: {}".format(e))
        return False
    if resp.status_code != 200:
        print("FAIL  HTTP {} - {}".format(resp.status_code, diagnose(resp)))
        print("       body: {}".format(resp.text[:500]))
        return False
    try:
        ids = [m.get("id") for m in resp.json().get("data", [])]
    except ValueError:
        print("WARN  HTTP 200 but response is not JSON: {}".format(resp.text[:500]))
        return True
    print("PASS  endpoint reachable, {} models available:".format(len(ids)))
    for i in ids:
        print("       - {}".format(i))
    if ids and MODEL not in ids:
        print("NOTE  model '{}' is NOT in the list above -- update MODEL to the exact id".format(MODEL))
    return True


def test_chat_ping():
    print("=" * 60)
    print("[test 2] POST {} (model = {}, max_tokens=16)".format(api_url("/chat/completions"), MODEL))
    r = chat([{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=16)
    if not r["ok"]:
        print("FAIL  {}".format(r["error"]))
        if r.get("body"):
            print("       body: {}".format(r["body"]))
        return False
    print("PASS  chat works ({})".format(describe_result(r)))
    print("       reply: {!r}".format(r["content"]))
    return True


def test_chat_divide_style(divide_prompt):
    print("=" * 60)
    print("[test 3] real divide prompt (prompts/divide_api.txt template), max_tokens={}".format(MAX_TOKENS))
    print("       expected reply shape:")
    print("         Who is Ann Druyan married to?")
    print("         Who is the employer of [ENT]?")
    r = chat([{"role": "user", "content": divide_prompt}])
    if not r["ok"]:
        print("FAIL  {}".format(r["error"]))
        if r.get("body"):
            print("       body: {}".format(r["body"]))
        print("       if this is a read timeout: generation ran longer than {}s.".format(TIMEOUT))
        print("       raise TIMEOUT, cap max_tokens, or pick a non-thinking model variant")
        return False
    print("PASS  ({})".format(describe_result(r)))
    print("       reply:")
    print_divide_reply(r["content"])
    return True


def test_thinking_control(divide_prompt):
    print("=" * 60)
    print("[test 4] thinking-control probe (informational)")
    print("       sending the same divide prompt with common thinking-disable parameters")
    for name, extra in THINK_CONTROL_ATTEMPTS:
        r = chat([{"role": "user", "content": divide_prompt}], extra=extra)
        if r["ok"]:
            has_think = "<think>" in (r.get("content") or "")
            print("ACCEPTED  {:45s} ({})".format(name, describe_result(r)))
            if has_think:
                print("          NOTE still contains <think>: the parameter was ignored by the backend")
        else:
            status = r.get("status", "")
            print("REJECTED  {:45s} ({})".format(name, r["error"] if not status else "HTTP {} - param not supported by this gateway".format(status)))
    print("       if every attempt still thinks or is rejected, simplest fix: pick an")
    print("       explicitly non-thinking model from test 1's list (e.g. *-non-thinking)")
    return True


def test_model_comparison(divide_prompt):
    print("=" * 60)
    print("[test 5] decomposition-model comparison with the REAL divide prompt (informational)")
    print("       expected reply shape: 2 sub-questions, second starting with '[ENT]'")
    results = []
    for cand in CANDIDATE_MODELS:
        print("-" * 60)
        print("--- {} ---".format(cand))
        r = chat([{"role": "user", "content": divide_prompt}],
                 timeout=COMPARISON_TIMEOUT, model=cand)
        if not r["ok"]:
            print("FAIL   {}".format(r["error"]))
            results.append((cand, None, None, False, False))
            continue
        print("PASS   {}".format(describe_result(r)))
        ok_fmt, has_ent = print_divide_reply(r["content"])
        results.append((cand, r["elapsed"], (r.get("usage") or {}).get("completion_tokens"), ok_fmt, has_ent))
    print("-" * 60)
    print("summary (sorted by latency):")
    scored = [(c, e, t, ok, ent) for (c, e, t, ok, ent) in results if e is not None]
    for c, e, t, ok, ent in sorted(scored, key=lambda x: x[1]):
        print("  {:30s} {:6.1f}s  {:>4} completion tokens  format {}  [ENT] {}".format(
            c, e, t if t is not None else "?", "OK" if ok else "BAD", "OK" if ent else "MISSING"))
    failed = [c for (c, e, t, ok, ent) in results if e is None]
    if failed:
        print("  failed/timeout: {}".format(", ".join(failed)))
    print("       pick a model with format OK + [ENT] OK and the lowest latency,")
    print("       then set it as DIVIDE_MODEL (and ANSWER_MODEL) in dynamic_exp.py")
    return True


if __name__ == "__main__":
    print("endpoint : {}".format(BASE_URL))
    print("model    : {}".format(MODEL))
    print("api key  : {}".format("set" if API_KEY else "(empty)"))
    print("timeout  : {}s, max_tokens: {}".format(TIMEOUT, MAX_TOKENS))
    divide_prompt = load_divide_prompt()
    ok = all([test_list_models(), test_chat_ping(), test_chat_divide_style(divide_prompt)])
    test_thinking_control(divide_prompt)
    test_model_comparison(divide_prompt)
    print("=" * 60)
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    sys.exit(0 if ok else 1)
