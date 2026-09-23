# -*- coding: utf-8 -*-
"""
Side-by-side fingerprint + verifier-latency probe for the qwen family on an
OpenAI-compatible gateway. Two questions answered:

  1. IDENTITY: what is the alias 'qwen-chat' actually? The gateway echoes the
     alias back in the response 'model' field, so identity must be resolved
     BEHAVIORALLY: run identical probes on every qwen id and compare
     (thinking on/off, latency profile, training-cutoff answer). Matching
     fingerprints = same underlying model.

  2. VERIFIER SELECTION: which qwen id can serve as the FAST verifier?
     Measures the REAL verify prompts (prompts/verify_support.txt /
     verify_refute.txt, same substitution + same system message + same
     temperature=0 + no max_tokens as run_llm_verify in the bridge scripts)
     on two cases taken from the actual experiment logs. Reports latency,
     parsed Confidence, and thinking length.

  3. /no_think SOFT SWITCH: Qwen3 hybrid models accept '/no_think' appended
     to the user message to disable thinking WITHOUT changing the model.
     If the gateway serves real Qwen3 hybrids, this turns a 29s/call
     thinking verifier into a ~4s/call one with the SAME weights (same
     calibration). Tested on the thinking candidates.

Pure `requests`. Run ON THE SERVER (the campus gateway is not reachable
from off-campus). Results print progressively; Ctrl-C anytime.

    python probe_verify_latency.py            # all qwen ids, all stages
    python probe_verify_latency.py qwen3.6-chat qwen3.8-chat   # subset
"""

import os
import re
import sys
import time

import requests

BASE_URL = "https://api.llm.ustc.edu.cn"
API_KEY = "sk-TVcESF80StWSNz7m874Keg" 
TIMEOUT = 300
VERIFY_SSL = True
HEADERS = {"Authorization": "Bearer " + API_KEY,
           "Content-Type": "application/json"}

ALL_QWEN = ["qwen-chat", "qwen-reasoner", "qwen3.5", "qwen3.5-non-thinking",
            "qwen3.5-thinking", "qwen3.6-chat", "qwen3.6-reasoner",
            "qwen3.8-chat", "qwen3.8-reasoner"]

# the two verify cases, copied from the real experiment logs (one adjudicated
# accept, one edit-bypass reject) -- realistic difficulty, not toy prompts
VERIFY_SYSTEM = ("You are a strict Fact-Checking Expert. Always follow the "
                 "required output format exactly.")
VERIFY_CASES = [
    ("support",
     "What is the country of citizenship of Diana Gabaldon?",
     "Diana Gabaldon is a citizen of United Kingdom of Great Britain and Ireland",
     "United Kingdom"),
    ("refute",
     "Which country is Zanjoe Marudo from?",
     "Zanjoe Marudo is associated with the sport of association football",
     "association football"),
]

_here = os.path.dirname(os.path.abspath(__file__))


def load_template(name):
    for p in (os.path.join(_here, "..", "prompts", name),
              os.path.join(_here, "prompts", name)):
        if os.path.exists(p):
            return open(p, encoding="utf-8").read()
    raise SystemExit("template not found: %s (run from the repo)" % name)


TEMPLATES = {"support": load_template("verify_support.txt"),
             "refute": load_template("verify_refute.txt")}


def chat(model, messages, max_tokens=None):
    payload = {"model": model, "messages": messages, "temperature": 0}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    t0 = time.time()
    r = requests.post(BASE_URL + "/v1/chat/completions", headers=HEADERS,
                      json=payload, timeout=TIMEOUT, verify=VERIFY_SSL)
    dt = time.time() - t0
    r.raise_for_status()
    j = r.json()
    msg = j["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    think = msg.get("reasoning_content") or ""
    return content, think, dt, msg, j


def parse_confidence(text):
    m = re.findall(r"Confidence\s*:?\s*\**\s*([01]?\.?[0-9]+)", text or "")
    if not m:
        return None
    try:
        v = float(m[-1])
        return v if 0.0 <= v <= 1.0 else None
    except ValueError:
        return None


models = sys.argv[1:] or ALL_QWEN
print("models under test: %s\n" % models)

# ---- stage A: ping fingerprint (thinking detection) ------------------------
print("=" * 78)
print("STAGE A  ping 'Say OK' -- thinking detector (reasoning_content + latency)")
results = {}
for m in models:
    try:
        content, think, dt, msg, j = chat(m, [{"role": "user",
                                               "content": "Say OK."}],
                                          max_tokens=64)
        psf = msg.get("provider_specific_fields")
        results[m] = {"think_ping": len(think), "lat_ping": dt}
        print("  %-22s %6.1fs  thinking=%5d chars  content=%r%s"
              % (m, dt, len(think), content[:40],
                 ("  psf=%s" % list(psf.keys())) if psf else ""))
    except Exception as e:
        results[m] = {"think_ping": -1, "lat_ping": -1}
        print("  %-22s FAILED: %r" % (m, e))

# ---- stage B: generation fingerprint (training cutoff) ---------------------
print("\n" + "=" * 78)
print("STAGE B  training cutoff -- generation fingerprint (match = same model)")
for m in models:
    try:
        content, think, dt, _, _ = chat(
            m, [{"role": "user",
                 "content": "What is your training data cutoff date? One line, no reasoning."}],
            max_tokens=200)
        results[m]["cutoff"] = content.replace("\n", " ")[:80]
        results[m]["lat_cutoff"] = dt
        print("  %-22s %6.1fs  %s" % (m, dt, results[m]["cutoff"]))
    except Exception as e:
        print("  %-22s FAILED: %r" % (m, e))

# ---- stage C: REAL verify prompts (the decision measurement) ---------------
print("\n" + "=" * 78)
print("STAGE C  real verify prompts (support+refute, faithful to run_llm_verify)")
for m in models:
    for kind, q, facts, ans in VERIFY_CASES:
        query = (TEMPLATES[kind]
                 .replace("<<<<QUESTION>>>>", q)
                 .replace("<<<<FACTS>>>>", facts)
                 .replace("<<<<ANSWER>>>>", ans))
        messages = [{"role": "system", "content": VERIFY_SYSTEM},
                    {"role": "user", "content": query}]
        try:
            content, think, dt, _, _ = chat(m, messages)
            conf = parse_confidence(content)
            results[m].setdefault("verify", {})[kind] = (dt, conf, len(think))
            print("  %-22s %-7s %6.1fs  conf=%-5s thinking=%d chars%s"
                  % (m, kind, dt, conf, len(think),
                     "  <-- NO conf parsed!" if conf is None else ""))
        except Exception as e:
            print("  %-22s %-7s FAILED: %r" % (m, kind, e))

# ---- stage D: /no_think soft switch on the thinking candidates -------------
print("\n" + "=" * 78)
print("STAGE D  '/no_think' soft switch -- same weights, thinking disabled?")
thinking = [m for m in models if results.get(m, {}).get("think_ping", 0) > 0]
if not thinking:
    print("  (no thinking models in this run -- nothing to test)")
for m in thinking:
    for kind, q, facts, ans in VERIFY_CASES:
        query = (TEMPLATES[kind]
                 .replace("<<<<QUESTION>>>>", q)
                 .replace("<<<<FACTS>>>>", facts)
                 .replace("<<<<ANSWER>>>>", ans)) + " /no_think"
        messages = [{"role": "system", "content": VERIFY_SYSTEM + " /no_think"},
                    {"role": "user", "content": query}]
        try:
            content, think, dt, _, _ = chat(m, messages)
            conf = parse_confidence(content)
            base = results.get(m, {}).get("verify", {}).get(kind)
            cmp = ("(was %.1fs, conf %s)" % base[:2]) if base else ""
            print("  %-22s %-7s %6.1fs  conf=%-5s thinking=%d chars  %s"
                  % (m, kind, dt, conf, len(think), cmp))
        except Exception as e:
            print("  %-22s %-7s FAILED: %r" % (m, kind, e))

# ---- summary ---------------------------------------------------------------
print("\n" + "=" * 78)
print("SUMMARY  (identity: match cutoff+behavior across ids;")
print("verifier: want low latency + sane conf on both cases)")
for m in models:
    r = results.get(m, {})
    v = r.get("verify", {})
    row = "  %-22s ping=%5.1fs think=%-4d cutoff=%s" % (
        m, r.get("lat_ping", -1), r.get("think_ping", -1),
        (r.get("cutoff") or "?")[:30])
    if v:
        row += " | verify: " + " ".join(
            "%s %.0fs conf=%s" % (k, x[0], x[1]) for k, x in sorted(v.items()))
    print(row)
