import os
import pickle
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import spacy
import random
from sentence_transformers import SentenceTransformer, util
import re
import openai
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm
import collections
import argparse
import time
import threading
import concurrent.futures
import datetime
import queue
import contextlib
import traceback
import requests


from build_divide_cache import DivideCache

parser = argparse.ArgumentParser()
parser.add_argument('--edit', type=int, default=1)
parser.add_argument('--dataset', type=str, default='MQuAKE-T', choices=['MQuAKE-CF-3k', 'MQuAKE-T'])
parser.add_argument('--tau', type=float, default=0.5, help='confidence gap threshold of the dialectical controller')
parser.add_argument('--max_iter', type=int, default=5, help='max dialectical verification rounds per sub-question')
parser.add_argument('--topk', type=int, default=3, help='top-k entities/relations kept as retrieval candidates')
parser.add_argument('--seed', type=int, default=42, help='random seed for dataset shuffling')
parser.add_argument('--no_dialectical', action='store_true', help='disable dialectical retrieval')
parser.add_argument('--no_chain_verify', action='store_true',
                    help='disable the chain-level meta-reflection + backtrack')
parser.add_argument('--divide_cache', type=str, default='auto')
parser.add_argument('--divide_model', type=str, default='')
parser.add_argument('--verify_model', type=str, default='',
                    help='override the VERIFY_MODEL constant')
parser.add_argument('--api_timeout', type=float, default=120.0,
                    help='item 16 (c): per-request transport timeout (seconds) for fast chat endpoints.')
parser.add_argument('--case_workers', type=int, default=1,
                    help='item 18: number of cases answered CONCURRENTLY')
parser.add_argument('--model_lock', action='store_true',
                    help='item 18 (e): serialize REBEL generate and the DistilBert judges across '
                         'threads. Off by default (inference-only model calls hold per-call state); '
                         'enable if a parallel run ever shows corrupted relation extractions.')

args = parser.parse_args()

TAU = args.tau
MAX_ITER = args.max_iter
TOPK = args.topk
RESCUE_ENTITIES = 5
SUPPORT = 0
REFUTE = 1
CHAIN_SUPPORT = 2  
CHAIN_REFUTE = 3
VERIFY_STATS = {'calls': 0, 'subq': 0, 'chain': 0, 'backtrack': 0, 'backtrack_adopt': 0, 'cache_hits': 0}


API_STATS = {
    'divide': {'calls': 0, 'sec': 0.0},   # live (cache-miss) divides only
    'verify': {'calls': 0, 'sec': 0.0},   # SUPPORT/REFUTE/chain verifier
    'answer': {'calls': 0, 'sec': 0.0},   # LLM fallback answering
    '429':    {'count': 0, 'sec': 0.0},   # quota stalls on any endpoint
    'retry':  {'count': 0, 'sec': 0.0},   # failed attempts + their wall time
}
_STATS_LOCK = threading.Lock()
_RESET_RE = re.compile(
    r'Limit resets at[:\s]+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*(?:UTC|GMT)?',
    re.IGNORECASE)


LOCAL_STATS = {
    'link':  {'calls': 0, 'hits': 0, 'miss_sec': 0.0},   # item 17
    'rebel': {'calls': 0, 'sec': 0.0},                   # REBEL extraction
    'judge': {'calls': 0, 'sec': 0.0},                   # DistilBERT judges
}
_LINK_CACHE = {}
_LINK_LOCK = threading.Lock()

_CTX = threading.local()
_METRICS_LOCK = threading.Lock()
_DIVIDE_LOCK = threading.Lock()
_MODEL_LOCK = threading.Lock()
_WORKER_ERRORS = {'count': 0}


def _bind_graph(qid2name, triples, qid_relations, registry, nlp):
    _CTX.qid2name = qid2name
    _CTX.triples = triples
    _CTX.qid_relations = qid_relations
    _CTX.registry = registry
    _CTX.nlp = nlp


def _thread_nlp():
    nlp = getattr(_CTX, 'nlp', None)
    if nlp is not None:
        return nlp
    print("[parallel] loading worker spaCy pipeline (en_core_web_md + entityLinker)...")
    nlp = spacy.load("../en_core_web_md")
    nlp.add_pipe("entityLinker", last=True)
    _CTX.nlp = nlp
    return nlp


def _model_guard():
    if args.model_lock:
        return _MODEL_LOCK
    return contextlib.nullcontext()


def _rate_limit_sleep(err, attempt=0):
    s = str(err)
    _is_quota = ('rate limit' in s.lower() or '429' in s
                 or 'ratelimit' in type(err).__name__.lower())
    if _is_quota:
        with _STATS_LOCK:
            API_STATS['429']['count'] += 1
        nap = None
        m = _RESET_RE.search(s)
        if m:
            try:
                reset = datetime.datetime.strptime(
                    m.group(1), '%Y-%m-%d %H:%M:%S').replace(
                        tzinfo=datetime.timezone.utc)
                now = datetime.datetime.now(datetime.timezone.utc)
                nap = (reset - now).total_seconds() + 2.0
            except ValueError:
                nap = None
        if nap is None:
            nap = 10.0
        nap = max(10.0, min(nap, 900.0))
        with _STATS_LOCK:
            API_STATS['429']['sec'] += nap
        print("[ratelimit] 429: sleeping %.0fs until the quota resets" % nap)
        time.sleep(nap)
    else:
        time.sleep(10)





ANSWER_API_BASE = "http://localhost:7001/v1"
ANSWER_API_KEY = ""
ANSWER_MODEL = "llama-2-7b"

# ANSWER_API_BASE = "http://localhost:7002/v1"
# ANSWER_API_KEY = ""
# ANSWER_MODEL = "vicuna-7b"

print("ANSWER_MODEL: ",ANSWER_MODEL)



DIVIDE_API_BASE = "https://.../v1"
DIVIDE_API_KEY = ""                          # <-- fill in your campus API key
DIVIDE_MODEL = "deepseek-v4-flash"





VERIFY_API_BASE = "http://localhost:7002/v1"
VERIFY_API_KEY = ""
VERIFY_MODEL = "qwen3-8b" 


print (VERIFY_MODEL,VERIFY_API_BASE)


if args.verify_model:
    VERIFY_MODEL = args.verify_model



if args.divide_model:
    DIVIDE_MODEL = args.divide_model


if not ANSWER_API_KEY or not DIVIDE_API_KEY:
    print("WARNING: ANSWER_API_KEY / DIVIDE_API_KEY is empty -- every request "
          "will be rejected with 401 and retried forever. Fill them in at the "
          "top of dynamic_exp_bridge.py before running.")

print("[config] ANSWER_MODEL: %s | DIVIDE_MODEL: %s | VERIFY_MODEL: %s%s"
      % (ANSWER_MODEL, DIVIDE_MODEL or '<endpoint default>',
         VERIFY_MODEL or ANSWER_MODEL,
         '' if VERIFY_MODEL else ' (same endpoint as answer model)'))
def _looks_reasoning(name):
    return any(k in (name or '').lower() for k in ('pro', 'r1', 'reason', 'thinking', 'qwen3-8b'))

if _looks_reasoning(VERIFY_MODEL or ANSWER_MODEL):
    print("[config] WARNING: the verifier '%s' looks like a REASONING model: each verify "
          "call spends tens of seconds thinking before emitting two confidences, and "
          "verification dominates the wall time. Switch to a fast chat verifier, e.g. "
          "--verify_model qwen3.8-chat (~3-4s/call), unless you are measuring verifier "
          "quality specifically." % (VERIFY_MODEL or ANSWER_MODEL))


VERIFY_REQ_TIMEOUT = 600.0 if _looks_reasoning(VERIFY_MODEL or ANSWER_MODEL) else float(args.api_timeout)
DIVIDE_REQ_TIMEOUT = 600.0 if _looks_reasoning(DIVIDE_MODEL) else float(args.api_timeout)
ANSWER_REQ_TIMEOUT = float(args.api_timeout)
print("[config] request timeouts: verify=%.0fs divide=%.0fs answer=%.0fs "
      "(--api_timeout %.0fs; reasoning models keep 600s)"
      % (VERIFY_REQ_TIMEOUT, DIVIDE_REQ_TIMEOUT, ANSWER_REQ_TIMEOUT, args.api_timeout))


try:
    # openai>=1.0
    from openai import OpenAI as _OpenAI
    _OPENAI_LEGACY = False
    _api_clients = {}

    def _get_client(api_key, api_base):
        ck = (api_key, api_base)
        if ck not in _api_clients:
            _api_clients[ck] = _OpenAI(api_key=api_key or "EMPTY", base_url=api_base, timeout=300.0)
        return _api_clients[ck]
except ImportError:
    # openai<1.0 
    _OPENAI_LEGACY = True
    openai.request_timeout = 300

print(f"[openai compat] legacy mode = {_OPENAI_LEGACY}, "
      f"has ChatCompletion = {hasattr(openai, 'ChatCompletion')}")

def chat_completion(api_key, api_base, model, messages, timeout=None, **kwargs):
    if _OPENAI_LEGACY:
        body = {"model": model, "messages": messages}
        body.update(kwargs)
        _t = float(timeout) if timeout is not None else 300.0
        r = requests.post(api_base.rstrip('/') + "/chat/completions",
                          headers={"Authorization": "Bearer " + (api_key or "")},
                          json=body,
                          timeout=(min(30.0, _t), _t))
        if r.status_code != 200:
            raise RuntimeError("HTTP %d from %s: %s"
                               % (r.status_code, api_base, r.text[:1000]))
        content = r.json()["choices"][0]["message"]["content"]
    else:
        client = _get_client(api_key, api_base)
        if timeout is not None:
            response = client.chat.completions.create(model=model, messages=messages,
                                                      timeout=timeout, **kwargs)
        else:
            response = client.chat.completions.create(model=model, messages=messages, **kwargs)
        content = response.choices[0].message.content

    if content is None or not str(content).strip():
        raise ValueError("empty chat completion content (content=None or blank)")
    return content

def list_first_model(api_key, api_base):
    if _OPENAI_LEGACY:

        r = requests.get(api_base.rstrip('/') + "/models",
                         headers={"Authorization": "Bearer " + (api_key or "")},
                         timeout=(10.0, 60.0))
        if r.status_code != 200:
            raise RuntimeError("HTTP %d from %s: %s"
                               % (r.status_code, api_base, r.text[:500]))
        return r.json()["data"][0]["id"]
    client = _get_client(api_key, api_base)
    return client.models.list().data[0].id

DATASET_SIZE = {'MQuAKE-CF-3k': 3000, 'MQuAKE-T': 1868}


DEVICE = "cuda:0"
ENTITY = 0
REL = 1
tokenizer = AutoTokenizer.from_pretrained("../model/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("../model/rebel-large")
model.to(DEVICE)
TRAIN_PATH = '../train/results/best_model'
ENTITY_TRAIN_PATH = '../train/results_entity_judge/best_model_entity_judge'
bert_tokenizer = DistilBertTokenizer.from_pretrained(TRAIN_PATH)
bert_model = DistilBertForSequenceClassification.from_pretrained(TRAIN_PATH, num_labels=2)
bert_model.to(DEVICE)



entity_bert_tokenizer = DistilBertTokenizer.from_pretrained(ENTITY_TRAIN_PATH)
entity_bert_model = DistilBertForSequenceClassification.from_pretrained(ENTITY_TRAIN_PATH, num_labels=2)
entity_bert_model.to(DEVICE)


def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations

class KB():
    def __init__(self):
        self.relations = []

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def add_relation(self, r):
        if not self.exists_relation(r):
            self.relations.append(r)

    def print(self):
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")

def from_small_text_to_kb(text, verbose=False):
    kb = KB()
    _t0 = time.time()
    with _model_guard():
        # Tokenizer text
        model_inputs = tokenizer(text, max_length=512, padding=True, truncation=True,
                                return_tensors='pt').to(DEVICE)
        # if verbose:
        #     print(f"Num tokens: {len(model_inputs['input_ids'][0])}")

        # Generate
        gen_kwargs = {
            "max_length": 216,
            "length_penalty": 0,
            "num_beams": 3,
            "num_return_sequences": 3
        }
        generated_tokens = model.generate(
            **model_inputs,
            **gen_kwargs,
        )
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    with _STATS_LOCK:
        LOCAL_STATS['rebel']['calls'] += 1
        LOCAL_STATS['rebel']['sec'] += time.time() - _t0

    # create kb
    for sentence_pred in decoded_preds:
        relations = extract_relations_from_model_output(sentence_pred)
        for r in relations:
            kb.add_relation(r)

    return kb

def link_entity(entity, nlp):
    with _LINK_LOCK:
        if entity in _LINK_CACHE:
            LOCAL_STATS['link']['calls'] += 1
            LOCAL_STATS['link']['hits'] += 1
            return _LINK_CACHE[entity]
    _t0 = time.time()
    pattern = r'Q\d+'
    name = entity
    try:
        linking = re.search(pattern, str(nlp(entity.capitalize())._.linkedEntities))
    except:
        linking = re.search(pattern, str(nlp(entity)._.linkedEntities))
    if linking:
        linking = linking.group(0) 
    else:
        linking = name
    _dt = time.time() - _t0
    with _LINK_LOCK:
        LOCAL_STATS['link']['calls'] += 1
        LOCAL_STATS['link']['miss_sec'] += _dt
        _LINK_CACHE[entity] = linking
    return linking


def run_llm_answer(query):
    messages = [{"role":"system","content":"You are an AI assistant that helps people find information. You must answer with ONLY the name of the entity -- never a full sentence, never an explanation, never extra context. The format is 'Answer: <entity name>'."}]
    message_prompt = {"role":"user","content":query}
    messages.append(message_prompt)
    f = 0
    while(f == 0):

        try:
            _t0 = time.time()
            result = chat_completion(ANSWER_API_KEY, ANSWER_API_BASE, ANSWER_MODEL,
                                     messages,
                                     timeout=ANSWER_REQ_TIMEOUT,
                                     temperature=0,
                                     frequency_penalty=0,
                                     presence_penalty=0)
            with _STATS_LOCK:
                API_STATS['answer']['calls'] += 1
                API_STATS['answer']['sec'] += time.time() - _t0
            f = 1
        except Exception as e:
            with _STATS_LOCK:
                API_STATS['retry']['count'] += 1
                API_STATS['retry']['sec'] += time.time() - _t0
            print("openai error (run_llm_answer), retry:", repr(e))
            _rate_limit_sleep(e)
    return result

def predict(question, relation, FLAG):
    _t0 = time.time()
    with _model_guard():
        if FLAG == ENTITY:
            inputs = entity_bert_tokenizer(question, relation, return_tensors='pt', truncation=True, padding='max_length').to(DEVICE)
            outputs = entity_bert_model(**inputs)
        elif FLAG == REL:
            inputs = bert_tokenizer(question, relation, return_tensors='pt', truncation=True, padding='max_length').to(DEVICE)
            outputs = bert_model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = logits.argmax().item()
    with _STATS_LOCK:
        LOCAL_STATS['judge']['calls'] += 1
        LOCAL_STATS['judge']['sec'] += time.time() - _t0
    return predicted_class, logits, probabilities


def get_scores(question, objects, FLAG):
    scores = []
    for object in objects:
        predicted_class, logits, probabilities = predict(question, object, FLAG)
        score = probabilities[0, 1].item()
        if score < 0.5:
            continue
        scores.append((score, object))
    return scores

def score_all(question, objects, FLAG):
    scores = []
    for object in objects:
        predicted_class, logits, probabilities = predict(question, object, FLAG)
        scores.append((probabilities[0, 1].item(), object))
    scores.sort(key=lambda x: (-x[0], str(x[1])))
    return scores

def retrieve_from_graph_greedy(q):
    qid2name, triples, qid_relations, nlp = (_CTX.qid2name, _CTX.triples,
                                             _CTX.qid_relations, _CTX.nlp)
    kb = from_small_text_to_kb(q, verbose=True)
    print(kb.relations)
    kb = kb.relations
    # choose entity

    entity_set = set()
    for k in kb:
        # capitalize
        entity_set.add(k['head'])
        entity_set.add(k['type'])
        entity_set.add(k['tail'])
    if len(entity_set) == 0:
        return None
    scores = get_scores(q, entity_set, ENTITY)
    if len(scores) == 0:
        return None
    scores.sort(key=lambda x: x[0], reverse=True)
    print(q, scores)
    head = scores[0][1]
    # choose rels

    head_linking = link_entity(head, nlp)
    tmp_rels = qid_relations[head_linking]
    if len(tmp_rels) == 0:
        return None
    scores = get_scores(q, tmp_rels, REL)
    if len(scores) == 0:
        return None
    scores.sort(key=lambda x: x[0], reverse=True)
    print(q, scores)
    rel = scores[0][1]
    # find rels
    if rel in tmp_rels:
        # get all posible tail entitys
        tail_linkings = list(triples[(head_linking, rel)])
        # return tail_linkings[0][1] # return sentence
        return qid2name[tail_linkings[0][0]] # return entity
    return None


def parse_confidence(text):
    # parses 'Confidence: <float>' (or any 0.0-1.0 float) from the verifier output
    matches = re.findall(r'[Cc]onfidence[:\s]*([01]?\.\d+|[01])', text)
    if matches:
        try:
            return max(0.0, min(1.0, float(matches[-1])))
        except ValueError:
            pass
    matches = re.findall(r'([01]\.\d+)', text)
    if matches:
        return max(0.0, min(1.0, float(matches[-1])))
    return 0.0


def run_llm_verify(question, facts, answer, FLAG):
    with _VERIFY_LOCK:
        VERIFY_STATS['calls'] += 1
    if FLAG == SUPPORT:
        template = verify_support_prompt
    elif FLAG == REFUTE:
        template = verify_refute_prompt
    elif FLAG == CHAIN_SUPPORT:
        template = verify_chain_support_prompt
    else:
        template = verify_chain_refute_prompt
    query = template.replace("<<<<QUESTION>>>>", question).replace("<<<<FACTS>>>>", facts).replace("<<<<ANSWER>>>>", answer)
    messages = [{"role":"system","content":"You are a strict Fact-Checking Expert. Always follow the required output format exactly."}]
    messages.append({"role":"user","content":query})
    f = 0
    while(f == 0):
        try:
            _t0 = time.time()
            result = chat_completion(VERIFY_API_KEY or ANSWER_API_KEY,
                                     VERIFY_API_BASE or ANSWER_API_BASE,
                                     VERIFY_MODEL or ANSWER_MODEL,
                                     messages,
                                     timeout=VERIFY_REQ_TIMEOUT,
                                     temperature = 0,
                                     frequency_penalty=0,
                                     presence_penalty=0)
            with _STATS_LOCK:
                API_STATS['verify']['calls'] += 1
                API_STATS['verify']['sec'] += time.time() - _t0
            f = 1
        except Exception as e:
            with _STATS_LOCK:
                API_STATS['retry']['count'] += 1
                API_STATS['retry']['sec'] += time.time() - _t0
            print("openai error (run_llm_verify), retry:", repr(e))
            _rate_limit_sleep(e)
    return result



_VERIFY_CACHE = {}
_VERIFY_LOCK = threading.Lock()
_VERIFY_POOL = None


def _verify_pool():
    global _VERIFY_POOL
    if _VERIFY_POOL is None:
        _VERIFY_POOL = concurrent.futures.ThreadPoolExecutor(
            max_workers=2 * max(1, args.case_workers))
    return _VERIFY_POOL


def verify_pair(question, facts, answer, flag_sup=SUPPORT, flag_ref=REFUTE):
    key = (question, facts, answer, flag_sup, flag_ref)
    with _VERIFY_LOCK:
        if key in _VERIFY_CACHE:
            VERIFY_STATS['cache_hits'] += 1
            return _VERIFY_CACHE[key]
    pool = _verify_pool()
    f_sup = pool.submit(run_llm_verify, question, facts, answer, flag_sup)
    f_ref = pool.submit(run_llm_verify, question, facts, answer, flag_ref)
    c_plus = parse_confidence(f_sup.result())
    c_minus = parse_confidence(f_ref.result())
    with _VERIFY_LOCK:
        _VERIFY_CACHE[key] = (c_plus, c_minus)
    return c_plus, c_minus


def build_candidates(q, entity_ranked, strict):
    qid2name, triples, qid_relations, nlp = (_CTX.qid2name, _CTX.triples,
                                             _CTX.qid_relations, _CTX.nlp)
    candidates = []
    seen_fact = set()
    for e_score, head in entity_ranked:
        head_linking = link_entity(head, nlp)
        tmp_rels = qid_relations[head_linking]
        if len(tmp_rels) == 0:
            continue
        if strict:
            rel_scores = get_scores(q, tmp_rels, REL)
            rel_scores.sort(key=lambda x: (-x[0], str(x[1])))
            chosen = rel_scores[:TOPK]
        else:
            chosen = score_all(q, tmp_rels, REL)[:1]
        for r_score, rel in chosen:
            for tail_linking, sentence in triples[(head_linking, rel)]:
                if sentence in seen_fact:
                    continue
                seen_fact.add(sentence)
                answer = qid2name[tail_linking]
                candidates.append({
                    'answer': answer,
                    'fact': sentence,
                    'score': e_score * r_score,
                    'head': head,
                    'rel': rel
                })
    return candidates


def print_miss_diagnostics(q, entity_set):
    qid_relations, nlp = _CTX.qid_relations, _CTX.nlp
    print("[miss] retrieval failed entirely for: " + q)
    for e_score, head in score_all(q, entity_set, ENTITY)[:RESCUE_ENTITIES]:
        head_linking = link_entity(head, nlp)
        rels = qid_relations[head_linking]
        if len(rels) == 0:
            print("[miss]   entity '%s' (score=%.2f) -> link '%s': NO relations in graph (linking failed or entity absent)"
                  % (head, e_score, head_linking))
            continue
        rel_ranked = score_all(q, rels, REL)
        rel_str = ", ".join("'%s' (%.2f)" % (r, s) for s, r in rel_ranked[:5])
        print("[miss]   entity '%s' (score=%.2f) -> link '%s': %d relations; ranked: %s"
              % (head, e_score, head_linking, len(rels), rel_str))


def _alt_pack(candidates, skip_idx, refuted):
    alts = []
    for i, c in enumerate(candidates):
        if i == skip_idx or c['answer'] in refuted:
            continue
        alts.append({'answer': c['answer'], 'fact': c['fact']})
        if len(alts) >= 3:
            break
    return alts


def dialectical_retrieve(q):
    with _VERIFY_LOCK:   
        VERIFY_STATS['subq'] += 1
    # ---- Retriever: relation-grounded top-k candidate generation ----
    kb = from_small_text_to_kb(q, verbose=True)
    print(kb.relations)
    entity_set = set()
    for k in kb.relations:
        entity_set.add(k['head'])
        entity_set.add(k['type'])
        entity_set.add(k['tail'])
    if len(entity_set) == 0:
        return None, False, None, []
    entity_scores = score_all(q, entity_set, ENTITY)
    strict_entities = [(s, e) for s, e in entity_scores if s >= 0.5][:TOPK]
    candidates = build_candidates(q, strict_entities, strict=True)


    if len(candidates) == 0:
        relaxed_entities = entity_scores[:RESCUE_ENTITIES]
        candidates = build_candidates(q, relaxed_entities, strict=False)
        if len(candidates) > 0:
            print("[rescue] strict channel empty, %d relaxed candidate(s) -> dialectical check"
                  % len(candidates))
    if len(candidates) == 0:
        print_miss_diagnostics(q, entity_set)
        return None, False, None, []
    # deterministic order for reproducibility
    candidates.sort(key=lambda x: (-x['score'], x['fact']))
    print(q + " candidates: " + str([c['fact'] for c in candidates[:MAX_ITER]]))

    # ---- Reasoner + Controller: dialectical bidirectional verification loop ----
    evidence = []
    best = None  # (c_plus, c_minus, idx) of the strongest supporting argument
    refuted = set()  # answers confidently refuted at hop level
    for t in range(min(MAX_ITER, len(candidates))):
        cand = candidates[t]
        evidence.append("- " + cand['fact'])
        facts = "\n".join(evidence)
        c_plus, c_minus = verify_pair(q, facts, cand['answer'])
        delta = abs(c_plus - c_minus)
        print(f"[dialectical] t={t+1} c+={c_plus:.2f} c-={c_minus:.2f} delta={delta:.2f}")
        if best is None or (c_plus - c_minus) > (best[0] - best[1]):
            best = (c_plus, c_minus, t)
        # controller: confident agreement in one direction -> early termination
        if delta >= TAU:
            if c_plus > c_minus:
                print("Found (verified): " + str(cand['answer']))
                return cand['answer'], True, cand['fact'], _alt_pack(candidates, t, refuted)
            # candidate refuted -> continue scanning the next candidate
            refuted.add(cand['answer'])

    if best is not None and best[0] >= best[1]:
        chosen = candidates[best[2]]
        print("Found (adjudicated): " + str(chosen['answer']))
        return chosen['answer'], False, chosen['fact'], _alt_pack(candidates, best[2], refuted)
    print("Unresolved by dialectical check, fall back to LLM")

    return None, False, None, _alt_pack(candidates, -1, refuted)


def retrieve_from_graph(q):
    if args.no_dialectical:
        return retrieve_from_graph_greedy(q), False, None, []
    return dialectical_retrieve(q)


def position_bridge_fallback(subq):
    if not re.search(r'(?i)\bsport\b', subq):
        return None
    kb = from_small_text_to_kb(subq)
    entity_set = set()
    for k in kb.relations:
        for v in (k['head'], k['type'], k['tail']):
            if v:
                entity_set.add(v)
    if not entity_set:
        return None
    ranked = score_all(subq, entity_set, ENTITY)
    if not ranked or ranked[0][0] < 0.5:
        return None
    person = ranked[0][1]
    if person.lower() in ('sport', 'sports', 'athlete', 'team', 'position'):
        return None
    pos, _, _, _ = retrieve_from_graph('What position does %s play?' % person)
    if pos is None:
        return None
    print('[pos-bridge] %s -> position %s -> sport' % (person, pos))
    sport, _, _, _ = retrieve_from_graph('Which sport is %s associated with?' % pos)
    if sport is None:
        return None
    return pos, sport


def verify_gold_path(entities, hops, path):
    if len(entities) != hops:
        return False
    for i in range(hops):
        if entities[i].lower() == path[i]["answer"].lower() or path[i]["answer"].lower() in entities[i].lower() or any(entities[i].lower() in p.lower() for p in path[i]["answer_alias"]):
            continue
        else:
            return False
    return True

def run_llm_divide(query):
    if DIVIDE_MODEL:
        divide_model = DIVIDE_MODEL
    else:
        divide_model = list_first_model(DIVIDE_API_KEY, DIVIDE_API_BASE)
    messages = [{"role":"system","content":"You are a question decomposition assistant. You decompose a multi-hop question into single-hop sub-questions, one per line, using [ENT] placeholders for entities resolved by previous sub-questions. Each sub-question must match exactly one knowledge-graph fact: never merge two facts into one, and never skip an intermediate lookup. When unsure whether a line covers one or two facts, split it into two lines -- under-splitting is worse than over-splitting, and typical questions need 2 to 4 sub-questions. If the asked relation plausibly belongs not to the named entity but to a related one (a song has no manager -- its performer does), first resolve that intermediate entity with its own sub-question. You output only the sub-questions, with no numbering and no extra text."}]
    message_prompt = {"role":"user","content":query}
    messages.append(message_prompt)
    f = 0
    while(f == 0):
        try:
            _t0 = time.time()
            result = chat_completion(DIVIDE_API_KEY, DIVIDE_API_BASE, divide_model,
                                     messages,
                                     timeout=DIVIDE_REQ_TIMEOUT,
                                     temperature=0,
                                     stop=["\nQuestion:"],
                                     frequency_penalty=0,
                                     presence_penalty=0)
            with _STATS_LOCK:
                API_STATS['divide']['calls'] += 1
                API_STATS['divide']['sec'] += time.time() - _t0
            f = 1
        except Exception as e:
            with _STATS_LOCK:
                API_STATS['retry']['count'] += 1
                API_STATS['retry']['sec'] += time.time() - _t0
            print("openai error (run_llm_divide), retry:", repr(e))
            _rate_limit_sleep(e)
    return result

def _mentions(text, name):
    tl = (text or '').strip().lower()
    nl = (name or '').strip().lower()
    if not tl or not nl:
        return False
    return tl == nl or nl in tl or (tl in nl and len(tl) >= 4)



GENERIC_RELS = {'instance of', 'facet of', 'main subject', 'applies to jurisdiction',
                'named after', 'subclass of', 'part of', 'country', 'type'}



def add_edit_to_graph(edit, qid2name, triples, qid_relations, nlp, clear_flag):
    subject = edit["subject"]
    target_new = edit["target_new"]["str"]
    sentence = edit["prompt"].format(subject) + ' ' + target_new
    kb = from_small_text_to_kb(sentence, verbose=True)

    rels, junk_rels = [], []
    for triple in kb.relations:
        head, rel, tail = triple['head'], triple['type'], triple['tail']
        if not head or not rel or not tail:
            continue
        hs = _mentions(head, subject)
        ts = _mentions(tail, subject)
        if hs and ts:
            continue
        if not hs and not ts:
            junk_rels.append(rel.strip())
            continue
        rel = rel.strip()
        if rel and rel not in rels:
            rels.append(rel)

    specific = [r for r in rels if r.lower() not in GENERIC_RELS]
    if specific:
        rels = specific
    if not rels:
        rels = [r for r in junk_rels if r][:1]
    if not rels:
        fallback_rel = edit["prompt"].replace('{}', '').strip()
        rels = [fallback_rel or 'fact']

    head_linking = link_entity(subject, nlp)
    if not re.match(r'Q\d+', str(head_linking)):
        head_linking = subject
    tail_linking = link_entity(target_new, nlp)
    if not re.match(r'Q\d+', str(tail_linking)):
        tail_linking = target_new

    qid2name[head_linking] = subject
    qid2name[tail_linking] = target_new
    for rel in rels:
        key = (head_linking, rel)
        if clear_flag is not None and clear_flag[key] == 0:
            # remove origin rel (edit overrides any other fact under this key)
            triples[key].clear()
            clear_flag[key] = 1
        triples[key].add((tail_linking, sentence))
        qid_relations[head_linking].add(rel)


    _entry = {
        'target': target_new,
        'fact': sentence,
        'rels': list(rels) + [edit["prompt"].replace('{}', '').strip()],
        'stoks': [t for t in re.findall(r'[a-z0-9]+', subject.lower()) if len(t) > 2],
    }

    _dup = [e for e in _CTX.registry[subject.lower()] if e['fact'] == sentence]
    if _dup:
        _dup[0].update(_entry)
    else:
        _CTX.registry[subject.lower()].append(_entry)


def construct_graph(batch, qid2name, triples, qid_relations, nlp, quiet=False):
    _it = range(len(batch)) if quiet else tqdm(range(len(batch)))
    for i in _it:
        d = batch[i]
        for edit in d["requested_rewrite"]:
            add_edit_to_graph(edit, qid2name, triples, qid_relations, nlp, None)


def modify_graph(d, qid2name, triples, qid_relations, nlp):
    clear_flag = collections.defaultdict(int)
    for edit in d["requested_rewrite"]:
        add_edit_to_graph(edit, qid2name, triples, qid_relations, nlp, clear_flag)


def _subject_in_question(subject, q):
    if _mentions(q, subject):
        return True
    qtok = set(re.findall(r'[a-z0-9]+', (q or '').lower()))
    stok = [t for t in re.findall(r'[a-z0-9]+', (subject or '').lower()) if len(t) > 2]
    if not stok:
        return False
    return all(t in qtok for t in stok)


_EDIT_STOPWORDS = {'the', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when',
                   'is', 'are', 'was', 'were', 'does', 'do', 'did', 'has', 'have',
                   'had', 'a', 'an', 'in', 'of', 'for', 'that', 'this', 'their',
                   'his', 'her', 'its', 'with', 'and', 'or', 'to', 'from', 'by',
                   'on', 'at', 'as', 'be', 'been', 'name', 'current', 'present',
                   'call', 'calls', 'called', 'also', 'known', 'plays'}


def edit_lookup(q):
    if not _CTX.registry:
        return None
    qtok = [t for t in re.findall(r'[a-z0-9]+', q.lower())
            if len(t) > 2 and t not in _EDIT_STOPWORDS]
    if not qtok:
        return None
    qtok_set = set(qtok) | {t[:-1] for t in qtok if t.endswith('s')}
    qfull = set(re.findall(r'[a-z0-9]+', q.lower()))
    best = None  # (overlap, entry)
    seen_fact = set()
    for subj, entries in _CTX.registry.items():
        stoks = entries[0].get('stoks') if entries else None
        if stoks and not any(t in qfull or (t + 's') in qfull for t in stoks):
            continue
        if not _subject_in_question(subj, q):
            continue
        for entry in entries:
            if entry['fact'] in seen_fact:
                continue
            seen_fact.add(entry['fact'])
            reltok = set()
            for r in entry['rels']:
                for t in re.findall(r'[a-z0-9]+', r.lower()):
                    if len(t) > 2 and t not in _EDIT_STOPWORDS:
                        reltok.add(t)
                        if t.endswith('s'):
                            reltok.add(t[:-1])
            overlap = len(qtok_set & reltok)
            if best is None or overlap > best[0]:
                best = (overlap, entry)
    if best is None:
        return None
    overlap, entry = best
    print("[edit-bypass] subject matched (overlap=%d), candidate '%s' -> dialectical check"
          % (overlap, entry['target']))
    c_plus, c_minus = verify_pair(q, "- " + entry['fact'], entry['target'])
    print(f"[edit-bypass] c+={c_plus:.2f} c-={c_minus:.2f}")
    if c_plus - c_minus >= TAU:
        return entry
    print("[edit-bypass] rejected by dialectical check (delta gate), fall back to LLM")
    return None


def chain_verify(question, answer, fact_chain):
    with _VERIFY_LOCK:   
        VERIFY_STATS['chain'] += 1
    facts = "\n".join("- " + f for f in fact_chain if f)
    c_plus, c_minus = verify_pair(question, facts, answer, CHAIN_SUPPORT, CHAIN_REFUTE)
    print(f"[chain-verify] c+={c_plus:.2f} c-={c_minus:.2f}")
    return c_plus, c_minus


def resolve_chain(sub_questions, entity):
    entity_list = []
    hop_alts = []     # per sub-question: untried / unrefuted candidate answers
    hop_facts = []    # per sub-question: fact sentence grounding the answer
    hop_entries = []  # per sub-question: entity_list entries it contributed
    for si, subq in enumerate(sub_questions):
        # replace with entity
        subq = subq.replace("[ENT]", entity)
        retrieve, verified, fact, alts = retrieve_from_graph(subq)
        bridge_hop = None
        if not args.no_dialectical and not verified:
            bridged = position_bridge_fallback(subq)
            if bridged is not None:
                bridge_hop, retrieve = bridged
                verified = True
                fact = None  
                print("Found (pos-bridge): " + str(bridge_hop) + " -> " + str(retrieve))

        if retrieve is None:
            prompt = answer_prompt.replace("<<<<QUESTION>>>>", subq)
            # not found, pass to LLM
            output = run_llm_answer(prompt)

            match = re.search(r'Answer: (.*)', output)
            if match:
                entity = match.group(1).strip()
            else:
                entity = output.strip()
            print("Not found in graph: " + entity)
        else:
            entity = retrieve
            print("Found: " + entity)
        entries = []
        if bridge_hop is not None:
            entries.append(bridge_hop)
        entries.append(entity)
        entity_list.extend(entries)
        hop_alts.append(alts)
        hop_facts.append(fact)
        hop_entries.append(entries)
    return entity, entity_list, hop_alts, hop_facts, hop_entries



answer_prompt_path = '../prompts/answer_time_llama2.txt' \
    if ANSWER_MODEL == "llama-2-7b"  else '../prompts/answer_time.txt'
with open(answer_prompt_path, 'r') as p:
    answer_prompt = p.read()

with open('../prompts/divide_api_bridge4.txt','r') as f:
    divide_prompt = f.read()

with open('../prompts/verify_support.txt','r') as f:
    verify_support_prompt = f.read()

with open('../prompts/verify_refute.txt','r') as f:
    verify_refute_prompt = f.read()

with open('../prompts/verify_chain_support.txt','r') as f:
    verify_chain_support_prompt = f.read()

with open('../prompts/verify_chain_refute.txt','r') as f:
    verify_chain_refute_prompt = f.read()



divide_cache = None
if args.divide_cache != 'off':
    if args.divide_cache == 'auto':
        _divide_cache_path = DivideCache.default_path(args.dataset, DIVIDE_MODEL)
        if not os.path.exists(_divide_cache_path):
            _divide_cache_path = None
            print("[divide-cache] no cache file for (dataset=%s, model=%s); dividing live. "
                  "Run build_divide_cache.py to precompute."
                  % (args.dataset, DIVIDE_MODEL or '<endpoint default>'))
    else:
        _divide_cache_path = args.divide_cache
    if _divide_cache_path:
        divide_cache = DivideCache(_divide_cache_path)
        divide_cache.set_prompt(divide_prompt)
        _n_loaded = divide_cache.load()
        divide_cache.meta.update({
            'dataset': args.dataset,
            'model': DIVIDE_MODEL,
            'prompt_file': 'divide_api_bridge4.txt',
            'prompt_sha1': divide_cache.prompt_sha1,
        })
        print("[divide-cache] loaded %d entries (%d usable with the current divide "
              "template) from %s" % (_n_loaded, divide_cache.usable_count(), _divide_cache_path))



import time

total = correct = 0
cor = tot = 0
ver_cor = 0
cor_list = [0,0,0]
ver_cor_list = [0,0,0]
tot_list = [0,0,0]
random.seed(args.seed)
with open("../datasets/" + args.dataset + ".json", 'r') as input:
    data = json.load(input)

random.shuffle(data)

batch_edits = args.edit
batch_size = DATASET_SIZE[args.dataset] // batch_edits
batch_data = []
idx = 0
for _ in range(batch_size):
    b = []
    for i in range(batch_edits):
        if idx < len(data):
            b.append(data[idx])
        idx += 1
    batch_data.append(b)
if not batch_data:
    batch_data = [list(data)]
    print("[all-edited] --edit %d >= %d cases in %s: ONE graph with all cases"
          % (args.edit, len(data), args.dataset))

nlp = None
if args.case_workers <= 1:
    nlp = spacy.load("../en_core_web_md") # entity linking model
    nlp.add_pipe("entityLinker", last=True)

start_time = time.time()
print("len batch_data = ",len(batch_data))

_CASE_BAR = None   # parallel modes only: one shared case-level progress bar
_BAR_LOCK = threading.Lock()


def _bar_tick():
    if _CASE_BAR is not None:
        with _BAR_LOCK:
            _CASE_BAR.update(1)


def run_case(d, batch_t0):
    global cor, tot, ver_cor
    with _METRICS_LOCK:
        tot += 1
        tot_list[len(d["single_hops"]) - 2] += 1
    found_ans = gold_path = False
    for q in d['questions']:
        entity_list = []
        # divide into several question
        prompt = divide_prompt.replace("<<<<QUESTION>>>>", q)


        _hit = False
        if divide_cache is not None:
            with _DIVIDE_LOCK:
                _hit = divide_cache.has(q)
                if _hit:
                    output = divide_cache.get_output(q)
        if _hit:
            print("[divide-cache] hit for: " + q)
        else:
            output = run_llm_divide(prompt)
            if divide_cache is not None:
                with _DIVIDE_LOCK:
                    divide_cache.put(q, output, model=DIVIDE_MODEL)
                    divide_cache.maybe_save(20)
                print("[divide-cache] miss -> live divide, recorded")

        sub_questions = []
        for s in output.split('\n'):
            s = re.sub(r'(?i)^subquestions?\s*:?\s*', '', s.strip())
            s = re.sub(r'^\s*(?:\d+[\.\)]\s*|[-\*]\s*)', '', s).strip()
            if s:
                sub_questions.append(s)
        print("subquesions: " +  str(sub_questions))

        ans, entity_list, hop_alts, hop_facts, hop_entries = resolve_chain(
            sub_questions, "")


        if not args.no_dialectical and not args.no_chain_verify \
                and ans and any(hop_alts):
            c_plus, c_minus = chain_verify(
                q, ans, [f for f in hop_facts if f])
            if c_minus - c_plus >= TAU:
                for j in range(len(hop_alts) - 1, -1, -1):
                    if not hop_alts[j]:
                        continue
                    with _VERIFY_LOCK:   # item 18 (d)
                        VERIFY_STATS['backtrack'] += 1
                    alt = hop_alts[j][0]
                    print("[backtrack] chain rejected; hop %d '%s' -> alternative '%s'"
                          % (j + 1, hop_entries[j][-1], alt['answer']))
                    prefix = [e for lst in hop_entries[:j] for e in lst]
                    if j + 1 < len(sub_questions):
                        new_ans, tail_list, _, tail_facts, _ = resolve_chain(
                            sub_questions[j + 1:], alt['answer'])
                        new_list = prefix + [alt['answer']] + tail_list
                    else:
                        new_ans = alt['answer']
                        new_list = prefix + [alt['answer']]
                    new_facts = [f for f in hop_facts[:j] if f] \
                        + ([alt['fact']] if alt.get('fact') else [])
                    if j + 1 < len(sub_questions):
                        new_facts += [f for f in tail_facts if f]
                    cp2, cm2 = chain_verify(q, new_ans, new_facts)
                    if cm2 - cp2 < TAU:
                        # not confidently rejected: strictly better than a
                        # trajectory the controller just rejected
                        with _VERIFY_LOCK:   # item 18 (d)
                            VERIFY_STATS['backtrack_adopt'] += 1
                        ans, entity_list = new_ans, new_list
                        print("[backtrack] adopted backtracked chain, answer '%s'" % ans)
                    else:
                        print("[backtrack] backtracked chain also rejected; keeping original")
                    break


        # if the answer is correct -> positive instance for Acc
        if ans.lower() == d["new_answer"].lower() or d["new_answer"].lower() in ans.lower() or any(ans.lower() in k.lower() for k in d["new_answer_alias"]):
            if not found_ans:
                with _METRICS_LOCK:
                    cor += 1
                    cor_list[len(d["new_single_hops"]) - 2] += 1
                found_ans = True


            gold_path = verify_gold_path(entity_list, len(d["single_hops"]), d["new_single_hops"])

            if gold_path:
                with _METRICS_LOCK:
                    ver_cor += 1
                    ver_cor_list[len(d["single_hops"]) - 2] += 1
                break



    with _STATS_LOCK:
        _api_snapshot = {k: dict(v) for k, v in API_STATS.items()}
        _link = dict(LOCAL_STATS['link'])
        _rebel = dict(LOCAL_STATS['rebel'])
        _judge = dict(LOCAL_STATS['judge'])
    with _METRICS_LOCK:
        # print(f"Total: {total}, Correct: {correct},  Total: {correct / total}")
        print(f'Acc = {cor / tot} ({cor} / {tot})')
        print(f'Hop-Acc = {ver_cor / tot} ({ver_cor} / {tot})')
        if VERIFY_STATS['subq'] > 0:
            print(f'Verify calls = {VERIFY_STATS["calls"]} ({VERIFY_STATS["calls"] / VERIFY_STATS["subq"]:.2f} per sub-question, {VERIFY_STATS["cache_hits"]} served from cache)')
            print(f'Chain verifies = {VERIFY_STATS["chain"]} | backtracks = {VERIFY_STATS["backtrack"]} ({VERIFY_STATS["backtrack_adopt"]} adopted)')
        print(f'2-hop-tot = {tot_list[0]} 2-hop-cor = {cor_list[0]} 2-hop-acc = {ver_cor_list[0]}')
        print(f'3-hop-tot = {tot_list[1]} 3-hop-cor = {cor_list[1]} 3-hop-acc = {ver_cor_list[1]}')
        print(f'4-hop-tot = {tot_list[2]} 4-hop-cor = {cor_list[2]} 4-hop-acc = {ver_cor_list[2]}')

        end_batch_time = time.time()
        print("batch_time: ",end_batch_time-batch_t0)
        print("current_total_time: ", end_batch_time-start_time )

        _misses = max(1, _link['calls'] - _link['hits'])
        _link_saved = _link['hits'] * (_link['miss_sec'] / _misses)
        _acc = (_api_snapshot['divide']['sec'] + _api_snapshot['verify']['sec']
                + _api_snapshot['answer']['sec'] + _api_snapshot['429']['sec']
                + _api_snapshot['retry']['sec'])
        print('[timing] divide: %d calls / %.0fs | verify: %d calls / %.0fs | '
              'answer: %d calls / %.0fs | 429 stalls: %d / %.0fs | '
              'failed attempts: %d / %.0fs | link-cache: %d/%d hits (~%.0fs saved) | '
              'rebel: %d calls / %.0fs | judge: %d calls / %.0fs | '
              'local+overlap remainder: %.0fs / batch wall %.0fs'
              % (_api_snapshot['divide']['calls'], _api_snapshot['divide']['sec'],
                 _api_snapshot['verify']['calls'], _api_snapshot['verify']['sec'],
                 _api_snapshot['answer']['calls'], _api_snapshot['answer']['sec'],
                 _api_snapshot['429']['count'], _api_snapshot['429']['sec'],
                 _api_snapshot['retry']['count'], _api_snapshot['retry']['sec'],
                 _link['hits'], _link['calls'], _link_saved,
                 _rebel['calls'], _rebel['sec'],
                 _judge['calls'], _judge['sec'],
                 max(0.0, (end_batch_time - batch_t0) - _acc),
                 end_batch_time - batch_t0))



if args.case_workers <= 1:
    for batch_idx,batch in tqdm(enumerate(batch_data), total=len(batch_data)):
        start_batch_time = time.time()

        qid2name = collections.defaultdict(str)
        triples = collections.defaultdict(set)
        qid_relations = collections.defaultdict(set)
        _bind_graph(qid2name, triples, qid_relations,
                    collections.defaultdict(list), nlp)
        # construct knowledge graph
        construct_graph(batch, qid2name, triples, qid_relations, nlp)

        for i in tqdm(range(len(batch))):
            d = batch[i]
            # modify knowledge graph
            modify_graph(d, qid2name, triples, qid_relations, nlp)

            print(f"[graph] {sum(1 for v in qid_relations.values() if v)} linked entities, "
                  f"{sum(len(v) for v in triples.values())} facts in graph")

            run_case(d, start_batch_time)
elif len(batch_data) == 1:
    # ---- single-batch (all-edited) regime: producer + snapshot workers ----
    batch = batch_data[0]
    start_batch_time = time.time()
    qid2name = collections.defaultdict(str)
    triples = collections.defaultdict(set)
    qid_relations = collections.defaultdict(set)
    _registry = collections.defaultdict(list)
    _bind_graph(qid2name, triples, qid_relations, _registry, _thread_nlp())
    print("[parallel] single-batch (all-edited) regime: serial producer "
          "(modify + snapshot) with %d answering worker(s); each case is "
          "answered against the exact graph state the serial loop would see"
          % args.case_workers)
    construct_graph(batch, qid2name, triples, qid_relations, _CTX.nlp)

    _CASE_BAR = tqdm(total=len(batch))
    _case_q = queue.Queue(maxsize=args.case_workers * 2)

    def _graph_snapshot():
        return (dict(qid2name),
                {k: set(v) for k, v in triples.items()},
                {k: set(v) for k, v in qid_relations.items()},
                {s: [dict(e) for e in v] for s, v in _registry.items()})

    def _snapshot_worker():
        nlp_w = _thread_nlp()
        while True:
            item = _case_q.get()
            if item is None:
                break
            d, snap = item
            try:
                _bind_graph(snap[0], snap[1], snap[2], snap[3], nlp_w)
                print(f"[graph] {sum(1 for v in snap[2].values() if v)} linked entities, "
                      f"{sum(len(v) for v in snap[1].values())} facts in graph")
                run_case(d, start_batch_time)
            except Exception:

                with _STATS_LOCK:
                    _WORKER_ERRORS['count'] += 1
                print("[parallel] worker exception on a case (counted as wrong), "
                      "continuing with the next:")
                traceback.print_exc()
            finally:
                _bar_tick()

    _workers = [threading.Thread(target=_snapshot_worker, daemon=True)
                for _ in range(args.case_workers)]
    for _t in _workers:
        _t.start()
    _producer_error = []
    try:
        for d in batch:
            # modify knowledge graph, then freeze this case's exact view
            modify_graph(d, qid2name, triples, qid_relations, _CTX.nlp)
            _case_q.put((d, _graph_snapshot()))
    except Exception:
        _producer_error.append(traceback.format_exc())
        print("[parallel] PRODUCER FAILED -- draining workers before aborting:")
        print(_producer_error[-1])
    finally:
        for _ in _workers:
            _case_q.put(None)
        for _t in _workers:
            _t.join()
    if _producer_error:
        raise RuntimeError("[parallel] producer failed; workers drained, aborting:\n"
                           + _producer_error[-1])
    _CASE_BAR.close()
else:
    # ---- multi-batch regime: one worker per batch, cases serial inside ----
    print("[parallel] multi-batch regime: %d batch(es) x ~%d cases, %d worker(s); "
          "batches are independent graphs so per-case semantics match the "
          "serial run exactly" % (len(batch_data), len(batch_data[0]),
                                  args.case_workers))
    _CASE_BAR = tqdm(total=sum(len(b) for b in batch_data))
    _batch_q = queue.Queue()
    for _b in batch_data:
        _batch_q.put(_b)
    for _ in range(args.case_workers):
        _batch_q.put(None)

    def _batch_worker():
        global tot   # the failed-batch handler below rebinds the module counter
        nlp_w = _thread_nlp()
        while True:
            _batch = _batch_q.get()
            if _batch is None:
                break
            try:
                _qid2name = collections.defaultdict(str)
                _triples = collections.defaultdict(set)
                _qid_relations = collections.defaultdict(set)
                _bind_graph(_qid2name, _triples, _qid_relations,
                            collections.defaultdict(list), nlp_w)
                construct_graph(_batch, _qid2name, _triples, _qid_relations,
                                nlp_w, quiet=True)
                print("[parallel] graph built for a batch of %d cases "
                      "(%d linked entities, %d facts)"
                      % (len(_batch),
                         sum(1 for v in _qid_relations.values() if v),
                         sum(len(v) for v in _triples.values())))
                _batch_t0 = time.time()
                for d in _batch:
                    try:
                        modify_graph(d, _qid2name, _triples, _qid_relations, nlp_w)
                        print(f"[graph] {sum(1 for v in _qid_relations.values() if v)} linked entities, "
                              f"{sum(len(v) for v in _triples.values())} facts in graph")
                        run_case(d, _batch_t0)
                    except Exception:
                        with _STATS_LOCK:
                            _WORKER_ERRORS['count'] += 1
                        print("[parallel] worker exception on a case (counted as wrong "
                              "if run_case had started), continuing:")
                        traceback.print_exc()
                    finally:
                        _bar_tick()
            except Exception:

                with _METRICS_LOCK:
                    for d in _batch:
                        tot += 1
                        tot_list[len(d["single_hops"]) - 2] += 1
                with _STATS_LOCK:
                    _WORKER_ERRORS['count'] += len(_batch)
                print("[parallel] graph build failed for a batch of %d cases "
                      "(all counted as wrong), continuing with the next batch:"
                      % len(_batch))
                traceback.print_exc()

    _workers = [threading.Thread(target=_batch_worker, daemon=True)
                for _ in range(args.case_workers)]
    for _t in _workers:
        _t.start()
    for _t in _workers:
        _t.join()
    _CASE_BAR.close()

if args.case_workers > 1:
    print("[parallel] worker exceptions: %d case(s) affected" % _WORKER_ERRORS['count'])

if divide_cache is not None:
    divide_cache.save()
    print("[divide-cache] saved %d entries to %s" % (len(divide_cache.entries), divide_cache.path))

end_time = time.time()
print("total time: " , (end_time-start_time)/60, " min")
