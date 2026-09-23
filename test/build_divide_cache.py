import os
import sys
import re
import json
import time
import hashlib
import argparse
import collections

import openai

try:
    from tqdm import tqdm
except ImportError:  
    def tqdm(x, **kwargs):
        return x


DIVIDE_API_BASE = ""
DIVIDE_API_KEY = ""      
DIVIDE_MODEL_DEFAULT = "deepseek-v4-flash"



print("DIVIDE_API_BASE: ", DIVIDE_API_BASE)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPT_FILE = os.path.join(_HERE, '..', 'prompts', 'divide_api_bridge4.txt')
DEFAULT_DATASET_DIR = os.path.join(_HERE, '..', 'datasets')


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
    # openai<1.0 (e.g. 0.26.1)
    _OPENAI_LEGACY = True
    # transport timeout for the old SDK: module-level only. A per-call
    # request_timeout kwarg is NOT understood by 0.26.x -- it would be
    # serialized into the request body and this gateway 400-rejects unknown
    # body params (proven by the enable_thinking probe).
    openai.request_timeout = 300


def chat_completion(api_key, api_base, model, messages, **kwargs):
    if _OPENAI_LEGACY:
        openai.api_key = api_key
        openai.api_base = api_base
        response = openai.ChatCompletion.create(model=model, messages=messages, **kwargs)
        content = response["choices"][0]["message"]["content"]
    else:
        client = _get_client(api_key, api_base)
        response = client.chat.completions.create(model=model, messages=messages, **kwargs)
        content = response.choices[0].message.content
    # some gateways occasionally return HTTP 200 with content=None (observed
    # on the divide endpoint during the full 322-case run); the caller's
    # string parsing would then crash (output.split / re.findall on None).
    # Surface it as an exception so the existing retry loops handle it like a
    # transport error instead of killing a multi-hour run.
    if content is None or not str(content).strip():
        raise ValueError("empty chat completion content (content=None or blank)")
    return content


def list_first_model(api_key, api_base):
    if _OPENAI_LEGACY:
        openai.api_key = api_key
        openai.api_base = api_base
        return openai.Model.list()["data"][0]["id"]
    client = _get_client(api_key, api_base)
    return client.models.list().data[0].id



DIVIDE_SYSTEM_MESSAGE = "You are a question decomposition assistant. You decompose a multi-hop question into single-hop sub-questions, one per line, using [ENT] placeholders for entities resolved by previous sub-questions. Each sub-question must match exactly one knowledge-graph fact: never merge two facts into one, and never skip an intermediate lookup. When unsure whether a line covers one or two facts, split it into two lines -- under-splitting is worse than over-splitting, and typical questions need 2 to 4 sub-questions. If the asked relation plausibly belongs not to the named entity but to a related one (a song has no manager -- its performer does), first resolve that intermediate entity with its own sub-question. You output only the sub-questions, with no numbering and no extra text."


def run_llm_divide(query, api_key=DIVIDE_API_KEY, api_base=DIVIDE_API_BASE, model=DIVIDE_MODEL_DEFAULT):
    if model:
        divide_model = model
    else:
        divide_model = list_first_model(api_key, api_base)
    messages = [{"role":"system","content":DIVIDE_SYSTEM_MESSAGE}]
    message_prompt = {"role":"user","content":query}
    messages.append(message_prompt)
    f = 0
    while(f == 0):
        try:
            result = chat_completion(api_key, api_base, divide_model,
                                     messages,
                                     temperature=0,
                                     stop=["\nQuestion:"],
                                     frequency_penalty=0,
                                     presence_penalty=0)
            f = 1
        except Exception as e:
            print("openai error (run_llm_divide), retry:", repr(e))
            time.sleep(10)
    return result


def parse_sub_questions(output):

    sub_questions = []
    for s in output.split('\n'):
        s = re.sub(r'(?i)^subquestions?\s*:?\s*', '', s.strip())
        s = re.sub(r'^\s*(?:\d+[\.\)]\s*|[-\*]\s*)', '', s).strip()
        if s:
            sub_questions.append(s)
    return sub_questions


class DivideCache:
    """JSON-backed cache: question text -> raw divide output (+ parsed copy).

    Shared by build_divide_cache.py (this file) and dynamic_exp_bridge3.py
    (which imports it), so the builder and the experiment can never disagree
    about the on-disk format.
    """

    FORMAT = 1

    def __init__(self, path):
        self.path = path
        self.entries = {}        # question -> {'model', 'prompt_sha1', 'output', 'sub_questions'}
        self.meta = {'format': self.FORMAT}
        self.prompt_sha1 = None  # set via set_prompt(); gates entry reuse
        self._dirty = 0          # entries added since the last save

    # -- path convention shared with dynamic_exp_bridge3.py ('auto' mode) ----
    @staticmethod
    def default_path(dataset, model):
        tag = re.sub(r'[^A-Za-z0-9._-]+', '_', (model or 'endpoint-default'))
        return os.path.join(_HERE, '..', 'output', 'divide_cache',
                            'divide_cache_%s_%s.json' % (dataset, tag))

    def set_prompt(self, prompt_text):
        self.prompt_sha1 = hashlib.sha1(prompt_text.encode('utf-8')).hexdigest()
        return self.prompt_sha1

    def load(self):
        self.entries = {}
        self.meta = {'format': self.FORMAT}
        if not os.path.exists(self.path):
            return 0
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                blob = json.load(f)
        except (ValueError, OSError) as e:
            backup = self.path + '.corrupt-' + time.strftime('%Y%m%d-%H%M%S')
            try:
                os.replace(self.path, backup)
                print("[divide-cache] WARNING: %s unreadable (%r); moved to %s, starting empty"
                      % (self.path, e, backup))
            except OSError:
                print("[divide-cache] WARNING: %s unreadable (%r); starting empty"
                      % (self.path, e))
            return 0
        self.entries = blob.get('entries', {}) or {}
        self.meta = blob.get('meta', {}) or {}
        return len(self.entries)

    def save(self):
        self.meta.update({
            'format': self.FORMAT,
            'entries': len(self.entries),
            'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
        })
        d = os.path.dirname(os.path.abspath(self.path))
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'meta': self.meta, 'entries': self.entries}, f,
                      ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)  # atomic: a crash never truncates the cache
        self._dirty = 0

    def _usable(self, entry):
        if not isinstance(entry, dict) or 'output' not in entry:
            return False
        if self.prompt_sha1 is not None and entry.get('prompt_sha1') != self.prompt_sha1:
            return False  
        return True

    def has(self, question):
        return self._usable(self.entries.get(question))

    def get_output(self, question):
        entry = self.entries.get(question)
        return entry['output'] if self._usable(entry) else None

    def get_entry(self, question):
        entry = self.entries.get(question)
        return entry if self._usable(entry) else None

    def usable_count(self):
        return sum(1 for e in self.entries.values() if self._usable(e))

    def put(self, question, output, model=None):
        self.entries[question] = {
            'model': model or self.meta.get('model', ''),
            'prompt_sha1': self.prompt_sha1,
            'output': output,
            'sub_questions': parse_sub_questions(output),
        }
        self._dirty += 1

    def maybe_save(self, every=20):
        if self._dirty and self._dirty >= every:
            self.save()


def collect_questions(data, limit=None):
    questions = []
    seen = set()
    hop_of = {}
    for i, d in enumerate(data):
        if limit is not None and i >= limit:
            break
        hops = len(d.get('single_hops') or [])
        for q in d.get('questions') or []:
            if q and q not in seen:
                seen.add(q)
                questions.append(q)
                hop_of[q] = hops
    return questions, hop_of


def main():
    try: 
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description='Pre-decompose every question of a dataset into a JSON cache '
                    'that dynamic_exp_bridge3.py replays instead of calling the '
                    'divide endpoint live (identical divide process, selectable model).')
    parser.add_argument('--dataset', type=str, default='MQuAKE-T',
                        choices=['MQuAKE-CF-3k', 'MQuAKE-T', 'MQuAKE-CF'])
    parser.add_argument('--model', type=str, default=DIVIDE_MODEL_DEFAULT,
                        help="divide model id, e.g. qwen3.8-chat (default, matches bridge3); "
                             "'auto' = first model listed by the endpoint")
    parser.add_argument('--api_base', type=str, default=DIVIDE_API_BASE)
    parser.add_argument('--api_key', type=str, default=os.environ.get('DIVIDE_API_KEY', DIVIDE_API_KEY))
    parser.add_argument('--prompt_file', type=str, default=DEFAULT_PROMPT_FILE,
                        help='divide prompt template')
    parser.add_argument('--cache', type=str, default='',
                        help='cache file (default: ../output/divide_cache/divide_cache_<dataset>_<model>.json)')
    parser.add_argument('--limit', type=int, default=0,
                        help='only decompose the questions of the first N cases (smoke test; 0 = all)')
    parser.add_argument('--overwrite', action='store_true',
                        help='re-decompose questions even if the cache already holds them')
    parser.add_argument('--save_every', type=int, default=20,
                        help='incrementally save the cache after every N new entries (default 20)')
    args = parser.parse_args()

    if not args.api_key:
        print("WARNING: api key is empty -- every request will be rejected with "
              "401 and retried forever. Pass --api_key or set DIVIDE_API_KEY.")
        
    model = '' if args.model == 'auto' else args.model
    if not model:
        for _ in range(3):
            try:
                model = list_first_model(args.api_key, args.api_base)
                break
            except Exception as e:
                print("model list error, retry:", repr(e))
                time.sleep(5)
        if not model:
            sys.exit("could not resolve the endpoint model list -- pass --model explicitly")
        print("[divide-cache] endpoint default model: %s" % model)

    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        divide_prompt = f.read()
    with open(os.path.join(DEFAULT_DATASET_DIR, args.dataset + '.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)

    cache_path = args.cache or DivideCache.default_path(args.dataset, model)
    questions, hop_of = collect_questions(data, args.limit or None)

    cache = DivideCache(cache_path)
    cache.load()
    cache.set_prompt(divide_prompt)
    cache.meta.update({
        'dataset': args.dataset,
        'model': model,
        'prompt_file': os.path.basename(args.prompt_file),
        'prompt_sha1': cache.prompt_sha1,
    })

    stale = len(cache.entries) - cache.usable_count()
    todo = [q for q in questions if args.overwrite or not cache.has(q)]
    print("[divide-cache] file       : %s" % cache_path)
    print("[divide-cache] dataset    : %s (%d cases scanned, %d unique questions)"
          % (args.dataset, min(len(data), args.limit or len(data)), len(questions)))
    print("[divide-cache] model      : %s" % model)
    print("[divide-cache] prompt     : %s (sha1 %s)"
          % (os.path.basename(args.prompt_file), cache.prompt_sha1[:10]))
    print("[divide-cache] cache      : %d entries loaded, %d usable, %d stale (other template)"
          % (len(cache.entries), cache.usable_count(), stale))
    print("[divide-cache] to-decompose: %d (%d already cached)"
          % (len(todo), len(questions) - len(todo)))
    if stale and not args.overwrite:
        print("[divide-cache] note: stale entries (built with a different divide template) "
              "are re-decomposed automatically; pass --overwrite to redo everything.")

    t0 = time.time()
    done = 0
    try:
        for q in tqdm(todo, desc='decompose', mininterval=2.0):
            prompt = divide_prompt.replace("<<<<QUESTION>>>>", q)
            output = run_llm_divide(prompt, args.api_key, args.api_base, model)
            cache.put(q, output, model=model)
            done += 1
            cache.maybe_save(args.save_every)
    except KeyboardInterrupt:
        print("\n[divide-cache] interrupted by user -- saving progress")
    finally:
        cache.save()
        print("[divide-cache] saved %d entries (%d decomposed this run) to %s"
              % (len(cache.entries), done, cache_path))


    hist = collections.Counter()
    fewer = equal = more = 0
    under_split = []
    for q in questions:
        entry = cache.get_entry(q)
        if entry is None:
            continue
        n = len(entry['sub_questions'])
        hist[n] += 1
        hops = hop_of.get(q) or 0
        if hops:
            if n < hops:
                fewer += 1
                under_split.append((q, n, hops))
            elif n == hops:
                equal += 1
            else:
                more += 1
    print("[report] sub-question count distribution: %s"
          % dict(sorted(hist.items())))
    comparable = fewer + equal + more
    if comparable:
        print("[report] sub-question count vs gold hops: fewer=%d equal=%d more=%d (of %d comparable)"
              % (fewer, equal, more, comparable))
    if under_split:
        print("[report] under-split questions (inspect before the run; first %d of %d):"
              % (min(10, len(under_split)), len(under_split)))
        for q, n, hops in under_split[:10]:
            print("  [%d sub-q vs %d gold hops] %s" % (n, hops, q))
    if done:
        print("[report] %d decompositions in %.1f min (%.2f s/question)"
              % (done, (time.time() - t0) / 60.0, (time.time() - t0) / done))
    print("[divide-cache] next: python dynamic_exp_bridge3.py --dataset %s ... "
          "(auto-uses this file; --divide_cache off disables it)" % args.dataset)


if __name__ == '__main__':
    main()
