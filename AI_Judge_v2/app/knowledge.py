"""Case catalog, legal knowledge base search, and the argument-pattern index used by demo mode.

The case definitions and the judge's prompt format come from ../generate_judge_dataset_v2.py,
the same file used to train the judge, so the app always talks to the model in its training format.
"""
import json
import math
import os
import re
import string
import sys
from collections import Counter

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, ROOT_DIR)
import generate_judge_dataset_v2 as gen  # noqa: E402

CASES = gen.CASES
build_prompt = gen.build_prompt
issue_proved = gen.issue_proved
finding_sentence = gen.finding_sentence
STRENGTH_LABEL = gen.STRENGTH_LABEL

with open(os.path.join(ROOT_DIR, "cases_catalog.json"), encoding="utf-8") as f:
    CATALOG = {c["id"]: c for c in json.load(f)}

# ----------------------------------------------------------------------------- text search
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the and or of to in on for by with from at as is are was were be been being it its this that
these those he she his her him they them their we our you your i my me not no nor but if then than so such
which who whom what when where why how all any each both few more most other some own same too very can will
just should would could may might must shall do does did done have has had having into over under again further
once here there about above below between through during before after up down out off also only per""".split())


def tokens(text):
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


class TfIdf:
    """Tiny TF-IDF search index (no external libraries)."""

    def __init__(self, texts):
        self.texts = texts
        docs = [Counter(tokens(t)) for t in texts]
        n = len(docs)
        df = Counter(w for d in docs for w in d)
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, counts):
        v = {w: (1 + math.log(c)) * self.idf.get(w, 0) for w, c in counts.items() if w in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def scores(self, query):
        q = self._vec(Counter(tokens(query)))
        return [sum(q.get(w, 0) * x for w, x in v.items()) for v in self.vecs]

    def top(self, query, k=4):
        s = self.scores(query)
        return sorted(range(len(s)), key=lambda i: -s[i])[:k], s


# ----------------------------------------------------------------------------- legal KB
def _load_kb():
    items = []
    kb_dir = os.path.join(APP_DIR, "legal_kb")
    for law in json.load(open(os.path.join(kb_dir, "laws.json"), encoding="utf-8")):
        items.append({"id": law["id"], "kind": "law", "title": law["title"], "text": law["text"]})
    for case in json.load(open(os.path.join(kb_dir, "cases.json"), encoding="utf-8")):
        items.append({"id": case["id"], "kind": "case", "title": case["title"], "text": case["summary"]})
    return items


KB = _load_kb()
_KB_INDEX = TfIdf([f"{k['title']}. {k['title']}. {k['text']}" for k in KB])


def search_kb(query, k=4):
    idx, s = _KB_INDEX.top(query, k)
    return [dict(KB[i], score=round(s[i], 3)) for i in idx if s[i] > 0]


# ----------------------------------------------------------------------------- argument patterns (demo mode)
class _Slots(dict):
    def __missing__(self, key):
        return {"amt": "the amount", "loss": "the loss", "goods": "the goods", "days": "the agreed",
                "purpose": "the stated purpose", "place": "the city", "project": "the project",
                "business": "the business", "accrole_cap": "The other party"}.get(key, key)


def _fill(text, case_id):
    c = CATALOG[case_id]
    return string.Formatter().vformat(text, (), _Slots(acc=c["acc"], comp=c["comp"]))


PATTERNS = {}   # case_id -> {"index": TfIdf, "entries": [...]}
for _cid, _case in CASES.items():
    entries = []
    for i, iss in enumerate(_case["issues"]):
        for side in ("A", "B"):
            for strength, wordings in iss[side].items():
                for text, gist in wordings:
                    entries.append({"issue": i, "side": side, "s": strength, "text": _fill(text, _cid), "gist": gist})
    for side in ("A", "B"):
        for text, gist in gen.IRRELEVANT[side]:
            entries.append({"issue": None, "side": side, "s": 0, "text": _fill(text, _cid), "gist": gist})
    PATTERNS[_cid] = {"entries": entries, "index": TfIdf([e["text"] for e in entries])}


def classify_sentence(case_id, sentence, threshold=0.2):
    """Nearest known argument pattern: returns (issue_index or None, strength 0-3, similarity)."""
    p = PATTERNS[case_id]
    s = p["index"].scores(sentence)
    best = max(range(len(s)), key=lambda i: s[i])
    if s[best] < threshold:
        return None, 0, s[best]
    e = p["entries"][best]
    return e["issue"], e["s"], s[best]


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])(?<!\bRs\.)(?<!\bNo\.)(?<!\bvs\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bSec\.)"
                     r"\s+(?=[A-Z\"'(])", text.strip())
    return [x.strip() for x in parts if len(x.strip()) > 3]


def public_case(case_id):
    c = CATALOG[case_id]
    return {k: c[k] for k in ("id", "title", "nature", "law", "lawyer_a_role", "lawyer_b_role", "facts",
                              "issues", "possible_verdicts", "acc", "comp")}
