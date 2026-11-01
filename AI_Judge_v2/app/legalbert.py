"""LegalBERT argument structuring (from 02_legalbert_finetuning / 03_argument_structuring).

Labels every sentence of every argument as Claim / Evidence / LegalRule / Rebuttal / Conclusion,
and groups each lawyer's sentences the same way as structured_arguments/case1_structured.json.

Optional: if torch/transformers or the model folder (app/models/legalbert) are missing, the app
simply runs without it. Runs on CPU (BERT-base, ~440 MB, a few milliseconds per sentence).
"""
import os
import threading

import knowledge as kb

MODEL_DIR = os.environ.get("LEGALBERT_DIR", os.path.join(kb.APP_DIR, "models", "legalbert"))
GROUP_KEY = {"Claim": "claims", "Evidence": "evidence", "LegalRule": "laws",
             "Rebuttal": "rebuttals", "Conclusion": "conclusions"}

_lock = threading.Lock()
_state = {"model": None, "tokenizer": None, "labels": None, "error": None, "tried": False}


def model_present():
    return os.path.exists(os.path.join(MODEL_DIR, "config.json"))


def load():
    """Load once; safe to call from several threads. Returns True if ready."""
    with _lock:
        if _state["model"] is not None:
            return True
        if _state["tried"]:
            return False
        _state["tried"] = True
        if not model_present():
            _state["error"] = f"model folder not found ({MODEL_DIR})"
            return False
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
            tok = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            model.eval()
            _state.update(model=model, tokenizer=tok, labels=model.config.id2label)
            return True
        except Exception as e:          # missing libraries, corrupt files, ...
            _state["error"] = f"{type(e).__name__}: {e}"
            return False


def ready():
    return _state["model"] is not None


def status():
    return {"ready": ready(), "present": model_present(), "error": _state["error"]}


def classify(sentences):
    """[(sentence)] -> [{"sentence", "role", "confidence"}]; [] if LegalBERT isn't available."""
    if not sentences or not load():
        return []
    import torch
    tok, model, labels = _state["tokenizer"], _state["model"], _state["labels"]
    with torch.no_grad():
        enc = tok(sentences, return_tensors="pt", truncation=True, max_length=128, padding=True)
        probs = torch.softmax(model(**enc).logits, dim=-1)
    out = []
    for s, p in zip(sentences, probs):
        i = int(p.argmax())
        out.append({"sentence": s, "role": labels.get(i, labels.get(str(i), str(i))), "confidence": round(float(p[i]), 3)})
    return out


def structure_turn(text):
    return classify(kb.split_sentences(text))


def structured_arguments(turns):
    """Group sentences per lawyer, in the format of 03_argument_structuring.ipynb."""
    out = {}
    for side in ("A", "B"):
        groups = {k: [] for k in GROUP_KEY.values()}
        for t in turns:
            if t["speaker"] != side:
                continue
            for item in t.get("structure") or []:
                groups[GROUP_KEY.get(item["role"], "claims")].append(item["sentence"])
        out[f"Lawyer_{side}"] = groups
    return out


def warm_up_in_background():
    threading.Thread(target=load, daemon=True).start()
