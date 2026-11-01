"""AI legal assistant: suggests next arguments for one lawyer, grounded in the legal knowledge base.
Uses the base Qwen3-4B-Instruct model in Ollama; falls back to pattern-based suggestions in demo mode."""
import os

import knowledge as kb
import llm

ASSISTANT_MODEL = os.environ.get("ASSISTANT_MODEL", "qwen3:4b-instruct-2507-q4_K_M")

SCHEMA = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue": {"type": "integer"},
                    "argument": {"type": "string"},
                    "strength": {"type": "string", "enum": ["strong", "medium", "weak"]},
                    "why": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["issue", "argument", "strength", "why"],
            },
        }
    },
    "required": ["suggestions"],
}


def _role(case_id, side):
    c = kb.CATALOG[case_id]
    return c["lawyer_a_role"] if side == "A" else c["lawyer_b_role"]


def _sources(case_id, turns, draft):
    c = kb.CATALOG[case_id]
    q = " ".join([c["title"], c["law"], " ".join(c["issues"]), draft or ""] + [t["text"] for t in turns[-3:]])
    return kb.search_kb(q, k=4)


def suggest_with_model(case_id, side, turns, draft):
    c = kb.CATALOG[case_id]
    sources = _sources(case_id, turns, draft)
    transcript = "\n".join(f"[Round {t['round']}] Lawyer {t['speaker']}: {t['text']}" for t in turns) or "(no arguments yet)"
    src_text = "\n".join(f"[{s['id']}] {s['title']}: {s['text']}" for s in sources)
    other = "B" if side == "A" else "A"
    tagged = [f"- ({x['role']}) {x['sentence']}" for t in turns if t["speaker"] == other for x in (t.get("structure") or [])]
    structure_text = ("\nLAWYER " + other + "'S SUBMISSIONS BY TYPE (tagged by LegalBERT):\n" + "\n".join(tagged) + "\n"
                      "Answer the opponent's Evidence first; their bare Claims are weak.\n") if tagged else ""
    goal = ("You must prove EVERY issue. Target issues that are not yet supported by strong evidence or that "
            "Lawyer B has rebutted." if side == "A" else
            "You win if Lawyer A fails to prove ANY ONE issue. Target the issue where Lawyer A's evidence is "
            "weakest, or rebut Lawyer A's strongest point with equal or stronger evidence.")
    user = f"""You are advising Lawyer {side}, who appears for {_role(case_id, side)}, in a simulated Indian court.

CASE: {c['title']}
LAW: {c['law']}
FACTS: {c['facts']}
ISSUES:
{chr(10).join(f"{i + 1}. {x}" for i, x in enumerate(c['issues']))}

TRANSCRIPT SO FAR:
{transcript}

LAWYER {side}'S CURRENT DRAFT: {draft.strip() if draft and draft.strip() else "(empty)"}

{structure_text}
LEGAL SOURCES YOU MAY CITE:
{src_text}

HOW THE JUDGE WEIGHS ARGUMENTS: documentary, forensic or independent evidence = strong; oral testimony or
circumstantial evidence = medium; bare assertions = weak; character attacks or appeals to sympathy = no weight.
{goal}

Suggest exactly 3 arguments Lawyer {side} could make next. Each argument: 1-2 sentences, written as the lawyer
would say it in court, consistent with the facts, naming the kind of evidence relied on. Give the issue number
it addresses, its likely strength, a one-line reason, and the ids of any sources above that support it."""
    data = llm.chat_json(ASSISTANT_MODEL, [
        {"role": "system", "content": "You are a precise legal research assistant for Indian criminal and contract law. Answer only in the requested JSON."},
        {"role": "user", "content": user}], SCHEMA)
    out = []
    valid_ids = {s["id"] for s in sources}
    for s in data.get("suggestions", [])[:3]:
        n = s.get("issue")
        out.append({"issue": n if isinstance(n, int) and 1 <= n <= len(c["issues"]) else None,
                    "argument": str(s.get("argument", "")).strip(),
                    "strength": s.get("strength", "medium"),
                    "why": str(s.get("why", "")).strip(),
                    "sources": [x for x in s.get("sources", []) if x in valid_ids]})
    return {"suggestions": [s for s in out if s["argument"]], "sources": sources, "engine": "model"}


def suggest_demo(case_id, side, turns, draft):
    """Pattern-based suggestions: finds issues this side still needs, proposes strong arguments for them."""
    case = kb.CASES[case_id]
    best = {i: {"A": 0, "B": 0} for i in range(len(case["issues"]))}
    for t in turns:
        for sent in kb.split_sentences(t["text"]):
            issue, s, _ = kb.classify_sentence(case_id, sent)
            if issue is not None:
                best[issue][t["speaker"]] = max(best[issue][t["speaker"]], s)
    if side == "A":
        order = sorted(best, key=lambda i: (kb.issue_proved(best[i]["A"], best[i]["B"]), best[i]["A"]))
    else:
        order = sorted(best, key=lambda i: (not kb.issue_proved(best[i]["A"], best[i]["B"]), -best[i]["A"]))
    out = []
    for i in order[:3]:
        text, gist = case["issues"][i][side][3][1]
        out.append({"issue": i + 1, "argument": kb._fill(text, case_id), "strength": "strong",
                    "why": ("This issue still needs strong evidence." if side == "A"
                            else "Rebutting this with documents can create doubt on the issue."),
                    "sources": []})
    return {"suggestions": out, "sources": _sources(case_id, turns, draft), "engine": "demo"}


def suggest(case_id, side, turns, draft, use_model):
    if use_model:
        try:
            return suggest_with_model(case_id, side, turns, draft)
        except llm.LLMUnavailable as e:
            res = suggest_demo(case_id, side, turns, draft)
            res["note"] = f"Assistant model unavailable, showing basic suggestions ({e})"
            return res
    return suggest_demo(case_id, side, turns, draft)
