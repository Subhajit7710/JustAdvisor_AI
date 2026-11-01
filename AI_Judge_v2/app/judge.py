"""The AI Judge: builds the prompt in the exact training format, runs the model, parses the judgment.
Falls back to a rule-based DEMO judge when the trained model isn't installed yet."""
import json
import os
import re
import time

import knowledge as kb
import llm

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "judge-v2")
_FMT_PATH = os.path.join(kb.APP_DIR, "models", "prompt_format.json")
_DEFAULT_FMT = {"prefix": "<|im_start|>user\n", "suffix": "<|im_end|>\n<|im_start|>assistant\n",
                "stop": ["<|im_end|>", "<|endoftext|>"]}


def prompt_format():
    try:
        with open(_FMT_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return _DEFAULT_FMT


def judge_prompt(case_id, turns):
    c = kb.CATALOG[case_id]
    return kb.build_prompt(case_id, c["facts"], [{"round": t["round"], "speaker": t["speaker"], "text": t["text"]}
                                                 for t in turns], acc=c["acc"], comp=c["comp"])


# ----------------------------------------------------------------------------- parsing
_VERDICT = re.compile(r"FINAL VERDICT:\s*(GUILTY|NOT GUILTY|CLAIM ALLOWED|CLAIM DISMISSED)")


def parse_judgment(text, case_id):
    issues_text = kb.CATALOG[case_id]["issues"]
    issues = []
    for m in re.finditer(r"^Issue (\d+):\s*(.*)$", text, flags=re.M):
        n = int(m.group(1))
        end = text.find("\nIssue ", m.end())
        for marker in ("\nSUBMISSIONS GIVEN NO WEIGHT", "\nFINAL VERDICT"):
            k = text.find(marker, m.end())
            if k != -1 and (end == -1 or k < end):
                end = k
        block = text[m.end(): end if end != -1 else len(text)]
        lines = [ln[2:].strip() for ln in block.splitlines() if ln.startswith("- ")]
        f = re.search(r"Finding:\s*(NOT PROVED|PROVED)\.?\s*(.*)", block)
        issues.append({
            "n": n,
            "issue": m.group(2).strip() or (issues_text[n - 1] if 0 < n <= len(issues_text) else ""),
            "submissions": [ln for ln in lines if not ln.startswith("Finding:")],
            "finding": f.group(1) if f else None,
            "reason": f.group(2).strip() if f else "",
        })
    no_weight = []
    k = text.find("SUBMISSIONS GIVEN NO WEIGHT")
    if k != -1:
        seg = text[k:].split("FINAL VERDICT")[0]
        no_weight = [ln[2:].strip() for ln in seg.splitlines() if ln.startswith("- ") and ln.strip() != "- None."]
    v = _VERDICT.search(text)
    verdict = v.group(1) if v else None
    order = text[v.end():].strip() if v else ""
    return {
        "verdict": verdict,
        "winner": ("A" if verdict in ("GUILTY", "CLAIM ALLOWED") else "B") if verdict else None,
        "issues": issues,
        "no_weight": no_weight,
        "order": order,
        "complete": bool(verdict) and len(issues) == len(issues_text),
    }


# ----------------------------------------------------------------------------- model judge
def judge_with_model(case_id, turns, on_token=None):
    fmt = prompt_format()
    prompt = fmt["prefix"] + judge_prompt(case_id, turns) + fmt["suffix"]
    return llm.generate_raw(JUDGE_MODEL, prompt, on_token=on_token, stop=fmt.get("stop"), max_tokens=1200)


# ----------------------------------------------------------------------------- demo judge
def judge_demo(case_id, turns, on_token=None):
    """Rule-based stand-in: matches each sentence to known argument patterns, then applies the
    same decision rule the model was trained on. Only for trying the app before the model is ready."""
    case = kb.CASES[case_id]
    c = kb.CATALOG[case_id]
    criminal = case["nature"] == "criminal"
    per_issue = {i: {"A": [], "B": []} for i in range(len(case["issues"]))}
    no_weight = []
    for t in turns:
        for sent in kb.split_sentences(t["text"]):
            issue, s, _ = kb.classify_sentence(case_id, sent)
            short = sent.rstrip(".")
            if len(short) > 110:
                short = short[:107].rstrip() + "..."
            if issue is None or s == 0:
                no_weight.append((t["round"], t["speaker"], short))
            else:
                per_issue[issue][t["speaker"]].append((t["round"], s, short))

    out = ["ANALYSIS"]
    failed = []
    for i, iss in enumerate(case["issues"]):
        a = max([x[1] for x in per_issue[i]["A"]], default=0)
        b = max([x[1] for x in per_issue[i]["B"]], default=0)
        out.append(f"Issue {i + 1}: Whether {iss['issue']}.")
        for side in ("A", "B"):
            pts = per_issue[i][side]
            if not pts:
                out.append(f"- Lawyer {side}: no submission on this issue.")
            for rnd, s, short in pts:
                out.append(f'- Lawyer {side} (Round {rnd}): "{short}" ({kb.STRENGTH_LABEL[s]}).')
        out.append(f"- Finding: {kb.finding_sentence(a, b, criminal)}")
        if not kb.issue_proved(a, b):
            failed.append(i + 1)
    out.append("SUBMISSIONS GIVEN NO WEIGHT")
    out += [f'- Lawyer {sp} (Round {r}): "{txt}". This is not relevant to any issue.' for r, sp, txt in no_weight] or ["- None."]
    fl = ", ".join(map(str, failed))
    pl = "s" if len(failed) > 1 else ""
    if criminal:
        if not failed:
            out += ["FINAL VERDICT: GUILTY",
                    f"Every issue is proved beyond reasonable doubt. The accused, {c['acc']}, is convicted under {case['section_short']}."]
        else:
            out += ["FINAL VERDICT: NOT GUILTY",
                    f"Issue{pl} {fl} {'are' if pl else 'is'} not proved beyond reasonable doubt. The accused, {c['acc']}, is acquitted of the charge under {case['section_short']}."]
    else:
        if not failed:
            out += ["FINAL VERDICT: CLAIM ALLOWED",
                    f"Every issue is proved on the balance of probabilities. The claim of the plaintiff, {c['comp']}, under {case['section_short']} is allowed."]
        else:
            out += ["FINAL VERDICT: CLAIM DISMISSED",
                    f"Issue{pl} {fl} {'are' if pl else 'is'} not proved on the balance of probabilities. The claim of the plaintiff, {c['comp']}, under {case['section_short']} is dismissed."]
    text = "\n".join(out)
    if on_token:                       # reveal gradually, like the real model
        for line in text.splitlines(keepends=True):
            on_token(line if line.endswith("\n") else line + "\n")
            time.sleep(0.05)
    return text


def deliver_judgment(case_id, turns, use_model, on_token=None):
    """Returns (text, engine). engine is 'model' or 'demo'."""
    if use_model:
        return judge_with_model(case_id, turns, on_token), "model"
    return judge_demo(case_id, turns, on_token), "demo"
