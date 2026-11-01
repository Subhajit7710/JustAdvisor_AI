#!/usr/bin/env python3
"""
AI Courtroom - web app server.

    python server.py            # uses the AI models in Ollama (falls back to demo mode if missing)
    python server.py --demo     # rule-based demo judge/assistant, no AI models needed
    python server.py --port 8000 --host 0.0.0.0   # let other devices on the same Wi-Fi join

Then open http://localhost:8000 in a browser.
"""
import argparse
import datetime
import json
import os
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import assistant
import judge
import knowledge as kb
import legalbert
import llm

# ----------------------------------------------------------------------------- rules of the courtroom
MAX_ROUNDS = 5          # each round = Lawyer A speaks, then Lawyer B
MIN_ROUNDS = 2          # verdict can be requested after this many complete rounds
MAX_CHARS = 600         # per argument
MIN_CHARS = 15
ASSIST_PER_SIDE = 5     # assistant requests per lawyer per case

DEMO = False            # set by --demo
SESSIONS = {}
LOCK = threading.Lock()
RECORDS_DIR = os.path.join(kb.APP_DIR, "records")
os.makedirs(RECORDS_DIR, exist_ok=True)

app = FastAPI(title="AI Courtroom")


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def engines():
    models = None if DEMO else llm.installed_models()
    return {
        "demo_forced": DEMO,
        "ollama_running": models is not None,
        "judge_model": judge.JUDGE_MODEL,
        "judge_ready": llm.has_model(judge.JUDGE_MODEL, models),
        "assistant_model": assistant.ASSISTANT_MODEL,
        "assistant_ready": llm.has_model(assistant.ASSISTANT_MODEL, models),
        "legalbert_ready": legalbert.ready(),
        "legalbert_present": legalbert.model_present(),
    }


def view(s):
    """Session as sent to the browser."""
    done_rounds = sum(1 for t in s["turns"] if t["speaker"] == "B")
    can_argue = s["status"] == "arguing" and s["round"] <= MAX_ROUNDS
    return {
        "id": s["id"], "case": kb.public_case(s["case_id"]), "status": s["status"],
        "turns": s["turns"], "round": min(s["round"], MAX_ROUNDS), "next_speaker": s["next_speaker"] if can_argue else None,
        "completed_rounds": done_rounds,
        "can_request_verdict": s["status"] == "arguing" and s["next_speaker"] == "A" and done_rounds >= MIN_ROUNDS,
        "assist_left": s["assist_left"], "judgment": s["judgment"], "error": s["error"],
        "structured": legalbert.structured_arguments(s["turns"]) if any(t.get("structure") for t in s["turns"]) else None,
        "limits": {"max_rounds": MAX_ROUNDS, "min_rounds": MIN_ROUNDS, "max_chars": MAX_CHARS, "min_chars": MIN_CHARS},
    }


def get_session(sid):
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "This hearing doesn't exist. It may have ended when the server restarted.")
    return s


# ----------------------------------------------------------------------------- API
class NewSession(BaseModel):
    case_id: str


class Turn(BaseModel):
    speaker: str
    text: str


class AssistReq(BaseModel):
    speaker: str
    draft: str = ""


@app.get("/api/status")
def status():
    return {**engines(), "limits": {"max_rounds": MAX_ROUNDS, "min_rounds": MIN_ROUNDS, "max_chars": MAX_CHARS}}


@app.get("/api/cases")
def cases():
    return [kb.public_case(cid) for cid in kb.CATALOG]


@app.post("/api/sessions")
def new_session(req: NewSession):
    if req.case_id not in kb.CATALOG:
        raise HTTPException(400, "Only the listed cases are supported in this version.")
    sid = uuid.uuid4().hex[:6].upper()
    s = {"id": sid, "case_id": req.case_id, "created": now(), "status": "arguing", "turns": [], "round": 1,
         "next_speaker": "A", "assist_left": {"A": ASSIST_PER_SIDE, "B": ASSIST_PER_SIDE},
         "judgment": None, "error": None}
    SESSIONS[sid] = s
    return view(s)


@app.get("/api/sessions/{sid}")
def read_session(sid: str):
    return view(get_session(sid.upper()))


@app.post("/api/sessions/{sid}/turns")
def add_turn(sid: str, req: Turn):
    s = get_session(sid.upper())
    text = " ".join(req.text.split())
    if MIN_CHARS <= len(text) <= MAX_CHARS and req.speaker == s["next_speaker"]:
        try:
            structure = legalbert.structure_turn(text)      # LegalBERT: Claim / Evidence / LegalRule / ...
        except Exception:
            structure = []
    else:
        structure = []
    with LOCK:
        if s["status"] != "arguing" or s["round"] > MAX_ROUNDS:
            raise HTTPException(409, "Arguments are closed for this hearing.")
        if req.speaker != s["next_speaker"]:
            raise HTTPException(409, f"It's Lawyer {s['next_speaker']}'s turn.")
        if len(text) < MIN_CHARS:
            raise HTTPException(400, f"An argument needs at least {MIN_CHARS} characters.")
        if len(text) > MAX_CHARS:
            raise HTTPException(400, f"Keep each argument under {MAX_CHARS} characters (this one has {len(text)}).")
        s["turns"].append({"round": s["round"], "speaker": req.speaker, "text": text, "at": now(),
                           "structure": structure})
        if req.speaker == "A":
            s["next_speaker"] = "B"
        else:
            s["next_speaker"] = "A"
            s["round"] += 1
    return view(s)


@app.post("/api/sessions/{sid}/assist")
def assist(sid: str, req: AssistReq):
    s = get_session(sid.upper())
    if req.speaker not in ("A", "B"):
        raise HTTPException(400, "speaker must be A or B")
    if s["status"] != "arguing":
        raise HTTPException(409, "The assistant is only available while arguments are open.")
    if s["assist_left"][req.speaker] <= 0:
        raise HTTPException(429, f"Lawyer {req.speaker} has used all {ASSIST_PER_SIDE} assistant requests for this case.")
    s["assist_left"][req.speaker] -= 1
    e = engines()
    res = assistant.suggest(s["case_id"], req.speaker, s["turns"], req.draft, use_model=e["assistant_ready"])
    res["assist_left"] = s["assist_left"][req.speaker]
    return res


@app.post("/api/sessions/{sid}/verdict")
def request_verdict(sid: str):
    s = get_session(sid.upper())
    v = view(s)
    if s["status"] in ("deliberating", "decided"):
        return v
    if not v["can_request_verdict"]:
        raise HTTPException(409, f"The judge can rule after at least {MIN_ROUNDS} complete rounds (Lawyer A and B each argue once per round).")
    e = engines()
    s["status"] = "deliberating"
    s["judgment"] = {"text": "", "engine": "model" if e["judge_ready"] else "demo", "started": time.time(),
                     "seconds": None, "parsed": None}

    def work():
        def on_token(piece):
            s["judgment"]["text"] += piece
        try:
            text, engine = judge.deliver_judgment(s["case_id"], s["turns"], e["judge_ready"], on_token)
            s["judgment"].update(text=text, engine=engine, parsed=judge.parse_judgment(text, s["case_id"]),
                                 seconds=round(time.time() - s["judgment"]["started"]))
            s["status"] = "decided"
            record = dict(s, structured_arguments=legalbert.structured_arguments(s["turns"]))
            with open(os.path.join(RECORDS_DIR, f"{s['created'][:10]}_{s['id']}_{s['case_id']}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        except Exception as ex:          # keep the hearing usable if the model fails
            s["status"] = "arguing"
            s["error"] = f"The judge could not deliver a verdict: {ex}"
            s["judgment"] = None

    s["error"] = None
    threading.Thread(target=work, daemon=True).start()
    return view(s)


# ----------------------------------------------------------------------------- web page
STATIC = os.path.join(kb.APP_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


def main():
    global DEMO
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="no AI models: rule-based demo judge and assistant")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    DEMO = args.demo
    print("\nLoading LegalBERT argument structuring ..." if legalbert.model_present() else "")
    legalbert.load()
    e = engines()
    print("\nAI Courtroom")
    print(f"  Judge     : {'trained model ' + e['judge_model'] if e['judge_ready'] else 'DEMO (rule-based)'}")
    print(f"  Assistant : {'model ' + e['assistant_model'] if e['assistant_ready'] else 'DEMO (pattern-based)'}")
    lb = legalbert.status()
    print(f"  LegalBERT : {'argument structuring ON' if lb['ready'] else 'off (' + (lb['error'] or 'not installed') + ')'}")
    if not DEMO and not e["ollama_running"]:
        print("  Note: Ollama is not running, so demo mode is used. Start Ollama for the real models.")
    print(f"\n  Open http://localhost:{args.port} in your browser.  (Ctrl+C to stop)\n")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
