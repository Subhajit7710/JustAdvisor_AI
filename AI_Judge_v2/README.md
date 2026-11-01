# AI Judge v2

The new AI judge (trained 2026) and the Nyaya Moot Court app. This folder is separate from the old v1 judge in `AI_Judge_Project/`. Nothing here uses or changes v1.

## Folder contents

| File / folder | Purpose | Used on |
|---|---|---|
| `generate_judge_dataset_v2.py` | Defines the 5 cases, builds the training data and the judge's prompt format. The app imports it, so keep it here. | everywhere |
| `cases_catalog.json` | The 5 cases the app offers (facts, parties, issues) | everywhere |
| `train_judge_v2.py` | Trains, tests and packages the judge (GPU) | lab PC |
| `export_for_cpu.py` | Turns the trained judge into one file that runs without a GPU | lab PC (after training) |
| `app/` | The web app: backend, UI, assistant, legal knowledge base | friend's PC |
| `app/START_HERE.md` | Setup guide for the app computer | friend's PC |
| `app/models/legalbert/` | Your trained LegalBERT (copied from `models/legalbert/final`). It tags each sentence of every argument. | friend's PC |

Created during the steps below (not in the folder yet):
`judge_model_v2/` + `judge_model_v2.zip` (trained adapter), `app/models/judge-v2-q8_0.gguf` (judge for CPU).

## The three steps

**1. Train (lab PC with GPU, in `~/judge_env`):**
```
nohup python -u train_judge_v2.py > train.log 2>&1 &
tail -f train.log
```
At the end it prints the verdict accuracy (aim for 90%+) and creates `judge_model_v2.zip`.

**2. Export for CPU (same lab PC, right after training):**
```
nohup python -u export_for_cpu.py > export.log 2>&1 &
tail -f export.log
```
It creates `app/models/judge-v2-q8_0.gguf` (~4.3 GB), `Modelfile` and `prompt_format.json`. Upload the `.gguf` to Drive. It's large, so upload it on its own, and the rest of the folder separately.

**3. Run the app (friend's PC, no GPU needed):** follow `app/START_HERE.md`. You can try it right away, before training, with `python app/server.py --demo`.

## How the pieces fit

```
Browser (two lawyers)
   │  arguments, "Ask assistant", "Ask for verdict"
   ▼
app/server.py ── rules: turns, 5 rounds max, 600 chars, 5 cases only
   ├─ legalbert.py ── every sentence → Claim / Evidence / LegalRule / Rebuttal / Conclusion
   │                  (shown as tags, grouped per lawyer, saved in records, given to the assistant)
   ├─ assistant.py ── searches app/legal_kb (laws + cases) + opponent's LegalBERT structure
   │                  → Ollama: qwen3:4b-instruct-2507
   └─ judge.py ────── builds the exact training prompt → Ollama: judge-v2 (your trained model)
                       parses issue-by-issue findings + FINAL VERDICT
```
