# Nyaya Moot Court: running the app

A web app where two lawyers argue one of 5 cases in turns, each helped by an AI assistant, and the trained AI judge delivers a reasoned verdict. It runs on an ordinary computer **without a graphics card** (16 GB+ RAM recommended; 24 GB is plenty).

## What runs where

| Part | What it is | Size in memory |
|---|---|---|
| Judge | your trained model, `judge-v2` (8-bit GGUF) | ~4.5 GB |
| Assistant | base Qwen3-4B-Instruct from Ollama | ~2.7 GB |
| Argument structuring | your trained **LegalBERT** (`app/models/legalbert`). It tags every sentence as Claim, Evidence, Legal rule, Rebuttal or Conclusion. | ~0.6 GB |
| App | `server.py` (Python) + web page | small |

**Ollama** runs both models on the processor. The app talks to Ollama and serves the web page.

## One-time setup

1. **Install Python 3.10 or newer** from python.org. On Windows, tick **"Add python.exe to PATH"** during install.
2. **Install Ollama** from https://ollama.com/download and start it. On Windows it runs in the system tray.
3. **Copy the whole `AI_Judge_v2` folder** to this computer. Check that `app/models/` contains:
   - `judge-v2-q8_0.gguf` and `Modelfile`, created by `export_for_cpu.py` after training
   - the `legalbert` folder with `model.safetensors` (about 440 MB) and its config and tokenizer files
4. **Open a terminal in `AI_Judge_v2/app/models`** and run:
   ```
   ollama create judge-v2 -f Modelfile
   ollama pull qwen3:4b-instruct-2507-q4_K_M
   ```
   The first command loads the judge into Ollama, which takes about a minute. The second downloads the assistant model, about 2.5 GB.

## Start the app

- **Windows:** double-click `app/start_app.bat`
- **Linux or macOS:** `bash app/start_app.sh`
- **Or manually:** `pip install -r requirements.txt -r requirements-legalbert.txt`, then `python server.py`

The first start installs PyTorch for LegalBERT, about 200 MB, so it takes a few minutes. If that install fails, the app still runs, just without the sentence tags.

Then open **http://localhost:8000** in a browser. The top-right corner shows whether the judge and assistant are using the AI models or demo mode.

### No model yet? Use demo mode
`python server.py --demo` (or `start_app.bat --demo`) runs the whole app with a rule-based judge and pattern-based suggestions. It's useful for trying the interface before training is finished. Demo verdicts are clearly labelled.

### Two devices (each lawyer on their own laptop or phone)
Start with `python server.py --host 0.0.0.0`. On the other device, open `http://<this computer's IP address>:8000`. To find the address, run `ipconfig` on Windows or `hostname -I` on Linux. Enter the **hearing code** shown at the top of the hearing, and pick "Lawyer A only" or "Lawyer B only" at the bottom of the page. Allow Python through the Windows firewall if asked.

## Rules of the hearing (in `server.py`, easy to change)

| Setting | Default |
|---|---|
| Rounds (A argues, then B) | at most 5 |
| Verdict can be requested after | 2 complete rounds |
| Length of one argument | 15 to 600 characters |
| Assistant requests per lawyer per case | 5 |
| Cases | the 5 in `cases_catalog.json` (the ones the judge was trained on) |

## How long things take (no graphics card)

- **First request after starting:** about 10–30 s while Ollama loads the model. Models then stay loaded for 30 minutes.
- **Assistant suggestions:** about 15–40 s.
- **Verdict:** about 1–3 minutes. The judgment text appears live as it's written.

## Records

Each finished hearing is saved as JSON in `app/records/`, with the full transcript and judgment.

## Troubleshooting

| What you see | Fix |
|---|---|
| "Judge: demo mode" although you set it up | Ollama isn't running, or `ollama list` doesn't show `judge-v2`. Start Ollama and rerun step 4. |
| `python` not found (Windows) | Reinstall Python with "Add to PATH" ticked, or use `py server.py`. |
| Verdict says the text was incomplete | Rare. Ask for the verdict again from a new hearing. If it keeps happening, the model may need retraining with more data. |
| Page doesn't load from another device | Use `--host 0.0.0.0` and allow Python through the firewall. |
| No Claim/Evidence tags on arguments | The terminal shows `LegalBERT : off (reason)`. Check that `app/models/legalbert/model.safetensors` exists, and run `pip install -r requirements-legalbert.txt`. |
