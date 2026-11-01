# JustAdvisor AI: LLM-Powered Moot Court and Verdict System

JustAdvisor AI is a moot court simulator for Indian law. Two human lawyers argue a case in turns, each helped by an AI legal assistant. When the arguments are done, a fine-tuned AI judge decides each legal issue separately and then gives a reasoned verdict.

Most legal AI projects read a finished judgment and predict the outcome. This one judges a live debate as it builds up: it works out which argument addresses which issue, rates the evidence behind it, and explains how it reached the verdict.

## Key results

| Model | Verdict accuracy | Issue accuracy |
|---|---|---|
| Always rule for Lawyer A | 50.0% | 75.6% |
| TF-IDF nearest-pattern baseline | 53.3% | 54.2% |
| **Fine-tuned judge (Qwen3-4B + LoRA)** | **85.0% (51/60)** | **85.1% (143/168)** |

All three are scored on held-out test debates whose argument wordings never appear in the training data.

## How it works

```
Browser (Lawyer A)   Browser (Lawyer B)
          \             /
           FastAPI server  (turn rules, limits, hearing records)
          /       |        \
   LegalBERT   Assistant    Judge
   (tags each  (TF-IDF over  (fine-tuned Qwen3-4B,
   sentence)   45 laws and   8-bit GGUF on CPU
               cases + Qwen3)  via Ollama)
```

1. **Judge.** Qwen3-4B-Instruct-2507, fine-tuned with LoRA, reads the case and the full transcript. It writes a finding (PROVED or NOT PROVED) for each issue and then a FINAL VERDICT.
2. **Assistant.** It retrieves relevant laws and precedents with TF-IDF and suggests 3 arguments the lawyer could make next. Each suggestion comes with a strength rating and its sources, returned as structured JSON.
3. **LegalBERT.** A fine-tuned `legal-bert-base-uncased` model tags every sentence as Claim, Evidence, LegalRule, Rebuttal or Conclusion. The assistant uses these tags to target the other side's evidence.

### The decision rule

Each argument gets an evidence grade:

| Grade | Meaning |
|---|---|
| 3 | Documentary, forensic or independent evidence |
| 2 | Oral testimony or circumstantial evidence |
| 1 | A bare assertion |
| 0 | Not relevant to the issue |

An issue is **proved** if Lawyer A's best argument on it is at least grade 2 and beats Lawyer B's best reply. Lawyer A wins only if **every** issue is proved, and ties go to the defence. The training labels are computed with this rule, so they never contradict each other.

## Supported cases

| Case | Law | Issues |
|---|---|---|
| CR-420 | Section 420 IPC, cheating | 3 |
| CR-406 | Section 406 IPC, criminal breach of trust | 2 |
| CR-FORGERY | Sections 468 and 471 IPC, forged sale deed | 3 |
| CV-73 | Section 73, Indian Contract Act, compensation for breach | 3 |
| CV-18 | Sections 18 and 19, Indian Contract Act, misrepresentation | 3 |

## Training details

- **Data:** 2,000 synthetic debates for training and 100 for testing, evenly split between wins for each side. The test set uses argument wordings that never appear in training (0 of 529 test turns repeat a training turn word for word).
- **Fine-tuning:**
  - LoRA with r=16, alpha=32 and dropout 0.05, applied to all 7 projection layers.
  - 33M trainable parameters, which is 0.81% of the model.
  - bf16, learning rate 2e-4 with a cosine schedule, effective batch size 16, 1 epoch (125 steps).
  - Loss counts only the judgment text, not the prompt.
- **Hardware:** NVIDIA RTX A4000 (16 GB).
- **Export:** the LoRA weights are merged into the model, which is converted to GGUF Q8_0 (4.3 GB) with llama.cpp. Ollama serves it in raw-prompt mode so the input is formatted exactly as in training.
- **Why the data was rebuilt:** the first dataset (folder `AI_Judge_Project`, Mistral-7B with QLoRA) picked verdicts at random. 47 of its 80 distinct prompts had contradictory labels, so no model could score above 82.9% on it.

## Repository structure

```
AI_Judge_v2/                     current version
  generate_judge_dataset_v2.py   cases, debate generator, prompt format
  train_judge_v2.py              LoRA fine-tuning and evaluation (GPU)
  export_for_cpu.py              merge and convert to GGUF for CPU
  cases_catalog.json             the 5 cases offered by the app
  train.log                      training and evaluation output
  app/
    server.py                    FastAPI backend and hearing rules
    judge.py                     judge prompt and verdict parsing
    assistant.py                 retrieval-grounded assistant
    legalbert.py                 sentence role tagging
    knowledge.py                 TF-IDF search over legal_kb
    legal_kb/                    25 laws and 20 case summaries
    static/                      web interface (HTML, CSS, JS)
    START_HERE.md                full setup guide
AI_Judge_Project/                first version (Mistral-7B, QLoRA)
01_pipeline.ipynb ...            original notebooks (data, LegalBERT, assistant)
```

## Running the app (no GPU needed)

Requirements: Python 3.10+, [Ollama](https://ollama.com), and about 16 GB of RAM.

Model weights are not in this repo because of GitHub's file size limits. Train the judge with `train_judge_v2.py` and `export_for_cpu.py`, then put the files in `AI_Judge_v2/app/models/`:

- `judge-v2-q8_0.gguf`, the judge.
- `legalbert/model.safetensors`, the LegalBERT weights.

```bash
cd AI_Judge_v2/app/models
ollama create judge-v2 -f Modelfile
ollama pull qwen3:4b-instruct-2507-q4_K_M

cd ..
pip install -r requirements.txt -r requirements-legalbert.txt
python server.py
```

Open http://localhost:8000. On Windows you can double-click `start_app.bat` instead.

To try the interface without any models, run `python server.py --demo`. See `AI_Judge_v2/app/START_HERE.md` for two-device setup and troubleshooting.

### Hearing rules

| Rule | Value |
|---|---|
| Rounds | up to 5 (Lawyer A argues, then Lawyer B) |
| Earliest verdict request | after 2 complete rounds |
| Length of one argument | 15 to 600 characters |
| Assistant requests | 5 per lawyer per hearing |

## Tech stack

Python, PyTorch, Hugging Face Transformers, TRL, PEFT (LoRA), Qwen3-4B, Legal-BERT, llama.cpp, GGUF, Ollama, FastAPI, JavaScript

## Limitations

- The training debates are synthetic and cover only 5 cases.
- The same proof rule is used for criminal and civil cases, even though their standards of proof differ.
- The cases use the IPC, which the Bharatiya Nyaya Sanhita replaced for offences committed from 1 July 2024.
- The LegalBERT training data is small (25 distinct sentences).
- This is an educational tool for moot court practice. It is not meant for real legal decisions.

## Author

**Subhajit Mahapatra**, MCA, Vellore Institute of Technology
[GitHub](https://github.com/Subhajit7710) · [LinkedIn](https://www.linkedin.com/in/subhajit-mahapatra-452136311)
