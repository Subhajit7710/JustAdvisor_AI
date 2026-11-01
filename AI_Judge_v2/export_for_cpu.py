#!/usr/bin/env python3
"""
AI Judge v2 - package the trained judge so it runs on a normal computer WITHOUT a GPU.

Run this AFTER train_judge_v2.py has finished, in the same folder and environment:

    source ~/judge_env/bin/activate
    python -u export_for_cpu.py

What it does:
  1. merges the trained LoRA adapter (judge_model_v2/) into the Qwen3-4B base model
  2. converts the merged model to a single GGUF file (8-bit, ~4.3 GB) using llama.cpp
  3. writes app/models/Modelfile (for Ollama) and app/models/prompt_format.json

Result: app/models/judge-v2-q8_0.gguf  -> copy the whole AI_Judge_v2 folder to the
computer that will run the app (see app/START_HERE.md).

Needs: ~20 GB free disk, ~16 GB RAM, internet (downloads llama.cpp from GitHub once).
Works on CPU too (slower, ~10-20 min). Options:
  --adapter DIR      trained adapter folder (default: judge_model_v2)
  --outtype q8_0     GGUF type: q8_0 (default, 4.3 GB) or f16 (8 GB) or bf16
  --keep_merged      keep the merged full model folder (8 GB) after converting
"""
import argparse
import functools
import json
import os
import shutil
import subprocess
import sys

print = functools.partial(print, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
PLACEHOLDER = "<<<JUDGE_PROMPT>>>"


def banner(t):
    print("\n" + "=" * 70 + f"\n {t}\n" + "=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="judge_model_v2")
    ap.add_argument("--base", default=None, help="base model (default: read from adapter_config.json)")
    ap.add_argument("--outdir", default=os.path.join("app", "models"))
    ap.add_argument("--outtype", default="q8_0", choices=["q8_0", "f16", "bf16"])
    ap.add_argument("--keep_merged", action="store_true")
    args = ap.parse_args()
    os.chdir(HERE)

    if not os.path.exists(os.path.join(args.adapter, "adapter_config.json")):
        sys.exit(f"ERROR: no trained adapter in '{args.adapter}'. Run train_judge_v2.py first "
                 f"(or unzip judge_model_v2.zip into a folder named judge_model_v2 here).")
    base = args.base or json.load(open(os.path.join(args.adapter, "adapter_config.json")))["base_model_name_or_path"]
    os.makedirs(args.outdir, exist_ok=True)
    merged_dir = "merged_judge_v2"
    gguf_path = os.path.join(args.outdir, f"judge-v2-{args.outtype}.gguf")

    # ---------------------------------------------------------------- merge
    banner(f"1/3  Merging adapter '{args.adapter}' into base '{base}'")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.adapter if os.path.exists(os.path.join(args.adapter, "tokenizer_config.json")) else base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    del model
    print("Merged model saved to", merged_dir)

    # exact prompt wrapper used in training (chat template), so the app sends identical text
    rendered = tok.apply_chat_template([{"role": "user", "content": PLACEHOLDER}], tokenize=False,
                                       add_generation_prompt=True)
    prefix, suffix = rendered.split(PLACEHOLDER)
    stops = sorted({t for t in [tok.eos_token, "<|im_end|>", "<|endoftext|>"] if t})
    fmt = {"prefix": prefix, "suffix": suffix, "stop": stops, "base_model": base}
    with open(os.path.join(args.outdir, "prompt_format.json"), "w", encoding="utf-8") as f:
        json.dump(fmt, f, indent=2)
    print("Prompt format:", json.dumps(fmt))

    # ---------------------------------------------------------------- convert
    banner(f"2/3  Converting to GGUF ({args.outtype})")
    if not os.path.exists(os.path.join("llama.cpp", "convert_hf_to_gguf.py")):
        print("Downloading llama.cpp converter from GitHub ...")
        subprocess.run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, "llama.cpp"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf", "sentencepiece", "protobuf",
                    "pyyaml", "tqdm", "numpy"], check=True)
    subprocess.run([sys.executable, os.path.join("llama.cpp", "convert_hf_to_gguf.py"), merged_dir,
                    "--outfile", gguf_path, "--outtype", args.outtype], check=True)
    print(f"Created {gguf_path} ({os.path.getsize(gguf_path) / 1e9:.1f} GB)")

    # ---------------------------------------------------------------- Modelfile
    banner("3/3  Writing Ollama Modelfile")
    modelfile = [f"FROM ./{os.path.basename(gguf_path)}",
                 "PARAMETER temperature 0",
                 "PARAMETER num_ctx 4096"]
    modelfile += [f'PARAMETER stop "{s}"' for s in stops]
    with open(os.path.join(args.outdir, "Modelfile"), "w", encoding="utf-8") as f:
        f.write("\n".join(modelfile) + "\n")
    shutil.copy(os.path.join(args.adapter, "MODEL_INFO.txt"), os.path.join(args.outdir, "MODEL_INFO.txt")) \
        if os.path.exists(os.path.join(args.adapter, "MODEL_INFO.txt")) else None

    if not args.keep_merged:
        shutil.rmtree(merged_dir, ignore_errors=True)

    print(f"""
ALL DONE.
  {gguf_path}
  {os.path.join(args.outdir, 'Modelfile')}
  {os.path.join(args.outdir, 'prompt_format.json')}

Next: copy the whole AI_Judge_v2 folder to the app computer and follow app/START_HERE.md
(the .gguf file is ~4.3 GB - upload it to Drive on its own if the folder is too big).""")


if __name__ == "__main__":
    main()
