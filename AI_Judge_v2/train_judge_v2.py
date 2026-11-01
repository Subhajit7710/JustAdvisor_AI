#!/usr/bin/env python3
"""
AI Judge v2 - train, test and package the new judge from the terminal.

Put this file in the AI_Judge_v2 folder next to generate_judge_dataset_v2.py
and cases_catalog.json, then run (inside the judge_env environment):

    nohup python -u train_judge_v2.py > train.log 2>&1 &
    tail -f train.log            # watch progress (Ctrl+C stops watching, NOT training)

What it does:
  1. generates the dataset (2000 train / 100 held-out test debates)
  2. fine-tunes Qwen3-4B-Instruct with LoRA (bf16) on the GPU
  3. tests the judge on held-out debates and prints the verdict accuracy
  4. writes MODEL_INFO.txt and creates judge_model_v2.zip

If it stops halfway, just run the same command again: it continues from the
last checkpoint automatically (use --fresh to start over).

Options (all optional):
  --n_train 1000     fewer training examples, about half the time
  --max_len 1536     use if you get "CUDA out of memory"
  --gpu 1            use a different GPU
  --model PATH       local model folder if huggingface.co is blocked
  --fresh            ignore old checkpoints and start from scratch
"""
import argparse
import datetime
import functools
import glob
import json
import os
import re
import shutil
import subprocess
import sys

print = functools.partial(print, flush=True)   # show output immediately in train.log

HERE = os.path.dirname(os.path.abspath(__file__))


def banner(text):
    print("\n" + "=" * 70 + f"\n {text}\n" + "=" * 70)


def main():
    ap = argparse.ArgumentParser(description="Train the AI Judge v2")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--n_eval", type=int, default=None, help="test cases to check (default: auto)")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output", default="judge_model_v2")
    ap.add_argument("--fresh", action="store_true", help="start from scratch, ignore checkpoints")
    ap.add_argument("--no_eval", action="store_true", help="skip the test step")
    ap.add_argument("--allow_cpu", action="store_true", help=argparse.SUPPRESS)  # for testing only
    args = ap.parse_args()

    os.chdir(HERE)
    for f in ("generate_judge_dataset_v2.py", "cases_catalog.json"):
        if not os.path.exists(f):
            sys.exit(f"ERROR: {f} is missing. Put it in the same folder as this script: {HERE}")

    # ------------------------------------------------------------------ GPU
    banner("1/5  Checking GPU")
    import torch
    import transformers, trl, peft
    print(f"torch {torch.__version__} | transformers {transformers.__version__} | trl {trl.__version__} | peft {peft.__version__}")
    on_gpu = torch.cuda.is_available()
    if not on_gpu and not args.allow_cpu:
        sys.exit("ERROR: PyTorch can't see the GPU. Check `nvidia-smi` works, and that torch was installed "
                 "with --index-url https://download.pytorch.org/whl/cu126")
    if on_gpu:
        torch.cuda.set_device(args.gpu)
        props = torch.cuda.get_device_properties(args.gpu)
        free, total = torch.cuda.mem_get_info(args.gpu)
        gpu_gb = total / 1e9
        print(f"GPU {args.gpu}: {props.name} | free {free/1e9:.1f} GB of {gpu_gb:.1f} GB")
        use_bf16 = torch.cuda.is_bf16_supported()
        if free / 1e9 < 13:
            print("WARNING: less than 13 GB free on this GPU - someone else may be using it (check nvidia-smi).")
    else:
        gpu_gb, use_bf16 = 0, False

    if gpu_gb >= 35:
        batch, accum, eval_batch, n_eval = 4, 4, 10, 100
    else:                       # RTX A4000 / 16 GB class (and CPU test)
        batch, accum, eval_batch, n_eval = 1, 16, 4, 60
    if args.n_eval is not None:
        n_eval = args.n_eval
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if on_gpu else torch.float32)
    print(f"Settings: batch {batch} x accumulation {accum} (effective 16), dtype {dtype}, max_len {args.max_len}")

    # ------------------------------------------------------------------ DATA
    banner("2/5  Generating dataset")
    subprocess.run([sys.executable, "generate_judge_dataset_v2.py",
                    "--train", str(args.n_train), "--test", str(args.n_test)], check=True)

    # ------------------------------------------------------------------ MODEL
    banner(f"3/5  Loading base model {args.model}  (first run downloads ~8 GB)")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {"dtype": dtype}
    if on_gpu:
        load_kwargs["device_map"] = {"": args.gpu}
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if on_gpu:
        print("GPU memory used by model: %.1f GB" % (torch.cuda.memory_allocated(args.gpu) / 1e9))

    # ------------------------------------------------------------------ TRAIN
    banner("4/5  Training  (loss should fall below ~0.1)")
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig

    train_ds = load_dataset("json", data_files={"train": "judge_train.jsonl"})["train"].remove_columns(["meta"])
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        max_length=args.max_len,
        completion_only_loss=True,              # learn the judgment, not the prompt
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=on_gpu and use_bf16,
        fp16=on_gpu and not use_bf16,
        optim="adamw_torch",
        report_to="none",
        use_cpu=not on_gpu,
    )
    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_ds,
                         processing_class=tokenizer, peft_config=peft_config)
    trainer.model.print_trainable_parameters()

    ckpts = glob.glob(os.path.join(args.output, "checkpoint-*"))
    if args.fresh and os.path.isdir(args.output):
        shutil.rmtree(args.output)
        ckpts = []
    if ckpts:
        print(f"Found {len(ckpts)} checkpoint(s) - RESUMING from the latest one")
    t0 = datetime.datetime.now()
    trainer.train(resume_from_checkpoint=True if ckpts else None)
    print(f"Training finished in {datetime.datetime.now() - t0}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("Saved adapter to", os.path.abspath(args.output))

    # ------------------------------------------------------------------ TEST
    results = []
    if not args.no_eval:
        banner(f"5/5  Testing on {n_eval} held-out debates (unseen wordings)")
        jm = trainer.model
        jm.eval()
        jm.config.use_cache = True
        tokenizer.padding_side = "left"

        def run_judge(prompts, max_new_tokens=900):
            texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                   add_generation_prompt=True) for p in prompts]
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(jm.device)
            with torch.no_grad():
                out = jm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
            return [tokenizer.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]

        def parse(text):
            m = re.search(r"FINAL VERDICT:\s*(GUILTY|NOT GUILTY|CLAIM ALLOWED|CLAIM DISMISSED)", text)
            return (m.group(1) if m else None), re.findall(r"Finding:\s*(NOT PROVED|PROVED)", text)

        test = [json.loads(l) for l in open("judge_test.jsonl", encoding="utf-8")][:n_eval]
        for i in range(0, len(test), eval_batch):
            batch_ex = test[i:i + eval_batch]
            outs = run_judge([ex["prompt"][0]["content"] for ex in batch_ex])
            for ex, out in zip(batch_ex, outs):
                verdict, findings = parse(out)
                results.append({"case": ex["meta"]["case_id"], "gold": ex["meta"]["verdict"], "pred": verdict,
                                "gold_issues": ex["meta"]["issue_findings"], "pred_issues": findings, "text": out})
            print(f"  checked {len(results)}/{len(test)}")

        v_ok = sum(r["pred"] == r["gold"] for r in results)
        i_tot = sum(len(r["gold_issues"]) for r in results)
        i_ok = sum(sum(p == g for p, g in zip(r["pred_issues"], r["gold_issues"])) for r in results)
        print(f"\n  VERDICT ACCURACY: {v_ok}/{len(results)} = {100 * v_ok / len(results):.1f}%   (aim: 90%+)")
        print(f"  ISSUE ACCURACY:   {i_ok}/{i_tot} = {100 * i_ok / max(i_tot, 1):.1f}%")
        for cid in sorted(set(r["case"] for r in results)):
            rs = [r for r in results if r["case"] == cid]
            print(f"    {cid:<11} {sum(r['pred'] == r['gold'] for r in rs)}/{len(rs)}")
        with open(os.path.join(args.output, "eval_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("\n----- sample judgment -----")
        print(test[0]["prompt"][0]["content"].split("DEBATE TRANSCRIPT:")[1].strip())
        print("\n" + results[0]["text"])

    # ------------------------------------------------------------------ PACKAGE
    banner("Packaging")
    for ck in glob.glob(os.path.join(args.output, "checkpoint-*")):
        shutil.rmtree(ck)
    shutil.copy("cases_catalog.json", args.output)
    shutil.copy("generate_judge_dataset_v2.py", args.output)
    info = [
        "AI JUDGE - VERSION 2 (NEW)",
        "==========================",
        f"Trained on     : {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"Base model     : {args.model}",
        f"Precision      : {'bf16' if use_bf16 else str(dtype)} LoRA (r=16)",
        f"Dataset        : generate_judge_dataset_v2.py ({args.n_train} train / {args.n_test} test, 5 cases)",
        f"Epochs         : {args.epochs}",
    ]
    if results:
        v_ok = sum(r["pred"] == r["gold"] for r in results)
        info.append(f"Test verdict accuracy: {v_ok}/{len(results)} = {100 * v_ok / len(results):.1f}%")
    info += ["",
             "This is NOT the old judge in AI_Judge_Project/judge_model (v1, Mistral-7B-v0.1,",
             "random-verdict dataset). Do not mix the two: they use different base models",
             "and different prompt formats. Use build_prompt() from generate_judge_dataset_v2.py."]
    with open(os.path.join(args.output, "MODEL_INFO.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(info) + "\n")
    print("\n".join(info))
    zip_path = shutil.make_archive("judge_model_v2", "zip", args.output)
    print(f"\nALL DONE. Upload this file to Drive:\n  {zip_path}  ({os.path.getsize(zip_path)/1e6:.0f} MB)")
    print("\nNext, to run the judge on a computer without a GPU:  python -u export_for_cpu.py")


if __name__ == "__main__":
    main()
