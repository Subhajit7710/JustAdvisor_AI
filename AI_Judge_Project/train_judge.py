import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# -------- MODEL --------
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# -------- TOKENIZER --------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# -------- MODEL LOAD (4-bit QLoRA) --------
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)


# -------- LoRA CONFIG --------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# -------- DATASET --------
dataset = load_dataset("json", data_files="judge_dataset.jsonl")

# -------- TRAINING CONFIG --------
training_args = TrainingArguments(
    output_dir="./judge_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,

    fp16=False,       
    bf16=False,       
    max_grad_norm=0.0, 

    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

# -------- TRAINER --------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    args=training_args,
)

trainer.train()
trainer.save_model("./judge_model")


