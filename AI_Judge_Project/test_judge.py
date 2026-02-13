import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ==================================================
# MODEL CONFIGURATION
# ==================================================
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
ADAPTER_PATH = "judge_model"

# ==================================================
# TOKENIZER
# ==================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# ==================================================
# 4-BIT CONFIG (SAME AS TRAINING)
# ==================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# ==================================================
# LOAD MODEL + LORA ADAPTER
# ==================================================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# ==================================================
# FINAL MANUAL PROMPT (CANONICAL FORMAT)
# ==================================================
prompt = (
    "<s>[INST] "
    "You are a neutral AI Judge presiding over a simulated courtroom.\n\n"
    "IMPORTANT RULES:\n"
    "- Decide strictly on the basis of stated facts and arguments.\n"
    "- Do NOT assume any unstated facts or external circumstances.\n"
    "- The verdict must be based on cumulative evaluation of all arguments.\n"
    "- No single argument shall be decisive on its own.\n\n"
    "Case facts:\n"
    "The accused entered into a contract to supply industrial machinery to the complainant.\n"
    "The accused received an advance payment of Rs. 7,50,000.\n"
    "The machinery was not delivered within the agreed contractual period.\n"
    "No written termination of the contract occurred.\n\n"
    "Lawyer A arguments:\n"
    "1. The accused accepted advance payment without having the capacity to supply machinery.\n"
    "2. No purchase orders or supplier invoices were produced.\n"
    "3. Communication ceased after receipt of money.\n"
    "4. Repeated false assurances of delivery were made.\n"
    "5. The advance amount was not refunded despite demands.\n"
    "6. Funds were diverted for unrelated personal use.\n"
    "7. The accused lacked prior experience in machinery supply.\n"
    "8. Material procurement difficulties were concealed.\n"
    "9. Similar complaints exist against the accused.\n"
    "10. Dishonest intention existed from the inception of the contract.\n\n"
    "Lawyer B arguments:\n"
    "1. A valid contract existed between the parties.\n"
    "2. Delay alone does not constitute cheating.\n"
    "3. Attempts were made to negotiate with suppliers.\n"
    "4. Market conditions affected availability.\n"
    "5. Obligation to deliver was never denied.\n"
    "6. Financial difficulty does not imply dishonest intent.\n"
    "7. The contract was never formally terminated.\n"
    "8. Civil remedies were available.\n"
    "9. Willingness to refund was expressed later.\n"
    "10. Criminal proceedings are being misused for recovery.\n\n"
    "Judicial Reasoning Instructions:\n"
    "1. List all arguments from both sides.\n"
    "2. Evaluate each argument independently.\n"
    "3. Assign relative weight (High / Medium / Low).\n"
    "4. Weigh arguments cumulatively.\n"
    "5. Deliver a reasoned verdict.\n\n"
    "OUTPUT FORMAT:\n"
    "Arguments Summary:\n"
    "Evaluation & Weighting:\n"
    "Final Verdict:\n"
    "Assumptions Check (Yes/No):\n"
    "[/INST] "
    "Arguments Summary:\n"
)

# ==================================================
# TOKENIZE & MOVE TO GPU
# ==================================================
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# ==================================================
# GENERATE VERDICT
# ==================================================
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=350,
        temperature=0.0,
        do_sample=False,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id
    )

# ==================================================
# PRINT OUTPUT
# ==================================================
verdict = tokenizer.decode(output[0], skip_special_tokens=True)

print("\n================ AI JUDGE VERDICT ================\n")
print(verdict)

