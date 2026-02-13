import json
import random

CRIMINAL_CASES = [
    {
        "fact": "The accused received an advance payment of Rs. 75,000 for supplying electronic goods but neither delivered the goods nor refunded the amount.",
        "section": "Section 420 IPC",
        "law_test": "dishonest intention at the inception of the transaction",
        "guilty_reason": "The accused accepted payment without any subsequent effort to perform the promise, indicating fraudulent intent from the beginning.",
        "not_guilty_reason": "The delay in delivery was caused by supply chain disruptions and does not conclusively prove dishonest intent at inception."
    },
    {
        "fact": "The accused was entrusted with company funds for investment purposes but diverted the money for personal use.",
        "section": "Section 406 IPC",
        "law_test": "dishonest misappropriation of entrusted property",
        "guilty_reason": "Bank records establish diversion of entrusted funds for personal expenditure.",
        "not_guilty_reason": "The funds were invested in good faith and losses occurred due to market volatility."
    },
    {
        "fact": "The accused forged property documents to unlawfully transfer ownership.",
        "section": "Sections 468 and 471 IPC",
        "law_test": "forgery with intent to cheat and use of forged documents",
        "guilty_reason": "Forensic examination confirms the falsification of signatures used for unlawful gain.",
        "not_guilty_reason": "There is insufficient proof linking the accused directly to the creation of forged documents."
    }
]

CIVIL_CASES = [
    {
        "fact": "The defendant failed to complete a construction project within the agreed timeline despite receiving partial payment.",
        "section": "Section 73 Contract Act",
        "law_test": "compensation for breach of contract",
        "verdict_reason": "The evidence indicates non-performance without fraudulent intent, attracting civil liability only."
    },
    {
        "fact": "The agreement was entered into based on inaccurate financial projections provided by one party.",
        "section": "Section 18 Contract Act",
        "law_test": "misrepresentation affecting free consent",
        "verdict_reason": "The misstatement materially influenced consent, rendering the contract voidable."
    }
]

LAWYER_A_ARGUMENTS = [
    "The conduct of the accused demonstrates clear mens rea and dishonest intention.",
    "The evidence on record satisfies the statutory ingredients of the offence.",
    "The accused abused the trust reposed in him for unlawful gain.",
    "The act goes beyond a mere contractual breach and attracts criminal liability."
]

LAWYER_B_ARGUMENTS = [
    "The dispute is purely civil in nature without criminal intent.",
    "There is no direct evidence establishing dishonest intention at inception.",
    "Procedural lapses weaken the prosecution's case.",
    "The benefit of doubt must be extended to the accused."
]

def generate_sample():
    criminal = random.choice([True, False])

    if criminal:
        case = random.choice(CRIMINAL_CASES)
        guilty = random.choice([True, False])

        verdict_reason = case["guilty_reason"] if guilty else case["not_guilty_reason"]
        verdict = (
            f"Upon evaluation of evidence, the court finds that the legal requirement of "
            f"{case['law_test']} is {'satisfied' if guilty else 'not conclusively established'}. "
            f"{verdict_reason} "
            f"{'The accused is convicted under ' + case['section'] + '.' if guilty else 'Accordingly, criminal liability is not established.'}"
        )

    else:
        case = random.choice(CIVIL_CASES)
        verdict = (
            f"The matter pertains to civil obligations governed by {case['section']}. "
            f"{case['verdict_reason']} Criminal prosecution is unwarranted."
        )

    text = f"""<s>[INST] You are a neutral AI Judge tasked with evaluating legal disputes objectively.

Case Facts:
{case['fact']}

Lawyer A:
{random.choice(LAWYER_A_ARGUMENTS)}

Lawyer B:
{random.choice(LAWYER_B_ARGUMENTS)}

[/INST]
Verdict:
{verdict}
</s>"""

    return {"text": text}

SAMPLES = 750  # You can safely increase to 1000+

with open("judge_dataset.jsonl", "w", encoding="utf-8") as f:
    for _ in range(SAMPLES):
        json.dump(generate_sample(), f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Generated {SAMPLES} realistic AI Judge training samples.")
