"""
AI Judge - dataset generator v2
================================

What changed from v1 (generate_advanced_judge_dataset.py):

* The verdict is no longer random. Every case has a list of ISSUES (the legal
  ingredients Lawyer A must prove). Every argument is tied to one issue and has
  an evidence strength:
        3 = strong  (documentary / forensic / independent evidence)
        2 = medium  (oral testimony or circumstantial evidence)
        1 = weak    (bare assertion, no evidence)
        0 = irrelevant (character attacks, sympathy, etc.)
  An issue is PROVED only if Lawyer A's best argument on it is at least medium
  AND stronger than Lawyer B's best rebuttal on it. Lawyer A wins only if every
  issue is proved. So the same transcript always gets the same verdict, and the
  model has to learn to read the arguments to get it right.

* Multi-round debates (Lawyer A and Lawyer B alternate), which is what the app
  will send to the judge.

* The judge writes a reasoned judgment: issue-by-issue analysis, which
  submissions were ignored, then "FINAL VERDICT: ...". The app parses that line.

* A held-out test set uses argument wordings the model never sees in training,
  so the test accuracy tells you whether it learned to reason or just memorised.

* Only 5 cases are supported (the "limited cases" for the first version of the
  app). `cases_catalog.json` lists them for the app.

Usage:
    python generate_judge_dataset_v2.py --train 2000 --test 100
Outputs: judge_train.jsonl, judge_test.jsonl, cases_catalog.json

The app should import `build_prompt` and `CASES` from this file so the prompt it
sends at runtime is exactly the format the model was trained on.
"""

import argparse
import json
import random

# --------------------------------------------------------------------------
# Strength labels used in the judgment text
# --------------------------------------------------------------------------
STRENGTH_LABEL = {
    3: "strong: documentary, forensic or independent evidence",
    2: "medium: oral testimony or circumstantial evidence",
    1: "weak: bare assertion without evidence",
}
STRENGTH_WORD = {3: "strong", 2: "medium", 1: "weak"}

NAMES = [
    "Ramesh Kumar", "Suresh Patil", "Anita Sharma", "Vikram Singh", "Pooja Iyer",
    "Rahul Mehta", "Farhan Qureshi", "Lakshmi Nair", "Arjun Reddy", "Neha Gupta",
    "Imran Shaikh", "Kavita Joshi", "Sanjay Das", "Meera Pillai", "Harpreet Kaur",
    "Rohit Banerjee", "Deepak Yadav", "Sunita Verma", "Joseph D'Souza", "Priya Menon",
    "Abhishek Tiwari", "Zoya Khan", "Manoj Pandey", "Shalini Rao", "Gaurav Chauhan",
]


def inr(n):
    """Format an integer in Indian style: 750000 -> Rs. 7,50,000"""
    s = str(n)
    if len(s) <= 3:
        return "Rs. " + s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return "Rs. " + ",".join(parts) + "," + tail


# --------------------------------------------------------------------------
# Argument pools.  Each pool entry is (text, gist).  Every list has 3 wordings:
# index 0 and 1 are used for training, index 2 is held out for the test set.
# Slots: {acc} accused/defendant, {comp} complainant/plaintiff, {amt}, {goods},
#        {days}, {purpose}, {place}, {project}, {loss}, {business}
# --------------------------------------------------------------------------

CASES = {}

# ============================ CASE 1: Cheating ==============================
CASES["CR-420"] = {
    "title": "Cheating over an unpaid advance for goods",
    "nature": "criminal",
    "law": "Section 420 IPC (cheating and dishonestly inducing delivery of property)",
    "section_short": "Section 420 IPC",
    "a_role": "the Prosecution (complainant {comp})",
    "b_role": "the Accused ({acc})",
    "facts": [
        "It is alleged that the accused, {acc}, received an advance payment of {amt} from the complainant, {comp}, for supplying {goods} within {days} days. The goods were not delivered and the advance has not been refunded. The complainant alleges cheating. The accused denies any dishonest intention.",
        "The complainant, {comp}, states that he paid {amt} to the accused, {acc}, as an advance for {goods} to be delivered within {days} days. No delivery was made and the money was not returned. The accused has been charged with cheating and pleads not guilty.",
    ],
    "slots": {"goods": ["electronic goods", "industrial machinery", "textile stock", "office furniture", "solar panels", "laptops"],
              "days": ["30", "45", "60", "90"]},
    "issues": [
        {
            "id": "deception",
            "issue": "the accused made a false representation to the complainant",
            "A": {
                3: [("The WhatsApp messages on record show {acc} told {comp} the goods were already in stock, while the supplier's letter proves no stock was ever ordered.", "WhatsApp messages and the supplier's letter showing the stock claim was false"),
                    ("The quotation issued by {acc} says 'ready stock available', but the GST records show his firm held no inventory of {goods} on that date.", "quotation claiming ready stock, contradicted by GST inventory records"),
                    ("Emails prove {acc} claimed a warehouse held the goods; the warehouse owner has deposed that no such goods were ever stored there.", "emails about warehouse stock contradicted by the warehouse owner")],
                2: [("{comp} testified that {acc} assured him orally that the goods were in stock, and his testimony stayed consistent in cross-examination.", "complainant's consistent testimony about a false stock assurance"),
                    ("A neighbouring trader deposed that {acc} was openly claiming to have stock he did not possess.", "a trader's testimony about false stock claims"),
                    ("{acc} repeatedly promised delivery dates he could not possibly have met, which points to a false representation.", "a pattern of impossible delivery promises")],
                1: [("The accused clearly lied to my client.", "bare allegation that the accused lied"),
                    ("Everyone knows the accused is not trustworthy, so he must have deceived the complainant.", "general allegation of untrustworthiness"),
                    ("It is obvious that a false promise was made.", "unsupported claim of a false promise")],
            },
            "B": {
                3: [("The purchase order {acc} placed with the manufacturer, and the manufacturer's written acknowledgement, prove the stock representation was true when made.", "manufacturer's acknowledged purchase order showing the representation was true"),
                    ("Bank records show {acc} had already paid the supplier for the goods before receiving the advance, so the representation was genuine.", "bank records of prior payment to the supplier"),
                    ("The supplier's invoice, dated before the deal, shows the goods were genuinely booked for the complainant.", "supplier invoice dated before the deal")],
                2: [("{acc}'s accountant testified that the order with the supplier was in process when the promise was made.", "accountant's testimony that the supply order was in process"),
                    ("The complainant admitted in cross-examination that the accused never claimed the goods were already in his godown.", "complainant's admission in cross-examination"),
                    ("A witness from the supplier's office said {acc} had checked availability before making the promise.", "supplier office witness on the prior availability check")],
                1: [("My client never lied.", "bare denial"),
                    ("The accused is an honest businessman and would never deceive anyone.", "assertion of good character"),
                    ("The complainant's story is simply false.", "bare denial of the complainant's version")],
            },
        },
        {
            "id": "intention",
            "issue": "the accused had a dishonest intention at the time of making the promise",
            "A": {
                3: [("Bank statements show the entire {amt} was withdrawn in cash the same day it was received and spent on personal expenses; no payment was ever made to any supplier.", "bank statements showing same-day diversion of the advance"),
                    ("Police recovered records showing {acc} collected similar advances from four other buyers that month without placing a single supplier order.", "recovered records of similar advances from other buyers with no supplier orders"),
                    ("{acc}'s firm had its GST registration cancelled a month before the deal, which he concealed, proving he never intended to supply.", "concealed cancellation of the firm's GST registration before the deal")],
                2: [("{acc} stopped answering calls immediately after receiving the money, which shows the intention was dishonest from the start.", "the accused cutting off contact right after payment"),
                    ("{acc} shifted his shop to an unknown address within a week of receiving the advance.", "shifting the shop soon after payment"),
                    ("No step at all was taken towards performance after the payment, which shows dishonest intention from the beginning.", "absence of any step towards performance")],
                1: [("The accused always intended to cheat.", "bare assertion of dishonest intent"),
                    ("Intention to cheat is self-evident in this case.", "unsupported claim that intent is self-evident"),
                    ("My client feels betrayed, which proves the accused meant to cheat him.", "complainant's feeling of betrayal")],
            },
            "B": {
                3: [("Bank records show {acc} transferred the money to his supplier within two days; the supplier then defaulted, as the supplier's own written admission confirms.", "bank transfer to the supplier and the supplier's written admission of default"),
                    ("The transport bill shows the goods were dispatched but seized at the state border, as the official seizure memo confirms.", "transport bill and official seizure memo"),
                    ("{acc} refunded part of the amount by cheque before the complaint was filed, as bank records show, which is inconsistent with an intention to cheat from the start.", "documented part-refund before the complaint")],
                2: [("{acc}'s employee testified that a fire in the godown disrupted the supply.", "employee's testimony about a godown fire"),
                    ("{acc} kept in touch with {comp} and offered revised delivery dates, as the complainant admitted.", "continued contact and revised delivery dates, admitted by the complainant"),
                    ("Two independent customers testified that {acc} has run the same business for ten years without a single complaint.", "independent customers' testimony of a clean business record")],
                1: [("Mere failure to deliver does not mean my client intended to cheat.", "bare submission that non-delivery is not cheating"),
                    ("My client had no bad intention.", "bare denial of dishonest intent"),
                    ("The delay happened due to circumstances beyond my client's control.", "unsupported claim of circumstances beyond control")],
            },
        },
        {
            "id": "inducement",
            "issue": "the complainant was induced by the representation to deliver money or property",
            "A": {
                3: [("The bank transfer receipt shows {comp} paid {amt} into {acc}'s account on the day of the representation, with the narration 'advance for {goods}'.", "bank transfer receipt with a narration linking the payment to the deal"),
                    ("The receipt signed by {acc} acknowledges {amt} received as an advance for {goods}.", "signed receipt acknowledging the advance"),
                    ("The written agreement records that the advance of {amt} was paid relying on {acc}'s assurance of supply.", "written agreement recording payment on the assurance")],
                2: [("{comp}'s brother witnessed the payment of {amt} to {acc}.", "eyewitness to the payment"),
                    ("{comp} testified that he paid the money only because of {acc}'s assurance.", "complainant's testimony on reliance"),
                    ("{comp}'s diary entry for that day records the payment to {acc}.", "complainant's diary entry of the payment")],
                1: [("My client gave the money because he was fooled.", "bare assertion of inducement"),
                    ("The payment obviously happened.", "unsupported claim of payment"),
                    ("The complainant would never have paid otherwise.", "speculation about the complainant's reasons")],
            },
            "B": {
                3: [("The loan agreement and bank records show the {amt} was repayment of an earlier unrelated loan, not an advance for the {goods}.", "loan agreement and bank records showing the payment was for an unrelated loan"),
                    ("The complainant's own ledger, produced in court, records the payment as repayment of an old debt.", "complainant's own ledger recording the payment as debt repayment"),
                    ("The bank-verified receipt is dated three months before the goods were ever discussed.", "bank-verified receipt dated before any discussion of the deal")],
                2: [("{acc}'s cousin, who was present, testified that the money was handed over as a family loan.", "witness testimony that the money was a family loan"),
                    ("The complainant admitted in cross-examination that he had given money to the accused earlier for unrelated ventures.", "complainant's admission of earlier unrelated payments"),
                    ("The amount paid does not match the price of the {goods}, which suggests it was paid for something else.", "mismatch between the payment and the price of the goods")],
                1: [("My client never took money on that basis.", "bare denial of inducement"),
                    ("The complainant paid voluntarily.", "bare assertion of voluntary payment"),
                    ("There was no inducement at all.", "bare denial")],
            },
        },
    ],
    "civil_note_if_only_failed": "intention",
    "civil_note": "The material on record shows at most a breach of contract. The dispute is civil in nature, and the complainant may pursue civil remedies.",
}

# ===================== CASE 2: Criminal breach of trust =====================
CASES["CR-406"] = {
    "title": "Criminal breach of trust over company funds",
    "nature": "criminal",
    "law": "Section 406 IPC (criminal breach of trust)",
    "section_short": "Section 406 IPC",
    "a_role": "the Prosecution (complainant {comp})",
    "b_role": "the Accused ({acc})",
    "facts": [
        "It is alleged that the accused, {acc}, an accounts manager in the firm of the complainant, {comp}, was entrusted with {amt} of company funds for {purpose}. The funds were not applied for that purpose. The firm alleges criminal breach of trust. The accused denies misappropriation.",
        "The complainant, {comp}, alleges that {amt} of company money meant for {purpose} was placed under the control of the accused, {acc}, the firm's accounts manager, and was not used for that purpose. The accused is charged with criminal breach of trust and pleads not guilty.",
    ],
    "slots": {"purpose": ["investment in fixed deposits", "payment of supplier dues", "purchase of raw material", "payment of employee salaries"]},
    "issues": [
        {
            "id": "entrustment",
            "issue": "the accused was entrusted with the property or had dominion over it",
            "A": {
                3: [("The board resolution and the authorisation letter signed by {acc} prove that {amt} was entrusted to him for {purpose}.", "board resolution and signed authorisation letter"),
                    ("Bank records show {amt} was transferred into an account operated solely by {acc} under the firm's mandate.", "bank records of transfer to an account the accused operated alone"),
                    ("The cash handover register, countersigned by {acc}, records his receipt of {amt}.", "cash handover register countersigned by the accused")],
                2: [("The firm's director testified that he handed {amt} to {acc} in his office.", "director's testimony about handing over the funds"),
                    ("A co-employee deposed that {acc} was in charge of the funds meant for {purpose}.", "co-employee's testimony that the accused was in charge"),
                    ("The HR manager testified that handling funds for {purpose} was part of {acc}'s duties.", "HR manager's testimony on the accused's duties")],
                1: [("The accused had the money, that is clear.", "bare assertion of entrustment"),
                    ("Everyone in the office knew the accused controlled the funds.", "unsupported claim of office knowledge"),
                    ("There is no doubt the accused was entrusted.", "bare assertion")],
            },
            "B": {
                3: [("The bank mandate required the joint signatures of the director and {acc}, and the director signed every withdrawal, so {acc} never had control of the funds.", "bank mandate requiring the director's joint signature on every withdrawal"),
                    ("The handwriting expert's report shows the signature of {acc} in the handover register is forged.", "handwriting expert's report showing the register signature is forged"),
                    ("Payroll records show {acc} had resigned and left the firm before the alleged date of entrustment.", "payroll records showing the accused had left before the entrustment")],
                2: [("{acc} testified that the funds were always kept by the director and he only maintained the ledger.", "accused's testimony that the director kept the funds"),
                    ("A clerk testified that the director kept the cash locker keys.", "clerk's testimony about the locker keys"),
                    ("The director admitted in cross-examination that he personally approved every payment.", "director's admission that he approved every payment")],
                1: [("My client was never given any money.", "bare denial of entrustment"),
                    ("The accused had nothing to do with the funds.", "bare denial"),
                    ("The entrustment story is made up.", "bare allegation of fabrication")],
            },
        },
        {
            "id": "misappropriation",
            "issue": "the accused dishonestly misappropriated the property or converted it to his own use",
            "A": {
                3: [("The forensic audit traces the {amt} from the firm's account to a flat bought in the name of {acc}'s wife.", "forensic audit tracing the funds to a flat bought by the accused's wife"),
                    ("Bank statements show the {amt} moved to {acc}'s personal account and was used to repay his personal car loan.", "bank statements showing transfer to the accused's personal account"),
                    ("The auditor's report shows the ledger entries for {purpose} are fictitious and the listed payees do not exist.", "auditor's report showing fictitious ledger entries")],
                2: [("Neighbours testified that {acc} started living lavishly and bought an expensive car soon after.", "neighbours' testimony about sudden lavish spending"),
                    ("{acc} could not explain where the funds went when questioned in the internal inquiry.", "failure to explain the funds in the internal inquiry"),
                    ("{acc} took leave on exactly the dates of the annual audit to avoid it.", "avoiding the annual audit")],
                1: [("The accused obviously took the money for himself.", "bare assertion of misappropriation"),
                    ("Where else could the money have gone?", "rhetorical speculation"),
                    ("The accused is greedy and must have used it.", "attack on the accused's character")],
            },
            "B": {
                3: [("Bank confirmations and receipts show the full {amt} was applied for {purpose} as instructed, and the auditor has verified each payment.", "bank confirmations and auditor-verified receipts for the authorised purpose"),
                    ("The director's own email instructed {acc} to transfer {amt} to the director's personal account, and bank records confirm the money went there.", "director's written instruction and bank records showing the funds went to the director"),
                    ("The statutory audit report for that period shows no shortfall at all in the firm's accounts.", "statutory audit report showing no shortfall")],
                2: [("{acc} testified that the funds were used in good faith and losses arose from market volatility.", "testimony of good-faith use and market losses"),
                    ("A former colleague testified that the firm's accounts were in disorder long before {acc} joined.", "colleague's testimony about pre-existing accounting disorder"),
                    ("The inquiry officer admitted that {acc} offered full accounts and cooperated with the inquiry.", "inquiry officer's admission of the accused's cooperation")],
                1: [("My client never misused a single rupee.", "bare denial of misappropriation"),
                    ("There is no misappropriation here.", "bare denial"),
                    ("The firm is blaming my client for its own losses.", "unsupported claim of scapegoating")],
            },
        },
    ],
    "civil_note_if_only_failed": "misappropriation",
    "civil_note": "Any dispute about the firm's accounts is a matter for civil accounting proceedings.",
}

# ============================ CASE 3: Forgery ===============================
CASES["CR-FORGERY"] = {
    "title": "Forged sale deed of a plot",
    "nature": "criminal",
    "law": "Sections 468 and 471 IPC (forgery for the purpose of cheating, and using a forged document as genuine)",
    "section_short": "Sections 468 and 471 IPC",
    "a_role": "the Prosecution (complainant {comp})",
    "b_role": "the Accused ({acc})",
    "facts": [
        "It is alleged that a sale deed transferring the complainant {comp}'s plot in {place} to the accused, {acc}, is forged, and that the accused used it to claim ownership and raise {amt}. The accused says the deed is genuine and that he bought the plot.",
        "The complainant, {comp}, owns a plot in {place}. He alleges that the accused, {acc}, produced a forged sale deed for the plot and used it to raise {amt}. The accused is charged with forgery and using a forged document, and pleads not guilty.",
    ],
    "slots": {"place": ["Pune", "Lucknow", "Indore", "Nagpur", "Bhopal", "Jaipur"]},
    "issues": [
        {
            "id": "forged",
            "issue": "the sale deed is a forged document",
            "A": {
                3: [("The handwriting expert's report confirms that {comp}'s signature on the sale deed is forged.", "handwriting expert's report that the signature is forged"),
                    ("The treasury office certifies that the stamp paper was printed two years after the date shown on the deed.", "treasury certificate that the stamp paper post-dates the deed"),
                    ("The sub-registrar's records have no entry for the registration number shown on the deed.", "sub-registrar's records with no entry for the deed")],
                2: [("{comp} testified that he never signed any sale deed, and his testimony was not shaken in cross-examination.", "complainant's unshaken testimony that he never signed"),
                    ("The attesting witness named in the deed testified that he never saw any deed being signed.", "attesting witness denying the signing"),
                    ("The signature on the deed looks visibly different from {comp}'s signatures on other documents before the court.", "visible difference from the complainant's other signatures")],
                1: [("The deed is fake.", "bare allegation that the deed is fake"),
                    ("My client would never sell his land.", "speculation about the complainant's intentions"),
                    ("Clearly the document is forged.", "bare assertion of forgery")],
            },
            "B": {
                3: [("The sub-registrar produced the registration record, with {comp}'s photograph and thumb impression taken at the time of registration.", "registration record with the complainant's photograph and thumb impression"),
                    ("The court-appointed handwriting expert found {comp}'s signature on the deed to be genuine.", "court-appointed expert finding the signature genuine"),
                    ("Bank records show the sale price was paid into {comp}'s account on the date of the deed.", "bank records of the sale price paid to the complainant")],
                2: [("The deed writer testified that {comp} was present and signed the deed before him.", "deed writer's testimony that the complainant signed"),
                    ("The attesting witness confirmed that he saw {comp} sign the deed.", "attesting witness confirming the signing"),
                    ("{comp} admitted in cross-examination that he had discussed selling the plot to {acc}.", "complainant's admission of sale discussions")],
                1: [("The deed is genuine.", "bare assertion that the deed is genuine"),
                    ("There is no forgery.", "bare denial"),
                    ("The complainant has simply changed his mind about the sale.", "unsupported claim that the complainant changed his mind")],
            },
        },
        {
            "id": "authorship",
            "issue": "the accused made the forged deed or used it knowing it to be forged",
            "A": {
                3: [("CCTV footage shows {acc} buying the stamp paper, and the stamp vendor's register bears his signature.", "CCTV footage and the stamp vendor's register"),
                    ("The cyber cell recovered the draft deed from {acc}'s laptop, and its edit history shows he typed it.", "cyber cell recovery of the draft from the accused's laptop"),
                    ("Messages recovered from {acc}'s phone show he knew {comp} had refused to sell, yet he presented the deed as genuine.", "recovered messages showing the accused knew there was no sale")],
                2: [("The deed writer testified that {acc} alone came to get the deed prepared.", "deed writer's testimony that the accused alone got the deed prepared"),
                    ("A witness testified that he saw {acc} practising signatures on blank paper.", "witness who saw the accused practising signatures"),
                    ("{acc} is the only person who benefited from the deed.", "the accused being the only beneficiary")],
                1: [("The accused must have made it.", "bare assertion of authorship"),
                    ("Who else would forge it?", "rhetorical speculation"),
                    ("The accused is the kind of person who forges documents.", "attack on the accused's character")],
            },
            "B": {
                3: [("The cyber cell report shows the draft deed was created on a computer belonging to a property broker, not {acc}.", "cyber cell report tracing the draft to a broker's computer"),
                    ("Travel records show {acc} was in another state on the day the deed was prepared and registered.", "travel records placing the accused in another state"),
                    ("Bank records show {acc} paid full consideration to the broker, who handed him the deed; the broker has been charged separately.", "bank records of payment to the broker, who has been charged separately")],
                2: [("{acc} testified that he bought the plot in good faith through a broker and had no reason to doubt the deed.", "accused's testimony of good-faith purchase through a broker"),
                    ("The broker's assistant testified that the broker handled all the paperwork.", "broker's assistant's testimony that the broker did the paperwork"),
                    ("{acc} reported the irregularity to the police himself once he learnt of the dispute.", "the accused reporting the irregularity himself")],
                1: [("My client did not make any document.", "bare denial of authorship"),
                    ("The accused had no idea about any of this.", "bare denial of knowledge"),
                    ("My client is innocent.", "bare assertion of innocence")],
            },
        },
        {
            "id": "purpose",
            "issue": "the forgery was committed for the purpose of cheating",
            "A": {
                3: [("The loan sanction letter shows {acc} mortgaged the plot using the deed and obtained {amt}, then defaulted.", "loan sanction letter showing a mortgage on the deed and default"),
                    ("Registry records show {acc} sold the plot to a third party for {amt} within a month, using the deed.", "registry records of a quick resale using the deed"),
                    ("Messages recovered from {acc}'s phone discuss using 'the papers' to raise money on the plot.", "recovered messages about using the papers to raise money")],
                2: [("A property dealer testified that {acc} tried to sell the plot quickly at a low price.", "property dealer's testimony about a hurried cheap sale"),
                    ("A neighbour testified that {acc} said he would 'make money from the plot soon'.", "neighbour's testimony about the accused's remark"),
                    ("{acc} tried to evict {comp}'s caretaker from the plot.", "attempt to evict the complainant's caretaker")],
                1: [("The purpose was obviously to cheat.", "bare assertion of a cheating purpose"),
                    ("The accused wanted the land, so he cheated.", "speculation about motive"),
                    ("Cheating was the goal.", "bare assertion")],
            },
            "B": {
                3: [("Bank records show the {amt} was raised against {acc}'s own salary, not the plot, and no transaction was ever made using the deed.", "bank records showing the money was raised against salary, not the deed"),
                    ("The court registry confirms {acc} voluntarily deposited the deed in court and has not dealt with the property since.", "court registry record of voluntary deposit of the deed"),
                    ("The revenue records show the plot has never been mortgaged or transferred by {acc}.", "revenue records showing no dealings with the plot")],
                2: [("{acc} testified that he only wanted to build a house for his family on the plot.", "accused's testimony about building a family home"),
                    ("The local revenue officer testified that {acc} never tried to sell or mortgage the plot.", "revenue officer's testimony of no attempted sale"),
                    ("{acc} offered to return the plot if the deed was found defective.", "offer to return the plot")],
                1: [("There was no intention to cheat.", "bare denial of a cheating purpose"),
                    ("My client is a simple man.", "assertion of simplicity"),
                    ("Cheating was never on his mind.", "bare denial")],
            },
        },
    ],
}

# ======================= CASE 4: Breach of contract =========================
CASES["CV-73"] = {
    "title": "Delayed construction contract and compensation",
    "nature": "civil",
    "law": "Section 73 of the Indian Contract Act, 1872 (compensation for breach of contract)",
    "section_short": "Section 73 of the Indian Contract Act, 1872",
    "a_role": "the Plaintiff ({comp})",
    "b_role": "the Defendant ({acc})",
    "facts": [
        "The plaintiff, {comp}, engaged the defendant contractor, {acc}, to build {project} for {amt}, to be completed within {days} days. The work was not completed within that period. The plaintiff claims compensation under Section 73 of the Indian Contract Act, 1872. The defendant disputes liability.",
        "The defendant, {acc}, a building contractor, was to construct {project} for the plaintiff, {comp}, for {amt} within {days} days. The project was not finished on time. The plaintiff has sued for compensation for breach of contract, and the defendant denies liability.",
    ],
    "slots": {"project": ["a two-storey house", "a warehouse", "an office interior", "a boundary wall and gate", "a school building extension"],
              "days": ["120", "180", "240", "300"]},
    "issues": [
        {
            "id": "contract",
            "issue": "a valid contract existed between the parties",
            "A": {
                3: [("The written agreement signed by both parties and two witnesses sets out the work, the price and the timeline.", "written agreement signed by both parties and witnesses"),
                    ("The work order signed by {acc} and his receipt for the advance prove the contract.", "work order and advance receipt signed by the defendant"),
                    ("The agreement on stamp paper, signed by both sides, records every term.", "agreement on stamp paper signed by both sides")],
                2: [("The architect testified that both parties agreed on the scope and price in his presence.", "architect's testimony on the agreed terms"),
                    ("{comp} testified that all terms were agreed at a meeting with {acc}.", "plaintiff's testimony about the agreed terms"),
                    ("{acc} started work on site, which shows a contract had been agreed.", "the defendant's conduct in starting work")],
                1: [("There was obviously a contract.", "bare assertion of a contract"),
                    ("Everyone knows they had a deal.", "unsupported claim of a deal"),
                    ("The defendant agreed, that is clear.", "bare assertion of agreement")],
            },
            "B": {
                3: [("The draft agreement relied upon was never signed by {acc}, and the plaintiff's own email says 'terms not yet finalised'.", "unsigned draft and the plaintiff's email saying terms were not finalised"),
                    ("The document relied upon leaves the price blank, so on its face no concluded contract exists.", "document with the price left blank"),
                    ("{acc}'s email, acknowledged by the plaintiff, expressly declines the proposed terms.", "defendant's acknowledged email declining the terms")],
                2: [("{acc} testified that the price was never finalised.", "defendant's testimony that the price was never fixed"),
                    ("The site supervisor testified that work began only on a trial basis.", "site supervisor's testimony about a trial start"),
                    ("The plaintiff admitted in cross-examination that some terms were still being negotiated.", "plaintiff's admission that terms were still under negotiation")],
                1: [("There was no contract.", "bare denial of a contract"),
                    ("My client never agreed to anything.", "bare denial of agreement"),
                    ("The plaintiff is inventing a contract.", "bare allegation of invention")],
            },
        },
        {
            "id": "breach",
            "issue": "the defendant breached the contract without lawful excuse",
            "A": {
                3: [("The completion certificate, dated well after the deadline, and the site inspection report prove the work was not finished within {days} days.", "completion certificate and inspection report showing the deadline was missed"),
                    ("The architect's certified progress reports show only 40 percent of the work was done by the deadline.", "architect's certified progress reports"),
                    ("{acc}'s own letter admits that the project was abandoned midway.", "defendant's own letter admitting abandonment")],
                2: [("A neighbour testified that work on the site stopped for months.", "neighbour's testimony that work stopped"),
                    ("{comp} testified that {acc} stopped coming to the site.", "plaintiff's testimony that the defendant stopped coming"),
                    ("Photographs taken by the plaintiff show the site lying idle.", "plaintiff's photographs of an idle site")],
                1: [("The defendant failed to perform.", "bare assertion of breach"),
                    ("The work was terrible.", "general complaint about the work"),
                    ("The defendant did not keep his word.", "bare assertion")],
            },
            "B": {
                3: [("The change orders signed by the plaintiff extended the scope and timeline, and the work was completed within the extended period.", "plaintiff's signed change orders extending the timeline"),
                    ("The payment record shows the plaintiff withheld instalments due under the contract, and the contract allows {acc} to suspend work for non-payment.", "payment record of withheld instalments and the suspension clause"),
                    ("The municipal stop-work order on record halted construction for reasons not attributable to {acc}.", "municipal stop-work order")],
                2: [("{acc} testified that heavy monsoon rains delayed the work.", "defendant's testimony about monsoon delays"),
                    ("A labour contractor testified that a market shortage of materials caused the delay.", "labour contractor's testimony about material shortages"),
                    ("The plaintiff admitted he made oral design changes during construction.", "plaintiff's admission of oral design changes")],
                1: [("My client did nothing wrong.", "bare denial of breach"),
                    ("The delay was not our fault.", "bare denial of fault"),
                    ("The plaintiff is exaggerating.", "bare allegation of exaggeration")],
            },
        },
        {
            "id": "loss",
            "issue": "the plaintiff suffered loss caused by the breach",
            "A": {
                3: [("The lease agreement and rent receipts show the plaintiff paid {loss} in rent for other premises because of the delay.", "lease agreement and rent receipts for substitute premises"),
                    ("Invoices from a replacement contractor show the plaintiff paid {loss} extra to finish the work.", "replacement contractor's invoices"),
                    ("A chartered accountant's certified statement quantifies the plaintiff's loss at {loss}.", "chartered accountant's certified loss statement")],
                2: [("{comp} testified that he had to live in rented accommodation during the delay.", "plaintiff's testimony about rented accommodation"),
                    ("The plaintiff's business partner testified that a planned business opening was postponed.", "business partner's testimony about a postponed opening"),
                    ("{comp} testified that material costs rose while he waited.", "plaintiff's testimony about rising costs")],
                1: [("My client suffered a lot.", "bare assertion of suffering"),
                    ("The plaintiff has been hugely inconvenienced.", "general claim of inconvenience"),
                    ("Obviously there was loss.", "bare assertion of loss")],
            },
            "B": {
                3: [("The plaintiff's own bank statements show no rent payments for the period claimed, and he was living in his family house.", "plaintiff's bank statements showing no rent was paid"),
                    ("The replacement contractor's invoice is for additional work outside the original contract.", "invoice showing the extra cost was for work outside the contract"),
                    ("The registered sale deed shows the plaintiff sold the completed property at a profit, so no loss was caused.", "registered sale deed showing a profitable sale")],
                2: [("{acc} testified that the plaintiff had not planned to occupy the building until the next year.", "defendant's testimony that occupation was not planned yet"),
                    ("A local broker testified that rents in the area are negligible.", "broker's testimony about negligible rents"),
                    ("The plaintiff admitted in cross-examination that he never actually moved out.", "plaintiff's admission that he never moved out")],
                1: [("There was no loss.", "bare denial of loss"),
                    ("The plaintiff lost nothing.", "bare denial"),
                    ("The loss claimed is imaginary.", "bare allegation that the loss is imaginary")],
            },
        },
    ],
}

# ======================== CASE 5: Misrepresentation =========================
CASES["CV-18"] = {
    "title": "Investment induced by misrepresented business figures",
    "nature": "civil",
    "law": "Sections 18 and 19 of the Indian Contract Act, 1872 (misrepresentation; contract voidable)",
    "section_short": "Sections 18 and 19 of the Indian Contract Act, 1872",
    "a_role": "the Plaintiff ({comp})",
    "b_role": "the Defendant ({acc})",
    "facts": [
        "The plaintiff, {comp}, bought a stake in the defendant {acc}'s business, {business}, for {amt} after the defendant shared its financial figures. The plaintiff seeks to set aside the contract for misrepresentation under Sections 18 and 19 of the Indian Contract Act, 1872, and a refund. The defendant disputes the claim.",
        "The defendant, {acc}, runs {business}. The plaintiff, {comp}, invested {amt} in it after receiving financial figures from the defendant. The plaintiff says the figures were false and seeks to avoid the contract and recover the money. The defendant denies any misrepresentation.",
    ],
    "slots": {"business": ["a bakery chain", "a logistics startup", "a coaching institute", "a textile unit", "a restaurant"]},
    "issues": [
        {
            "id": "false_statement",
            "issue": "the defendant made a false representation of fact",
            "A": {
                3: [("The audited accounts show actual revenue was less than half of what {acc}'s investor presentation stated.", "audited accounts contradicting the investor presentation"),
                    ("The GST returns filed by the business contradict the turnover figure in {acc}'s written statement to the plaintiff.", "GST returns contradicting the stated turnover"),
                    ("The business's bank statements show the 'existing contracts' listed in {acc}'s pitch never produced a single payment.", "bank statements showing the claimed contracts were fictitious")],
                2: [("{comp} testified that {acc} told him the business was profitable.", "plaintiff's testimony about an oral claim of profitability"),
                    ("A former employee testified that the business was running at a loss at the time.", "former employee's testimony about losses"),
                    ("The plaintiff's accountant testified that the figures appeared inflated.", "plaintiff's accountant's opinion that the figures looked inflated")],
                1: [("The defendant lied about the numbers.", "bare allegation of lying"),
                    ("The figures were false.", "bare assertion of falsity"),
                    ("Everything the defendant said was untrue.", "sweeping unsupported allegation")],
            },
            "B": {
                3: [("The audited accounts at the time of the deal match exactly the figures given to the plaintiff.", "audited accounts matching the figures given"),
                    ("The chartered accountant's certificate confirms that the revenue figures were accurate on that date.", "chartered accountant's certificate of accuracy"),
                    ("The written projections were labelled 'estimates, not guaranteed', and the actual figures supplied match the audited accounts.", "projections labelled as estimates, with actual figures matching the audit")],
                2: [("{acc} testified that he gave honest estimates based on experience.", "defendant's testimony of honest estimates"),
                    ("The business manager testified that the figures came straight from the internal books.", "manager's testimony that the figures came from the books"),
                    ("The plaintiff admitted that {acc} called the future figures only hopes.", "plaintiff's admission that future figures were described as hopes")],
                1: [("My client never lied.", "bare denial"),
                    ("The numbers were fine.", "bare assertion that the numbers were correct"),
                    ("There was no misrepresentation.", "bare denial of misrepresentation")],
            },
        },
        {
            "id": "reliance",
            "issue": "the representation induced the plaintiff to enter into the contract",
            "A": {
                3: [("The plaintiff's investment memo, prepared before signing, relies expressly on {acc}'s revenue figures.", "plaintiff's pre-signing investment memo relying on the figures"),
                    ("The share purchase agreement recites that the plaintiff invested relying on the financial statements supplied by {acc}.", "recital in the share purchase agreement"),
                    ("Emails show the plaintiff agreed to invest within a day of receiving {acc}'s figures, stating 'based on these numbers, I am in'.", "plaintiff's email agreeing to invest based on the numbers")],
                2: [("{comp} testified that he invested only because of the figures.", "plaintiff's testimony on reliance"),
                    ("The plaintiff's wife testified that he decided after seeing the figures.", "wife's testimony about the decision"),
                    ("The plaintiff had no other source of information about the business.", "absence of any other source of information")],
                1: [("My client relied on the figures, obviously.", "bare assertion of reliance"),
                    ("Anyone would have invested after seeing those figures.", "speculation"),
                    ("The figures were the reason.", "bare assertion")],
            },
            "B": {
                3: [("The plaintiff's own due-diligence report by his chartered accountant independently verified the business, and he relied on that report.", "plaintiff's independent due-diligence report"),
                    ("The term sheet signed by the plaintiff is dated two months before the figures were shared.", "term sheet signed before the figures were shared"),
                    ("The share purchase agreement contains the plaintiff's signed declaration that he did not rely on any projections.", "plaintiff's signed non-reliance declaration")],
                2: [("{acc} testified that the plaintiff was keen to invest even before seeing any figures.", "defendant's testimony about prior keenness"),
                    ("The plaintiff admitted that he invested because a friend recommended the business.", "plaintiff's admission that a friend's recommendation drove the investment"),
                    ("A mutual friend testified that the plaintiff had decided to invest a month earlier.", "mutual friend's testimony about an earlier decision")],
                1: [("The plaintiff did not rely on anything.", "bare denial of reliance"),
                    ("The plaintiff wanted to invest anyway.", "unsupported claim"),
                    ("Reliance is not proved.", "bare submission")],
            },
        },
        {
            "id": "not_discoverable",
            "issue": "the plaintiff could not have discovered the truth with ordinary diligence",
            "A": {
                3: [("{acc}'s email on record refuses the plaintiff's request to inspect the books.", "defendant's email refusing inspection of the books"),
                    ("Registrar of Companies records show the true accounts were filed only after the deal.", "ROC records showing the true accounts were filed after the deal"),
                    ("{acc}'s signed letter told the plaintiff that inspection was unnecessary and denied access to the bank statements.", "defendant's signed letter denying access")],
                2: [("{comp} testified that he asked for the books but was put off.", "plaintiff's testimony that he was put off"),
                    ("The defendant's manager testified that the books were kept locked.", "manager's testimony that the books were locked"),
                    ("The plaintiff is not a finance professional and could not have spotted the problem.", "plaintiff's lack of financial expertise")],
                1: [("My client could not have known.", "bare assertion"),
                    ("The truth was hidden.", "bare allegation of concealment"),
                    ("It was impossible to find out.", "bare assertion")],
            },
            "B": {
                3: [("The data-room access logs show the plaintiff had full access to the books and bank statements before signing.", "data-room logs showing full access before signing"),
                    ("Audited accounts with the true figures were publicly filed with the Registrar of Companies before the deal.", "public ROC filing of the true figures before the deal"),
                    ("The plaintiff's own adviser's report, prepared before signing, flagged the revenue gap.", "plaintiff's adviser's report flagging the gap before signing")],
                2: [("{acc} testified that he offered the books to the plaintiff.", "defendant's testimony that he offered the books"),
                    ("The defendant's accountant testified that the plaintiff visited the office and saw the ledgers.", "accountant's testimony that the plaintiff saw the ledgers"),
                    ("The plaintiff admitted that he chose not to hire an accountant.", "plaintiff's admission that he hired no accountant")],
                1: [("The plaintiff could have checked.", "bare assertion"),
                    ("It was the plaintiff's own fault.", "bare allegation of fault"),
                    ("Due diligence was the plaintiff's job.", "bare submission")],
            },
        },
    ],
}

# Irrelevant submissions (no legal weight).  {accrole} = "the accused"/"the defendant"
IRRELEVANT = {
    "A": [("{accrole_cap} has a bad reputation in the locality.", "attack on the other party's reputation"),
          ("{comp} is a respected member of society.", "praise of my client's standing"),
          ("This case has caused my client great mental stress.", "appeal to sympathy"),
          ("{accrole_cap} drives an expensive car and lives in a big house.", "remarks about the other party's lifestyle"),
          ("A strong decision is needed to send a message to society.", "appeal to deterrence"),
          ("My client has suffered sleepless nights over this.", "appeal to sympathy")],
    "B": [("My client has a family to support.", "appeal to sympathy"),
          ("The other side has filed cases against other people before.", "attack on the other party's litigation history"),
          ("My client is a senior citizen.", "the party's age"),
          ("My client donates to charity regularly.", "the party's charitable work"),
          ("The other side's lawyer is being very aggressive.", "complaint about opposing counsel"),
          ("My client is well known in the community.", "the party's reputation")],
}

PREFIXES = ["", "", "", "", "Your Honour, ", "My Lord, ", "It is respectfully submitted that ",
            "I submit that ", "With respect, ", "Your Honour, I would point out that "]


# --------------------------------------------------------------------------
# Prompt (the app MUST use this exact function at inference time)
# --------------------------------------------------------------------------
def build_prompt(case_id, facts, turns, acc=None, comp=None):
    """
    case_id : key in CASES
    facts   : facts paragraph (string)
    turns   : list of dicts {"round": int, "speaker": "A" or "B", "text": str}
    acc/comp: party names (used in the role lines)
    """
    c = CASES[case_id]
    criminal = c["nature"] == "criminal"
    a_role = c["a_role"].format(acc=acc or "", comp=comp or "")
    b_role = c["b_role"].format(acc=acc or "", comp=comp or "")
    lines = [
        "You are a neutral AI Judge presiding over a simulated Indian court. Decide ONLY on the facts and the arguments in the transcript. Do not assume facts that are not stated.",
        "Weigh each submission by its quality: documentary, forensic or independent evidence is strong; oral testimony or circumstantial evidence is medium; bare assertions are weak; submissions unrelated to the issues carry no weight. An issue is proved only if Lawyer A's best submission on it is at least medium and outweighs Lawyer B's best rebuttal on it.",
        "",
        f"CASE: {c['title']}",
        f"LAW: {c['law']}",
        f"NATURE: {'Criminal' if criminal else 'Civil'}. STANDARD OF PROOF: {'beyond reasonable doubt' if criminal else 'balance of probabilities'}. The burden of proof is on Lawyer A.",
        f"LAWYER A appears for {a_role}. LAWYER B appears for {b_role}.",
        "",
        "FACTS:",
        facts,
        "",
        "ISSUES (each must be proved by Lawyer A):",
    ]
    for i, iss in enumerate(c["issues"], 1):
        lines.append(f"{i}. Whether {iss['issue']}.")
    lines += ["", "DEBATE TRANSCRIPT:"]
    for t in turns:
        lines.append(f"[Round {t['round']}] Lawyer {t['speaker']}: {t['text']}")
    lines += ["", "Deliver a reasoned judgment."]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Verdict logic
# --------------------------------------------------------------------------
def issue_proved(a, b):
    return a >= 2 and a > b


def finding_sentence(a, b, criminal):
    if a == 0:
        return "NOT PROVED. Lawyer A made no submission on this issue, so the burden of proof is not discharged."
    if a == 1:
        tail = " even though it is not rebutted" if b == 0 else ""
        return f"NOT PROVED. A bare assertion cannot discharge the burden of proof{tail}."
    if b == 0:
        return f"PROVED. Lawyer A's {STRENGTH_WORD[a]} evidence stands unrebutted."
    if b < a:
        return f"PROVED. Lawyer A's {STRENGTH_WORD[a]} evidence outweighs Lawyer B's {STRENGTH_WORD[b]} response."
    if criminal:
        return (f"NOT PROVED. Lawyer B's {STRENGTH_WORD[b]} rebuttal is at least as weighty as Lawyer A's "
                f"{STRENGTH_WORD[a]} evidence, which raises a reasonable doubt.")
    return (f"NOT PROVED. Lawyer B's {STRENGTH_WORD[b]} rebuttal is at least as weighty as Lawyer A's "
            f"{STRENGTH_WORD[a]} evidence, so the issue is not established on the balance of probabilities.")


PROVED_PAIRS = [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
FAILED_PAIRS = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3),
                (2, 2), (2, 3), (3, 3)]
FAILED_WEIGHTS = [1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3]


def fill(text, ctx):
    return text.format(**ctx)


def with_prefix(text, rng):
    p = rng.choice(PREFIXES)
    if not p:
        return text
    # lower-case the first letter after a prefix, but keep names / "I" capitalised
    first_word = text.split(" ")[0]
    if first_word in ("The", "A", "An", "There", "It", "Everyone", "Bank", "My", "This", "Who", "Where",
                      "Mere", "No", "Intention", "Clearly", "Obviously", "Any", "Anyone", "Due", "Reliance",
                      "Cheating", "Payroll", "Travel", "Messages", "Emails", "Invoices", "Registry",
                      "Neighbours", "Photographs", "Audited", "Police", "CCTV", "Two"):
        text = text[0].lower() + text[1:]
    return p + text


def make_example(case_id, rng, split, target=None):
    c = CASES[case_id]
    criminal = c["nature"] == "criminal"
    wording_ids = [0, 1] if split == "train" else [2]

    acc, comp = rng.sample(NAMES, 2)
    amt_n = rng.choice([45000, 75000, 120000, 250000, 400000, 750000, 1200000, 2500000])
    loss_n = int(amt_n * rng.choice([0.1, 0.15, 0.2, 0.25, 0.3]))
    loss_n = max(5000, (loss_n // 1000) * 1000)
    ctx = {"acc": acc, "comp": comp, "amt": inr(amt_n), "loss": inr(loss_n),
           "accrole_cap": "The accused" if criminal else "The defendant"}
    for k, vals in c.get("slots", {}).items():
        ctx[k] = rng.choice(vals)
    facts = fill(rng.choice(c["facts"]), ctx)

    # --- choose outcome first (balanced), then strengths consistent with it
    n_issues = len(c["issues"])
    if target is None:
        target = rng.random() < 0.5          # True = Lawyer A wins
    if target:
        failing = set()
    else:
        k = rng.choices(range(1, n_issues + 1), weights=[3] + [1] * (n_issues - 1))[0]
        failing = set(rng.sample(range(n_issues), k))

    # args[side] = list of dicts {issue_idx or None, strength, text, gist}
    args = {"A": [], "B": []}
    strengths = []
    for i, iss in enumerate(c["issues"]):
        if i in failing:
            a, b = rng.choices(FAILED_PAIRS, weights=FAILED_WEIGHTS)[0]
        else:
            a, b = rng.choice(PROVED_PAIRS)
        strengths.append((a, b))
        for side, s in (("A", a), ("B", b)):
            if s == 0:
                continue
            text, gist = iss[side][s][rng.choice(wording_ids)]
            args[side].append({"issue": i, "s": s, "text": fill(text, ctx), "gist": gist})
            # sometimes a side also makes a weaker extra point on the same issue
            if s >= 2 and rng.random() < 0.15:
                s2 = rng.randint(1, s - 1)
                text2, gist2 = iss[side][s2][rng.choice(wording_ids)]
                args[side].append({"issue": i, "s": s2, "text": fill(text2, ctx), "gist": gist2})

    # irrelevant submissions
    irr_ids = list(range(len(IRRELEVANT["A"])))
    irr_pool_ids = irr_ids[:4] if split == "train" else irr_ids[4:]
    for side in ("A", "B"):
        n_irr = rng.choices([0, 1, 2], weights=[5, 4, 1])[0]
        if not args[side]:
            n_irr = max(n_irr, 1)
        for j in rng.sample(irr_pool_ids, n_irr):
            text, gist = IRRELEVANT[side][j]
            args[side].append({"issue": None, "s": 0, "text": fill(text, ctx), "gist": gist})

    # --- arrange into turns (sometimes two points in one turn)
    side_turns = {}
    for side in ("A", "B"):
        pts = args[side][:]
        rng.shuffle(pts)
        turns = []
        while pts:
            if len(pts) >= 2 and rng.random() < 0.25:
                turns.append([pts.pop(0), pts.pop(0)])
            else:
                turns.append([pts.pop(0)])
        side_turns[side] = turns

    n_rounds = max(len(side_turns["A"]), len(side_turns["B"]))
    transcript = []
    for r in range(n_rounds):
        for side in ("A", "B"):
            if r < len(side_turns[side]):
                pts = side_turns[side][r]
                for p in pts:
                    p["round"] = r + 1
                text = " ".join(with_prefix(p["text"], rng) if k == 0 else p["text"] for k, p in enumerate(pts))
                transcript.append({"round": r + 1, "speaker": side, "text": text})

    prompt = build_prompt(case_id, facts, transcript, acc=acc, comp=comp)

    # --- judgment
    out = ["ANALYSIS"]
    failed_names = []
    for i, iss in enumerate(c["issues"]):
        a, b = strengths[i]
        out.append(f"Issue {i + 1}: Whether {iss['issue']}.")
        for side in ("A", "B"):
            pts = sorted([p for p in args[side] if p["issue"] == i], key=lambda p: p["round"])
            if not pts:
                out.append(f"- Lawyer {side}: no submission on this issue.")
            for p in pts:
                out.append(f"- Lawyer {side} (Round {p['round']}): {p['gist']} ({STRENGTH_LABEL[p['s']]}).")
        out.append(f"- Finding: {finding_sentence(a, b, criminal)}")
        if not issue_proved(a, b):
            failed_names.append(i + 1)

    out.append("SUBMISSIONS GIVEN NO WEIGHT")
    irr = sorted([(p["round"], side, p["gist"]) for side in ("A", "B") for p in args[side] if p["issue"] is None])
    if irr:
        for rnd, side, gist in irr:
            out.append(f"- Lawyer {side} (Round {rnd}): {gist}. This is not relevant to any issue.")
    else:
        out.append("- None.")

    won = not failed_names
    assert won == target
    fl = ", ".join(str(x) for x in failed_names)
    plural = "s" if len(failed_names) > 1 else ""
    if criminal:
        if won:
            out.append("FINAL VERDICT: GUILTY")
            out.append(f"Every issue is proved beyond reasonable doubt. The accused, {acc}, is convicted under {c['section_short']}.")
        else:
            out.append("FINAL VERDICT: NOT GUILTY")
            msg = (f"Issue{plural} {fl} {'are' if plural else 'is'} not proved beyond reasonable doubt. "
                   f"The accused, {acc}, is acquitted of the charge under {c['section_short']}.")
            only = c.get("civil_note_if_only_failed")
            if only and [c["issues"][x - 1]["id"] for x in failed_names] == [only]:
                msg += " " + c["civil_note"]
            out.append(msg)
    else:
        if won:
            out.append("FINAL VERDICT: CLAIM ALLOWED")
            if case_id == "CV-73":
                out.append(f"Every issue is proved on the balance of probabilities. The defendant, {acc}, is liable to compensate the plaintiff, {comp}, for the loss proved, under {c['section_short']}.")
            else:
                out.append(f"Every issue is proved on the balance of probabilities. The contract is voidable at the option of the plaintiff, {comp}, under {c['section_short']}. The plaintiff may rescind it and recover the {ctx['amt']} paid.")
        else:
            out.append("FINAL VERDICT: CLAIM DISMISSED")
            out.append(f"Issue{plural} {fl} {'are' if plural else 'is'} not proved on the balance of probabilities. The claim of the plaintiff, {comp}, against the defendant, {acc}, under {c['section_short']} is dismissed.")

    completion = "\n".join(out)
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": completion}],
        "meta": {"case_id": case_id, "split": split, "winner": "A" if won else "B",
                 "verdict": out[-2].replace("FINAL VERDICT: ", ""),
                 "issue_findings": ["PROVED" if issue_proved(a, b) else "NOT PROVED" for a, b in strengths]},
    }


def build_catalog():
    """The fixed list of cases the app offers, with one canonical set of facts each."""
    canon = {
        "CR-420": {"acc": "Suresh Patil", "comp": "Ramesh Kumar", "amt": inr(750000), "goods": "industrial machinery", "days": "60"},
        "CR-406": {"acc": "Vikram Singh", "comp": "Anita Sharma", "amt": inr(1200000), "purpose": "payment of supplier dues"},
        "CR-FORGERY": {"acc": "Rahul Mehta", "comp": "Lakshmi Nair", "amt": inr(2500000), "place": "Pune"},
        "CV-73": {"acc": "Arjun Reddy", "comp": "Neha Gupta", "amt": inr(2500000), "project": "a two-storey house", "days": "180"},
        "CV-18": {"acc": "Imran Shaikh", "comp": "Kavita Joshi", "amt": inr(1200000), "business": "a bakery chain"},
    }
    cat = []
    for cid, c in CASES.items():
        ctx = dict(canon[cid])
        cat.append({
            "id": cid,
            "title": c["title"],
            "nature": c["nature"],
            "law": c["law"],
            "lawyer_a_role": c["a_role"].format(**ctx),
            "lawyer_b_role": c["b_role"].format(**ctx),
            "acc": ctx["acc"],
            "comp": ctx["comp"],
            "facts": c["facts"][0].format(**ctx),
            "issues": [f"Whether {i['issue']}." for i in c["issues"]],
            "possible_verdicts": ["GUILTY", "NOT GUILTY"] if c["nature"] == "criminal" else ["CLAIM ALLOWED", "CLAIM DISMISSED"],
        })
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    case_ids = list(CASES)
    for split, n, fname in (("train", args.train, "judge_train.jsonl"), ("test", args.test, "judge_test.jsonl")):
        with open(f"{args.outdir}/{fname}", "w", encoding="utf-8") as f:
            for k in range(n):
                cid = case_ids[k % len(case_ids)]
                target = (k // len(case_ids)) % 2 == 0   # exact 50/50 balance per case
                ex = make_example(cid, rng, split, target=target)
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"wrote {n} {split} examples -> {fname}")

    with open(f"{args.outdir}/cases_catalog.json", "w", encoding="utf-8") as f:
        json.dump(build_catalog(), f, indent=2, ensure_ascii=False)
    print("wrote cases_catalog.json")


if __name__ == "__main__":
    main()
