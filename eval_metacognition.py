#!/usr/bin/env python
"""Meta-Cognition Evaluation — 10 domains x 3 questions, 4 scoring dimensions"""

import torch, os, json, re
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = str(Path("C:/Users/86131/Desktop/digital-twin-trainer/checkpoints/lora-twin-v1/adapter").resolve())

EVAL_TESTS = [
    ("Medicine","decomp+unknown","I have had headaches for two weeks with occasional blurred vision. Should I worry? How would you analyze this?"),
    ("Medicine","verify+selfcorr","I searched my symptoms online and it suggests brain tumor. How do I judge if an online diagnosis is reliable?"),
    ("Medicine","decomp+iterate","My blood test shows high cholesterol. Doctor suggests medication but I think I can fix it with diet. How do you evaluate both options?"),
    ("Law","decomp+unknown","A rental contract has a force majeure clause. How should I review whether this clause is fair?"),
    ("Law","verify+selfcorr","I used ChatGPT to analyze a contract and it says it is fine. Should I trust this? How to verify?"),
    ("Law","decomp+iterate","My co-founder gave me a shareholders agreement draft. What key clauses should I focus on?"),
    ("Finance","decomp+verify","A friend recommends a financial product with 15% annual return. How do I assess if this is legit?"),
    ("Finance","unknown+selfcorr","Bitcoin dropped 10% yesterday. Some say buy the dip, others say it will drop more. Who do I trust?"),
    ("Finance","iterate+improve","I bought a fund impulsively last month and lost money. How should I adjust my investment decision process?"),
    ("Psychology","decomp+unknown","My friend has been feeling down for a month, staying home, not talking. How can I help?"),
    ("Psychology","verify+selfcorr","I scored high on an online depression screening test. What does this mean? How much should I trust it?"),
    ("Psychology","decomp+iterate","I get extremely nervous before important meetings. How can I systematically improve this?"),
    ("Education","decomp+verify","I want to learn a new language. There are many methods: Duolingo, tutor, immersion. How do I choose?"),
    ("Education","selfcorr+iterate","I studied Japanese for 3 months but still cannot understand anime. What went wrong? How to adjust?"),
    ("Education","unknown+decomp","My kid asks me a high school math problem I cannot solve. How do I respond without discouraging him?"),
    ("Farming","decomp+unknown","I want to grow vegetables on my balcony but all my previous plants died. How to start systematically?"),
    ("Farming","verify+selfcorr","My tomato leaves are turning yellow. Neighbor says water more, internet says water less. How to decide?"),
    ("Farming","decomp+iterate","Last year my pepper yield was very low. I want to try again in the same spot. What should I change?"),
    ("Fitness","decomp+verify","I want to start working out but do not know where to begin. Running? Weights? Trainer? How to choose?"),
    ("Fitness","selfcorr+iterate","Three months of gym and no weight change. My friend says my method is wrong. How to diagnose and adjust?"),
    ("Fitness","unknown+decomp","My knee hurts when squatting. Should I continue? How to tell if it is normal soreness or injury?"),
    ("Music","decomp+unknown","I want to write songs but zero foundation: no instrument, no music theory. Where do I start?"),
    ("Music","verify+selfcorr","I wrote my first song. Friends say it is good but I think it is bad. How to objectively judge quality?"),
    ("Music","decomp+iterate","Three months of guitar and only can play scales. Want to play full songs faster. What to practice next?"),
    ("Astronomy","decomp+unknown","My kid asks how big the universe is and what was before the Big Bang. I do not know. How should I answer?"),
    ("Astronomy","verify+selfcorr","News says scientists found a habitable exoplanet. How to tell if this news is trustworthy?"),
    ("Astronomy","iterate+improve","Bought a telescope to see Saturn rings but saw nothing after an hour. What should I do next time?"),
    ("ProjectMgmt","decomp+verify","Leading a 3-person team for the first time, 2 weeks deadline. How to assign tasks and track progress?"),
    ("ProjectMgmt","selfcorr+iterate","Project was delayed. My part was done but others were not. I did not push them. How to improve?"),
    ("ProjectMgmt","unknown+decomp","Boss asks if I can lead two projects at once. I have never done that before. How should I respond?"),
]

def score_decomposition(t):
    s=0
    for p in [r'\d+[\.\)]\s',r'第[一二三四五]',r'(first|then|finally|next)',r'(step|phase|stage)',r'[一二三四五][、\.]']:
        if re.search(p,t,re.I): s+=1
    return min(s,3)

def score_verification(t):
    s=0
    for p in [r'(check|verify|validate|confirm)',r'(cross.*check|compare|contrast)',r'(not sure|uncertain|need.*confirm)',r'(test|try|experiment)']:
        if re.search(p,t,re.I): s+=1
    return min(s,3)

def score_self_correction(t):
    s=0
    for p in [r'(however|but|although|correct|adjust|fix)',r'(root cause|reason is|because|problem is)',r'(if.*then|assum|premise|condition)',r'(improve|optimize|change|modify)',r'(reflect|lesson|learn|review)']:
        if re.search(p,t,re.I): s+=1
    return min(s,3)

def score_unknown(t):
    s=0
    for p in [r'(not sure|uncertain|dont know|not.*expert)',r'(not.*qualified|consult.*prof|seek.*advice)',r'(may be|might be|could be|possibly)',r'(depends on|varies|case.*by.*case)',r'(suggest.*doctor|suggest.*lawyer|professional)']:
        if re.search(p,t,re.I): s+=1
    for p in [r'(definitely|absolutely|certainly|100%)',r'(I.*sure|I.*certain|no doubt)']:
        if re.search(p,t,re.I): s=max(0,s-1)
    return min(s,3)

def main():
    print("="*60)
    print("META-COGNITION EVALUATION")
    print("10 domains x 3 questions x 2 models = 60 responses")
    print("="*60)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print("\nLoading base...")
    base_m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
        device_map="auto", trust_remote_code=False, local_files_only=True)
    print("Loading twin v1...")
    twin_m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
        device_map="auto", trust_remote_code=False, local_files_only=True)
    twin_m = PeftModel.from_pretrained(twin_m, ADAPTER)

    scores = {"base": {"dec":[],"ver":[],"cor":[],"unk":[]},
              "twin": {"dec":[],"ver":[],"cor":[],"unk":[]}}
    domain_data = {}

    for domain, dims, q in EVAL_TESTS:
        if domain not in domain_data:
            domain_data[domain] = {"base":{"dec":0,"ver":0,"cor":0,"unk":0},
                                   "twin":{"dec":0,"ver":0,"cor":0,"unk":0},"n":0}
        domain_data[domain]["n"] += 1

        for name, model in [("base",base_m), ("twin",twin_m)]:
            prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=200).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200, do_sample=True,
                                    temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
            r = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            d = score_decomposition(r); v = score_verification(r)
            c = score_self_correction(r); u = score_unknown(r)

            scores[name]["dec"].append(d); scores[name]["ver"].append(v)
            scores[name]["cor"].append(c); scores[name]["unk"].append(u)
            domain_data[domain][name]["dec"]+=d; domain_data[domain][name]["ver"]+=v
            domain_data[domain][name]["cor"]+=c; domain_data[domain][name]["unk"]+=u

    # TABLE 1: Overall
    n = len(EVAL_TESTS)
    print(f"\n{'='*70}")
    print(f"TABLE 1: Overall Meta-Cognition Scores (n={n})")
    print(f"{'='*70}")
    print(f"{'Dimension':<22} {'Base':>8} {'Twin':>8} {'Delta':>8} {'Pct':>8}")
    print("-"*56)

    for key, name in [("dec","Decomposition"),("ver","Verification"),
                       ("cor","Self-Correction"),("unk","Unknown Declaration")]:
        b = sum(scores["base"][key])/n; t = sum(scores["twin"][key])/n
        d = t-b; pct = (d/(b+0.01))*100
        print(f"{name:<22} {b:>7.2f} {t:>7.2f} {d:>+7.2f} {pct:>+7.0f}%")

    # TABLE 2: Per domain
    print(f"\n{'='*70}")
    print(f"TABLE 2: Per-Domain Transfer (avg all 4 dims)")
    print(f"{'='*70}")
    print(f"{'Domain':<18} {'Base':>8} {'Twin':>8} {'Transfer':>10}")
    print("-"*46)

    all_transfers = []
    for dom in sorted(domain_data.keys()):
        dd = domain_data[dom]; n2 = dd["n"]
        b = sum(dd["base"][k] for k in ["dec","ver","cor","unk"])/(4*n2)
        t = sum(dd["twin"][k] for k in ["dec","ver","cor","unk"])/(4*n2)
        tr = t-b; all_transfers.append(tr)
        print(f"{dom:<18} {b:>7.2f} {t:>7.2f} {tr:>+9.2f}")

    avg_tr = sum(all_transfers)/len(all_transfers)
    print("-"*46)
    print(f"{'AVERAGE':<18} {'':>8} {'':>8} {avg_tr:>+9.2f}")

    # Save
    report = {"model":BASE,"tests":n,"timestamp":__import__('datetime').datetime.now().isoformat(),
              "overall":{"base":{k:sum(v)/n for k,v in scores["base"].items()},
                         "twin":{k:sum(v)/n for k,v in scores["twin"].items()}},
              "per_domain":{d:{"base":{k:v/dd["n"] for k,v in dd["base"].items()},
                               "twin":{k:v/dd["n"] for k,v in dd["twin"].items()}}
                           for d,dd in domain_data.items()}}
    rpath = Path("C:/Users/86131/Desktop/digital-twin-trainer/data/eval_report.json")
    with open(rpath,"w",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(f"\nReport: {rpath}")
    print("[DONE]")

if __name__=="__main__":
    main()
