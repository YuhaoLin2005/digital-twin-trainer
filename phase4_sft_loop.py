#!/usr/bin/env python
"""
Phase 4: 双池专家引导 SFT 循环 (rejection sampling)
  v1 生成 → 双池专家评审 → 选中最佳 → SFT训练 → v2 → 循环
用法: python phase4_sft_loop.py --rounds 3 [--no-api] [--dry-run]
"""

import argparse, json, time, random, os, torch
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, BitsAndBytesConfig,
)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from openai import OpenAI
from config import (MERGE_CONFIG, LORA_CONFIG, LOOP_CONFIG, API_CONFIG, CHECKPOINT_DIR, DATA_DIR)
from experts.pool import ExpertPool
from utils.gate import DeliveryGate, make_checkpoint_check, make_winrate_check


def load_model(base_model, adapter_path=None):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.float16)
    except (RuntimeError, OSError):
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb, device_map="auto",
            trust_remote_code=False, torch_dtype=torch.float16, local_files_only=True)
    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        lora_cfg = LoraConfig(r=LORA_CONFIG["r"], lora_alpha=LORA_CONFIG["lora_alpha"],
                              target_modules=LORA_CONFIG["target_modules"],
                              lora_dropout=LORA_CONFIG["lora_dropout"],
                              bias=LORA_CONFIG["bias"], task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, lora_cfg)
    return model, tokenizer


def generate_responses(model, tokenizer, prompts, max_new_tokens=200):
    responses = []
    model.eval()
    for prompt in prompts:
        inputs = tokenizer(
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=True, temperature=0.7, top_p=0.9,
                                     pad_token_id=tokenizer.eos_token_id)
        r = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        responses.append(r.strip())
    return responses


def load_prompts():
    data_file = DATA_DIR / "persona_training_data.jsonl"
    if data_file.exists():
        prompts = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                prompts.append(json.loads(line)["instruction"])
        random.shuffle(prompts)
        return list(dict.fromkeys(prompts))[:20]
    return [
        "开始工作前，你第一步做什么？", "描述你的代码审查流程。",
        "你如何做技术决策？", "介绍一下你自己，以及你的技术方向。",
        "你做过的最有技术挑战的项目是什么？",
        "你的方案被客户拒绝了三次，怎么判断继续改还是换方向？",
        "你如何快速学习一项新技术？", "三年后你希望自己成为什么样的工程师？",
    ]


def train_sft_step(model, tokenizer, chosen_pairs, round_num):
    if len(chosen_pairs) < 3:
        print(f"  [SKIP] only {len(chosen_pairs)} pairs")
        return model
    texts = [f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>"
             for p, r in chosen_pairs]
    ds = Dataset.from_dict({"text": texts})
    def tokenize(ex):
        tok = tokenizer(ex["text"], truncation=True, max_length=512, padding="max_length")
        tok["labels"] = [ids.copy() for ids in tok["input_ids"]]
        return tok
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    print(f"  SFT: {len(ds)} samples")
    output_dir = str(CHECKPOINT_DIR / f"sft-round{round_num}")
    args = TrainingArguments(output_dir=output_dir, per_device_train_batch_size=1,
                             gradient_accumulation_steps=4, num_train_epochs=1,
                             learning_rate=5e-5, fp16=True, gradient_checkpointing=False,
                             logging_steps=5, save_strategy="no", report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()
    adapter_path = str(CHECKPOINT_DIR / f"sft-round{round_num}" / "adapter")
    model.save_pretrained(adapter_path)
    print(f"  saved: {adapter_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=LOOP_CONFIG["max_rounds"])
    parser.add_argument("--base", type=str, default=MERGE_CONFIG["base_model"])
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-base", type=str, default="https://api.deepseek.com")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-api", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 4: dual-pool expert-guided SFT loop")
    print("=" * 60)
    print(f"  base: {args.base}  rounds: {args.rounds}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    if args.dry_run:
        print("\n[DRY RUN] expert pool:\n")
        for eid, e in ExpertPool().fixed_pool.items():
            print(f"  fixed: {eid} ({e['lens']}) w={e['weight']}")
        for eid, e in ExpertPool().random_pool.items():
            print(f"  random: {eid} ({e['lens']}) domains={e['domains']}")
        return

    client = None
    if not args.no_api:
        api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key, base_url=args.api_base)
            print(f"  API: {args.api_base}")
        else:
            print("  [WARN] no API key, heuristic voting")

    pool = ExpertPool(client=client)
    print("\n[1/4] loading model...")
    adapter_path = args.adapter or str(CHECKPOINT_DIR / "lora-twin-v1" / "adapter")
    model, tokenizer = load_model(args.base, adapter_path)

    prompts = load_prompts()
    print(f"[2/4] {len(prompts)} prompts")

    print(f"[3/4] SFT loop ({args.rounds} rounds)...\n")
    growth_log, best_wr = [], 0.0

    for rnd in range(1, args.rounds + 1):
        print(f"{'#'*60}\n# ROUND {rnd}/{args.rounds}\n{'#'*60}")
        rp = random.sample(prompts, min(10, len(prompts)))

        print(f"\n  [gen] {len(rp)} x 2...")
        base_model, _ = load_model(args.base)
        curr = generate_responses(model, tokenizer, rp)
        base = generate_responses(base_model, tokenizer, rp)
        del base_model; torch.cuda.empty_cache()

        print(f"  [judge] expert pool...")
        chosen, votes = [], {"curr_wins": 0, "base_wins": 0, "ties": 0}

        for i, (p, c, b) in enumerate(zip(rp, curr, base)):
            swap = random.random() < 0.5
            ra, rb = (b, c) if swap else (c, b)
            ci = not swap
            result = pool.judge_pair(p, ra, rb)
            pref, conf = result.weighted_preference
            if pref == "A": votes["curr_wins" if ci else "base_wins"] += 1
            elif pref == "B": votes["base_wins" if ci else "curr_wins"] += 1
            else: votes["ties"] += 1
            if pref != "tie" and pool.should_include(result):
                chosen.append((p, ra if pref == "A" else rb))
            if i < 2: print(pool.summary(result))

        total = votes["curr_wins"] + votes["base_wins"]
        wr = votes["curr_wins"] / max(total, 1)
        print(f"\n  win_rate: {votes['curr_wins']}/{total}={wr:.2f}, ties={votes['ties']}")

        print(f"\n  [SFT] {len(chosen)} chosen...")
        if chosen: model = train_sft_step(model, tokenizer, chosen, rnd)

        gate = DeliveryGate(f"sft-r{rnd}")
        ckpt = str(CHECKPOINT_DIR / f"sft-round{rnd}" / "adapter" / "adapter_config.json")
        gate.add_check("ckpt", lambda: make_checkpoint_check(ckpt))
        gate.add_check("wr", lambda: make_winrate_check(wr, 0.4))
        gr = gate.run(); print(gate.report(gr))

        entry = {"round": rnd, "timestamp": datetime.now().isoformat(),
                 "prompts": len(rp), "chosen": len(chosen),
                 "win_rate": round(wr, 3), "votes": votes, "gate_passed": gr.passed}
        growth_log.append(entry)
        (DATA_DIR / "growth-log.jsonl").parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_DIR / "growth-log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if wr > best_wr + LOOP_CONFIG["min_improvement"]: best_wr = wr
        elif wr > 0.7 and wr <= best_wr + 0.02:
            print(f"\n  [converged] wr={wr:.2f}"); break
        torch.cuda.empty_cache()

    print(f"\n[4/4] {'='*60}\naudit report\n{'='*60}")
    for e in growth_log:
        print(f"  R{e['round']}: wr={e['win_rate']:.2f} n={e['chosen']} [{'PASS' if e['gate_passed'] else 'FAIL'}]")
    with open(DATA_DIR / "final-report.json", "w", encoding="utf-8") as f:
        json.dump({"rounds": len(growth_log), "log": growth_log, "best_wr": best_wr}, f, ensure_ascii=False, indent=2)
    print(f"\n  report: {DATA_DIR / 'final-report.json'}")
    print(f"\n[DONE] {len(growth_log)} rounds")


if __name__ == "__main__":
    main()
