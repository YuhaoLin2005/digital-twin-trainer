#!/usr/bin/env python
"""
Phase 3: QLoRA 个性注入 → 数字分身 v1
用法: python phase3_lora_train.py [--base MODEL] [--data DATA.jsonl]
显存: ~4.5GB (RTX 3060 6GB 验证通过)
"""

import argparse, json, time, torch
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, TrainerCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from config import MERGE_CONFIG, LORA_CONFIG, TRAIN_CONFIG, CHECKPOINT_DIR, DATA_DIR
from utils.gate import DeliveryGate, make_loss_check, make_checkpoint_check


def load_training_data(data_path: str) -> Dataset:
    path = Path(data_path) if data_path else DATA_DIR / "persona_training_data.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"训练数据不存在: {path}")
    instructions, outputs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            instructions.append(item["instruction"])
            outputs.append(item["output"])
    return Dataset.from_dict({"instruction": instructions, "output": outputs})


def format_prompt(examples):
    texts = [
        f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
        for inst, out in zip(examples["instruction"], examples["output"])
    ]
    return {"text": texts}


class Monitor(TrainerCallback):
    def __init__(self):
        self.history = []
        self.best_loss = float("inf")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs["loss"]
            self.history.append({"step": state.global_step, "loss": loss})
            if loss < self.best_loss:
                self.best_loss = loss
            print(f"  step={state.global_step:3d}  loss={loss:.4f}  best={self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=MERGE_CONFIG["base_model"])
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 3: QLoRA 个性注入 → 数字分身 v1")
    print("=" * 60)
    print(f"  基座: {args.base}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    if args.dry_run:
        print("\n[DRY RUN] 跳过训练")
        return

    # [1/4] 数据
    print("\n[1/4] 加载训练数据...")
    dataset = load_training_data(args.data)
    print(f"  样本数: {len(dataset)}")

    # [2/4] 模型 (4-bit QLoRA)
    print("\n[2/4] 加载模型 (4-bit QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True, torch_dtype=torch.float16,
        )
    except (RuntimeError, OSError) as e:
        print(f"  [WARN] Network error, falling back to offline mode: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=False, torch_dtype=torch.float16,
            local_files_only=True,
        )
    lora_cfg = LoraConfig(
        r=LORA_CONFIG["r"], lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"], task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # [3/4] 格式化
    print("\n[3/4] 格式化数据...")
    dataset = dataset.map(format_prompt, batched=True)

    def tokenize(examples):
        tok = tokenizer(examples["text"], truncation=True,
                       max_length=TRAIN_CONFIG["max_seq_length"], padding="max_length")
        tok["labels"] = [ids.copy() for ids in tok["input_ids"]]
        return tok
    dataset = dataset.map(tokenize, batched=True, remove_columns=["instruction", "output", "text"])

    # [4/4] 训练
    print("\n[4/4] 训练中...")
    monitor = Monitor()
    output_dir = args.output or str(CHECKPOINT_DIR / "lora-twin-v1")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=TRAIN_CONFIG["per_device_batch_size"],
        gradient_accumulation_steps=TRAIN_CONFIG["gradient_accumulation_steps"],
        num_train_epochs=TRAIN_CONFIG["num_epochs"],
        learning_rate=TRAIN_CONFIG["learning_rate"],
        warmup_ratio=TRAIN_CONFIG["warmup_ratio"],
        lr_scheduler_type=TRAIN_CONFIG["lr_scheduler_type"],
        fp16=TRAIN_CONFIG["fp16"],
        gradient_checkpointing=TRAIN_CONFIG["gradient_checkpointing"],
        logging_steps=TRAIN_CONFIG["logging_steps"],
        save_strategy=TRAIN_CONFIG["save_strategy"],
        optim=TRAIN_CONFIG["optim"],
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, callbacks=[monitor])
    t0 = time.time()
    trainer.train()
    train_time = (time.time() - t0) / 60

    # 保存
    adapter_path = str(CHECKPOINT_DIR / "lora-twin-v1" / "adapter")
    model.save_pretrained(adapter_path)
    print(f"\n  适配器: {adapter_path}")

    # 交付门禁
    gate = DeliveryGate("phase3-lora")
    gate.add_check("loss_decreased", lambda: make_loss_check(monitor.history))
    gate.add_check("checkpoint", lambda: make_checkpoint_check(f"{adapter_path}/adapter_config.json"))
    result = gate.run()
    print(gate.report(result))

    # 摘要
    summary = {
        "phase": "3-qlora", "timestamp": datetime.now().isoformat(),
        "base_model": args.base, "samples": len(dataset),
        "trainable_params": trainable, "train_time_min": round(train_time, 1),
        "loss_first": round(monitor.history[0]["loss"], 4) if monitor.history else None,
        "loss_final": round(monitor.history[-1]["loss"], 4) if monitor.history else None,
        "gate_passed": result.passed,
    }
    summary_path = DATA_DIR / "phase3-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  摘要: {summary_path}")
    print(f"\n[DONE] 数字分身 v1 ({train_time:.1f} min)")


if __name__ == "__main__":
    main()
