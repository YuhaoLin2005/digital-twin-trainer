#!/usr/bin/env python
"""
Phase 2: .claude 配置 → 脱敏 → 训练指令对
用法: python phase2_data_prep.py [--max-samples 300] [--dry-run]
"""

import argparse, json
from pathlib import Path
from utils.data import load_persona_data, save_training_data
from config import DATA_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 2: 数据准备 — .claude 配置 → 训练指令对")
    print("=" * 60)

    pairs = load_persona_data(max_samples=args.max_samples)

    if not pairs:
        print("\n[FAIL] 未提取到训练数据。检查 .claude/ 路径。")
        return

    sources = {}
    for p in pairs:
        s = p["source"]
        sources[s] = sources.get(s, 0) + 1

    print(f"\n  来源分布:")
    for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"    {src}: {count} 条")

    high = sum(1 for p in pairs if p.get("priority") == "high")
    print(f"  High priority: {high}/{len(pairs)}")

    print(f"\n  --- 预览前3条 ---")
    for pair in pairs[:3]:
        print(f"  Q: {pair['instruction'][:80]}...")
        print(f"  A: {pair['output'][:80]}...\n")

    if not args.dry_run:
        output_path = Path(args.output) if args.output else None
        path = save_training_data(pairs, output_path)
        print(f"\n[DONE] 训练数据: {path}")

        stats = {
            "total_samples": len(pairs), "sources": sources,
            "high_priority": high,
            "avg_instruction_len": sum(len(p["instruction"]) for p in pairs) / max(len(pairs), 1),
            "avg_output_len": sum(len(p["output"]) for p in pairs) / max(len(pairs), 1),
        }
        stats_path = DATA_DIR / "data_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"  统计: {stats_path}")


if __name__ == "__main__":
    main()
