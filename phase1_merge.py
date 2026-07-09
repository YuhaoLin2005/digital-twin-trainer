#!/usr/bin/env python
"""
Phase 1: 模型融合 — MergeKit TIES 多模型合并
用法: python phase1_merge.py [--dry-run]
"""

import argparse, yaml, subprocess
from pathlib import Path
from config import MERGE_CONFIG, CHECKPOINT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 1: MergeKit 模型融合")
    print("=" * 60)
    cfg = MERGE_CONFIG
    print(f"  基座: {cfg['base_model']}")
    for m in cfg["aux_models"]:
        print(f"  辅助: {m}")
    print(f"  方法: {cfg['merge_method']} (density={cfg['density']}, weight={cfg['weight']})")

    if args.dry_run:
        merge_yaml = {
            "models": [{"model": cfg["base_model"]}] + [{"model": m} for m in cfg["aux_models"]],
            "merge_method": cfg["merge_method"],
            "base_model": cfg["base_model"],
            "parameters": {"density": cfg["density"], "weight": cfg["weight"]},
            "dtype": "float16",
        }
        print("\n  预览 merge_config.yaml:")
        print(yaml.dump(merge_yaml, default_flow_style=False))
        return

    config_path = CHECKPOINT_DIR / "merge_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    merge_config = {
        "models": [{"model": cfg["base_model"]}] + [{"model": m} for m in cfg["aux_models"]],
        "merge_method": cfg["merge_method"],
        "base_model": cfg["base_model"],
        "parameters": {"density": cfg["density"], "weight": cfg["weight"]},
        "dtype": "float16",
    }
    with open(config_path, "w") as f:
        yaml.dump(merge_config, f)

    output_dir = args.output or str(CHECKPOINT_DIR / cfg["output_name"])
    print(f"\n  配置: {config_path}")
    print(f"  输出: {output_dir}")
    print(f"  运行 mergekit-merge (CPU, ~5-10min)...\n")

    result = subprocess.run(
        ["mergekit-merge", str(config_path), output_dir, "--copy-tokenizer"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"\n[DONE] 融合模型: {output_dir}")
    else:
        print(f"\n[ERROR] MergeKit:")
        print(result.stderr[-500:])


if __name__ == "__main__":
    main()
