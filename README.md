# Digital Twin Trainer

**Train meta-cognition into model weights.** Not RAG. Not prompts. Internalization.

[Technical Report](paper/paper.md) | [DEV.to](https://dev.to/yuhaolin2005/meta-cognition-is-the-future-of-ai-personalization-a-4-quadrant-framework-to-build-it-5fki) | [Juejin](https://juejin.cn/spost/7660395581106782234)

## Research Pipeline (for evaluators)

| Stage | Paper | GitHub | Article |
|-------|-------|--------|---------|
| Cognitive Architecture | Sec 2 | .claude/ SOUL/BODY/INTERFACE | [DEV.to](https://dev.to/yuhaolin2005/i-built-a-self-referential-ai-system-then-anthropic-discovered-the-same-architecture-in-claude-3m73) |
| J-space Replication | Sec 3 | Hermes Workspace | [Juejin](https://juejin.cn/post/7659251094817341490) |
| Quality Gates | Sec 4 | training-gate, behavioral_drift | [DEV.to](https://dev.to/yuhaolin2005/my-loss-went-down-but-my-model-still-broke-so-i-built-a-drift-metric-e8f) |
| Digital Twin Training | Sec 5 | digital-twin-trainer | [DEV.to](https://dev.to/yuhaolin2005/meta-cognition-is-the-future-of-ai-personalization-a-4-quadrant-framework-to-build-it-5fki) |
| Evaluation | Sec 6 | eval_metacognition.py | [Juejin](https://juejin.cn/spost/7660395581106782234) |

## Quick Start

pip install transformers peft bitsandbytes datasets accelerate
python phase2_data_prep.py
python phase3_lora_train.py        # 5 min, 6GB VRAM

## Key Finding

Meta-cognition transfers across untrained domains when distilled from cognitive architecture. RAG beats knowledge internalization but cannot change thinking patterns. Three-layer architecture: RAG (WHAT) + QLoRA (HOW) + Foundation.

## License

MIT