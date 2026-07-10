# digital-twin-trainer

> QLoRA fine-tuning pipeline for AI behavioral drift detection. RTX 3060 6GB.

## The Problem

Fine-tuning language models is easy. Knowing whether the fine-tuned model actually behaves better is hard. Loss goes down — everyone celebrates. But behavior can collapse while loss improves.

Discovered the hard way: 6 rounds of QLoRA fine-tuning on Qwen2.5-1.5B-Instruct. Loss 8.11→0.77. All four behavioral metrics degraded. Model was confidently wrong.

## What This Is

Training pipeline pairing loss tracking with behavioral evaluation:
- QLoRA fine-tuning on consumer GPU (RTX 3060 6GB)
- Multiplicative behavioral drift metric (self-BLEU × digit_density × repetition_ratio)
- Submitted as HuggingFace evaluate PR [#778](https://github.com/huggingface/evaluate/pull/778)
- Detects catastrophic forgetting that perplexity misses

## Related

- [hermes-workspace](https://github.com/YuhaoLin2005/hermes-workspace) — Dual-layer gate architecture
- [compact-counter](https://github.com/YuhaoLin2005/compact-counter) — Compaction tracker
- [Juejin: loss从8.11降到0.77](https://juejin.cn/post/7660007537018617883)

## Status

- [x] QLoRA pipeline (Qwen2.5 0.5B/1.5B-Base/1.5B-Instruct)
- [x] Behavioral drift metric (HF Evaluate PR #778)
- [x] 6-round experiment data
- [ ] Cross-model validation
- [ ] Larger model experiments (7B+)

## License

MIT
