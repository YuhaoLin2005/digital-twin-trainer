# Behavioral Pattern Transfer via QLoRA: From External Scaffolding to Weight-Internalized Agent Constraints

**Yuhao Lin** — Fujian Agriculture and Forestry University, Spatial Information & Digital Technology
**Technical Report v2.0** — July 2026 — github.com/YuhaoLin2005/digital-twin-trainer

> **Series Note:** This is Part 2 of a two-part series. Part 1 (hermes-workspace) established that external scaffolding causally reduces identity drift (n=30, p=0.0092). Part 2 (this) tests whether the same behavioral pattern can be transferred via QLoRA into a 1.5B model.

## Abstract

We investigate whether agent behavioral constraints, first validated as external file-system scaffolding (Part 1), can be internalized into model weights via QLoRA fine-tuning. Using 253 instruction-response pairs derived from 50+ sessions of agent operation, we train a Qwen2.5-1.5B-Instruct model (4-bit QLoRA, r=16, 6GB VRAM, ~5 min) and evaluate behavioral pattern transfer across 10 completely untrained domains. Qualitative analysis reveals structured decomposition and multi-perspective reasoning patterns absent from the base model. Quantitative automated evaluation shows comparable structural metrics, with lower uncertainty declaration rates in the fine-tuned model. **Limitation:** We have not validated whether the observed mechanism constitutes human-equivalent meta-cognition; our results demonstrate only behavioral pattern transfer at proof-of-concept scale (n=10 domains, single responses, regex-based scoring). All code and data are open-source.

## Repository & Article Map

| Stage | Paper | GitHub | Article |
|-------|-------|--------|---------|
| External Scaffolding (Part 1) | Sec 2-4 | hermes-workspace, gategrow, training-gate | [DEV.to](https://dev.to/yuhaolin2005/i-built-a-self-referential-ai-system-then-anthropic-discovered-the-same-architecture-in-claude-3m73) |
| J-space Replication | Sec 3 | Hermes Workspace | [Juejin](https://juejin.cn/post/7659251094817341490) |
| Quality Gates | Sec 4 | training-gate, behavioral_drift | [DEV.to](https://dev.to/yuhaolin2005/my-loss-went-down-but-my-model-still-broke-so-i-built-a-drift-metric-e8f) |
| Weight Internalization (Part 2) | Sec 5-6 | digital-twin-trainer | [DEV.to](https://dev.to/yuhaolin2005/meta-cognition-is-the-future-of-ai-personalization-a-4-quadrant-framework-to-build-it-5fki) |
| Evaluation | Sec 6 | eval_metacognition.py | [Juejin](https://juejin.cn/spost/7660395581106782234) |

## 1. Introduction

Part 1 (hermes-workspace) demonstrated that external file-system scaffolding (SOUL/BODY/INTERFACE) causally alters model behavior (n=30, Fisher exact p=0.0092) and received community validation through ECC (200K+ stars) and HuggingFace mergers. However, external scaffolding must be reloaded each session. Part 2 asks: can the same behavioral patterns be internalized into model weights?

## 2-4. Background: External Scaffolding (see Part 1 for details)

Hermes Workspace: file-system cognitive architecture. J-space: p=0.0092 causal validation. Quality gates: training-gate, behavioral_drift HF PR #778, ECC delivery-gate hook.

## 5. Behavioral Pattern Transfer via QLoRA

253 pairs from 50+ agent sessions. QLoRA (4-bit, r=16). Qwen2.5-1.5B-Instruct. RTX 3060 6GB. ~5 min training.

## 6. Cross-Domain Evaluation

10 untrained domains. Qualitative: structured decomposition transfers. Quantitative: structural metrics comparable (100% vs 100% decomposition, 80% vs 70% verification). Uncertainty declaration lower in fine-tuned model (10% vs 40%).

**Limitations (explicit):**
1. n=10 domains with single responses per test — pilot study scale
2. Regex-based automated scoring cannot capture semantic quality
3. No baseline comparison with standard SFT (pending)
4. No ROUGE-L/BLEU quantitative metrics (pending)
5. 1.5B parameter model may have insufficient capacity
6. **We have NOT validated whether observed patterns constitute human-equivalent meta-cognition.** They represent behavioral pattern transfer at proof-of-concept level.
7. Single temperature setting (t=0.7)

## 7. Four-Quadrant Framework

Q1(Rules)->Q2(Blind Spots)->Q3(Hidden)->Q4(Unknown). Session cycle with precipitation events.

## 8. Three-Layer Architecture (Proposed)

Layer 1: RAG (WHAT, daily). Layer 2: QLoRA Behavioral Patterns (HOW, monthly). Layer 3: Foundation (base, annual).

## 9. Future Work & Required Baselines

- Baseline A: Raw Qwen2.5-1.5B (done)
- Baseline B: External prompt only, no fine-tuning (pending)
- Baseline C: Standard SFT comparison (pending)
- Scale to 7B-70B models
- LLM-as-judge evaluation replacing regex
- Human-subject studies
- ROUGE-L/BLEU quantitative metrics

## References

Anthropic J-space (2026) · Constitutional AI (Bai 2022) · SPIN (Chen 2024) · LoRA (Hu 2021) · QLoRA (Dettmers 2023)

---
github.com/YuhaoLin2005/digital-twin-trainer · Part 2 of 2 · All experiments on 6GB consumer GPU
