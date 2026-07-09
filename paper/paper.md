# Externalized Cognitive Architecture: From File-System State Machines to Weight-Internalized Meta-Cognition

**Yuhao Lin** — Fujian Agriculture and Forestry University, Spatial Information & Digital Technology
**Technical Report v1.0** — July 2026 — github.com/YuhaoLin2005/digital-twin-trainer

## Abstract

We present a pipeline for externalizing then internalizing human cognitive architecture into AI: (1) Hermes Workspace — file-system cognitive architecture with mechanical feedback loops, (2) Independent J-space replication via prompt engineering (n=30, p=0.0092), (3) Mechanical quality-gate systems for output enforcement, (4) QLoRA distillation of cognitive architecture into model weights (253 pairs, r=16, 6GB VRAM), (5) Cross-domain meta-cognition evaluation across 10 untrained domains. We propose a three-layer architecture (RAG/knowledge + QLoRA/meta-cognition + foundation/capability) and a 4-quadrant meta-cognition model.

## Repository & Article Map

| Stage | Paper | GitHub | Article |
|-------|-------|--------|---------|
| Cognitive Architecture | Sec 2 | .claude/ config | [Self-Referential AI](https://dev.to/yuhaolin2005/i-built-a-self-referential-ai-system-then-anthropic-discovered-the-same-architecture-in-claude-3m73) |
| J-space Replication | Sec 3 | Hermes Workspace | [J-space](https://juejin.cn/post/7659251094817341490) |
| Quality Gates | Sec 4 | training-gate, behavioral_drift | [Drift Metric](https://dev.to/yuhaolin2005/my-loss-went-down-but-my-model-still-broke-so-i-built-a-drift-metric-e8f) |
| Digital Twin | Sec 5 | digital-twin-trainer | [Meta-Cognition](https://dev.to/yuhaolin2005/meta-cognition-is-the-future-of-ai-personalization-a-4-quadrant-framework-to-build-it-5fki) |
| Evaluation | Sec 6 | eval_metacognition.py | [4-Quadrant](https://juejin.cn/spost/7660395581106782234) |

## 1. Introduction

Current AI personalization: system prompts, RAG, skills. These are external crutches. The model never changes. We ask: can cognitive architecture be compiled, validated at file-system level, then distilled into weights?

## 2. Hermes Workspace

Layered file-system design: SOUL.md (identity) -> BODY.md (rules) -> INTERFACE.md (9-line behavioral calibration) -> self-model.md (versioned self-description) -> growth-log/ (50+ session records).

Strange Loop: quality-gate detects staleness -> writes flag -> health-check on startup -> AI regenerates self-model -> audit log. 4/5 steps mechanized.

Dual-Pool Review: 5 fixed experts + rotating random pool for multi-perspective validation.

## 3. J-space Independent Replication

n=30 controlled trials. Fisher exact p=0.0092. 9-line INTERFACE causally alters behavior. Structural isomorphism with Anthropic J-space — achieved via public API only.

## 4. Quality Gate System

training-gate (pip), behavioral_drift (HF evaluate PR #778), ECC delivery-gate hook (21k+ star project). Philosophy: machines check, humans judge.

## 5. Digital Twin Training

253 instruction-response pairs from 50+ sessions. QLoRA (4-bit, r=16). Qwen2.5-1.5B-Instruct. RTX 3060 6GB. ~5 min training. Expert-guided SFT refinement loop.

## 6. Meta-Cognition Transfer

10 untrained domains. Qualitative: structured decomposition, multi-perspective reasoning transfer. Quantitative: structured 100%vs100%, verification 80%vs70%, uncertainty 40%vs10%. Proof-of-concept. Regex scoring limitation acknowledged.

## 7. Four-Quadrant Model

Q1(Rules)->Q2(Blind Spots)->Q3(Hidden Patterns)->Q4(Unknown). Session cycle: startup loads Q1->checks Q2->surfaces Q3->verifies Q4. Shutdown: Q3->Q1 precipitation, Q2 archiving.

Critical distinction: Meta-cognition (HOW) != Knowledge (WHAT). RAG beats knowledge internalization. RAG cannot change thinking patterns.

## 8. Three-Layer Architecture

Layer 1: RAG (dynamic knowledge, daily updates). Layer 2: QLoRA Meta-Cognition (thinking patterns, monthly). Layer 3: Foundation (base capability, annual).

## 9. Future Work

Scale to 7B-70B. LLM-as-judge evaluation. Human-subject studies. Continued pre-training. Formalize externalized-neural connection.

## References

Anthropic J-space (2026) · Constitutional AI (Bai 2022) · SPIN (Chen 2024) · LoRA (Hu 2021) · QLoRA (Dettmers 2023) · GEB (Hofstadter 1979)

---
github.com/YuhaoLin2005/digital-twin-trainer · All experiments on 6GB consumer GPU
