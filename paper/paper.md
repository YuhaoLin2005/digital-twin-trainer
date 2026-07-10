# Behavioral Pattern Transfer via QLoRA: A Pilot Study

**Yuhao Lin** — Fujian Agriculture and Forestry University, Spatial Information & Digital Technology
**Technical Report v3.0** — July 2026 — github.com/YuhaoLin2005/digital-twin-trainer

> **Series Note:** This is Part 2 of a two-part series. Part 1 (hermes-workspace) established that external scaffolding causally reduces identity drift (n=30, Fisher exact p=0.0092; 2x2 contingency: 11/4 vs 3/12, OR=11.0, 95%CI [2.0,60.6]). Part 2 (this) is a **pilot study** testing whether behavioral patterns can be transferred via QLoRA into a 1.5B model.

## Abstract

**Objective:** Test whether agent behavioral constraints, first validated as external file-system scaffolding (Part 1), can be internalized into model weights via QLoRA fine-tuning.

**Method:** 253 instruction-response pairs from 50+ Claude Code sessions (single user) → Qwen2.5-1.5B-Instruct, 4-bit QLoRA (r=16), RTX 3060 6GB, ~5 min. Evaluation: 10 untrained domains, comparing raw Qwen vs. prompt-only Qwen vs. QLoRA-tuned model.

**Results:** No behavioral metric reached statistical significance (McNemar tests, all p>0.05 after correction). ROUGE-L between twin and raw Qwen outputs: mean 0.095, bootstrap 95% CI [0.064, 0.122]. Prompt-only ROUGE-L vs raw Qwen: mean 0.107 — directionally but not significantly higher (paired t-test p=0.13). Structured decomposition showed a significant *decrease* in the fine-tuned model (90% vs 30%, McNemar p=0.031 uncorrected) — which may indicate less formulaic output, or measurement noise.

**Power analysis:** With n=10 domains, the study can only detect effect sizes >47pp at 80% power. All observed effects are below this threshold. A properly powered replication requires n≥48 domains.

**Limitations:** Single human subject (n=1), regex-based scoring, single responses per domain, no SFT baseline, 1.5B model, non-blind outcome assessment in Part 1. We have NOT validated human-equivalent meta-cognition.

**Conclusion:** This pilot study demonstrates the feasibility of the QLoRA behavioral transfer pipeline on consumer hardware, but provides insufficient statistical evidence to confirm behavioral pattern transfer. The quantitative results are consistent with both "genuine behavioral shift" and "sampling noise." A scaled replication (48+ domains, multiple human subjects, semantic evaluation, SFT baseline) is the required next step. All code, data, and raw outputs are open-source.

## Repository & Article Map

| Stage | Paper Section | GitHub | Article |
|-------|--------------|--------|---------|
| External Scaffolding (Part 1) | Sec 2-4 | hermes-workspace, gategrow, training-gate | [DEV.to](https://dev.to/yuhaolin2005/i-built-a-self-referential-ai-system-then-anthropic-discovered-the-same-architecture-in-claude-3m73) |
| Causal Validation | Sec 3 | hermes-workspace (EXPERIMENT.md) | [Juejin](https://juejin.cn/post/7659251094817341490) |
| Quality Gates | Sec 4 | training-gate | [DEV.to](https://dev.to/yuhaolin2005/my-loss-went-down-but-my-model-still-broke-so-i-built-a-drift-metric-e8f) |
| Weight Internalization (Part 2) | Sec 5-6 | digital-twin-trainer | [DEV.to](https://dev.to/yuhaolin2005/meta-cognition-is-the-future-of-ai-personalization-a-4-quadrant-framework-to-build-it-5fki) |
| Evaluation | Sec 6 | eval_metacognition.py, run_baseline.py | [Juejin](https://juejin.cn/spost/7660395581106782234) |

## 1. Introduction

Part 1 (hermes-workspace) demonstrated that external file-system scaffolding causally alters model behavior (n=30, Fisher exact p=0.0092; 2x2 contingency: 11/4 vs 3/12, OR=11.0, 95%CI [2.0,60.6]; **limitation:** non-blind single-rater scoring, alternating assignment) and received community validation through ECC (228k stars, PRs #2377+#2378 merged) and claude-skills (21.9k stars, PR #867 merged). However, external scaffolding must be reloaded each session, consuming context window space. Part 2 asks: can the same behavioral patterns be internalized into model weights, eliminating the per-session reload cost?

**This is a pilot study.** The goal is to establish feasibility, quantify effect sizes, and compute the sample sizes required for a definitive experiment. All limitations are stated explicitly.

## 2-4. Background: External Scaffolding (see Part 1 for details)

Hermes Workspace: file-system cognitive architecture (SOUL/INTERFACE/BODY). J-space: p=0.0092 causal validation (2x2 contingency table in EXPERIMENT.md; **note:** single-rater non-blind, alternating assignment — see Part 1 limitations). Quality gates: ECC delivery-gate hook (merged #2377+#2378), behavioral_drift (HF evaluate PR #778, pending review).

## 5. Behavioral Pattern Transfer via QLoRA

**Data:** 253 instruction-response pairs from 50+ Claude Code sessions of a single user. PII-sanitized, principle-extracted, converted to instruction-following format. **Limitation:** n=1 human subject — results may reflect one individual's interaction style rather than generalizable behavioral patterns.

**Training:** Qwen2.5-1.5B-Instruct, 4-bit QLoRA (r=16, alpha=32), learning rate 2e-4, 3 epochs, batch size 4. RTX 3060 6GB, ~5 minutes. Merged adapter weights.

**Baselines:**
- **A (Raw Qwen):** Qwen2.5-1.5B-Instruct with minimal system prompt
- **B (Prompt-only):** Same model with external behavioral prompt (no fine-tuning)
- **C (Standard SFT):** Pending — critical missing baseline

## 6. Cross-Domain Evaluation

**10 untrained domains:** Med, Law, Finance, Psych, Edu, Agri, Fit, Music, Astro, Mgmt — spanning STEM, social sciences, and practical domains. Single response per domain per model at t=0.7.

### 6.1 Behavioral Metrics (Regex Scoring)

Three binary metrics scored via regex pattern matching (**limitation:** unvalidated instrument; no gold-standard human labels; see Limitations):

| Metric | Raw Qwen | Prompt-only | QLoRA Twin | McNemar p (Raw vs Twin) |
|--------|----------|-------------|------------|--------------------------|
| Structured Decomposition | 9/10 (90%) | 9/10 (90%) | 3/10 (30%) | **p=0.031** (uncorrected) |
| Uncertainty Declaration | 3/10 (30%) | 7/10 (70%) | 1/10 (10%) | p=0.625 (ns) |
| Verification Suggestion | 4/10 (40%) | 6/10 (60%) | 2/10 (20%) | p=0.500 (ns) |

**Interpretation:**
- **Structured decomposition** shows a significant *decrease* in the twin model (90%→30%, p=0.031 uncorrected). This could indicate the twin produces less formulaically structured output — potentially more natural reasoning. Alternatively, it may reflect measurement noise or regex pattern mismatch. This metric should NOT be interpreted as evidence of "better" or "worse" behavior without semantic validation.
- **Uncertainty** (30%→10%) and **verification** (40%→20%) trend lower in the twin, but neither reaches significance at n=10. The twin model is directionally more confident and less self-verifying — whether this reflects internalized pattern confidence or overconfidence is unclear.
- **None of these results survive multiple-comparison correction** (Bonferroni: α=0.05/3=0.017). The structured decomposition result (p=0.031) becomes non-significant after correction.

### 6.2 ROUGE-L Lexical Divergence

| Comparison | Mean ROUGE-L Recall | Bootstrap 95% CI | Range |
|------------|-------------------|-------------------|-------|
| Raw Qwen vs Twin | 0.095 | [0.064, 0.122] | [0.000, 0.157] |
| Prompt-only vs Twin | 0.107 | [0.073, 0.137] | [0.018, 0.162] |

Paired t-test (prompt-twin vs raw-twin): t=1.66, p=0.13 (ns).

**Interpretation:** Both raw Qwen and prompt-only Qwen produce outputs with ~90% lexical divergence from the twin model. The prompt-only ROUGE-L (0.107) is directionally but not significantly higher than raw-twin ROUGE-L (0.095), suggesting QLoRA fine-tuning and prompt engineering produce similar *magnitudes* of output change. Low ROUGE-L proves lexical divergence — it does NOT prove behavioral transfer. Models can produce lexically different text with identical reasoning quality, or lexically similar text with different reasoning. ROUGE-L measures surface form only.

### 6.3 Qualitative Observations

Despite the quantitative ambiguity, qualitative inspection reveals patterns consistent with training data influence:

**Finance domain** (never trained): Twin output showed three-factor structured analysis (growth rate, reinvestment, exit value) with explicit quantification. Raw Qwen gave generic advice. Prompt-only Qwen gave structured but formulaic analysis.

**Psychology domain** (never trained): Twin output produced a numbered action plan with concrete steps (listen, help find resources, get them out). Raw Qwen gave a general paragraph. Prompt-only gave a longer but less actionable response.

These qualitative patterns are suggestive but do not constitute statistical evidence. They motivate the scaled replication proposed in Section 9.

### 6.4 Statistical Power Analysis

| Effect Size (pp) | Required n (80% power, α=0.05, McNemar) |
|------------------|------------------------------------------|
| 47 (minimum detectable at n=10) | 10 |
| 30 (observed uncertainty effect) | 48 |
| 20 | 194 |
| 10 | 778 |

The current study (n=10) is adequately powered only for effect sizes >47pp — larger than any plausible behavioral intervention effect. All observed effects (10-30pp) fall below the detection threshold. **A definitive experiment requires n≥48 domains with multiple responses each.**

## 7. Four-Quadrant Framework (Proposed, Untested)

A conceptual model derived from 50+ sessions of agent operation:
- **Q1 (Rules):** Explicit constraints (BODY.md, INTERFACE.md)
- **Q2 (Blind Spots):** Discovered through adversarial review (persona-pool)
- **Q3 (Hidden Patterns):** Extracted from growth-logs, awaiting formalization
- **Q4 (Unknown):** Assumptions and uncharted territory (assumption.md)

Session cycle: Q1 loading → Q2 scanning → Q3 surfacing → Q4 monitoring. "Precipitation events" occur when Q3 patterns crystallize into Q1 rules. **This framework is observational, not experimentally validated.** It describes the author's configuration architecture and may not generalize.

## 8. Three-Layer Architecture (Proposed, Untested)

A design hypothesis for integrating external knowledge and internalized patterns:
- **Layer 1 (RAG — WHAT, daily update):** Facts, documents, policies
- **Layer 2 (QLoRA — HOW, monthly retrain):** Behavioral patterns, decision frameworks
- **Layer 3 (Foundation — base, annual update):** Model capability upgrades

**Status:** Only Layer 2 has been tested (this pilot study). Layer 1+3 are design hypotheses. The three-layer integration remains entirely speculative and requires validation.

## 9. Future Work: Required for Definitive Validation

### Immediate (feasible on current hardware):
- [x] Baseline A: Raw Qwen + ROUGE-L (done, mean 0.095)
- [x] Baseline B: Prompt-only Qwen + ROUGE-L (done, mean 0.107)
- [x] Statistical power analysis (done, n≥48 required)
- [ ] Baseline C: Standard SFT comparison (same data, full fine-tuning or same-rank LoRA without scaffolding-derived patterns)
- [ ] Multiple responses per domain (≥10) with variance reporting
- [ ] BLEU, BERTScore as complementary metrics

### Requires external resources:
- [ ] Multi-subject replication (n≥10 participants, diverse interaction styles)
- [ ] Blind multi-rater behavioral coding (≥2 raters, Cohen's κ reported)
- [ ] LLM-as-judge evaluation replacing regex scoring
- [ ] Scale to 7B-70B models
- [ ] Randomized controlled assignment with allocation concealment (Part 1 replication)
- [ ] Pre-registered analysis plan
- [ ] IRB approval for human subjects research

## Limitations (Complete)

1. **n=1 human subject:** All training data from single user. Results may reflect individual interaction style, not generalizable behavioral patterns.
2. **n=10 domains, single response each:** Within-domain variance cannot be estimated. All per-domain ROUGE-L values are single draws from unknown distributions.
3. **Regex-based scoring without validation:** No gold-standard human labels. False positive/negative rates unknown. Structural decomposition metric shows ceiling effects.
4. **No standard SFT baseline:** Cannot attribute observed effects to scaffolding-derived data vs. simple instruction-following fine-tuning.
5. **1.5B parameter model:** Findings may not transfer to larger models.
6. **No statistical significance on behavioral metrics:** All McNemar tests non-significant after multiple-comparison correction.
7. **Part 1 limitations propagate:** Non-blind single-rater scoring, alternating (non-random) assignment, single model family (DeepSeek V4 Pro).
8. **ROUGE-L measures surface form:** Low lexical overlap ≠ behavioral transfer. Semantic evaluation required.
9. **Single temperature (t=0.7):** Results may vary with decoding parameters.
10. **Training data not publicly accessible:** Source .claude files contain personal data. Extraction methodology documented in utils/data.py.
11. **We have NOT validated human-equivalent meta-cognition.** Observed patterns may reflect surface-level style mimicry, not deep cognitive transfer.

## References

Anthropic J-space (2026) · Constitutional AI (Bai 2022) · SPIN (Chen 2024) · LoRA (Hu 2021) · QLoRA (Dettmers 2023) · McNemar (1947)

---
github.com/YuhaoLin2005/digital-twin-trainer · Part 2 of 2 · Pilot Study · All experiments on 6GB consumer GPU
