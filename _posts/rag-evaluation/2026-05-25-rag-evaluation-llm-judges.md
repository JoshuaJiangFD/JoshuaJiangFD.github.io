---
title: "RAG Evaluation Part 2: LLM Judges and Training Better Evaluators"
date: 2026-05-25 12:00:00 +0000
categories: [RAG Evaluation]
tags: [RAG, Evaluation, LLM-as-Judge, Prometheus, ARES, DPO, Fine-Tuning]
mermaid: true
math: true
---

The [previous post]({% post_url rag-evaluation/2026-05-25-rag-evaluation-faithfulness-and-relevance %}) defined the three evaluation surfaces of a RAG pipeline and the metrics for each. Every metric in that post requires someone or something to make a judgment: is this chunk relevant? Is this claim supported? Is the answer correct? The metrics are only as reliable as whoever is making those judgments. This post examines how LLMs are used as judges in RAG evaluation (Section 1), where they systematically fail (Section 2), and whether training dedicated judge models closes the remaining gap with humans (Section 3).

---

## 1. How LLM-as-Judge Is Used in RAG Evaluation

### 1.1 Where LLM Judges Appear

In the metrics described in [Part 1]({% post_url rag-evaluation/2026-05-25-rag-evaluation-retrieval-metrics %}), an LLM replaces a human annotator at multiple points:

| Evaluation Task | What the LLM Judges | Output |
|----------------|--------------------|---------| 
| RAGAS Context Precision | "Was this chunk useful for producing the reference answer?" | Binary verdict per chunk (0 or 1) |
| RAGAS Context Recall | "Can this sentence in the reference be attributed to the retrieved context?" | Binary verdict per sentence |
| RAGAS Faithfulness | "Is this claim in the response supported by the context?" | Binary verdict per claim |
| NDCG approximation | "How relevant is this document to the query?" | Graded score (0-3) |
| Databricks answer evaluation | "Is this answer correct, comprehensive, and readable?" | Integer scores per dimension |

In each case, the LLM receives a structured prompt with the inputs (query, context, response, reference) and returns a judgment. The format of that judgment varies: binary verdicts, Likert scores, or chain-of-thought reasoning followed by a score.

### 1.2 Judging Modes

LLM judges operate in two modes:

**Pointwise scoring.** The LLM evaluates a single item in isolation. Given a query and a retrieved chunk, it scores the chunk's relevance. Given a response and its context, it scores faithfulness. Most RAG evaluation metrics use this mode because each chunk or claim is evaluated independently.

**Pairwise comparison.** The LLM receives two responses and picks the better one (or declares a tie). This mode is common in general LLM evaluation (MT-Bench, Chatbot Arena) but less common in RAG evaluation, where the question is usually "how good is this response?" rather than "which of these two responses is better?"

### 1.3 Why It Works (and How Well)

The MT-Bench paper (Zheng et al., NeurIPS 2023) established the empirical foundation: GPT-4 as a judge achieves >80% agreement with human preferences, matching the rate at which humans agree with each other (~80%). A strong LLM is statistically indistinguishable from a third human annotator, at over 100x lower cost (~$30 per 1000 queries via RAGAS zero-shot vs. ~$3,500 for human pairwise annotation).

However, >80% agreement is an average across tasks. On specific RAG evaluation tasks (graded relevance, faithfulness detection), agreement can be lower: Cohen's kappa of 0.4-0.6, compared to human-human at 0.4-0.7. The next section examines why.

---

## 2. Failure Modes, Mitigations, and Calibration

LLM judges have systematic biases. These are not theoretical concerns. They are quantified in peer-reviewed work.

### 2.1 Position Bias

"Large Language Models are not Fair Evaluators" (Wang et al., 2023) demonstrated that in pairwise evaluation (judge picks the better of two responses), the presentation order can determine the outcome:

- Vicuna-13B could "beat" ChatGPT on 66/80 queries (82.5%) simply by being presented first
- The judge's preference can be entirely determined by which response appears in position A vs. position B

**Mitigation:** Evaluate in both orders, average the results. This doubles evaluation cost but removes order dependence.

### 2.2 Verbosity Bias

"Length-Controlled AlpacaEval" (Dubois et al., COLM 2024) showed that auto-annotators systematically prefer longer responses regardless of quality:

- Longer responses receive higher scores even when content quality is equivalent
- After applying length debiasing (a GLM-based regression that computes counterfactual preference if outputs had equal length): Spearman correlation with Chatbot Arena improves from 0.94 to 0.98

**Mitigation:** Length-controlled scoring. Penalize padding explicitly in the judge prompt, or apply statistical debiasing post-hoc.

### 2.3 Self-Enhancement Bias

Panickssery, Bowman, and Feng (2024) demonstrated that LLMs recognize and prefer their own outputs:

- GPT-4 and Llama 2 can distinguish their own outputs from other models' outputs with non-trivial accuracy
- There is a linear correlation between self-recognition capability and strength of self-preference
- This bias occurs "even when human annotators consider outputs of equal quality"

**Mitigation:** Use a different model family as judge than as generator. If your RAG system uses GPT-4 for generation, use Claude or an open-source model for judging.

### 2.4 Cross-Dataset Variance

JUDGE-BENCH (Bavaresco et al., ACL 2025) tested 11 LLMs as judges across 20 NLP datasets and found:

- "Substantial variance across models and datasets"
- Performance varies based on the property being evaluated, the expertise level of reference human judges, and whether the text is human-generated or model-generated
- Conclusion: "LLMs should be carefully validated against human judgments before being used as evaluators"

**Mitigation:** Validate your LLM judge against human labels on your specific task before trusting it. An LLM judge calibrated on one domain cannot be assumed to work on another without recalibration.

### 2.5 Non-Determinism

LLM outputs vary across runs even at low temperature. The same query can receive different evaluation scores on different days.

**Mitigation:** Low temperature (0.1), structured output format (JSON with constrained fields), and averaging across multiple runs.

### 2.6 Few-Shot Calibration

Including human-labeled examples in the judge prompt improves alignment without any model training:

- **UMBRELA** (used in TREC RAG 2024) supports few-shot prompting with human-graded examples alongside zero-shot evaluation
- **TALEC** (2024) found that "fine-tuning can be replaced by in-context learning" for teaching evaluation criteria, achieving >80% correlation with human judgments
- Thomas et al. (SIGIR 2024) found that prompt selection calibrated against gold labels significantly affects accuracy, but also that "simple paraphrases" cause unpredictable variance

Few-shot calibration requires only 5-20 labeled examples, making it the most accessible improvement over zero-shot judging. However, the sensitivity to prompt phrasing means results may not be reproducible across prompt versions.

### 2.7 Summary of Mitigations

| Bias | Mitigation | Cost | Source |
|------|-----------|------|--------|
| Position | Evaluate in both orders, average | 2x evaluation cost | Wang et al. (2023) |
| Verbosity | Length-controlled scoring or statistical debiasing | Moderate complexity | Dubois et al. (2024) |
| Self-enhancement | Use different model family as judge | Introduces different biases | Panickssery et al. (2024) |
| Cross-dataset | Validate on your specific task with human labels | 20-50 human annotations | JUDGE-BENCH (2025) |
| Non-determinism | Low temperature, structured output, multiple runs | 3-5x evaluation cost | MT-Bench (2023) |
| General misalignment | Few-shot calibration with 5-20 examples | Minimal | UMBRELA, TALEC (2024) |

These mitigations reduce bias but do not eliminate it. Even with all mitigations applied, zero-shot and few-shot LLM judges achieve moderate agreement with humans (Cohen's kappa 0.4-0.6 for graded relevance, compared to human-human at 0.4-0.7). This remaining gap motivates training dedicated judge models.

---

## 3. Training Better Judges

### 3.1 Statistical Calibration (ARES)

ARES (Saad-Falcon et al., NAACL 2024, arXiv:2311.09476) does not improve the judge model itself. Instead, it corrects for the judge's systematic bias using a statistical layer:

1. **Generate synthetic training data** for evaluating context relevance, answer faithfulness, and answer relevance
2. **Fine-tune lightweight classifiers** (DeBERTa-v3-large, 437M parameters) on synthetic data
3. **Apply Prediction-Powered Inference (PPI)**: use a small set (~150-300) of human annotations to produce confidence intervals that correct for systematic bias in model predictions

PPI characterizes the relationship between model predictions and human labels on the small labeled set, then debiases estimates across the full unlabeled evaluation set. The result: statistical confidence intervals for RAG quality metrics without exhaustive human annotation.

Validated on 8 knowledge-intensive tasks in KILT, SuperGLUE, and AIS. "ARES judges remain effective across domain shifts." The tradeoff: PPI confidence intervals widen when model predictions are poorly calibrated, and the ~150-300 human annotations must be representative of the query distribution.

### 3.2 Fine-Tuned Judge Models

Multiple research groups have fine-tuned LLMs specifically for evaluation tasks. These models target the gap between zero-shot performance and human agreement.

**General-purpose evaluation judges (trained on LLM output quality data):**

| Model | Scale | Training Data | Key Result | Venue |
|-------|-------|--------------|-----------|-------|
| **Prometheus** (Kim et al.) | 13B | 100K responses with GPT-4-generated feedback + 1000 custom rubrics (SFT) | Pearson 0.897 with humans (GPT-4: 0.882) | ICLR 2024 |
| **Prometheus 2** (Kim et al.) | 13B | Extended dataset for both assessment modes (SFT) | Handles direct assessment + pairwise ranking | EMNLP 2024 |
| **JudgeLM** (Zhu et al.) | 7B/13B/33B | GPT-4-generated judgments with swap augmentation (SFT) | >90% agreement with GPT-4; 7B judges 5K samples in 3 min on 8 A100s | ICLR 2025 |

These make LLM-as-judge feasible without per-query API costs. The tradeoff: they require GPU infrastructure and may not generalize to domains far from their training distribution.

**IR relevance-specific judges (trained on retrieval relevance data):**

| Paper | Venue | Approach | Key Finding |
|-------|-------|----------|-------------|
| Fitte-Rey et al. | WSDM LLM4EVAL 2025 | Fine-tune small LLMs on augmented relevance datasets | Fine-tuned small models "outperform certain closed source models" for relevance labeling |
| Wang et al. (Pinterest) | RecSys EARL 2025 | Fine-tune open-source LLMs with relevance prediction objective | Validates alignment between LLM judgments and human annotations at production scale |
| Self-Taught Evaluators (Meta) | 2024 | Iteratively train on synthetic contrastive pairs, no human annotations | Improves Llama3-70B from 75.4 to 88.3 on RewardBench, outperforming GPT-4 |

The trajectory is clear: fine-tuned small models (8B-13B) are approaching or exceeding zero-shot GPT-4 performance for evaluation tasks, at a fraction of the inference cost.

### 3.3 Preference Optimization (DPO) for Judge Consistency

**ConsJudge** (Liu et al., ACL 2025 Findings) uses Direct Preference Optimization to train more consistent judges without human annotation:

1. Prompt an LLM to generate judgments across different combinations of evaluation dimensions (e.g., relevance + faithfulness, relevance + completeness)
2. Measure judge-consistency: does the LLM arrive at the same verdict regardless of which dimension combination is used?
3. Consistent judgments become "chosen" examples, inconsistent ones become "rejected"
4. Apply DPO training on these preference pairs

The key insight: consistency across prompting variations is a proxy for correctness, eliminating the need for human preference data. Results show "judgments generated by ConsJudge have a high agreement with the superior LLM."

---

## 4. Choosing an Approach

| Approach | Human Annotation | Model Training | Agreement with Humans | Cost per 1K Queries |
|----------|-----------------|---------------|----------------------|-------------------|
| Zero-shot GPT-4 | None | No | Kappa 0.4-0.6 | ~$30-150 |
| Few-shot calibration | 5-20 examples | No | Moderate improvement (fragile) | ~$30-150 |
| Statistical calibration (ARES) | 150-300 labels | Classifier only | Confidence intervals, debiased | ~$1 |
| Fine-tuned SFT model | 1K+ labels (or synthetic) | Yes | Matches or exceeds GPT-4 | ~$1 |
| DPO judge (ConsJudge) | None (self-supervised) | Yes | Improved consistency | ~$1 |

For IR relevance assessment specifically, zero-shot GPT-4/GPT-4o remains the dominant approach in practice (Thomas et al., UMBRELA, TREC RAG 2024). But the 2025 fine-tuning results suggest a transition: teams that need high-volume, low-cost evaluation are beginning to replace API-based judges with fine-tuned open-source models.

The choice depends on volume and maturity:

**Low volume, starting out.** Zero-shot GPT-4 with few-shot calibration against 20 human-labeled examples. Validate agreement on your specific task before trusting the scores. This is sufficient for comparing retrieval strategies, chunk sizes, and prompt variations during development.

**Medium volume, need reliability.** ARES-style statistical calibration with ~200 human annotations for confidence bounds. This gives you statistical guarantees on your evaluation accuracy without requiring a large annotation budget. Suitable for regular regression testing and model selection.

**High volume, cost-sensitive.** Fine-tuned open-source judge (Prometheus, JudgeLM, or domain-specific) deployed on owned infrastructure. The upfront investment in training data and GPU infrastructure pays off at scale through ~$1 per 1000 queries and deterministic, reproducible scores.

---

## References

| Paper | Venue/Year | Key Contribution |
|---|---|---|
| Zheng et al., "Judging LLM-as-a-Judge" (MT-Bench) | NeurIPS 2023 | GPT-4 judge matches human agreement at >80% |
| Saad-Falcon et al., "ARES" | NAACL 2024 | Lightweight judges + PPI with ~150 human labels |
| Kim et al., "Prometheus" | ICLR 2024 | Open-source 13B evaluator, Pearson 0.897 with humans |
| Kim et al., "Prometheus 2" | EMNLP 2024 | Direct assessment + pairwise ranking |
| Zhu et al., "JudgeLM" | ICLR 2025 | Fine-tuned 7B-33B judges, >90% GPT-4 agreement |
| Wang et al., "Large Language Models are not Fair Evaluators" | 2023 | Position bias quantification |
| Dubois et al., "Length-Controlled AlpacaEval" | COLM 2024 | Verbosity debiasing, correlation 0.94→0.98 |
| Panickssery et al., "Self-Enhancement Bias" | 2024 | LLMs recognize and favor own outputs |
| Bavaresco et al., "JUDGE-BENCH" | ACL 2025 | Variance across judge models and datasets |
| Thomas et al., "LLMs can Accurately Predict Searcher Preferences" | SIGIR 2024 | LLM-based NDCG correlates with human at tau > 0.9 |
| Upadhyay et al., "UMBRELA" | 2024 | Large-scale LLM assessor benchmark for TREC |
| Liu et al., "ConsJudge" | ACL 2025 Findings | DPO-trained judge using consistency signal |
