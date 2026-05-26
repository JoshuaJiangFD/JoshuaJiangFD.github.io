---
title: "RAG Evaluation Part 2: Faithfulness and Answer Relevance"
date: 2026-05-25 12:00:00 +0000
categories: [RAG Evaluation]
tags: [RAG, Evaluation, Faithfulness, Answer Relevance, DeepEval, RAGAS, NLI, Hallucination]
mermaid: true
math: true
---

[Part 1]({% post_url rag-evaluation/2026-05-25-rag-evaluation-retrieval-metrics %}) covered the RAG pipeline structure and retrieval metrics (edge 1: did the retriever find relevant passages?). This post covers the remaining two edges: faithfulness (edge 2: did the generator stay faithful to the evidence?) and answer relevance (edge 3: does the output address the user's question?). It explains how RAGAS and DeepEval implement each, how NLI-based classifiers offer a cheaper alternative for faithfulness, and where all current metrics have blind spots.

---

## 1. Faithfulness / Groundedness Metrics

Faithfulness measures whether the generated response is supported by the retrieved context, or whether the LLM fabricated information. This is the hallucination check, and arguably the most critical dimension for trust.

Two families of approaches exist: LLM-based (prompt a general-purpose LLM to verify claims) and classifier-based (use a trained model specifically for entailment detection). They trade off flexibility against cost and determinism.

### 1.1 [RAGAS](https://arxiv.org/abs/2309.15217) Faithfulness

$$\text{Faithfulness} = \frac{\text{Claims in response supported by context}}{\text{Total claims in response}}$$

**How it works:**
1. An LLM decomposes the generated answer into atomic claims with pronoun resolution. For example, "Paris is the capital of France and has 2.1 million people" becomes two claims: "Paris is the capital of France" and "Paris has 2.1 million people."
2. For each claim, the LLM judges whether it is supported by the retrieved context. Binary verdict: supported or not supported.
3. Score = supported claims / total claims.

If an answer produces 5 claims and 3 are grounded in the passages, the faithfulness score is 0.6.

**Inputs required:** query, response, retrieval_context. No reference answer needed.

Strength: fine-grained. Identifies exactly which claims are unsupported. Flexible across domains because the LLM generalizes.

Weakness: expensive (multiple LLM calls per response). Non-deterministic. The claim decomposition step itself can introduce errors if the LLM splits claims poorly.

### 1.2 DeepEval Faithfulness

DeepEval's faithfulness metric uses a similar claim-verification structure but with a three-way verdict system instead of binary.

**How it works:**
1. Extract **truths** from the retrieval_context (factual statements the context establishes).
2. Extract **claims** from the actual_output (factual assertions the response makes). Steps 1 and 2 run in parallel.
3. For each claim, judge it against all truths with a three-way verdict:
   - "yes" = claim agrees with context
   - "no" = claim **directly contradicts** context
   - "idk" = claim is not addressed by context (neither supported nor contradicted)
4. Score = non-contradicting claims / total claims.

$$\text{Faithfulness} = \frac{\text{count}(\text{verdict} \neq \text{"no"})}{\text{total claims}}$$

**Inputs required:** query, response, retrieval_context. No reference answer needed.

**The critical difference from RAGAS:** DeepEval only penalizes direct contradictions by default. A claim that is not mentioned anywhere in the context still passes (verdict = "idk" counts as non-contradicting). This makes DeepEval's faithfulness more lenient than RAGAS, which requires positive support.

An optional `penalize_ambiguous_claims` flag changes this behavior. When enabled, "idk" verdicts also reduce the score, making it more similar to RAGAS's binary approach.

**Example:**

Query: "Tell me about the Eiffel Tower."
Response: "The Eiffel Tower was built in 1889. It is 330 meters tall. It was designed by Alexander Gustave Eiffel for the Olympics."
Retrieval context: ["The Eiffel Tower, completed in 1889, is a 330-meter wrought-iron tower built for the 1889 World's Fair."]

Truths extracted: "completed in 1889", "330 meters tall", "built for the 1889 World's Fair"
Claims extracted: "built in 1889", "330 meters tall", "designed by Gustave Eiffel for the Olympics"

Verdicts:
- "built in 1889" → yes (agrees with "completed in 1889")
- "330 meters tall" → yes (directly stated)
- "designed by Gustave Eiffel for the Olympics" → no (context says World's Fair, not Olympics. Direct contradiction.)

Score = 2/3 = 0.67 (one direct contradiction).

### 1.3 Classifier-Based: NLI and Trained Models

Natural Language Inference (NLI) is a classification task where a model determines the relationship between two texts: given a *premise* and a *hypothesis*, the model predicts whether the premise **entails** the hypothesis, **contradicts** it, or is **neutral**. NLI models are trained on large datasets of human-annotated text pairs (e.g., SNLI, MultiNLI).

Applied to faithfulness: the retrieved context is the premise, the generated response (or individual sentences from it) is the hypothesis. If the context entails the response, the response is grounded. If not, it may be hallucinated.

The [TRUE benchmark (Honovich et al., 2022)](https://arxiv.org/abs/2204.04991) validated this approach, finding that "large-scale NLI and question generation-and-answering-based approaches achieve strong and complementary results."

Production-oriented classifiers in this category:

| Model | Type | Key Property |
|-------|------|-------------|
| **Vectara HHEM-2.1-Open** | Small binary classifier | Free, fast, deterministic. Replaces LLM-based verification for production use. |
| **Fine-tuned DeBERTa** (as in ARES) | NLI model fine-tuned on domain data | ROC-AUC improved from 0.56 to 0.85 with ~1000 task-specific training examples |

Strength: deterministic, fast, cheap (no LLM API calls). Can run on CPU or small GPU.

Weakness: less flexible than LLM-based approaches. NLI models trained on general entailment data may not transfer well to domain-specific language. Requires fine-tuning on task-specific data to achieve strong performance.

### 1.4 How Accurate Is Faithfulness Detection?

Detecting hallucination is itself an unsolved problem. Current accuracy of different approaches:

| Approach | Accuracy / Performance | Source |
|----------|----------------------|--------|
| ChatGPT (zero-shot) | 53.8-58.5% accuracy on HaluEval | HaluEval benchmark |
| Claude 2 (zero-shot) | 53.8-58.5% accuracy on HaluEval | HaluEval benchmark |
| Off-the-shelf NLI model | ROC-AUC 0.56 | TRUE benchmark |
| NLI fine-tuned on ~1000 domain examples | ROC-AUC 0.85 | TRUE benchmark follow-up |
| ChainPoll (Galileo) | AUROC 0.781 | arXiv:2310.18344 |

The gap between zero-shot LLM detection (~55% accuracy, barely above random) and fine-tuned classifiers (ROC-AUC 0.85) is striking. It suggests that faithfulness detection requires task-specific training to be reliable. General-purpose LLMs are surprisingly poor at detecting hallucinations in their own outputs without specialized prompting or fine-tuning.

### 1.5 What Faithfulness Metrics Cannot Catch

A response can be perfectly faithful (every claim supported by context) while being *wrong*, because the reasoning that connects those facts is incorrect. Faithfulness checks attribution, not inference quality.

Example: if the context says "Company X revenue was $10M in 2023" and "Company X revenue was $8M in 2022", a response stating "Company X revenue declined by 50%" is faithful (both numbers are in the context) but the arithmetic is wrong. No faithfulness metric catches this because each individual claim ("revenue was $10M", "revenue was $8M", "revenue declined") can be attributed to the context. The error is in the reasoning between claims, not in the claims themselves.

---

## 2. Answer Relevance Metrics

Answer relevance measures whether the final response actually addresses what the user asked. A system can retrieve perfectly and generate faithfully, but still fail if it answers the wrong question. This edge (query → response) captures end-to-end utility.

### 2.1 RAGAS Answer Relevancy

RAGAS uses a unique approach based on reverse question generation and embedding similarity:

$$\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \cos(E_{q_i}, E_{q_{orig}})$$

**How it works:**
1. The LLM generates $N$ artificial questions (default 3) that the response could plausibly be answering (given only the response, not the original question).
2. Embed the original question and each generated question.
3. Compute cosine similarity between the original question embedding and each generated question embedding.
4. Score = average cosine similarity.

**Intuition:** If the response correctly addresses the original question, then the original question should be reconstructable from the answer alone. If the response wanders off-topic, the generated questions will diverge from the original.

**Inputs required:** query, response. No context or reference answer needed.

Strength: penalizes both incompleteness (answer doesn't cover the question) and verbosity (answer includes irrelevant information that suggests a different question).

Weakness: depends on embedding quality. Two semantically similar questions with different phrasing might have low cosine similarity if the embedding model is weak. Also, this is an indirect measurement. The metric does not explicitly judge "does this answer the question?" It infers relevance through a generative round-trip.

### 2.2 DeepEval Answer Relevancy

DeepEval takes a more direct approach using LLM-as-judge:

**How it works:**
1. Extract **statements** from the actual_output (atomic information units).
2. For each statement, the LLM judges: "Is this statement relevant to addressing the input query?" Three-way verdict:
   - "yes" = relevant to the input
   - "no" = irrelevant to the input
   - "idk" = ambiguous or supporting information
3. Score = non-irrelevant statements / total statements.

$$\text{Answer Relevancy} = \frac{\text{count}(\text{verdict} \neq \text{"no"})}{\text{total statements}}$$

Note: "idk" verdicts count as relevant (not penalized), similar to DeepEval's faithfulness handling.

**Inputs required:** query, response. No context or reference answer needed.

**Example:**

Query: "What year was Python created?"
Response: "Python was created in 1991 by Guido van Rossum. It is a high-level, interpreted programming language. The latest version is Python 3.12."

Statements extracted: "Python was created in 1991", "created by Guido van Rossum", "high-level interpreted language", "latest version is Python 3.12"

Verdicts:
- "Python was created in 1991" → yes (directly answers the question)
- "created by Guido van Rossum" → idk (supporting info, not asked but related)
- "high-level interpreted language" → no (irrelevant to the year question)
- "latest version is Python 3.12" → no (irrelevant to the year question)

Score = 2/4 = 0.5 (two irrelevant statements dilute the answer).

### 2.3 Comparison: RAGAS vs. DeepEval Answer Relevancy

| | RAGAS | DeepEval |
|---|---|---|
| Approach | Reverse question generation + embedding similarity | Direct LLM judgment per statement |
| LLM calls | N (generate questions) + embedding calls | 2 (extract statements + judge verdicts) |
| Requires embeddings? | Yes | No |
| Interpretability | Low (cosine similarity is opaque) | High (each statement gets a verdict with optional reason) |
| What it penalizes | Off-topic content (generated questions diverge from original) | Irrelevant statements in the response |

### 2.4 Weighted Composite Scoring (Databricks)

Databricks published a methodology using a different decomposition that collapses faithfulness and answer relevance into one evaluation: Correctness (60%), Comprehensiveness (20%), Readability (20%), scored on a 0-3 integer scale with LLM-as-judge. Their key findings:

- LLM-human exact agreement: >80%
- LLM-human within 1-score distance: >95%
- GPT-3.5 with few-shot examples: 10x cheaper, 3x faster than GPT-4
- GPT-3.5 without grading rubric: "completely unusable"
- "Evaluation results can't be transferred between use cases"

This illustrates a general pattern: answer-level evaluation benefits from task-specific decomposition rather than generic metrics. A team evaluating a customer support bot needs different criteria than a team evaluating a research assistant.

### 2.5 The Problem with Answer Relevance as a Separate Dimension

In practice, answer relevance is hard to disentangle from correctness. An answer that is relevant but wrong (addresses the right topic, gives incorrect information) scores well on answer relevance and fails on faithfulness. Users experience it as simply "wrong." The decomposition creates a clean analytical framework that does not always map to user experience.

This is why Databricks collapses the three edges into a single "Correctness" judgment with 60% weight. For production evaluation, a composite "is this a good answer?" judgment may be more aligned with user satisfaction than separate scores for faithfulness and relevance.

---

## 3. Failure Modes and Metric Coverage

Sections 1-2 (and Part 1's retrieval metrics) introduced metrics for each evaluation surface. A natural question follows: if I deploy these metrics, what failures will I catch, and what will slip through undetected?

**Failures that current metrics detect:**

| Failure Mode | Description | Detected By |
|---|---|---|
| **Irrelevant retrieval** | Retrieved chunks unrelated to query | Contextual Relevancy (DeepEval), Precision@K |
| **Missed retrieval** | Relevant docs not retrieved | Recall@K, Context Recall (RAGAS/DeepEval) |
| **Hallucination** | LLM invents facts not in context | Faithfulness (RAGAS/DeepEval), NLI classifiers |
| **Lost in the middle** | LLM ignores relevant info in middle of context | Positional analysis |
| **Keyword failure** | Embeddings miss exact-match (IDs, codes) | Hybrid search comparison |
| **Context distraction** | Irrelevant chunks cause hallucination | Noise Sensitivity (RAGAS) |

**Failures that no current metric catches:**

| Failure Mode | Description | Why No Metric Catches It |
|---|---|---|
| **Reasoning errors** | Correct context, wrong conclusion | Faithfulness verifies that claims are attributed to context. It does not check whether the logical inference connecting those claims is valid. |
| **Stale information** | Retrieved content is outdated | No metric compares document timestamps against the question's temporal requirements. A factually correct but outdated passage scores well on all retrieval and faithfulness metrics. |
| **Cross-doc contradictions** | Sources disagree, system picks one silently | No metric checks consistency across retrieved documents. Each document is evaluated independently. |
| **Partial answers** | Answers one part, ignores the rest | Comprehensiveness scoring exists but has low inter-rater reliability. Annotators disagree on what "complete" means. |
| **Confident refusal** | System capable but refuses to answer | Faithfulness metrics score refusals as grounded (no claims made = no unsupported claims). The metric sees a "correct" output. |

These undetectable failures require either human review or task-specific validation logic (e.g., timestamp checks for freshness, arithmetic verification for reasoning).

Beyond blind spots, metrics can also be actively gamed. A system that copies context verbatim scores perfect faithfulness but produces useless answers. LLM judges prefer longer responses ([Dubois et al., COLM 2024](https://arxiv.org/abs/2404.04475)), so systems optimized against them drift toward verbosity. Optimizing for comprehensiveness produces padded answers that technically cover everything but bury the useful signal. Evaluation scores should be used as diagnostics for identifying where quality degrades, not as optimization targets.

