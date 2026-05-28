---
title: "RAG Evaluation Part 3: Training Reliable Faithfulness Judges"
date: 2026-05-25 12:00:00 +0000
categories: [RAG Evaluation]
tags: [RAG, Evaluation, Faithfulness, Hallucination Detection, FaithLens, GRPO, LLaMA-Factory, veRL]
mermaid: true
math: true
---

LLM judges are used throughout RAG evaluation: grading retrieval relevance, scoring answer quality, verifying faithfulness. Not all of these tasks need dedicated training. For retrieval relevance grading, zero-shot LLM judges already correlate with human annotators at [Kendall's tau > 0.9](https://dl.acm.org/doi/10.1145/3626772.3657707) (Thomas et al., SIGIR 2024). Training barely improves on that. Faithfulness detection is different. Zero-shot frontier LLMs achieve only ~76% balanced accuracy here ([LLM-AggreFact](https://arxiv.org/abs/2404.10774)), while specialized fine-tuned models reach 84%. On adversarially hard cases ([FaithBench](https://arxiv.org/abs/2410.13210)), even the best approaches hit ~50%. This is also where research is most active: 20+ papers in the last 18 months focused specifically on training better faithfulness detectors.

RAG makes this problem uniquely tractable. Unlike open-domain hallucination detection (where you'd need to verify claims against all of world knowledge), RAG faithfulness has a *closed* grounding set: the retrieved chunks. Every claim in the response can be checked against a finite, known context. This constraint is what makes training effective.

This post is a deep dive into [FaithLens](https://arxiv.org/abs/2512.20182) (Si et al., ACL 2026 Findings), an 8B faithfulness judge trained with SFT + GRPO that outperforms GPT-5.2 and o3 across 12 faithfulness tasks. The project is [open-sourced](https://github.com/S1s-Z/FaithLens) with training scripts, datasets, and model weights. The SFT stage uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), the RL stage uses [veRL](https://github.com/volcengine/verl) (ByteDance's RL library). The structure mirrors the post-training pipeline from the [VLM series]({% post_url post-training-vlm/2026-05-20-learning-sft-rl-series-introduction %}): supervised fine-tuning teaches the task format, then reinforcement learning sharpens accuracy where SFT plateaus.

---

## 1. Why RL After SFT

SFT teaches the model to imitate the training data. Given a document and a claim, it learns to produce a verdict and explanation that match the annotated examples. But imitation has limits. The FaithLens ablation quantifies this:

| Method | Avg F1 | Std (cross-task variance) |
|--------|--------|---------------------------|
| Llama-3.1-8B-Instruct (zero-shot) | 56.3 | 10.9 |
| SFT on unfiltered 52K examples | 79.1 | 6.1 |
| SFT on filtered 12K examples | 82.6 | 6.0 |
| **SFT + GRPO (FaithLens)** | **86.4** | **4.6** |

SFT alone lifts performance from 56.3 to 82.6 F1. GRPO adds another +3.8 F1 and reduces cross-task variance from 6.0 to 4.6. The variance reduction is significant: it means the model generalizes more consistently across different faithfulness tasks (summarization, QA, data-to-text) rather than performing well on some and poorly on others.

The conceptual difference: during SFT, the model is told exactly what explanation to produce for each example. During RL, the model can produce any explanation it wants, as long as the final verdict is correct *and* the explanation is genuinely useful. This freedom allows the model to discover verification strategies not present in the training data.

---

## 2. The Training Data Pipeline

FaithLens generates its own training data through a three-stage filtering pipeline. The raw source is [FactCG](https://arxiv.org/abs/2501.17144)'s open-source datasets (ANLI subset, Claim-to-Doc, Doc-to-Claim pairs). A synthesis LLM (DeepSeek-V3.2-Think, temperature=1.0) generates chain-of-thought reasoning, an explanation, and a predicted label for each (document, claim) pair.

### 2.1 The Raw Data

Each training example has the structure:

- **Input:** a source document + a claim to verify
- **Output:** `<think>` chain-of-thought `</think>` + `<reason>` explanation `</reason>` + `<answer>` verdict `</answer>`

Starting pool: 52,268 examples total (35,554 for SFT, 16,714 for RL).

### 2.2 Three-Stage Filtering

The filtering is what makes FaithLens's data quality high enough for SFT to reach 82.6 F1 (vs. 79.1 on unfiltered data).

**Stage 1: Label Correctness.** Discard any example where the synthesis LLM's predicted label does not match the ground-truth label. This removes cases where the LLM hallucinated during data generation. 35,554 → 14,258 examples survive.

**Stage 2: Explanation Quality.** For each surviving example, compute the perplexity of the correct label under Llama-3.1-8B-Instruct in two conditions: with and without the explanation prepended. Keep only examples where `PPL_with_explanation < PPL_without_explanation`. This ensures the explanation actually helps the model arrive at the correct answer (if the explanation is vague or misleading, it doesn't reduce perplexity). 14,258 → 4,363 examples survive.

**Stage 3: Data Diversity.** Apply K-Medoids clustering (K=10) on sentence embeddings (Llama-Embed-Nemotron-8B) of the (document, claim) pairs. For each candidate example, check whether it reduces perplexity for at least K/2 of the cluster medoids (probe samples). This prevents the dataset from being dominated by similar examples from one domain.

Final dataset sizes:

| Split | Count |
|-------|-------|
| SFT data | 11,929 |
| RL data | 16,714 |
| Total | 28,643 |

The aggressive filtering (52K → 12K for SFT) is a deliberate tradeoff: fewer examples, but each one has a correct label, a genuinely helpful explanation, and contributes diversity. The ablation confirms this works: filtered 12K outperforms unfiltered 52K by +3.5 F1.

---

## 3. Stage 1: SFT via LLaMA-Factory

The SFT stage uses LLaMA-Factory with DeepSpeed ZeRO-3, the same framework used in the [VLM series]({% post_url post-training-vlm/2026-05-23-the-sft-training-loop %}).

### 3.1 Training Configuration

From [`training/sft/sft.yaml`](https://github.com/S1s-Z/FaithLens):

| Parameter | Value |
|-----------|-------|
| Base model | Llama-3.1-8B-Instruct |
| Training stage | SFT (full-parameter) |
| Learning rate | 1e-5 |
| Scheduler | Cosine with 0.1 warmup |
| Epochs | 3 |
| Per-device batch size | 2 |
| Gradient accumulation | 2 |
| Effective batch size | 32 (2 × 2 × 8 GPUs) |
| Precision | BF16 |
| DeepSpeed | ZeRO-3 |
| GPUs | 8x A800 80GB |

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train ./sft.yaml
```

### 3.2 What the Model Learns

The inference prompt (from `faithlens/prompts.py`) defines the task:

```python
USER_PROMPT_OURS = """Determine whether the provided claim is consistent with
the corresponding document. Consistency in this context implies that all
information presented in the claim is substantiated by the document. If not,
it should be considered inconsistent.

- First, think step by step about whether all the information in the claim is
  fully supported by the document within <think> and </think> tags.
- Then, please provide an easy-to-understand explanation for your answer
  within <reason> and </reason> tags.
- Finally, assess the claim's consistency with the document by responding with
  either "Yes" or "No" and wrap your final answer in <answer> and </answer> tags.

Document: [DOCUMENT]
Claim: [CLAIM]
"""
```

Each training example teaches the model to produce structured output in three parts:

```
<think>
The document states that the merger was completed in Q3 2024. The claim says
the merger was finalized in Q2 2024. These are different time periods.
</think>
<reason>
The document explicitly says "completed in Q3 2024" while the claim states
"finalized in Q2 2024." This is a direct temporal contradiction.
</reason>
<answer>No</answer>
```

The `<think>` block contains chain-of-thought reasoning (internal analysis). The `<reason>` block contains the explanation to be evaluated by the RL reward. The `<answer>` block contains the binary verdict ("Yes" or "No").

Like the VLM series' "think before you score" pattern, forcing the model to reason before producing a verdict improves accuracy. The reasoning serves two purposes: it makes the evaluation interpretable, and it forces the model to articulate its observations before committing to a label.

### 3.3 Training Outcome

The SFT checkpoint is saved at step 1119 (approximately 3 epochs over 11,929 examples at batch size 32). This checkpoint achieves:

- **82.6 avg F1** across 12 faithfulness tasks
- **83.8 explainability score**
- Cross-task standard deviation of 6.0

This is already competitive with [MiniCheck](https://arxiv.org/abs/2404.10774) (80.7 avg F1) despite being a general-purpose architecture rather than a task-specific classifier. But it plateaus here because SFT only teaches imitation of the training explanations. The RL stage pushes beyond this ceiling.

---

## 4. Stage 2: GRPO via veRL

The RL stage uses [veRL](https://github.com/volcengine/verl) (ByteDance/Volcano Engine's RL library) with GRPO. The SFT checkpoint serves as both the starting policy and the reference model for KL regularization.

### 4.1 The GRPO Algorithm

GRPO (Group Relative Policy Optimization, [Shao et al., 2024](https://arxiv.org/abs/2402.03300)) is the same algorithm used in the [VLM series' GRPO blog]({% post_url post-training-vlm/2026-05-24-grpo-reinforcement-learning %}). For each training input, the model generates G completions. Each completion is scored by a reward function. Completions scoring above the group mean are reinforced; those scoring below are suppressed.

For FaithLens, one GRPO step proceeds as follows:

```
1. Sample a (document, claim, ground_truth_label) from the RL dataset
2. Generate G=7 completions (each containing <think> + <reason> + <answer>)
3. Score each completion with the reward function → 7 reward values
4. Compute group mean and std of the 7 rewards
5. Normalize: advantage_i = (reward_i - mean) / (std + epsilon)
6. Update policy: increase probability of high-advantage completions
```

### 4.2 Training Configuration

From [`training/verl/examples/grpo_trainer/train_faithlens.sh`](https://github.com/S1s-Z/FaithLens):

| Parameter | Value |
|-----------|-------|
| Group size (G) | 7 |
| KL coefficient (beta) | 0.001 |
| KL loss type | `low_var_kl` |
| Clip parameter (epsilon) | 0.2 |
| Learning rate | 1e-6 |
| Train batch size | 112 |
| PPO mini-batch size | 16 |
| PPO micro-batch size per GPU | 2 |
| Rollout temperature | 0.6 |
| Max response length | 8196 tokens |
| GPUs | 7x A800 80GB |
| Epochs | 1 |
| Save frequency | Every 50 steps |
| Gradient checkpointing | Enabled |

Key differences from SFT:

- **Much lower learning rate** (1e-6 vs 1e-5). RL makes smaller updates to avoid catastrophic forgetting.
- **Very low KL coefficient** (0.001 vs VideoScore2's 0.04). The explanation reward already constrains reasoning quality, so less KL pressure is needed.
- **7 GPUs instead of 8.** The 8th GPU serves the novice model used in the explanation reward (Section 5.2).

### 4.3 Total Training Steps

With 16,714 RL examples, batch size 112, and 1 epoch:

$$\text{Steps} = \frac{16{,}714}{112} \approx 149$$

The training saves checkpoints every 50 steps. The best checkpoint is selected based on validation performance.

---

## 5. The Reward Function

The reward function is rule-based and deterministic. No learned reward model is involved. Each completion receives a score that is the sum of three components:

$$R = R_{\text{pred}} + R_{\text{exp}} + R_{\text{format}}$$

Maximum possible reward: 3.0.

Implementation: [`training/verl/verl/utils/reward_score/ours_reward_with_reason.py`](https://github.com/S1s-Z/FaithLens)

### 5.1 Prediction Reward (R_pred)

Extract the verdict from `<answer>...</answer>` tags. Normalize (lowercase, remove punctuation and articles, collapse whitespace). Compare against ground truth via exact match.

From `ours_reward_with_reason.py`:

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)

    if answer is None:
        em_score = 0.0
    else:
        if em_check(answer, ground_truth):
            if open_count > 10 or close_count > 10:  # prevent degenerate output
                em_score = 1 / 4
            else:
                em_score = 1
        else:
            em_score = 0
```

```
Correct verdict:                R_pred = 1.0
Wrong verdict:                  R_pred = 0.0
Output contains >10 answer tags: R_pred = 0.25 (penalize degenerate output)
```

### 5.2 Explanation Reward (R_exp)

This is the most novel component. It measures whether the explanation actually helps a weaker model arrive at the correct answer.

The mechanism: a "novice" model (vanilla Llama-3.1-8B-Instruct, untuned) is served via vLLM on GPU 7, separate from the 7 GRPO training GPUs (0-6). From `activate_vllm.sh`:

```bash
CUDA_VISIBLE_DEVICES=7 \
vllm serve Llama-3.1-8B-Instruct \
  --served-model-name llama3-8b-instruct \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.7
```

For each completion, the reward function queries the novice model twice via OpenAI-compatible API:

1. **Without explanation:** Give the novice (document, claim) and ask for a verdict.
2. **With explanation:** Give the novice (document, claim, explanation from the `<reason>` block) and ask for a verdict.

From `ours_reward_with_reason.py`:

```python
def reason_reward(solution_str, ground_truth, client, extra_info):
    doc = extra_info['doc']
    claim = extra_info['claim']

    # Query 1: novice without explanation
    ori_prompt = USER_PROMPT_ori.replace("[DOCUMENT]", doc).replace("[CLAIM]", claim)
    ori_response = query_response(ori_prompt, client).strip()
    ori_answer = extract_solution(ori_response)
    ori_score = em_check(ori_answer, ground_truth) if ori_answer else 0

    # Query 2: novice with explanation
    extracted_explain = extract_explanation(solution_str)
    if extracted_explain is None:
        return 0
    explain_prompt = USER_PROMPT_with_reason.replace("[DOCUMENT]", doc) \
        .replace("[CLAIM]", claim).replace("[EXPLANATION]", extracted_explain)
    explain_response = query_response(explain_prompt, client).strip()
    explain_answer = extract_solution(explain_response)
    explain_score = em_check(explain_answer, ground_truth)

    # Scoring logic
    if ori_score == 1 and explain_score == 1:
        return 1.0
    elif ori_score == 0 or explain_score == 1:
        return 1.0
    else:
        return 0.0
```

The scoring logic in plain terms:

```
If novice_without == correct AND novice_with == correct: R_exp = 1.0 (no harm)
If novice_without == wrong AND novice_with == correct:   R_exp = 1.0 (explanation helped)
If novice_without == wrong AND novice_with == wrong:     R_exp = 1.0 (novice too weak)
If novice_without == correct AND novice_with == wrong:   R_exp = 0.0 (explanation misled)
```

The only case that receives zero reward is when the explanation *misleads* the novice. An explanation that doesn't help is not penalized (the novice may simply be too weak). But an explanation that actively hurts is penalized. This encourages the model to produce explanations that are at worst harmless and at best genuinely clarifying.

### 5.3 Format Reward (R_format)

Check that the output matches the expected structure. From `ours_reward_with_reason.py`:

```python
def format_reward(predict_str: str) -> float:
    pattern = re.compile(
        r"^<think>[^<>]*</think>\s*<reason>[^<>]*</reason>\s*<answer>[^<>]*</answer>\s*$",
        re.DOTALL
    )
    match_result = re.fullmatch(pattern, predict_str.strip())
    return 1.0 if match_result else 0.0
```

The `[^<>]*` pattern (no angle brackets allowed inside tags) prevents nested XML tags.

```
Correct format:  R_format = 1.0
Malformed output: R_format = 0.0
```

### 5.4 Why This Reward Design Works

The three-component reward addresses three distinct failure modes:

| Component | What it prevents |
|-----------|-----------------|
| R_pred | Wrong verdicts (the obvious failure) |
| R_exp | Vague or misleading explanations (correct verdict but useless reasoning) |
| R_format | Degenerate output (repeated tags, missing structure) |

The explanation reward is the key difference from simpler approaches. Without it, the model could learn to produce correct verdicts with meaningless explanations like "the claim is not supported by the context." With it, the model must produce explanations specific enough to actually guide a weaker model. The ablation confirms: removing R_exp drops avg F1 from 86.4 to 85.7 and explainability from 90.4 to 84.7.

---

## 6. One GRPO Step: Walkthrough

To make the training loop concrete, here is what happens for one (document, claim) pair from the RL dataset.

**Input:**

- Document: "SpaceX launched 23 Starlink satellites into orbit on March 15, 2025, from Cape Canaveral Space Force Station."
- Claim: "SpaceX launched 23 Starlink satellites from Kennedy Space Center in March 2025."
- Ground truth: `not_supported` (Cape Canaveral ≠ Kennedy Space Center, though they are adjacent facilities)

**Step 1: Generate G=7 completions** at temperature 0.6.

```
Completion 1: <think>The document says Cape Canaveral, claim says Kennedy Space Center.
These are different facilities.</think>
<reason>Cape Canaveral Space Force Station and Kennedy Space Center are
adjacent but distinct launch facilities.</reason>
<answer>not_supported</answer>

Completion 2: <think>23 satellites, March 2025, SpaceX — all match. The launch
site is Cape Canaveral in the document.</think>
<reason>The main facts are consistent.</reason>
<answer>supported</answer>

Completion 3: <think>The document says Cape Canaveral, the claim says Kennedy
Space Center. While close geographically, these are technically different.</think>
<reason>The launch site in the claim (Kennedy Space Center) does not match the
document (Cape Canaveral Space Force Station).</reason>
<answer>not_supported</answer>

... (4 more completions)
```

**Step 2: Score each completion.**

| Completion | R_pred | R_exp | R_format | Total |
|-----------|--------|-------|----------|-------|
| 1 (correct, specific explanation) | 1.0 | 1.0 | 1.0 | 3.0 |
| 2 (wrong verdict) | 0.0 | 1.0 | 1.0 | 2.0 |
| 3 (correct, specific explanation) | 1.0 | 1.0 | 1.0 | 3.0 |
| 4 (correct, vague explanation) | 1.0 | 0.0 | 1.0 | 2.0 |
| 5 (correct, specific) | 1.0 | 1.0 | 1.0 | 3.0 |
| 6 (wrong verdict) | 0.0 | 1.0 | 1.0 | 2.0 |
| 7 (correct, specific) | 1.0 | 1.0 | 1.0 | 3.0 |

**Step 3: Compute advantages.**

```
rewards    = [3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0]
mean       = 2.57
std        = 0.53

advantages = [0.81, -1.08, 0.81, -1.08, 0.81, -1.08, 0.81]
```

**Step 4: Update policy.** Completions 1, 3, 5, 7 (correct verdict + specific explanation) receive positive advantages and are reinforced. Completions 2, 6 (wrong verdict) and Completion 4 (correct but vague explanation that misled the novice) receive negative advantages and are suppressed.

The model learns two things simultaneously: (1) Cape Canaveral ≠ Kennedy Space Center is a meaningful distinction, and (2) explanations must cite the specific discrepancy to be useful.

---

## 7. Results

### 7.1 Comparison with Baselines

Full results across 12 faithfulness tasks (from Table 1):

| Model | Size | Avg F1 | Std |
|-------|------|--------|-----|
| GPT-4o | — | 76.1 | 7.0 |
| o1 | — | 76.8 | 5.9 |
| o3 | — | 82.1 | 6.0 |
| GPT-5.2 | — | 86.1 | 5.9 |
| [MiniCheck](https://arxiv.org/abs/2404.10774) | 770M | 80.7 | 7.5 |
| [FactCG](https://arxiv.org/abs/2501.17144) | — | 78.2 | 7.0 |
| **FaithLens** | **8B** | **86.4** | **4.6** |

FaithLens exceeds GPT-5.2 by +0.3 F1 with substantially lower variance (4.6 vs 5.9). It outperforms MiniCheck by +5.7 F1. The low variance means consistent performance across summarization, QA, and data-to-text tasks rather than excelling on some and failing on others.

### 7.2 Ablation: Each Component's Contribution

| Configuration | Avg F1 | Explainability |
|---------------|--------|----------------|
| Base model (zero-shot) | 56.3 | 71.9 |
| SFT on unfiltered 52K | 79.1 | — |
| SFT on filtered 12K | 82.6 | 83.8 |
| SFT + GRPO without R_exp | 85.7 | 84.7 |
| **SFT + GRPO (full FaithLens)** | **86.4** | **90.4** |

Each component contributes:
- Data filtering: +3.5 F1 (52K unfiltered → 12K filtered)
- GRPO with all rewards: +3.8 F1 (SFT-only → SFT+GRPO)
- Explanation reward specifically: +0.7 F1 and +5.7 explainability

### 7.3 Generalization Across Base Models

The paper also tests the recipe on Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct. Both show consistent improvement from SFT + GRPO, suggesting the training approach transfers across model families and sizes.

---

## 8. Practical Considerations

### 8.1 Compute Requirements

| Stage | Hardware | Time |
|-------|----------|------|
| Data synthesis (DeepSeek-V3.2-Think) | API calls | ~hours |
| Data filtering (perplexity computation) | 1 GPU | ~hours |
| SFT (3 epochs, 12K examples) | 8x A800 | ~2-3 hours |
| RL (298 steps, G=7) | 7x A800 + 1 for novice model | ~8-12 hours |

The RL stage is more expensive than SFT because each step generates 7 completions per input and queries the novice model for explanation scoring. The novice model adds latency to each reward computation.

### 8.2 When to Use This Approach

The FaithLens recipe is appropriate when:
- You need the highest detection accuracy on difficult cases (subtle contradictions, plausible inferences)
- You need explanations (for debugging, audit trails, or user-facing rationales)
- You have GPU infrastructure for serving an 8B model
- Your domain has faithfulness patterns not covered by existing detectors

### 8.3 Simpler Alternatives

If you don't need the full SFT + RL pipeline:

| Model | Size | Avg F1 | Use Case |
|-------|------|--------|----------|
| [MiniCheck](https://arxiv.org/abs/2404.10774) | 770M | 80.7 | Fast binary classification, CPU-runnable |
| [LettuceDetect](https://arxiv.org/abs/2502.17125) | ~150M | F1 79.2 (RAGTruth) | Token-level localization, 30-60 ex/sec |
| [HHEM 2.1-Open](https://vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model/) | ~100M | 76.6 BAcc | Lightweight monitoring, <600MB RAM |
| [HalluGuard](https://arxiv.org/abs/2510.00880) | 4B | 84.0 BAcc | SFT + preference optimization, reasoning-based |

These are all SFT-only approaches. They are cheaper to train and faster at inference, but they do not produce explanations and plateau at lower accuracy on difficult cases.

---

## 9. Open Questions

**Hard cases.** [FaithBench](https://arxiv.org/abs/2410.13210) showed that adversarially difficult examples defeat all current detectors (~50% accuracy). FaithLens's improvement to 86.4 F1 is on the LLM-AggreFact benchmark, which contains a mix of easy and hard cases. On the hardest tier specifically, the gap remains.

**Long-context faithfulness.** [HalluMix](https://arxiv.org/abs/2505.00506) demonstrated that detection accuracy degrades as context length increases. FaithLens's max prompt length is 32,768 tokens, but its training data may not include enough long-context examples to fully exploit this window.

**Cross-document reasoning.** Current detectors verify each claim against the full context independently. They cannot catch contradictions *between* retrieved documents, or errors that arise from incorrectly synthesizing information across multiple sources.

**Domain transfer.** FaithLens is trained on general-domain data (news, Wikipedia, QA datasets). Faithfulness patterns in specialized domains (legal, medical, financial) may require domain-specific training data.

## References

| Resource | Link |
|----------|------|
| FaithLens paper | [arXiv 2512.20182](https://arxiv.org/abs/2512.20182) |
| FaithLens code | [github.com/S1s-Z/FaithLens](https://github.com/S1s-Z/FaithLens) |
| FaithLens model | [HuggingFace ssz1111/FaithLens](https://huggingface.co/ssz1111/FaithLens) |
| FaithLens dataset | [HuggingFace ssz1111/FaithLens](https://huggingface.co/datasets/ssz1111/FaithLens) |
| veRL (RL framework) | [github.com/volcengine/verl](https://github.com/volcengine/verl) |
| LLaMA-Factory (SFT framework) | [github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| DeepSeekMath (GRPO paper) | [arXiv 2402.03300](https://arxiv.org/abs/2402.03300) |
| MiniCheck | [arXiv 2404.10774](https://arxiv.org/abs/2404.10774) |
| FactCG | [arXiv 2501.17144](https://arxiv.org/abs/2501.17144) |
