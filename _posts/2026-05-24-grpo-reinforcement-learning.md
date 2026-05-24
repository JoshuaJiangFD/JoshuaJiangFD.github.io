---
title: "GRPO: Teaching the Model to Score Accurately"
date: 2026-05-24 14:00:00 +0000
categories: [Post-Training VLM]
tags: [VideoScore2, GRPO, Reinforcement Learning, Video-R1, Qwen2.5-VL, TRL]
mermaid: true
math: true
---

This is Blog 4 in the Post-Training VLM series. [Blog 3]({% post_url 2026-05-23-the-sft-training-loop %}) described the SFT training loop that teaches the model to produce structured reasoning and scores. This post covers the second training stage: Group Relative Policy Optimization (GRPO), where the model generates its own responses and receives reward signal based on score accuracy. The SFT checkpoint from Blog 3 serves as the starting point.

## 1. Why RL After SFT

SFT teaches the model to imitate the training data. Given a video, it learns to produce reasoning and scores that match the annotated examples. But imitation has limits. The paper cites research ([Chu et al., 2025](https://arxiv.org/abs/2501.17161)) arguing that "SFT memorizes, RL generalizes." The SFT-trained model may produce correct-looking reasoning that leads to slightly wrong scores, because it learned the format and style of reasoning without being directly optimized for score accuracy.

GRPO adds a second optimization signal. Instead of "produce text that matches this target" (SFT's objective), the RL objective is "produce text that results in correct scores." The model is free to develop whatever reasoning strategy leads to accurate scores. It may discover reasoning patterns not present in the training data, or learn to self-correct when its initial assessment seems inconsistent.

The ablation from the paper quantifies this. SFT alone achieves ~39.81 accuracy on VideoScore-Bench-v2. Adding 300 steps of GRPO raises this to 44.53, a +4.7 point improvement. The improvement is larger on out-of-domain benchmarks, consistent with the "RL generalizes" hypothesis.

## 2. The GRPO Algorithm

GRPO (Group Relative Policy Optimization) was introduced in the [DeepSeekMath paper (Shao et al., 2024)](https://arxiv.org/abs/2402.03300). It is a simpler alternative to PPO (Proximal Policy Optimization) that eliminates the need for a separate critic/value model.

### The core idea

For each training example (one video + prompt), the model generates G=8 candidate responses. Each response is scored by a reward function. Rather than using an absolute reward baseline (which PPO's critic provides), GRPO uses the group's own statistics as the baseline. Responses that score above the group mean are reinforced. Responses that score below the group mean are suppressed.

### The training step

For a single video, one GRPO step proceeds as follows:

```
1. Generate G=8 completions for the same prompt
2. Score each completion with the reward function → 8 reward values
3. Compute group mean and std of the 8 rewards
4. Normalize: advantage_i = (reward_i - mean) / (std + epsilon)
5. Update policy: increase probability of high-advantage completions,
   decrease probability of low-advantage completions
```

The advantage normalization (step 4) means the model always has some completions to reinforce and some to suppress, regardless of the absolute reward level. Even if all 8 completions score poorly, the best among them still receives positive advantage.

### The loss function

The loss function has three goals. First, increase the probability of tokens that appeared in high-reward completions (reinforce good responses). Second, decrease the probability of tokens from low-reward completions (suppress bad responses). Third, prevent the model from drifting too far from the SFT checkpoint (maintain reasoning quality). The GRPO loss combines these into a single formula using the policy gradient scaled by advantages (goals 1 and 2) and a KL divergence penalty (goal 3):

$$\mathcal{L} = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|c_i|} \sum_{t=1}^{|c_i|} \left[ \hat{A}_i \cdot r_t(\theta) - \beta \cdot \text{KL}_t \right]$$

Reading this formula from outside in:

**Outer layer: average over the group.**

$$-\frac{1}{G} \sum_{i=1}^{G} (\text{...per-completion loss...})$$

The loss averages over all $G = 8$ completions generated for this video. The negative sign converts this into a loss that, when minimized, reinforces high-advantage completions and suppresses low-advantage ones. Specifically:

- When the inner expression is positive (good completion with high advantage), the loss becomes negative. Minimizing the loss means making this more negative, which means making the inner expression more positive. So the model increases probability of tokens in good completions.
- When the inner expression is negative (bad completion with low advantage), the loss becomes positive. Minimizing the loss means reducing this, which means making the inner expression less negative. So the model suppresses tokens in bad completions.

**Middle layer: average over tokens in one completion.**

$$\frac{1}{|c_i|} \sum_{t=1}^{|c_i|} (\text{...per-token policy objective...})$$

For a single completion $i$ (the $i$-th of the 8 generated responses), the loss computes the policy objective (the combination of policy gradient and KL penalty) at every token position $t$, then averages across all positions. The $\frac{1}{|c_i|}$ normalization ensures short and long completions contribute equally regardless of their token count.

**Inner layer: per-token policy gradient + KL penalty.**

$$\hat{A}_i \cdot r_t(\theta) - \beta \cdot \text{KL}_t$$

Each token contributes two terms:

1. $\hat{A}_i \cdot r_t(\theta)$ — the advantage-weighted policy gradient. The advantage $\hat{A}_i$ is the same for every token in the same completion (the entire response was either good or bad as a unit). If the completion got a high advantage (good reward relative to the group), this term pushes the model to increase the probability of each token in that completion. If low advantage, it pushes to decrease them.

2. $-\beta \cdot \text{KL}_t$ — the KL penalty at this position. This pulls the model back toward the reference model (SFT checkpoint) regardless of the advantage. Even if a token is in a high-reward completion, the penalty discourages drifting too far from SFT behavior.

**Definitions of each element:**

- $G = 8$ is the group size (number of completions per prompt)

- $|c_i|$ is the length of completion $i$ in tokens. Completion $i$ refers to the $i$-th of the 8 generated responses for the same video prompt.

- $\hat{A}_i$ is the normalized advantage for completion $i$ (from step 4 in the training step overview above). Positive for above-average completions, negative for below-average.

- $r_t(\theta) = \exp(\log \pi_\theta(a_t \mid s_t) - \log \pi_\theta^{\text{old}}(a_t \mid s_t))$ is the probability ratio at token $t$. Same token $a_t$, same context $s_t$ (the prompt + all generated tokens before position $t$), different model weights. The ratio measures how much the weight update changed the model's probability for token $a_t$ given context $s_t$. Breaking this down further:
  - $\pi_\theta(a_t \mid s_t)$ is the probability the current model (with updated weights) assigns to token $a_t$ at position $t$, given all previous tokens $s_t$. This is the same softmax probability from Blog 3's loss function.
  - $\pi_\theta^{\text{old}}(a_t \mid s_t)$ is the probability the model assigned to the same token at the time the 8 completions were generated. The weights may have since been updated, so the two probabilities differ.
  - The ratio $\pi_\theta / \pi_\theta^{\text{old}}$ measures how much the model's behavior has shifted. A ratio > 1 means the current model is more likely to produce this token than before. A ratio < 1 means less likely.
  - In practice, the code implements this as `exp(logps - logps.detach())`. At the start of each step (before any weight update within this step), the ratio is exactly 1.0, but the `exp(x - x.detach())` form preserves the gradient for backpropagation.

- $\beta = 0.04$ is the KL penalty coefficient, controlling how strongly the model is pulled back toward the SFT checkpoint.

- $\text{KL}_t = \exp(\log \pi_{\text{ref}}(a_t \mid s_t) - \log \pi_\theta(a_t \mid s_t)) - (\log \pi_{\text{ref}}(a_t \mid s_t) - \log \pi_\theta(a_t \mid s_t)) - 1$ is the per-token KL divergence from the reference model. Breaking this down:
  - $\pi_{\text{ref}}(a_t \mid s_t)$ is the probability the reference model (the frozen SFT checkpoint) assigns to token $a_t$ given context $s_t$. This never changes during RL training.
  - $\pi_\theta(a_t \mid s_t)$ is the probability the current model (being trained) assigns to the same token in the same context. This changes as weights are updated.
  - $\log \pi_{\text{ref}} - \log \pi_\theta$ is the log-ratio between the reference and current model. If the current model assigns higher probability to this token than the reference did, this value is negative. If lower, it is positive.
  - The formula $\exp(x) - x - 1$ (where $x = \log \pi_{\text{ref}} - \log \pi_\theta$) is KL divergence expressed in log-probability space (see Appendix for derivation). It is always >= 0 and equals 0 only when $x = 0$ (i.e., the two models agree perfectly on this token). The further the current model drifts from the reference, the larger $\text{KL}_t$ becomes.
  - The purpose: multiplied by $\beta = 0.04$ in the loss, this term penalizes every token where the current model's behavior differs from the SFT checkpoint, regardless of whether the completion received high or low reward.

The KL penalty prevents the model from drifting too far from the SFT checkpoint. Without it, the model could find degenerate strategies (e.g., always outputting the most common score) that maximize reward but lose reasoning quality.

## 3. The Reward Function

This section details Step 2 from Section 2's training step overview. The reward function is rule-based, not learned. It compares the model's predicted scores against ground-truth scores from the dataset. No neural reward model is involved.

### Accuracy reward

The accuracy reward uses a multi-tiered scheme based on how close the predicted scores are to the ground truth across all three dimensions:

```
All three dimensions match exactly:     R_acc = 1.0
Two match, one differs by 1:            R_acc = 0.7
One matches, two differ by 1:           R_acc = 0.4
All three differ by 1:                  R_acc = 0.1
Any dimension differs by more than 1:   R_acc = 0.0
```

The key design principle is that only predictions within ±1 of the ground truth on all dimensions receive non-zero reward. A prediction of (3, 3, 2) against a ground truth of (3, 4, 2) receives 0.7 (one dimension off by 1). A prediction of (3, 3, 2) against (5, 4, 2) receives 0.0 (visual quality off by 2).

The reward function extracts scores from the model's generated text using a regex pattern:

```python
pattern = r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+).*?physical/common-sense consistency:\s*(\d+)"
```

If the pattern fails to match (the model produced malformed output), the reward is 0.0.

### Format reward

The format reward checks whether the response contains a `<think>` block:

```python
pattern = r"<think>.*?</think>"
```

Since VideoScore2 starts RL from the SFT checkpoint (which already produces correctly formatted output), the format reward weight is set to 0. It is only used when training from the base model without SFT.

### Length control

The code includes a length bonus for completions that fall within 320-512 tokens (when accuracy > 0.1). Correct answers with concise reasoning receive a +0.2 bonus. This encourages the model to be efficient in its reasoning without being too terse.

## 4. Advantage Computation

This section details Steps 3-4 from Section 2's training step overview. After the reward function (Section 3) assigns a scalar reward to each of the 8 completions, the advantage computation converts these raw rewards into normalized values that the loss function uses to decide which completions to reinforce and which to suppress.

### Why not use raw rewards directly?

Raw rewards are absolute values (e.g., 1.0, 0.7, 0.4). If all 8 completions happen to score high (e.g., all get 1.0), the model has nothing to learn from this example. If all score low (e.g., all get 0.0), there's no signal either. The advantage normalization makes the signal relative within each group. Even if all completions are mediocre, the best among them gets reinforced and the worst gets suppressed.

### The formula

$$\mu = \frac{1}{G} \sum_{i=1}^{G} R_i$$

$$\sigma = \sqrt{\frac{1}{G} \sum_{i=1}^{G} (R_i - \mu)^2}$$

$$\hat{A}_i = \frac{R_i - \mu}{\sigma + \epsilon}$$

Where:

- $R_i$ is the scalar reward for completion $i$ from Section 3's reward function
- $G = 8$ is the group size
- $\mu$ is the mean reward across all 8 completions for this video
- $\sigma$ is the standard deviation of the 8 rewards
- $\epsilon = 10^{-4}$ is a small constant to prevent division by zero when all rewards are identical
- $\hat{A}_i$ is the normalized advantage for completion $i$, which enters the loss function (Section 2)

For example, using the treadmill video rewards from Section 6:

```
rewards     = [1.0, 0.4, 0.7, 0.7, 0.7, 1.0, 0.4, 1.0]

mu          = (1.0 + 0.4 + 0.7 + 0.7 + 0.7 + 1.0 + 0.4 + 1.0) / 8
            = 0.7375

sigma       = std([1.0, 0.4, 0.7, 0.7, 0.7, 1.0, 0.4, 1.0])
            = 0.235

advantage_1 = (1.0 - 0.7375) / (0.235 + 1e-4) =  1.12  (reinforce)
advantage_2 = (0.4 - 0.7375) / (0.235 + 1e-4) = -1.44  (suppress)
advantage_3 = (0.7 - 0.7375) / (0.235 + 1e-4) = -0.16  (slightly suppress)
```

### Properties

The advantage is positive for completions scoring above the group mean (these get reinforced) and negative for those scoring below (these get suppressed). The division by standard deviation ensures the magnitude of the advantage is consistent across different examples regardless of the reward scale. An example where rewards spread from 0.0 to 1.0 produces advantages of similar magnitude to one where rewards spread from 0.4 to 0.7.

## 5. SFT vs GRPO

### Configuration differences

The RL training uses a different framework (Video-R1) on different hardware than SFT:

| Parameter            | SFT (Blog 3)           | GRPO (this post)                          |
| -------------------- | ---------------------- | ----------------------------------------- |
| Framework            | LLaMA-Factory          | Video-R1                                  |
| Starting model       | Qwen2.5-VL-7B-Instruct | SFT checkpoint                            |
| GPUs                 | 8x A800                | 4x A100                                   |
| Learning rate        | 5e-5                   | 2e-6                                      |
| Epochs               | 2                      | 1                                         |
| Effective batch size | 64                     | 32 (4 GPUs × 1 × 8 accumulation)          |
| Total steps          | 832                    | ~300 (best checkpoint)                    |
| Frozen modules       | ViT + merger           | None specified                            |
| DeepSpeed            | ZeRO-3                 | ZeRO-3                                    |
| Additional           | -                      | gradient_checkpointing, flash_attention_2 |

Key differences:

- **Much lower learning rate** (2e-6 vs 5e-5). RL makes smaller updates to avoid catastrophic forgetting of the SFT-learned behavior.
- **Fewer total steps** (300 vs 832). Performance peaks at 300 steps then degrades.
- **Weight decay enabled** (0.01). Regularization to prevent reward hacking.
- **Gradient checkpointing** enabled. Needed because each step generates 8 completions, multiplying activation memory.
- **Beta = 0.04**. KL penalty coefficient constraining drift from the SFT checkpoint.

### Conceptual differences

The shift from SFT to RL changes what the model is optimized for:

| Aspect                | SFT (Blog 3)                        | GRPO (this post)                           |
| --------------------- | ----------------------------------- | ------------------------------------------ |
| Objective             | Match target text token-by-token    | Produce scores that match ground truth     |
| Reasoning supervision | Every reasoning token has a target  | No supervision on reasoning content        |
| What's rewarded       | Exact next-token prediction         | Score accuracy (±1 tolerance)              |
| Generation            | Teacher forcing (sees ground truth) | Free generation (model's own output)       |
| Reference model       | None (no KL constraint)             | SFT checkpoint (KL penalty prevents drift) |

The critical difference is that reasoning is no longer supervised. During SFT, the model is told exactly what reasoning to produce ("The visual quality is moderate because..."). During RL, the model can produce any reasoning it wants, as long as the final scores are correct. This freedom allows the model to discover reasoning patterns not present in the training data.

## 6. One GRPO Step: The Treadmill Video

This section walks through all five steps from Section 2's training step overview with concrete numbers. Continuing the running example from Blogs 2-3, here is what happens when `004250_a.mp4` ("A man jogs on a treadmill...") is selected for one GRPO step.

**Step 1: Generate 8 completions.** The model receives the video + evaluation prompt and generates 8 independent responses. Each response contains a `<think>` block followed by three scores. Because generation is stochastic (temperature sampling), the 8 responses differ:

```
Completion 1: <think>...moderate quality...</think> visual: 3, alignment: 3, physical: 2
Completion 2: <think>...clear rendering...</think>  visual: 4, alignment: 3, physical: 3
Completion 3: <think>...poor motion...</think>       visual: 3, alignment: 2, physical: 2
Completion 4: <think>...adequate...</think>          visual: 3, alignment: 3, physical: 3
Completion 5: <think>...inconsistent...</think>      visual: 2, alignment: 3, physical: 2
Completion 6: <think>...moderate...</think>          visual: 3, alignment: 3, physical: 2
Completion 7: <think>...good detail...</think>       visual: 3, alignment: 4, physical: 3
Completion 8: <think>...average...</think>           visual: 3, alignment: 3, physical: 2
```

**Step 2: Score with reward function.** Ground truth is (3, 3, 2). Computing accuracy rewards:

```
Completion 1: (3,3,2) vs (3,3,2) → exact match    → R = 1.0
Completion 2: (4,3,3) vs (3,3,2) → two off by 1   → R = 0.0 (visual off by 1 AND physical off by 1... 
              wait: sum of diffs = 1+0+1 = 2, all within ±1)  → R = 0.4
Completion 3: (3,2,2) vs (3,3,2) → one off by 1   → R = 0.7
Completion 4: (3,3,3) vs (3,3,2) → one off by 1   → R = 0.7
Completion 5: (2,3,2) vs (3,3,2) → one off by 1   → R = 0.7
Completion 6: (3,3,2) vs (3,3,2) → exact match    → R = 1.0
Completion 7: (3,4,3) vs (3,3,2) → two off by 1   → R = 0.4
Completion 8: (3,3,2) vs (3,3,2) → exact match    → R = 1.0
```

**Step 3: Compute advantages.**

```
rewards = [1.0, 0.4, 0.7, 0.7, 0.7, 1.0, 0.4, 1.0]
mean    = 0.7375
std     = 0.235

advantages = [(1.0-0.7375)/0.235, (0.4-0.7375)/0.235, ...]
           = [1.12, -1.44, -0.16, -0.16, -0.16, 1.12, -1.44, 1.12]
```

**Step 4: Update policy.** Completions 1, 6, 8 (exact matches) receive positive advantages and are reinforced. Completions 2, 7 (two dimensions off) receive negative advantages and are suppressed. The model learns that reasoning leading to (3, 3, 2) is better than reasoning leading to (4, 3, 3) for this video.

## 7. The Video-R1 Framework

VideoScore2's GRPO training uses [Video-R1 (Feng et al., 2025)](https://github.com/tulerfeng/Video-R1), an open-source video reinforcement learning framework. Video-R1 is built on top of [HuggingFace TRL](https://github.com/huggingface/trl) (Transformer Reinforcement Learning) library and provides multimodal extensions for Qwen2-VL and Qwen2.5-VL models.

The key file is `grpo_trainer.py`, which implements `Qwen2VLGRPOTrainer` extending HuggingFace's `Trainer`. It handles the multimodal-specific complexities that TRL's standard `GRPOTrainer` does not support: processing video inputs during generation, repeating pixel values across the G=8 completions, and managing video_grid_thw tensors.

The data format (from Blog 2, Section 10) is consumed directly:

```json
{
  "problem_id": "004250_a",
  "problem": "[evaluation prompt]",
  "solution": "<answer>visual quality: 3; text-to-video alignment: 3; physical/common-sense consistency: 2</answer>",
  "path": "./vs2_videos/004250_a.mp4",
  "data_type": "video",
  "problem_type": "video eval"
}
```

The `solution` field provides the ground truth for the accuracy reward. The `problem_type` field routes to the correct reward computation logic (VideoScore2 uses "video eval").

## 8. Training Dynamics

The paper reports that performance peaks at 300 steps and degrades beyond. This is characteristic of RL fine-tuning. Early steps improve score accuracy as the model learns to self-correct. Later steps risk reward hacking, where the model finds shortcuts that maximize the reward function without genuine quality assessment.

Three mechanisms constrain this degradation:

1. **KL penalty (beta=0.04)** prevents the policy from drifting far from the SFT checkpoint in token probability space.
2. **Low learning rate (2e-6)** limits how much weights change per step.
3. **Early stopping at 300 steps** based on empirical observation of the reward curve.

The training takes approximately 8 hours per 100 steps on 4x A100 GPUs. The total training time for 300 steps is roughly 24 hours.

## 9. Summary

The two-stage pipeline is now complete:

```
Qwen2.5-VL-7B-Instruct
    │
    ▼ SFT (Blog 3): 832 steps, 8x A800
    │ Learn to produce reasoning + scores (imitation)
    │
SFT Checkpoint
    │
    ▼ GRPO (this post): 300 steps, 4x A100
    │ Optimize for score accuracy (reinforcement)
    │
VideoScore2 (final model)
```

SFT provides the behavioral foundation. The model learns the format (think block + scores), the evaluation dimensions, and a baseline level of assessment quality. GRPO then sharpens the model's scoring accuracy by rewarding correct predictions and suppressing incorrect ones, without constraining how the model reasons about video quality.

## Appendix: RL vs SFT Notation

Blog 3 (SFT) and Blog 4 (GRPO) use different notation conventions for the same underlying computation (softmax probability of a token given preceding context). The difference comes from their respective traditions: SFT uses probability/statistics notation, while RL uses policy notation.

| Blog 3 (SFT notation) | Blog 4 (RL notation) | Meaning |
| ---------------------- | -------------------- | ------- |
| $P(y_t \mid y_{<t}, x)$ | $\pi_\theta(a_t \mid s_t)$ | Softmax probability of the token at position $t$ |
| $y_t$ | $a_t$ | The token at position $t$ (RL calls this an "action") |
| $y_{<t}, x$ | $s_t$ | All preceding context (RL calls this the "state") |
| (implicit in $P$) | $\theta$ subscript | The model weights. RL makes this explicit because it compares multiple versions of the policy ($\pi_\theta$ vs $\pi_\theta^{\text{old}}$ vs $\pi_{\text{ref}}$) |

The computation is identical in both cases. At position $t$, the model's `lm_head` produces a 152,064-dimensional logit vector. Softmax converts it to a probability distribution over all possible tokens. Both $P(y_t \mid ...)$ and $\pi_\theta(a_t \mid ...)$ look up the probability assigned to one specific token in that distribution. The difference is where that specific token comes from:

- In SFT, the token is the ground-truth label from the training data (the annotated response from Blog 2). The model never generated it.
- In RL, the token is one the model produced itself during a "rollout" (RL term for letting the model generate freely from a prompt until it stops). In GRPO, each of the 8 completions is one rollout.

## Appendix: KL Divergence in the GRPO Loss

KL divergence measures how different two probability distributions are. In GRPO, it measures how far the current model's token probabilities have drifted from the reference model (the SFT checkpoint).

### The standard definition

For two distributions $P$ and $Q$ over a discrete set of outcomes, KL divergence is defined as:

$$D_{\text{KL}}(P \| Q) = \sum_x P(x) \cdot \log \frac{P(x)}{Q(x)}$$

This is always >= 0 and equals 0 only when $P = Q$ (the distributions are identical).

### Applying it to a single token position

In GRPO, we want to measure how much the current policy $\pi_\theta$ differs from the reference policy $\pi_{\text{ref}}$ at one token position. Let:

```
x = log(pi_ref(a_t | s_t)) - log(pi_theta(a_t | s_t))
```

This is the log-ratio between the reference and current model's probability for a specific token $a_t$ in context $s_t$.

### Rewriting KL in terms of x

Starting from the KL definition and rewriting in terms of log-probabilities, KL divergence at a single token can be expressed as:

$$\text{KL}_t = \exp(x) - x - 1$$

This is not an approximation. It is KL divergence expressed in log-probability space. The properties of this form:

- When $x = 0$ (models agree perfectly on this token): $\exp(0) - 0 - 1 = 0$. No penalty.
- When $x > 0$ (current model assigns lower probability than reference): penalty grows exponentially.
- When $x < 0$ (current model assigns higher probability than reference): penalty grows, but more gently.
- For small $|x|$ (slight drift): the function behaves like $x^2/2$ (quadratic, gentle penalty).
- For large $|x|$ (significant drift): the function grows exponentially (harsh penalty).

### In the code

```python
x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
per_token_kl = torch.exp(x_clamped) - x_clamped - 1
```

The `clamp` limits $x$ to [-10, 10] for numerical safety (preventing overflow in `exp()`). This is not part of the KL definition, just a practical safeguard.

### Its role in the GRPO loss

In the loss function, $\text{KL}_t$ is multiplied by $\beta = 0.04$ and subtracted from the policy gradient term:

$$\hat{A}_i \cdot r_t(\theta) - \beta \cdot \text{KL}_t$$

The effect: even if a completion receives high reward (positive $\hat{A}_i$), every token in that completion still incurs a KL penalty proportional to how much the model has drifted from SFT. This prevents the model from finding reward-maximizing strategies that completely abandon the SFT-learned reasoning behavior.

## References

| Resource                  | Link                                                                               |
| ------------------------- | ---------------------------------------------------------------------------------- |
| Video-R1                  | [github.com/tulerfeng/Video-R1](https://github.com/tulerfeng/Video-R1)             |
| DeepSeekMath (GRPO paper) | [arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)                       |
| TRL (Transformer RL)      | [github.com/huggingface/trl](https://github.com/huggingface/trl)                   |
| VideoScore2               | [github.com/TIGER-AI-Lab/VideoScore2](https://github.com/TIGER-AI-Lab/VideoScore2) |
| VideoScore2 paper         | [arxiv.org/abs/2509.22799](https://arxiv.org/abs/2509.22799)                       |
