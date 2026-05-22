---
title: "The VideoFeedback2 Dataset: What the Model Learns From"
date: 2026-05-22 14:00:00 +0000
categories: [Post-Training VLM]
tags: [VideoScore2, VideoFeedback2, VLM, Video Evaluation, Dataset, SFT]
---

This post is the third in the Post-Training VLM series. The [previous post (Blog 1)]({% post_url 2026-05-21-from-video-frames-to-tokens %}) described how a raw video is transformed into visual tokens ready for the LLM decoder. This post covers the VideoFeedback2 dataset -- the 26,634 annotated examples that define both the input content (video + evaluation prompt) and the target output (reasoning + scores) for SFT. A running example -- one text prompt and its 10 generated videos -- is threaded throughout to show exactly how the dataset is constructed, annotated, and formatted for supervised fine-tuning.

## 1. The Supervision Problem

Blog 1 ended with 3,520 visual tokens entering the LLM's input sequence alongside a text prompt. But the LLM does not know what to do with those tokens until it is trained. Supervised fine-tuning requires input-output pairs: given a video and an evaluation prompt (input), produce reasoning and quality scores (output). The VideoFeedback2 dataset provides 26,634 such pairs.

The challenge is not just volume. The model must learn to evaluate across three distinct dimensions (visual quality, text-to-video alignment, physical consistency), across videos from 22 different T2V models spanning vastly different quality levels, and produce both a reasoning trace explaining its assessment and integer scores. The dataset's design addresses each of these requirements.

## 2. Prompt Collection

The foundation of VideoFeedback2 is 2,933 unique text-to-video prompts. Each prompt describes a scene that T2V models will attempt to generate. The prompts come from five sources:

| Source | Description | Purpose |
| ------ | ----------- | ------- |
| VidProM | Real user queries from generative model communities | Representative of actual T2V usage |
| Koala-36M | Structured captions from real video segments | Grounded in physically plausible scenes |
| OCR-Text | ~200 prompts requiring visible text in the video | Tests a known weakness of T2V models |
| Multi-Action | ~200 prompts with 2-3 connected actions | Tests temporal coherence |
| Camera Motion | Existing prompts augmented with zoom/pan/tilt/tracking | Tests motion understanding |

All prompts pass through two filtering stages. A rule-based filter removes NSFW content (probability > 0.2), prompts referencing specific people, prompts outside the 15-100 word range, and prompts containing image-to-video cues or aspect ratio specifications. A second LLM-based filter (GPT-4o-mini) removes prompts that are vague, lack motion, or describe more than three actions.

The running example for this post uses the prompt:

> A man jogs on a treadmill, increases the speed, and then bursts into a full sprint.

This is a multi-action prompt (three distinct phases: jog, speed increase, sprint) that tests both text alignment and physical consistency -- the model must generate a plausible acceleration sequence.

## 3. Video Generation

For each prompt, 10 T2V models are randomly sampled from a pool of 22 models. The sampling is stratified across four quality tiers so that the model sees the full quality spectrum for any given prompt:

| Tier | Label | Proportion | Example Models |
| ---- | ----- | ---------- | -------------- |
| 1 | Modern | 10% | Kling-1.6, Sora, Pika-2.2, StepVideo-T2V |
| 2 | Good | 35% | Wanx-2.1 (1.3B), MagicTime, Mochi1-Preview, CogVideoX (5B) |
| 3 | Moderate | 42% | CogVideoX (2B), LTX-Video, OpenSora v1.2, VideoCrafter2, AnimateDiff |
| 4 | Poor/Early | 13% | ModelScope, ZeroScope, T2V-Zero |

Typically 1-2 videos come from the Poor tier, 3-4 from Moderate, 3-4 from Good, and 1 from Modern. This means the training data for any single prompt contains videos ranging from barely recognizable (Poor tier) to near-photorealistic (Modern tier), teaching the model to discriminate quality levels within the same semantic content.

For the treadmill prompt, 10 videos were generated. Their final scores span:

```
004250_a.mp4: visual=3, alignment=3, physical=2
004250_n.mp4: visual=4, alignment=3, physical=3
004250_h.mp4: visual=3, alignment=3, physical=3
004250_z.mp4: visual=2, alignment=2, physical=3
004250_u.mp4: visual=3, alignment=2, physical=2
004250_e.mp4: visual=3, alignment=3, physical=2
004250_c.mp4: visual=3, alignment=3, physical=3
004250_j.mp4: visual=2, alignment=3, physical=2
004250_y.mp4: visual=3, alignment=2, physical=3
004250_b.mp4: visual=3, alignment=2, physical=3
```

The scores range from 2 to 4, with no video achieving 5 on any dimension. This is typical -- even for a straightforward prompt, most T2V models struggle with at least one aspect (in this case, the speed-increase transition is hard to depict convincingly, dragging down alignment and physical consistency scores).

The generated videos vary in resolution (256x256 to 1920x982), frame rate (8-30 fps), and duration (1-6 seconds). Total: 2,933 prompts x ~10 models = 27,168 videos, of which 26,634 survive deduplication and annotation quality control.

## 4. The Annotation Pipeline

Converting 27,168 raw videos into usable training data requires three stages: human scoring, LLM rationale generation, and alignment reconciliation.

**Stage 1: Human scoring.** 15 trained annotators scored each video on the three dimensions (integer 1-5) and wrote short diagnostic comments describing what they observed. For the treadmill video `004250_a.mp4`, an annotator might write: "running motion present but no visible speed change; treadmill belt speed appears constant throughout." These comments capture the specific evidence behind a score without committing to the score's numerical value.

Annotators went through pilot rounds (30-50 videos each) with reviewer feedback before working on the main dataset. Periodic audits spot-checked 10-20% of annotations.

**Stage 2: LLM rationale generation.** Claude-4-Sonnet (with thinking enabled) generated chain-of-thought reasoning traces and its own scores for each video. The LLM received: sampled video frames, the evaluation instructions, the annotator's diagnostic comments (but not scores), and 2-3 few-shot examples. This is a semi-blind approach -- the LLM sees what annotators noticed but must independently construct its assessment and arrive at its own scores.

For `004250_a.mp4`, Claude produced reasoning like:

> **Visual Quality Analysis:** The man is reasonably clear, and the treadmill and gym setting are recognizable. Lighting is generally consistent and colors appear believable. The resolution is adequate for identifying the scene, but the detail level is not high.
>
> **Text-to-Video Alignment:** The prompt specifies a sequence: jog, speed increase, then sprint. The scene depicts a man on a treadmill and clearly shows running/jogging, which aligns with the initial portion. However, the key elements -- observing a speed increase and a transition to a sprint -- are not observable.
>
> **Physical Consistency:** The static frames suggest a plausible gym workout scene with normal running motion, but there is no demonstrated speed increase or sprint, and some posture appearances could be interpreted as inconsistent with rapid acceleration on a treadmill.

**Stage 3: Score-rationale alignment.** The human scores and LLM-generated scores are reconciled. If the difference between them is <= 1 on a dimension, the human score is kept. If the difference is 2, they are averaged. If any dimension differs by >= 3, the entry is re-scored up to three times, then discarded if still misaligned (this accounts for the gap between 27,168 generated videos and 26,634 training examples).

After reconciliation, GPT-5-mini makes minor edits to the rationale text when the final score differs from what the reasoning implies. For instance, if the human score is 3 but the rationale says "very poor resolution," GPT-5-mini might soften it to "moderate resolution" so the reasoning logically supports the final score.

## 5. The Training Format

Each of the 26,634 examples is formatted as a ShareGPT-style two-turn conversation. The JSON structure:

```json
{
  "conversations": [
    {"from": "human", "value": "<video>[evaluation prompt with definitions]...\n\ntext prompt:\n[specific prompt]\n\n..."},
    {"from": "gpt", "value": "<think>\n[reasoning]\n</think>\n\n(1) visual quality: [score]\n(2) text-to-video alignment: [score]\n(3) physical/common-sense consistency: [score]"}
  ],
  "videos": ["videos/004250_a.mp4"]
}
```

The human turn contains a fixed template (~500 tokens when tokenized) with three parts: the `<video>` placeholder (replaced by 3,520 visual tokens during training, as Blog 1 described), definitions of the three evaluation dimensions, and the specific text prompt for this video.

The assistant turn is the target output the model must learn to generate. It has two parts:

1. A `<think>` block containing the reasoning trace (mean length ~620 tokens, ranging from 25 to 1,350 tokens)
2. Three lines of integer scores in a fixed format

During training, only the assistant turn contributes to the loss. The human turn (including the visual tokens) is masked with `IGNORE_INDEX = -100`, as described in Blog 1, Section 7. The model learns to predict the reasoning and scores given the video and prompt, but is never asked to reconstruct the input.

## 6. The Think Block Structure

The `<think>` block varies in detail based on what there is to say about a video. A low-quality video with obvious problems receives detailed criticism:

```
<think>
multi-dimensional analysis:
I need to analyze this AI-generated video based on three dimensions.

The prompt was: "A man jogs on a treadmill, increases the speed,
and then bursts into a full sprint"

Visual Quality Analysis:
Looking at the frames, the visual quality is overall moderate. The man
is reasonably clear, and the treadmill and gym setting are recognizable.
Lighting is generally consistent. The resolution is adequate but the
detail level is not high.

Text-to-Video Alignment:
The prompt specifies a sequence: jog, speed increase, then sprint. The
scene depicts a man on a treadmill which aligns with the initial portion.
However, the speed increase and transition to sprint are not observable.

Physical Consistency:
The scene suggests a plausible gym workout with normal running motion,
but there is no demonstrated speed increase, and some posture appearances
could be interpreted as inconsistent with rapid acceleration.
</think>

(1) visual quality: 3
(2) text-to-video alignment: 3
(3) physical/common-sense consistency: 2
```

A video that fails completely on all dimensions may receive a minimal think block:

```
<think>
multi-dimensional analysis:
Visual Quality: 1; Text-to-Video Alignment: 1; Physical Consistency: 1
</think>

(1) visual quality: 1
(2) text-to-video alignment: 1
(3) physical/common-sense consistency: 1
```

Approximately 35% of examples (9,341 out of 26,634) include explicit ground-truth score labels within the think block -- these are cases where the reconciled scores were injected into the reasoning trace during the alignment stage (Section 4, Stage 3). The remaining 65% contain reasoning that naturally arrives at scores consistent with the final labels.

## 7. Score Distributions

The score distributions across all 26,634 training examples:

| Score | Visual Quality | T2V Alignment | Physical Consistency |
| ----- | -------------- | ------------- | -------------------- |
| 1     | 2,535 (10%)    | 2,831 (11%)   | 2,675 (10%)          |
| 2     | 5,291 (20%)    | 4,917 (18%)   | 6,212 (23%)          |
| 3     | 10,397 (39%)   | 8,911 (33%)   | 9,910 (37%)          |
| 4     | 6,899 (26%)    | 7,726 (29%)   | 5,490 (21%)          |
| 5     | 1,512 (6%)     | 2,249 (8%)    | 2,347 (9%)           |
| Mean  | 2.98           | 3.06          | 2.95                 |

The means cluster near 3.0 because the tier-stratified video generation (Section 3) deliberately produces videos spanning the full quality range. Score 5 is rare (6-9%) because even Modern-tier models rarely achieve perfect scores on all dimensions simultaneously -- generating a physically consistent multi-action sequence with high visual fidelity remains challenging.

The distributions are roughly bell-shaped with slight left skew (more 2s than 4s), reflecting the fact that three of the four quality tiers (Poor, Moderate, Good) tend to produce videos scoring 2-3, while only the Modern tier regularly produces 4s and 5s.

## 8. The Test Set (VideoScore-Bench-v2)

500 videos are held out entirely from training and form the evaluation benchmark. The test set has a different format from the training data:

| Field | Content |
| ----- | ------- |
| `video_name` | Video identifier (e.g., "004882_h") |
| `prompt` | The text-to-video prompt |
| `visual_score`, `t2v_score`, `phy_score` | Human-annotated ground-truth scores |
| `thinking` | Reference reasoning trace |

The test set uses direct human annotations as ground truth, without the LLM rationale generation and reconciliation process applied to training data. This ensures evaluation measures the model's ability to match human judgement rather than its ability to reproduce LLM-generated reasoning.

The 500 test videos come from 472 unique prompts (some prompts have multiple test videos). These prompts do not overlap with the 2,933 training prompts.

## 9. The RL Training Format

The same videos are also formatted for reinforcement learning (GRPO), used in the second training stage after SFT. The RL format strips out the reasoning trace and provides only ground-truth scores as the reward signal:

```json
{
  "problem_id": "004250_a",
  "problem": "[evaluation prompt without <video> tag]",
  "solution": "<answer>visual quality: 3; text-to-video alignment: 3; physical/common-sense consistency: 2</answer>",
  "path": "./vs2_videos/004250_a.mp4"
}
```

During GRPO training, the model generates its own reasoning and scores multiple times (group size = 8), and a rule-based reward function compares the generated scores against the ground-truth in `solution`. The RL data contains 26,669 entries (slightly more than SFT due to 35 duplicated problem_ids). Blog 4 will cover the GRPO training in detail.

## 10. Summary

The VideoFeedback2 dataset translates the abstract task "evaluate video quality" into 26,634 concrete input-output pairs. For any given training step, the model sees:

```
Input:  3,520 visual tokens + ~500 prompt tokens
Output: <think>[reasoning]</think> + three integer scores
```

The dataset's key design decisions:

1. **Stratified generation** (22 models across 4 tiers) ensures every prompt has videos spanning scores 1-5
2. **Hybrid annotation** (human scores + LLM reasoning + alignment reconciliation) produces training targets that are both accurate and explanatory
3. **Two output formats** (SFT with reasoning traces, RL with scores only) support the two-stage training pipeline

The next post (Blog 3) covers how LLaMA-Factory consumes this dataset during SFT -- the training loop, loss computation, and the hyperparameter choices that turn 26,634 examples into a trained video evaluator over 2 epochs on 8 GPUs.

## References

| Resource | Link |
| -------- | ---- |
| VideoFeedback2 dataset | [huggingface.co/datasets/TIGER-Lab/VideoFeedback2](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback2) |
| VideoScore2 paper | [arxiv.org/abs/2509.22799](https://arxiv.org/abs/2509.22799) |
| VideoScore2 code | [github.com/TIGER-AI-Lab/VideoScore2](https://github.com/TIGER-AI-Lab/VideoScore2) |
