---
title: "Learning SFT+RL on a Multimodal LLM: Series Introduction"
date: 2026-05-20 10:00:00 +0000
categories: [Post-Training VLM]
tags: [VideoScore2, Qwen2.5-VL, VLM, Video Evaluation, SFT, Reinforcement Learning, GRPO, LLaMA-Factory]
---

This series documents the full post-training pipeline — supervised fine-tuning followed by reinforcement learning — applied to a multimodal large language model. Every concept is grounded in a single real project, VideoScore2, which fine-tunes Qwen2.5-VL-7B-Instruct to evaluate AI-generated videos. The series spans six posts covering model architecture, dataset construction, SFT mechanics, the motivation for RL, GRPO training, and inference deployment.

## 1. Goal of This Series

Post-training techniques like supervised fine-tuning (SFT) and reinforcement learning (RL) are the primary means of specializing a general-purpose language model for a specific task. This series teaches both from first principles, not in the abstract, but through the lens of a concrete project that required both stages to achieve strong performance.

The running example is VideoScore2, a video evaluation model that scores AI-generated videos on visual quality, text-to-video alignment, and physical consistency. Building it required understanding how a multimodal model encodes video frames into tokens, how training data must be structured for SFT, why SFT alone produces a model with limited score discrimination, and how reinforcement learning with rule-based rewards sharpens the model's outputs. Each of these topics becomes one post in the series.

The six posts are structured so that each builds on the previous, but can also be read independently by someone already familiar with earlier concepts.

| Blog | Title | Covers |
|------|-------|--------|
| 1 | From Video Frames to Tokens | How Qwen2.5-VL processes video input: dynamic resolution, patch embedding, temporal pooling, mRoPE positional encoding |
| 2 | Building a Training Dataset | How VideoFeedback2 is structured — prompt collection, model sampling, human annotation, chain-of-thought formatting |
| 3 | Supervised Fine-Tuning | The mechanics of SFT in LLaMA-Factory: tokenization, loss masking, layer freezing, DeepSpeed configuration |
| 4 | Why SFT Isn't Enough | The limitations of SFT — score clustering, lack of calibration — and the motivation for reinforcement learning |
| 5 | GRPO in Practice | The RL training loop: group relative policy optimization, rule-based reward functions, training dynamics |
| 6 | Inference and Evaluation | Deploying the trained model, benchmarking against baselines, and analyzing failure modes |

## 2. The Project: VideoScore2

VideoScore2 is a model that evaluates AI-generated videos along three dimensions — visual quality, text-to-video alignment, and physical/common-sense consistency — producing a score from 1 to 5 on each dimension. It is built on Qwen2.5-VL-7B-Instruct and trained in two stages: first supervised fine-tuning on 27,168 annotated videos, then GRPO reinforcement learning with rule-based rewards. The resulting model achieves state-of-the-art correlation with human judgments across multiple video generation benchmarks.

The training data comes from VideoFeedback2, a dataset constructed by collecting 2,933 unique text prompts and generating approximately 10 videos per prompt using different text-to-video models (CogVideoX, Open-Sora, Vchitect, and others). Human annotators then scored each video on the three dimensions and wrote natural-language reasoning explaining their scores. This produces 27,168 training examples, each consisting of a video, a prompt, reasoning text, and three integer scores.

The key design decision in VideoScore2 is "think before you score" — the model generates chain-of-thought reasoning analyzing the video before outputting its numerical scores. This serves two purposes: it makes the evaluation interpretable (users can read why the model assigned a particular score), and it improves accuracy by forcing the model to articulate its observations before committing to a rating. The reasoning and scores are structured in a specific XML-like format that both stages of training enforce. The paper describing this work is "VideoScore2: Think Before You Score in Generative Video Evaluation" (arXiv 2509.22799).

## 3. The Libraries

Three codebases appear throughout the series, each handling a different layer of the system. Understanding their boundaries clarifies which code is responsible for what behavior at any point in the pipeline.

**HuggingFace transformers** provides the model implementation itself. The files `modeling_qwen2_5_vl.py` and `image_processing_qwen2_vl.py` define everything the model computes: the Vision Transformer encoder that converts pixel patches into embeddings, the merger layer that projects and pools visual tokens, the 28-layer LLM decoder that produces text, and the mRoPE positional encoding scheme that gives the model spatial and temporal awareness of visual inputs. Preprocessing functions like `smart_resize` (which determines how to resize frames to valid patch dimensions) and `get_rope_index` (which assigns 3D position indices to each token) also live here. When the series discusses what a forward pass computes or how visual tokens are encoded, transformers is the source of truth.

**LLaMA-Factory** is the training framework that orchestrates everything around the model during fine-tuning. It loads datasets from JSON, samples video frames at a configured fps, tokenizes multi-turn conversations according to chat templates, constructs training labels with proper loss masking (so the model only learns to predict assistant responses, not user prompts), manages DeepSpeed ZeRO-3 distributed training across multiple GPUs, and runs the training loop with gradient accumulation and checkpointing. The relevant code spans `src/llamafactory/data/mm_plugin.py` (video frame extraction and pixel-value construction), `template.py` (chat formatting), `processor/supervised.py` (label construction), and `src/llamafactory/model/` (adapter loading and layer freezing via `visual.py`). LLaMA-Factory calls into transformers for all model-level operations — it never reimplements the forward pass or attention mechanism.

**The VideoScore2 repository** contains the project-specific configuration and scripts that tie the other two together. This includes the YAML training configuration (specifying learning rate, batch size, frozen layers, fps settings), data preparation scripts that convert raw annotations into the format LLaMA-Factory expects, RL training scripts (GRPO reward functions that check score accuracy and format compliance, plus trainer modifications), and inference code for running the trained model on new videos. The VideoScore2 repo makes the project-specific decisions — which layers to freeze, what learning rate to use, how to structure the reward signal — while delegating model computation to transformers and training orchestration to LLaMA-Factory.

The relationship between these three forms a stack. Transformers defines HOW the model computes (architecture, forward pass, attention). LLaMA-Factory defines HOW training is orchestrated (data loading, tokenization, distributed training). VideoScore2 defines WHAT specifically is trained and with what configuration (dataset choice, hyperparameters, reward design). Each layer calls into the one below it.

| Library | Role | Key Files |
|---------|------|-----------|
| HuggingFace transformers | Model implementation | `modeling_qwen2_5_vl.py`, `image_processing_qwen2_vl.py` |
| LLaMA-Factory | Training orchestration | `mm_plugin.py`, `template.py`, `supervised.py`, `visual.py` |
| VideoScore2 | Project config + scripts | YAML config, `prepare_sft_data.py`, `grpo_vs2_sft.py` |

## 4. The Base Model: Qwen2.5-VL-7B-Instruct

Qwen2.5-VL-7B-Instruct is a 7-billion-parameter multimodal model from Alibaba's Qwen team, designed to process both text and visual inputs (images and video). It consists of three components: a Vision Transformer (ViT) with approximately 675 million parameters that encodes visual patches into embeddings, a merger projector that compresses and projects those embeddings into the LLM's input space, and a 28-layer causal language model decoder that processes the combined text and visual tokens to generate output.

The model handles arbitrary image and video resolutions through dynamic patching — rather than resizing all inputs to a fixed resolution, it computes valid dimensions based on patch size constraints and processes the actual content at near-native resolution. For temporal encoding of video frames, it uses multi-dimensional rotary position embedding (mRoPE), which assigns each token a 3D position index covering temporal (which frame), height (row within the frame), and width (column within the frame) dimensions. This allows the model to understand both spatial layout within frames and temporal ordering across frames.

Because the base model is already instruction-tuned for general multimodal tasks (visual question answering, OCR, video understanding), VideoScore2's training does not start from scratch. Instead, it further specializes an already-capable model for the narrow task of structured video evaluation. Blog 1 covers the architecture in full detail, tracing the complete path from raw video frames through the vision encoder, merger, and LLM to understand exactly what VideoScore2 inherits from its base model.

## 5. The Training Pipeline at a Glance

The training pipeline has two stages, each addressing a different aspect of model capability. The first stage teaches the model the task — what format to produce, what reasoning to generate, and roughly what scores to assign. The second stage sharpens the model's score accuracy and format compliance by reinforcing outputs that match ground truth.

Stage 1 is supervised fine-tuning. The base Qwen2.5-VL model is loaded with its vision encoder and merger layers frozen (only the 28-layer LLM is trainable). It is then fine-tuned on 27,168 video-score examples, each formatted as a conversation where the model must produce chain-of-thought reasoning followed by three scores. Training uses LLaMA-Factory with DeepSpeed ZeRO-3 across 8 GPUs, a learning rate of 5e-5, and runs for 2 epochs over approximately 6 hours. The output is a checkpoint that can produce well-formatted video evaluations but tends to cluster scores near the mean.

Stage 2 is reinforcement learning via Group Relative Policy Optimization (GRPO). The SFT checkpoint serves as the starting policy. For each training prompt, the model generates 8 candidate completions. A rule-based reward function scores each completion on two criteria: whether the predicted scores match ground truth (accuracy reward) and whether the output follows the required XML format (format reward). GRPO then updates the policy to increase the probability of higher-reward completions relative to lower-reward ones within each group. This stage uses a learning rate of 2e-6 and runs for 300 steps on 4 GPUs using the Video-R1 framework.

```
Qwen2.5-VL-7B-Instruct --> [SFT, 27K examples, 2 epochs] --> SFT checkpoint --> [GRPO RL, 300 steps] --> VideoScore2
```

Blogs 3 through 5 cover each stage in detail: Blog 3 explains the SFT mechanics (tokenization, loss masking, distributed training configuration), Blog 4 analyzes why the SFT checkpoint alone is insufficient and motivates reinforcement learning, and Blog 5 walks through the GRPO training loop including reward function design and training dynamics.

To begin, proceed to Blog 1: From Video Frames to Tokens, which traces how a raw video file becomes a sequence of tokens that the Qwen2.5-VL model can process.
