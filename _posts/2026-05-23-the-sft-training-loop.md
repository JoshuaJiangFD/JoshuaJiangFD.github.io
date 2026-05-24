---
title: "The SFT Training Loop: Full-Parameter Fine-Tuning via LLaMA-Factory"
date: 2026-05-23 14:00:00 +0000
categories: [Post-Training VLM]
tags: [VideoScore2, LLaMA-Factory, SFT, DeepSpeed, Qwen2.5-VL, Fine-Tuning, ZeRO-3]
mermaid: true
math: true
---

This is Blog 3 in the Post-Training VLM series. Blog 1 described the vision pipeline that converts a video into 3,520 visual tokens. Blog 2 detailed the VideoFeedback2 dataset -- 26,634 examples of chain-of-thought reasoning and scores that define the training signal. This post covers the machinery that converts that dataset into weight updates: the full SFT training loop as implemented by LLaMA-Factory, running on 8x A800 GPUs with DeepSpeed ZeRO-3.

## 1. The Training Configuration

The entire training run is specified by a single LLaMA-Factory YAML configuration file:

```yaml
### model
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
video_fps: 2.0
video_maxlen: 10
video_max_pixels: 691200  # 960*720

### method
stage: sft
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
template: qwen2_vl
cutoff_len: 8192

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
```

## 2. What Gets Trained

The first decision in any fine-tuning configuration is what parameters receive gradients. VideoScore2's YAML specifies three freeze flags:

```yaml
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false
```

The result is a clean partition. The ViT (675M parameters) and merger (a small MLP projection) are entirely frozen -- they contribute to the forward pass but receive no gradients. The LLM decoder (28 transformer layers, hidden size 3584, 28 attention heads, approximately 7.6B parameters) is the sole trainable component. Of the model's total 8.3B parameters, only the 7.6B in the decoder receive weight updates.

This partition means the vision pipeline functions as a fixed feature extractor. The visual tokens entering the LLM have the same quality and meaning they had after Qwen2.5-VL's original multimodal pretraining. What changes during VideoScore2 training is how the LLM interprets those tokens -- learning to attend to visual features that indicate quality, alignment, and physical consistency, and to generate structured reasoning followed by integer scores.

The following diagram shows the end-to-end training loop for one optimization step, indicating which components are frozen and where the loss/gradient flow:

```mermaid
graph TD
    subgraph "Data Loading"
        A[VideoFeedback2 JSON] --> B[Data Processor]
        B --> C[input_ids + labels]
        B --> D[pixel_values]
        C --> E[Data Collator]
        D --> E
        E --> F[Tensor Dictionary]
    end

    subgraph "Forward Pass"
        F --> G[ViT - FROZEN]
        F --> H[Embedding Layer]
        G --> I[Merger - FROZEN]
        I --> J[Visual Embeddings]
        H --> K[Text Embeddings]
        J --> L[Embedding Injection]
        K --> L
        L --> M[LLM Decoder 28 layers - TRAINABLE]
        M --> N[Logits]
    end

    subgraph "Loss and Update"
        N --> O[CrossEntropyLoss]
        C --> O
        O --> P[loss.backward]
        P --> M
        P -.->|no gradient| G
        P -.->|no gradient| I
        M --> Q[Gradient Accumulation x8]
        Q --> R[AdamW Optimizer Step]
        R --> M
    end
```

The implementation details of how LLaMA-Factory maps these YAML flags to `requires_grad_(False)` calls are covered in the Appendix.

## 3. Label Construction: What the Loss Sees

The training objective is next-token prediction, but not every token in the sequence contributes to the loss. The Data Processor (shown in Section 2's diagram) constructs a `labels` tensor alongside `input_ids`. Each position in `labels` is either -100 (meaning "ignore this position during loss computation") or a real token ID (meaning "the model should have predicted this token"). The loss function later uses this tensor to determine which positions generate gradient signal.

The split follows a simple rule: everything the model receives as context (the video, the system prompt, the evaluation instructions) is masked. Everything the model must learn to generate (the reasoning trace and scores) carries real labels. For the treadmill video example from Blog 2, the sequence looks like this:

```
Positions 0-3:        <|im_start|>system\n          → labels: [-100, -100, -100, -100]
Positions 4-28:       You are a helpful assistant...  → labels: [-100, ..., -100]
Positions 29-3548:    <|vision_start|>...<|vision_end|>  → labels: [-100 x 3520]
Positions 3549-4019:  evaluation instruction text    → labels: [-100 x ~471]
Position 4020:        <|im_start|>assistant\n        → labels: [-100]
Positions 4021-4640:  <think>...reasoning...</think> → labels: [real IDs x ~620]
Positions 4641-4643:  score tokens (3, 3, 2)         → labels: [real IDs x 3]
```

The boundary falls at the start of the assistant turn. Everything before it — system prompt, the 3,520 visual token placeholders, evaluation instructions, role tags — gets -100. Everything after it — the chain-of-thought reasoning and the three integer scores — gets real token IDs that the model must learn to predict.

Of the ~4,643 tokens in this sequence, only ~623 participate in the loss (620 reasoning tokens + 3 score tokens). The remaining ~4,020 tokens provide context but generate zero gradient signal. This 7:1 ratio of context to supervision means each training example carries substantial conditioning from the video and instruction, while the backward pass only needs to compute gradients through the ~623 supervised positions.

The output of this step is the `labels` tensor — one of the five tensors in the batch dictionary (Section 4).

## 4. The Model Input: Tensor Dictionary

The data collator (`MultiModalDataCollatorForSeq2Seq` in `data/collator.py`) assembles the final tensor dictionary that the model's forward pass receives. For the treadmill example, this dictionary contains:

| Tensor           | Shape             | Content                                                                                                                                          |
| ---------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `input_ids`      | [1, ~4643]        | Token IDs for the full sequence (system prompt + visual placeholders + instruction + assistant response)                                         |
| `labels`         | [1, ~4643]        | Same length as input_ids; -100 for prompt positions, real token IDs for assistant positions (Section 3)                                          |
| `attention_mask` | [1, ~4643]        | 1 for all real tokens, 0 for padding (no padding needed here since batch_size=1)                                                                 |
| `position_ids`   | [3, ~4643]        | Three-dimensional mRoPE coordinates (temporal, height, width) per token, computed via `model.get_rope_index()` as described in Blog 1, Section 8 |
| `pixel_values`   | [8, 3, 616, 1120] | Raw video frames (8 sampled frames, 3 color channels, resized to 616x1120) for the ViT to process                                                |

These five tensors are everything the model needs. The forward pass (Section 5) consumes `pixel_values`, `input_ids`, `position_ids`, and `attention_mask` to produce logits. The loss computation (Section 6) then uses `labels` to determine which positions contribute.

## 5. The Forward Pass

The forward pass consumes the tensor dictionary from Section 4 in a specific order. Each step transforms one or more input tensors into intermediate representations, culminating in a logits tensor that feeds the loss function.

**Step 1: Vision encoding.** `pixel_values` [8, 3, 616, 1120] → ViT → merger → visual embeddings.

The ViT receives the 8 raw frames, patches them into 14,080 embeddings, and processes them through 32 transformer layers to produce 14,080 vectors of dimension 1280. The merger pools these in 2x2 spatial blocks and projects to dimension 3584, yielding 3,520 visual token embeddings. Because `requires_grad = False` on all vision parameters (Section 2), PyTorch does not build a computation graph for this step — the visual embeddings are computed but treated as constants for gradient purposes.

**Step 2: Text embedding.** `input_ids` [1, ~4643] → embedding layer → hidden states [1, ~4643, 3584].

The model's embedding layer maps every token ID to a 3584-dimensional vector. At this point, the 3,520 positions that contain the `<|video_pad|>` placeholder ID have meaningless embeddings — they are about to be replaced.

**Step 3: Embedding injection.** The 3,520 visual embeddings from Step 1 overwrite the placeholder embeddings from Step 2 at the corresponding positions. After this replacement, the sequence is a heterogeneous mix of text-derived and vision-derived vectors, all in the same 3584-dimensional space.

**Step 4: LLM decoder.** hidden states [1, ~4643, 3584] + `position_ids` [3, ~4643] + `attention_mask` [1, ~4643] → 28 transformer layers → logits [1, ~4643, 152064].

The 28 transformer layers process the full sequence with causal self-attention (each position attends only to itself and earlier positions). The `position_ids` tensor feeds into the rotary embedding computation at every layer, encoding spatial and temporal relationships via mRoPE. The `attention_mask` prevents attending to padding positions (trivial here with batch_size=1). The final linear layer (`lm_head`) projects hidden states to vocabulary size, producing a 152,064-dimensional logit vector at each position.

**Step 5: Output.** The forward pass produces logits at all ~4,643 positions. At position `t`, the logit vector represents the model's unnormalized prediction for what token comes at position `t+1`, conditioned on all tokens at positions 0 through `t`. The `labels` tensor is not used during the forward pass — it is consumed only in the loss computation (Section 6) to determine which positions contribute.

## 6. The Loss Function

The SFT loss is standard next-token-prediction cross-entropy, applied selectively to the unmasked positions from Section 4. The `SupervisedDatasetProcessor._encode_data_example` method constructs the labels:

```python
source_label = [IGNORE_INDEX] * source_len   # prompt/input: masked
target_label = target_ids                      # assistant response: supervised
```

HuggingFace's `CrossEntropyLoss(ignore_index=-100)` skips positions where labels equal -100. This means the loss is computed exclusively over the assistant response -- approximately 623 tokens per example (620 reasoning tokens + 3 score tokens).

The loss computation proceeds in two steps applied to the logits from Section 3:

**Step 1: Softmax.** At each position `t`, the logit vector is converted into a probability distribution over the vocabulary:

$$P(y \mid y_{<t}, x) = \frac{\exp(z_y)}{\sum_{j=1}^{V} \exp(z_j)}$$

where:

- $y$ — any token in the vocabulary; during loss computation we evaluate this at the correct token $y_t$ from the labels tensor
- $y_{<t}$ — all tokens before position $t$ in the sequence (visual tokens, prompt tokens, and the ground-truth reasoning tokens at positions before $t$). During training, the model always sees the correct previous tokens from the labels, not its own predictions — this is known as teacher forcing
- $x$ — the input conditioning (video frames + evaluation prompt); these are part of $y_{<t}$ but called out separately to emphasize that the model sees the video content
- $z_y$ — the logit (raw score) at the vocabulary index of token $y$. This is one element of the 152,064-dimensional vector output by `lm_head` at position $t$ (from Section 3)
- $\exp(z_y)$ — exponentiation ensures all values are positive before normalization
- $\sum_{j=1}^{V} \exp(z_j)$ — the sum of exponentiated logits across the entire vocabulary, used as the normalizing denominator so all probabilities sum to 1
- $V = 152{,}064$ — Qwen2.5-VL's vocabulary size (the number of possible next tokens)

The output is a probability distribution over all 152,064 tokens. The model's prediction for position $t$ is whichever token receives the highest probability, but during training we only care about the probability assigned to the correct token $y_t$.

**Step 2: Cross-entropy loss.** The loss indexes into this distribution at the correct token $y_t$, takes the negative log, and averages over all unmasked positions:

$$\mathcal{L} = -\frac{1}{N_{\text{target}}} \sum_{t \in \text{target}} \log P(y_t \mid y_{<t}, x)$$

where:

- $\mathcal{L}$ — the scalar loss for this single training example (one video)
- $N_{\text{target}}$ — the number of unmasked positions (where `labels[t] != -100`), approximately 623 (620 reasoning tokens + 3 score tokens)
- $t \in \text{target}$ — iteration over only the unmasked positions (the assistant turn)
- $P(y_t \mid y_{<t}, x)$ — the probability from Step 1's softmax, evaluated at the correct token $y_t$. The visual tokens, system prompt, and evaluation instructions all influence this probability through attention (they are part of the conditioning context $x$), but their own label positions are -100 so they never contribute to the loss

Why **log**: the probability $P(y_t \mid y_{<t}, x)$ is between 0 and 1. The log converts this into a range from $-\infty$ to 0. Critically, log penalizes low probabilities much more harshly than a linear function would — if the model assigns P = 0.01 to the correct token, $\log(0.01) = -4.6$, but if it assigns P = 0.1, $\log(0.1) = -2.3$. A 10x improvement in probability only halves the penalty. This creates strong gradients when the model is confidently wrong, pushing it to fix its worst predictions first.

Why **negative**: $\log P$ is always negative or zero (since $P \leq 1$). The negative sign flips this to a positive value, giving us a loss we can minimize. When P = 1 (perfect prediction), $-\log(1) = 0$ (zero loss). When P is near 0 (terrible prediction), $-\log P$ is large (high loss). Minimizing the negative log-probability is mathematically equivalent to maximizing the probability the model assigns to every correct token in the reasoning trace and scores.

In practice, PyTorch's `CrossEntropyLoss(ignore_index=-100)` fuses Steps 1 and 2 internally — it takes raw logits and labels directly, without materializing the full softmax matrix (using the log-sum-exp trick for numerical stability).

An important implication: 620/623 = 99.5% of the per-token loss comes from reasoning tokens. The 3 score tokens contribute only 0.5% of the gradient signal per example. The model is overwhelmingly trained to reason about video quality, not merely to emit scores.


## 7. Training Arithmetic

The effective batch size and optimization step count determine how much the model trains. These numbers are derived from the training parameters in Section 1's YAML (`per_device_train_batch_size`, `gradient_accumulation_steps`, `num_train_epochs`) combined with 8 GPUs and 26,634 dataset examples from Blog 2:

```
Dataset size:              26,634 examples
GPUs:                      8 (A800 80GB)
Per-device batch size:     1
Gradient accumulation:     8 steps
Effective batch size:      8 GPUs * 1 * 8 = 64 examples/step
Steps per epoch:           ceil(26,634 / 64) = 417 steps
Total epochs:              2
Total optimization steps:  ~832
Warmup steps:              0.1 * 832 = ~83 steps
Save checkpoint every:     300 steps
```

Each optimization step processes gradients accumulated from 64 examples. With ~623 target tokens per example, each step incorporates loss information from approximately 39,872 tokens. Over the full training run:

```
Total target tokens:       26,634 * 623 * 2 epochs = ~33.2M tokens
Total forward passes:      26,634 * 2 = 53,268
Total optimization steps:  ~832
```

For context, 33M tokens is modest by language model standards -- GPT-3 trained on 300B tokens. But this is targeted fine-tuning of an already-capable model on a narrow task. The 832 optimization steps are sufficient because the model starts from a strong initialization (Qwen2.5-VL-7B-Instruct) and the task requires adapting existing capabilities rather than learning them from scratch.

## 8. DeepSpeed ZeRO Stage 3

DeepSpeed is a distributed training library (from Microsoft) that enables training models too large to fit on a single GPU. ZeRO (Zero Redundancy Optimizer) is its core memory optimization strategy. In standard data-parallel training, every GPU holds a full copy of the model parameters, gradients, and optimizer states. This is redundant and scales poorly. ZeRO eliminates this redundancy by partitioning (sharding) these states across GPUs so each GPU holds only a fraction. ZeRO defines three stages of increasing aggressiveness. Stage 1 shards only optimizer states. Stage 2 adds gradient sharding. Stage 3 shards all three (parameters, gradients, and optimizer states). VideoScore2 uses Stage 3.

### The memory problem

The training config sets `bf16: true`, meaning model weights and computations use bfloat16 (bf16). bf16 is a 16-bit floating point format (2 bytes per number) originally developed by Google Brain. It keeps the same exponent range as fp32 (32-bit, 4 bytes) but reduces decimal precision. For neural network training, the range matters more than precision, so bf16 halves memory with negligible impact on training quality.

The optimizer states use fp32 (4 bytes) rather than bf16 (2 bytes) because the weight update step accumulates many small gradient changes over time. bf16's limited precision would cause these small updates to round to zero, preventing the model from learning. This is a default behavior in DeepSpeed's bf16 optimizer, not an explicit configuration choice.

The memory required for training this model without sharding:

```
Parameters (full model, bf16):       8.3B × 2 bytes  = 16.6 GB
Gradients (trainable only, bf16):    7.6B × 2 bytes  = 15.2 GB
Optimizer states (trainable, fp32):  7.6B × 4 bytes × 2 (Adam m + v) = 60.8 GB
Activations (per sequence):          ~10-15 GB
Total:                               ~107 GB
```

A single A800 has 80 GB of memory. The training state alone (92.6 GB) already exceeds this before activations are counted.

### How ZeRO-3 solves it

Since 92.6 GB cannot fit on a single 80 GB A800, the training state must be distributed. With 8 GPUs available, ZeRO-3 shards parameters, gradients, and optimizer states so each GPU holds only 1/8 of each category (roughly 11.6 GB of model state per GPU, leaving ample room for activations). During the forward and backward pass, each layer's full parameters are temporarily gathered from other GPUs via all-gather, used, then discarded.

The DeepSpeed config:

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "bf16": { "enabled": "auto" },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": false,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

Key behaviors:

- **Parameter sharding**: Each GPU holds ~1/8 of the model's parameters at rest. During forward/backward, parameters are all-gathered on demand and discarded after use.
- **Gradient sharding**: Each GPU computes gradients for all parameters during backward, but only retains the shard it owns. Gradients are reduce-scattered across GPUs.
- **Optimizer state sharding**: AdamW's first and second moment estimates are sharded. Each GPU updates only its 1/8 of the optimizer state.
- **Frozen modules and ZeRO-3**: The 675M frozen ViT parameters are still sharded by ZeRO-3 (they exist in the model's parameter list), but they never generate gradients or optimizer states. This means ZeRO-3 gathers them during forward pass but skips them during backward. The memory saved from not storing optimizer states for frozen parameters is: 675M * 8 bytes (two fp32 moments) = ~5.4 GB total, or ~675 MB per GPU.
- **overlap_comm: false**: Communication does not overlap with computation. This simplifies debugging and avoids potential race conditions at the cost of ~10-15% throughput.

Per-GPU memory budget (approximate):

```
Sharded parameters (8.3B / 8 * 2 bytes):    ~2.1 GB
Optimizer states (7.6B / 8 * 8 bytes):       ~7.6 GB
Gathered parameters during forward:          varies (up to full layer)
Activations (1 example, 8192 tokens):        ~8-12 GB
Gradient buffers:                            ~2-4 GB
Overhead (fragmentation, CUDA context):      ~4-6 GB
Total:                                       ~24-32 GB / 80 GB available
```

### Why batch size is 1

ZeRO-3 solves the model state memory problem (parameters, gradients, optimizer states are sharded). But there is a second memory consumer that ZeRO does not shard: **activations**. Activations are the intermediate tensors saved during the forward pass that are needed to compute gradients in the backward pass. At each of the 28 transformer layers, the model must save hidden states, attention scores, and feed-forward intermediates for every token in the sequence.

Activation memory scales with sequence length, hidden dimension, and number of layers:

```
activations_per_example ≈ seq_len × hidden_dim × num_layers × bytes_per_value
                        ≈ 8,192 × 3,584 × 28 × ~several tensors
                        ≈ 10-15 GB per example
```

With batch_size=1, a single example already consumes 10-15 GB of activation memory. If batch_size were 2, activations would double to 20-30 GB, pushing total per-GPU memory toward 40+ GB. While this might technically fit on an 80 GB A800, it leaves little headroom for memory fragmentation, CUDA context overhead, and communication buffers that DeepSpeed needs for all-gather/reduce-scatter operations.

The binding constraint is the combination of long sequences (up to 8,192 tokens, of which ~4,643 are typical for VideoScore2 examples) and a large hidden dimension (3,584) across many layers (28). This is why the config uses `per_device_train_batch_size: 1` with `gradient_accumulation_steps: 8` to achieve an effective batch size of 64 without increasing per-GPU activation memory.

## 9. Learning Rate Schedule

The learning rate follows a cosine decay schedule with linear warmup:

```
Phase 1 (steps 0-83):    Linear warmup from 0 to 5e-5
Phase 2 (steps 83-832):  Cosine decay from 5e-5 to 0
```

The choice of lr=5e-5 is relatively aggressive for full fine-tuning of a 7B model. For comparison, typical full fine-tuning of LLaMA-7B uses 1e-5 to 2e-5. The higher rate is justified by three factors:

1. **Short training duration**: 832 steps provides limited time for the model to converge. A higher learning rate ensures the parameters move far enough from initialization.
2. **Frozen vision tower**: The LLM decoder's gradients are not propagated through the vision encoder, eliminating one source of gradient instability. The gradient norms are more predictable when the input representation is fixed.
3. **Task narrowness**: Video quality assessment is a constrained output space (5^3 = 125 possible score combinations). The model needs to adapt its response distribution but not fundamentally restructure its representations.

The 10% warmup (83 steps) prevents large initial updates when the learning rate is applied to a model whose gradients may be poorly calibrated for the new task distribution. After warmup, cosine decay provides a smooth reduction that allows fine-grained convergence in later steps without the abrupt transitions of step-based schedules.


## 10. Gradient Flow

During each forward pass, the computation graph is:

```
Video frames -> ViT (frozen) -> Merger (frozen) -> Visual tokens
                                                       |
Text prompt tokens ---------------------------------> LLM Decoder -> Loss
                                                       |
                                                   Backward pass
                                                       |
                                              Gradients to LLM only
```

Because `requires_grad=False` on all ViT and merger parameters, PyTorch's autograd does not construct gradient computation nodes for the vision pipeline. The visual tokens enter the LLM decoder as leaf tensors with no gradient history. This has two consequences:

1. **No gradient instability from vision encoder**: The 675M-parameter ViT is a deep network (32 transformer blocks). Backpropagating through it would introduce gradient scale issues and increase the risk of catastrophic forgetting of visual features. Freezing eliminates this.
2. **Reduced activation memory**: Activations in the vision encoder do not need to be retained for backward pass. Only the LLM decoder's 28 layers of activations are stored.

Within the LLM decoder, gradients flow through all 28 layers of the standard transformer architecture: self-attention (Q, K, V projections + output projection), feed-forward network (gate, up, down projections), and RMSNorm layers. With 3584 hidden dimension, 28 attention heads, and an intermediate size of 18944, each layer has approximately 270M parameters.

## 11. Overfitting Prevention

Full-parameter fine-tuning of a 7.6B model on only 26,634 examples creates a significant overfitting risk. The model has far more capacity than the dataset can constrain. Common strategies to prevent SFT overfitting include:

| Strategy | Used in VideoScore2? | Notes |
| -------- | -------------------- | ----- |
| Few epochs (1-3) | Yes (2 epochs) | Primary safeguard. The model already has strong priors from pretraining; too many passes overwrite them. |
| Low learning rate | Partially (5e-5) | Moderate for full fine-tuning of a 7B model. Typical range is 1e-5 to 5e-5. |
| Cosine decay to ~0 | Yes | Learning rate diminishes toward the end, making late-training updates tiny. |
| Early stopping with validation | No | `val_size` is commented out in the config. The model trains blind for 832 steps. |
| Weight decay | No | Defaults to 0.0. For a 2-epoch run, regularization from weight decay would have minimal effect. |
| LoRA/adapters (fewer trainable params) | No | Full fine-tuning is used, giving the model maximum freedom to adapt. |

VideoScore2 relies primarily on the combination of few epochs and cosine decay. The 2-epoch limit means the model sees each example only twice. Combined with the cosine schedule that reduces the learning rate toward zero in the final steps, the model makes progressively smaller updates as training proceeds. This limits how far the parameters can drift from the pretrained state.

The absence of early stopping means there is no mechanism to detect overfitting during training. Evaluation happens only post-hoc on VideoScore-Bench-v2 (the 500-video test set from Blog 2). If overfitting occurs, it would be caught after training completes, not during.

## 12. What This Training Does Not Include

Several notable absences in this configuration:

- **No gradient clipping specified in YAML**: DeepSpeed's `gradient_clipping: "auto"` defers to the HuggingFace Trainer's default of 1.0. This caps the global gradient norm, preventing explosion from outlier examples.
- **No evaluation during training**: The YAML contains commented-out eval settings (`val_size: 0.1`, `eval_strategy: steps`, `eval_steps: 500`). Uncommenting these would hold out 10% of training data and evaluate every 500 steps, reporting validation loss. Without this, the model trains blind for 832 steps with no signal about whether it is overfitting. Evaluation happens only post-hoc on VideoScore-Bench-v2 (the 500-video test set from Blog 2). This choice trades the ability to detect overfitting mid-training for simplicity and full use of the training data.
- **No data packing**: Each example occupies its own sequence. LLaMA-Factory supports packing via `packing: true` (or `neat_packing: true` for block diagonal attention), but VideoScore2 does not enable it. At ~4,643 tokens per typical example, two examples would total ~9,286 tokens, exceeding the 8,192-token window. Only very short videos would be packable, making the optimization impractical for this dataset.
- **No weight decay specified**: Defaults to 0.0 in HuggingFace Trainer. For a 2-epoch run, regularization from weight decay would have minimal effect.

## 13. From SFT to RL

The SFT training produces a checkpoint at `saves/vs2_qwen2_5vl_sft_27k_5e-5_2fps_960_720_8192`. This checkpoint is a complete Qwen2.5-VL-7B model with modified language model weights and unchanged vision weights. It serves as the initialization for the subsequent GRPO reinforcement learning stage, where the model generates its own reasoning (no longer supervised) and receives reward signal based solely on score accuracy.

The SFT stage's role is to establish the model's ability to produce structured video quality reasoning. The RL stage then optimizes the policy for score accuracy without constraining the reasoning format. This two-stage approach (SFT then RL) is standard in RLHF pipelines -- SFT provides the behavioral prior, RL sharpens it toward the objective.

## Appendix: Training Loop Call Chain

The entry point for the entire training loop is `run_sft()` in LLaMA-Factory's `train/sft/workflow.py`. This single function orchestrates initialization, data processing, and the training loop:

```python
def run_sft(model_args, data_args, training_args, finetuning_args, ...):
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft")
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
    )

    trainer = CustomSeq2SeqTrainer(model=model, args=training_args, ...)
    train_result = trainer.train()
    trainer.save_model()
```

The sequence diagram below expands what happens inside `run_sft()`, showing which library owns each step:

```mermaid
sequenceDiagram
    participant LF as LLaMA-Factory
    participant HF as HuggingFace Transformers
    participant DS as DeepSpeed
    participant Model as Qwen2.5-VL (HF Transformers)

    Note over LF,Model: Initialization
    LF->>HF: load model (Qwen2.5-VL-7B-Instruct)
    LF->>LF: freeze vision tower + merger
    LF->>DS: wrap model with ZeRO Stage 3
    LF->>HF: create Trainer

    Note over LF,Model: Per-Step Training Loop
    LF->>LF: Data Processor: tokenize + label masking
    LF->>LF: Data Collator: assemble tensor dictionary
    LF->>Model: get_rope_index (compute mRoPE position_ids)
    Model-->>LF: position_ids

    Note over HF,Model: Forward Pass
    HF->>Model: model.forward(pixel_values, input_ids, position_ids)
    Model->>Model: ViT encoding (frozen)
    Model->>Model: Merger (frozen)
    Model->>Model: Embedding injection
    Model->>Model: LLM decoder 28 layers (trainable)
    Model-->>HF: logits

    Note over HF,DS: Loss and Update
    HF->>HF: CrossEntropyLoss(logits, labels, ignore_index=-100)
    HF->>DS: loss.backward()
    DS->>DS: all-reduce gradients across 8 GPUs
    DS->>DS: AdamW optimizer step (update LLM params only)
```

Ownership summary:

| Step                            | Owner                                 | Notes                                                |
| ------------------------------- | ------------------------------------- | ---------------------------------------------------- |
| Data processing + label masking | LLaMA-Factory                         | `data/processor/supervised.py`                       |
| Data collation + mRoPE          | LLaMA-Factory + Qwen2.5-VL model      | `data/collator.py` calls `model.get_rope_index()`    |
| Freezing                        | LLaMA-Factory                         | `model/adapter.py` sets `requires_grad_(False)`      |
| Forward pass (ViT, merger, LLM) | HuggingFace Transformers (Qwen2.5-VL) | Model code in `transformers` package                 |
| Loss computation                | HuggingFace Transformers              | `Trainer` calls `CrossEntropyLoss`                   |
| Gradient sync + optimizer       | DeepSpeed                             | ZeRO-3 shards params/gradients/optimizer across GPUs |

## Appendix: Alternatives to DeepSpeed

DeepSpeed is not the only option for distributed training at this scale. The main alternatives are:

| Library              | Approach                                                                   | When to use                                                            |
| -------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| DeepSpeed ZeRO       | Shards params/gradients/optimizer across GPUs (data-parallel)              | HuggingFace ecosystem, LLaMA-Factory default                           |
| PyTorch FSDP         | PyTorch's native equivalent of ZeRO-3, built into PyTorch                  | No external dependency, increasingly popular                           |
| Megatron-LM (NVIDIA) | Tensor parallelism and pipeline parallelism (splits model by layer/tensor) | Very large models (100B+) where data-parallel sharding is insufficient |
| ColossalAI           | Multiple parallelism strategies                                            | Alternative to DeepSpeed with similar scope                            |

For VideoScore2's use case (7.6B trainable parameters on 8 GPUs), either DeepSpeed ZeRO-3 or PyTorch FSDP would work. DeepSpeed is the default in LLaMA-Factory's configuration templates and has native integration with HuggingFace Trainer (just point to a JSON config file), which is likely why it was chosen here.

## References

| Resource                    | Link                                                                                                                   |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| LLaMA-Factory               | [github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)                                           |
| DeepSpeed ZeRO              | [deepspeed.ai/tutorials/zero](https://www.deepspeed.ai/tutorials/zero/)                                                |
| VideoScore2 training config | [github.com/TIGER-AI-Lab/VideoScore2/training/SFT](https://github.com/TIGER-AI-Lab/VideoScore2/tree/main/training/SFT) |
| Qwen2.5-VL                  | [github.com/QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)                                                   |
