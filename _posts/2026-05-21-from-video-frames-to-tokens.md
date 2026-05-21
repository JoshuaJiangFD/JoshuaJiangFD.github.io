---
title: "From Video Frames to Tokens: How Qwen2.5-VL Sees Video"
date: 2026-05-20 14:00:00 +0000
categories: [Post-Training VLM]
tags: [VideoScore2, Qwen2.5-VL, VLM, Video Evaluation, Vision Transformer, mRoPE, LLaMA-Factory]
mermaid: true
---

This post traces the complete transformation from an MP4 file to a token sequence ready for the LLM decoder in Qwen2.5-VL-7B-Instruct, the base model behind VideoScore2. The series introduction (Blog 0) established the three-component architecture -- ViT, merger, LLM -- and identified the key design choices (frozen vision tower, full-parameter LLM training, mRoPE positional encoding). This post provides the implementation details: the exact formulas, config values, and worked arithmetic that determine how many tokens the LLM receives for a given video and how those tokens are positioned in 3D space. A running example is carried throughout: a 4-second video at 1280x720 native resolution encoded at 30fps.

## 1. The Scale Problem

The fundamental challenge of video understanding in a language model is one of scale. A language model operates on sequences of tokens, and its context window -- the maximum sequence length it can process at once -- is finite. For VideoScore2's configuration, that limit is 8,192 tokens. The raw video data exceeds this capacity by a factor of roughly 40,000.

The example video contains 4 seconds at 30 frames per second, producing 120 individual frames. Each frame is a grid of 1,280 by 720 pixels, and each pixel stores three color values (red, green, blue), yielding 1,280 x 720 x 3 = 2,764,800 numbers per frame. Across all 120 frames, the total is 120 x 2,764,800 = 331,776,000 numbers. The model must reduce this to a few thousand tokens while retaining enough visual information to judge quality, alignment, and physical consistency. Sections 2 through 7 describe the stages that accomplish this reduction, each operating on a different axis of the data.

## 2. Frame Sampling

The first reduction operates along the time axis, discarding the vast majority of frames. At 30 frames per second, consecutive frames are nearly identical. The pipeline exploits this redundancy by keeping only a sparse temporal sample.

LLaMA-Factory implements this in `_get_video_sample_indices` within `mm_plugin.py`. The computation proceeds in two steps: a desired frame count is derived from video duration and a configured frames-per-second rate, then that count is capped against hard limits:

```python
sample_frames = max(1, floor(duration * video_fps))
sample_frames = min(total_frames, video_maxlen, sample_frames)
indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
```

VideoScore2's training configuration sets `video_fps: 2.0` and `video_maxlen: 10`. The `video_fps` parameter is the model's desired sampling rate -- not the source video's native frame rate. The `np.linspace` call distributes sampled frames uniformly across the full frame range, ensuring temporal coverage regardless of where motion occurs.

| Video duration | Source frames | Desired = floor(dur x 2) | Actual = min(total, 10, desired) | Sampled indices |
|---|---|---|---|---|
| 2s | 60 | 4 | 4 | [0, 20, 40, 59] |
| 4s | 120 | 8 | 8 | [0, 17, 34, 51, 68, 85, 102, 119] |
| 6s | 180 | 12 | 10 | [0, 20, 40, 60, 80, 100, 120, 140, 160, 179] |
| 10s | 300 | 20 | 10 | [0, 33, 66, 100, 133, 166, 200, 233, 266, 299] |

For videos longer than 5 seconds, the `video_maxlen=10` cap becomes the binding constraint. This is the first mechanism that bounds downstream token count -- regardless of video length, the model never sees more than 10 frames.

One additional detail: the ViT's `temporal_patch_size=2` requires frames to be processed in pairs. If the sampled frame count is odd, LLaMA-Factory pads it to the next even number by duplicating the last frame. For the running example, 8 frames are already even, so no padding is needed.

## 3. Frame Resizing

Two resize operations constrain the spatial dimensions of each frame before patching. These operations determine the final spatial grid size and therefore have direct impact on the visual token count computed in Section 5.

The first resize is a pixel budget check. Frames whose total pixel count exceeds `video_max_pixels` are scaled down. VideoScore2 sets this to 691,200 (960 x 720). For the running example at 1280x720: the source pixel count is 1,280 x 720 = 921,600, which exceeds 691,200. A uniform scale factor of sqrt(691200 / 921600) = 0.866 is applied, producing intermediate dimensions of approximately 1108 x 623 pixels.

The second resize is a grid-alignment snap. The `smart_resize` function (from HuggingFace transformers' `image_processing_qwen2_vl.py`) rounds dimensions to the nearest multiple of `factor = patch_size x merge_size = 14 x 2 = 28`. The factor of 28 exists because the ViT divides the frame into 14-pixel patches, and the merger then groups patches in 2x2 blocks -- both divisions must produce whole numbers. For the intermediate dimensions: width becomes round(1108 / 28) x 28 = 40 x 28 = 1120, and height becomes round(623 / 28) x 28 = 22 x 28 = 616.

Each frame is resized to **1120 x 616 pixels**. These two steps together guarantee that every frame (a) fits within the pixel budget, preventing memory blowup in the ViT, and (b) divides evenly through both the patching and merging stages that follow.

## 4. The Patch Grid (video_grid_thw)

After resizing, the video's spatial and temporal structure is fully determined and can be described by three integers -- T, H, W -- that form the `video_grid_thw` tensor. This tensor is the central piece of metadata that flows through the rest of the pipeline: it determines the visual token count (Section 6), controls token injection (Section 7), and provides the coordinate system for positional encoding (Section 8).

The three dimensions are computed as follows:

- T = ceil(num_frames / temporal_patch_size) = ceil(8 / 2) = **4**. Each temporal position corresponds to a pair of consecutive frames.
- H = resized_height / patch_size = 616 / 14 = **44**. The number of non-overlapping 14-pixel rows.
- W = resized_width / patch_size = 1120 / 14 = **80**. The number of non-overlapping 14-pixel columns.

The resulting grid is **video_grid_thw = [4, 44, 80]**, representing 4 x 44 x 80 = 14,080 total patches that the ViT must process.

Additional examples illustrate how resolution and duration interact:

| Video | Frames sampled | After resize | video_grid_thw | Total patches |
|---|---|---|---|---|
| 2s, 720x480 | 4 | 728x476 | [2, 34, 52] | 3,536 |
| 4s, 1280x720 | 8 | 1120x616 | [4, 44, 80] | 14,080 |
| 6s, 1920x1080 | 10 | 1120x616 | [5, 44, 80] | 17,600 |

For the 1920x1080 case: 1920 x 1080 = 2,073,600 pixels far exceeds 691,200, so the scale factor is sqrt(691200/2073600) = 0.577, giving approximately 1108 x 623, which snaps to the same 1120x616 as the 1280x720 case. The pixel budget effectively normalizes high-resolution videos to the same spatial grid.

## 5. The Vision Encoder (ViT)

As introduced in Blog 0, the ViT encoder transforms raw pixel patches into dense feature vectors. This section details its architecture and how it handles the variable-resolution inputs produced by Sections 2-4.

The ViT in Qwen2.5-VL-7B-Instruct has approximately 675M parameters. Its architecture consists of 32 transformer layers with hidden_size=1280 and 16 attention heads. Each patch input is a 3D volume: 2 frames x 14 pixels x 14 pixels x 3 channels, flattened and linearly projected into a 1280-dimensional embedding by the patch embedding layer.

A distinctive feature is the attention pattern: the ViT uses **windowed attention** with `window_size=112` patches for most layers, switching to **full (global) attention** at layers 7, 15, 23, and 31. This design balances efficiency (most layers attend only within local spatial neighborhoods) with global coherence (periodic full-attention layers allow information to propagate across the entire frame). Using full attention at every layer would be quadratic in the patch count -- impractical for 14,080 patches.

Rather than using absolute position embeddings (which would fix the model to a single resolution), the ViT applies 2D Rotary Position Embeddings over the spatial height and width coordinates. This is what enables Qwen2.5-VL's "dynamic resolution" capability -- the same ViT weights can process frames of any size that satisfies the patch alignment constraints from Section 3.

For the running example, the ViT processes 14,080 patches and produces **14,080 vectors of dimension 1280**. In VideoScore2's configuration, the ViT is entirely frozen (`freeze_vision_tower: true`) -- its 675M parameters remain exactly as they were after Qwen2.5-VL's original multimodal pretraining.

## 6. The Merger

The 14,080 patch vectors from the ViT would be far too many tokens for the LLM to process within its context window. As introduced in Blog 0, the merger performs 2x2 spatial pooling to reduce this count by a factor of 4.

The merger operates with `merge_size=2`: at each temporal position, it takes every 2x2 block of spatially adjacent patch embeddings (each 1280-dimensional), concatenates them into a single 5120-dimensional vector (4 x 1280), and passes this through an MLP that projects down to 3584 dimensions (the LLM's hidden size). The output is one token embedding that the LLM can directly consume.

The visual token count after merging follows a simple formula:

```
num_tokens = T x H x W / merge_size^2 = T x (H/2) x (W/2)
```

For the [4, 44, 80] grid: 4 x 22 x 40 = **3,520 tokens**. These 3,520 vectors of dimension 3584 are what ultimately enter the LLM sequence.

The merger is also frozen in VideoScore2 (`freeze_multi_modal_projector: true`). It serves as a fixed dimensionality bridge: mapping from the ViT's 1280-dimensional representation space to the LLM's 3584-dimensional embedding space using weights optimized during Qwen2.5-VL's original multimodal pretraining and never updated during VideoScore2 training.

## 7. Token Injection

With visual embeddings produced by the ViT and merger, the remaining question is how they enter the LLM's input sequence alongside text tokens. This happens through a two-stage mechanism: placeholder expansion at data processing time, and embedding replacement during the forward pass.

The starting point is the `<video>` placeholder in training data. During data processing, `Qwen2VLPlugin.process_messages()` in LLaMA-Factory replaces this placeholder with a precisely-sized sequence of special tokens:

```python
merge_length = merge_size ** 2  # = 4
video_seqlen = video_grid_thw[i].prod() // merge_length
# For [4, 44, 80]: 4*44*80 // 4 = 3520

content = content.replace(
    VIDEO_PLACEHOLDER,
    f"<|vision_start|>{'<|video_pad|>' * video_seqlen}<|vision_end|>",
    1,
)
```

The result is: `<|vision_start|><|video_pad|>x3520<|vision_end|>`. The 3,520 `<|video_pad|>` tokens serve as positional placeholders in the tokenized sequence. During the forward pass, the model runs the ViT and merger to produce visual embeddings of shape [3520, 3584], then locates all `video_pad` positions in `input_ids` and replaces their embeddings with the corresponding visual vectors. From that point forward, the LLM's self-attention treats visual tokens identically to text tokens -- there is no cross-attention pathway.

The full tokenized sequence layout for a VideoScore2 training example follows the `qwen2_vl` chat template:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|video_pad|>x3520<|vision_end|>
You are an expert for evaluating AI videos...
<|im_end|>
<|im_start|>assistant
<think>...chain-of-thought reasoning...</think>
(1) visual quality: 4
(2) text-to-video alignment: 3
(3) physical/common-sense consistency: 5<|im_end|>
```

The model reads this left to right. By the time it generates the assistant response, it has attended to both the video tokens and the text instruction through self-attention.

## 8. Positional Encoding (mRoPE)

There is a subtle problem with inserting 3,520 video tokens into a text sequence. If the model simply numbers all tokens sequentially (position 1, 2, 3, ..., 3520, ...), it loses the video's spatial and temporal structure entirely. Consider two tokens that are sequentially adjacent in the flattened list: token 200 and token 201. They might be horizontally neighboring patches within the same frame row. But token 200 and token 1080 might represent the same spatial position viewed one time step later (since each time step contributes 22 x 40 = 880 tokens after merging). With sequential numbering, the model would treat token 200 and 201 as "close" and token 200 and 1080 as "far apart," even though the latter pair represents the same location across time and might be more semantically related for evaluating physical consistency.

Qwen2.5-VL solves this with Multimodal Rotary Position Embedding (mRoPE). Instead of assigning each token a single position number, mRoPE assigns **three position numbers**: a time coordinate, a height coordinate, and a width coordinate. These three coordinates encode each token's actual location in the three-dimensional grid described by `video_grid_thw` from Section 4.

Two tokens in the same row of the same frame share identical time and height coordinates but differ in width. The model's attention mechanism perceives them as horizontally adjacent. Two tokens at the same spatial location but in different time steps share height and width coordinates but differ in time. The model perceives them as temporally adjacent. This three-dimensional encoding preserves the spatial and temporal relationships that the pipeline worked to maintain through all preceding steps.

For text tokens, all three coordinates are set to the same incrementing value. If a text token is at sequence position 3721, its coordinates are (3721, 3721, 3721). This makes the three-dimensional system degenerate to standard one-dimensional positioning for text, since all three dimensions agree. The dual behavior -- 3D for visual tokens, 1D-equivalent for text -- emerges naturally from how coordinates are assigned rather than from separate mechanisms.

The implementation works as follows. Each attention head has dimension 128 (3584 hidden / 28 heads). These 128 dimensions are split into three groups: the first 32 dimensions (16 pairs) receive rotary embeddings based on the temporal coordinate, the next 48 dimensions (24 pairs) use the height coordinate, and the final 48 dimensions (24 pairs) use the width coordinate. Within each group, standard rotary position embedding math applies -- the only difference from regular RoPE is that three different position values feed into three different portions of the head dimension.

A critical detail for context budget reasoning is how the position counter advances after a video region. Rather than advancing by the full visual token count (3,520), `current_pos` advances by only `max(H/merge_size, W/merge_size)`. For the [4, 44, 80] grid, this is max(22, 40) = 40. This design keeps subsequent text tokens at moderate position values rather than pushing them to extremely high positions that would degrade attention patterns.

The `second_per_grid_ts` parameter provides real-time spacing between temporal patches: `temporal_patch_size / fps = 2 / 2.0 = 1.0` second per temporal grid position. This allows the model to encode actual temporal duration rather than just ordinal frame indices.

The mRoPE computation happens per-batch in the data collator, using the `video_grid_thw` values from Section 4 and `mm_token_type_ids` (which marks each token as text=0, image=1, or video=2) to determine coordinate assignment.

## 9. Context Budget

The practical constraint that ties all previous sections together is the context window. VideoScore2 sets `cutoff_len=8192`, meaning every component of the input -- system prompt, visual tokens, user query, and model response -- must fit within 8,192 tokens. Because visual tokens can dominate this budget (as the arithmetic in Sections 4 and 6 shows), understanding the budget breakdown explains why VideoScore2 uses the specific configuration values it does.

| Video | Frames | Grid | Visual tokens | + Instruction (~500) | + Response (~500) | Total | Headroom |
|---|---|---|---|---|---|---|---|
| 2s, 720x480 | 4 | [2, 34, 52] | 884 | 1,384 | 1,884 | 1,884 | 6,308 |
| 4s, 1280x720 | 8 | [4, 44, 80] | 3,520 | 4,020 | 4,520 | 4,520 | 3,672 |
| 6s, 1920x1080 | 10 | [5, 44, 80] | 4,400 | 4,900 | 5,400 | 5,400 | 2,792 |

The visual token count formula from Section 6 -- `T x H x W / 4` -- makes it clear how the two configuration parameters `video_max_pixels=691200` and `video_maxlen=10` jointly constrain the budget. The pixel budget caps the spatial grid (preventing H and W from growing with resolution), while the frame cap limits T. Together they ensure that even the most demanding videos (long duration, high resolution) produce at most approximately 4,400 visual tokens, leaving sufficient room for text.

If a sequence still exceeds 8,192 tokens after these constraints, LLaMA-Factory's `infer_seqlen` function truncates it to `cutoff_len`. The conservative parameter choices in VideoScore2's YAML are specifically designed to make this truncation extremely rare in practice.

## 10. Summary of the Compression Journey

The table below summarizes each stage of the pipeline as applied to the running example (4-second, 1280x720, 30fps video):

| Stage | Input | Output | Reduction |
|-------|-------|--------|-----------|
| Frame sampling (Section 2) | 120 frames | 8 frames | 15x |
| Resizing (Section 3) | 921,600 pixels/frame | 689,920 pixels/frame | 1.3x |
| Patching (Section 4) | 689,920 pixels/frame | 3,520 patches/frame | reorganization |
| Temporal grouping (Section 4) | 8 frames | 4 time steps | 2x |
| Vision encoding (Section 5) | 14,080 patches (588-dim) | 14,080 vectors (1,280-dim) | transformation |
| Merging (Section 6) | 14,080 vectors (1,280-dim) | 3,520 tokens (3,584-dim) | 4x |

Net result: 331,776,000 raw numbers become 3,520 tokens. After this pipeline, the LLM receives a flat sequence of 3584-dim vectors -- some from text embeddings, some from the vision pipeline -- and processes them identically through self-attention. The vision pipeline is entirely frozen; the LLM decoder alone learns to interpret these representations for the scoring task.

Blog 2 covers the training dataset, VideoFeedback2, that provides the target outputs the model learns to generate: chain-of-thought reasoning followed by three integer scores for visual quality, text-to-video alignment, and physical consistency.

## References

| Resource | Link |
|---|---|
| HuggingFace Transformers | [github.com/huggingface/transformers](https://github.com/huggingface/transformers) |
| Qwen2.5-VL-7B-Instruct model card | [huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| LLaMA-Factory | [github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| VideoScore2 | [github.com/TIGER-AI-Lab/VideoScore2](https://github.com/TIGER-AI-Lab/VideoScore2) |
