---
title: "Ranking Models Deep Dive: Input Representation"
date: 2026-06-01 19:00:00 +0000
categories: [Ranking Models]
tags: [Deep Learning, Recommendation Systems, Feature Engineering, Embeddings, Sequence Encoding]
mermaid: true
math: true
---

This is Part 2 of a series on deep learning for ranking models. [Part 1 (the overview)](/posts/ranking-models-overview) presents the full pipeline and taxonomy. This post deep-dives into the first stage: how raw features become dense vectors that the model can process.

---

## The Role of Input Representation

A single ad impression or video recommendation request carries dozens to hundreds of signals: the advertiser's identity, the user's device, the product category, the price, the time of day, and the user's entire click or viewing history from the past week. These signals arrive in fundamentally different forms — some are discrete labels with no numeric meaning, some are continuous measurements at wildly different scales, and some are variable-length sequences of past events. The input representation stage converts all of them into a uniform format the model can process: fixed-size dense vectors of floating-point numbers.

This stage is where the model's "vocabulary" is defined. Every categorical value the system has ever seen gets an entry in an embedding table. Every numerical feature gets a scaling rule. Every sequence gets a length limit and an aggregation strategy. The quality of these representations sets a ceiling on everything downstream — a poorly encoded feature cannot be rescued by a brilliant interaction module later.

### The goal: uniform dense vectors

Regardless of what type a feature starts as — a discrete label, a float, a pre-computed vector, or a 128-item history — after input representation, every feature becomes a dense float tensor of shape `[B, dim]` (one vector per sample in the batch, with a feature-specific number of dimensions). The subsequent stages of the pipeline never see raw categories or variable-length lists; they only see these uniform dense vectors, eventually concatenated or grouped for further processing.

```
Raw inputs (heterogeneous)          After input representation (uniform)
─────────────────────────           ────────────────────────────────────
advertiser_id = 7823       →        [0.12, -0.34, ..., 0.91]     (12 dims)
genre = "action"           →        [0.05, 0.22, ..., -0.18]     (32 dims)
price = 29.99              →        [0.73]                        (1 dim)
stream_volume_7d = 47000   →        [-0.31, 0.47, ..., 0.08]     (8 dims)
viewing_history = [128 items] →     [0.09, -0.15, ..., 0.44]     (~1200 dims)
```

The sections that follow explain how each feature type reaches this uniform format.

---

## 1. Categorical Features → Dense Vectors

**The problem:** Features like `genre=action` or `advertiser_id=38271` are discrete symbols with no inherent numeric meaning. A neural network operates entirely through arithmetic — addition, multiplication, comparison of numbers. A discrete label like "action" or "advertiser #38271" has no numeric form the network can compute with. Before the model can process these features at all, they must be converted into numbers — and the choice of *how* to convert them has major consequences for what the model can learn.

### The abandoned approach: one-hot encoding

For a feature with vocabulary size $V$, one-hot produces a binary vector of length $V$ with a single 1 at the feature's index. For example, if there are 20,000 unique advertisers in the system:

$$\text{one\_hot}(\text{advertiser\_id}=7823) = [0, 0, \ldots, \underbrace{1}_{\text{index 7823}}, \ldots, 0] \in \mathbb{R}^{20000}$$

With 100 such features, the input vector reaches millions of dimensions, almost entirely zeros. The first layer would need millions of parameters, most learning nothing because their corresponding input is permanently zero.

### The standard approach: learned embeddings

Replace the sparse lookup with a dense one. Store a trainable matrix $E \in \mathbb{R}^{V \times d}$ where $d \ll V$. The forward pass is a simple index operation:

$$\text{embed}(\text{advertiser\_id}=7823) = E[7823] \in \mathbb{R}^{12}$$

This is mathematically equivalent to one-hot times a weight matrix — selecting row 7823 — but without constructing the sparse vector or performing the multiply-by-zero.

```python
# Integer in → dense vector out
embedding = nn.Embedding(num_embeddings=20001, embedding_dim=12)
output = embedding(torch.tensor([7823]))  # → tensor of shape [12]
```

**Why embeddings are powerful beyond efficiency:**

1. **Semantic similarity emerges.** "Nike" and "Adidas" learn similar vectors because they appear in similar contexts. One-hot treats them as maximally distant.
2. **Generalization to rare IDs.** A rare advertiser sharing context with common ones benefits from the shared representation structure.
3. **Composability.** Dense vectors can be concatenated, added, attended over, or crossed — enabling the entire downstream architecture.

### Embedding dimension selection

The embedding dimension — how many numbers represent each ID — follows a logarithmic heuristic:

| Vocabulary Size | Embedding Dim | Rationale |
|----------------|---------------|-----------|
| $\leq 10$ | 8 | Very few values, minimal information content |
| $\leq 50$ | 16 | Small set, moderate capacity needed |
| $\leq 1000$ | 32 | Standard categorical features |
| $\leq 20000$ | 64 | Medium-cardinality features |
| $> 20000$ | 128-256 | High-cardinality (title IDs, product IDs) |

Production systems make different choices based on their total feature count. A system with 100+ categorical features (ad prediction) uses small dimensions (4-16) to keep the concatenated vector manageable, relying on feature interaction modules (FM, DCN) to build expressiveness from combinations. A system with fewer features (video recommendation, ~65 categorical) can afford larger dimensions (up to 256) for richer individual representations.

### Examples from production systems

| Role | Feature | Cardinality | Embedding Dim | System |
|------|---------|-------------|---------------|--------|
| Item | `ad_id` | 300,000 | 8 | Ad prediction |
| Item | `title_id` (content ID) | ~2,000,000 | 256 | Video recommendation |
| Item | `genre` | ~500 | 32 | Video recommendation |
| Advertiser | `advertiser_id` | 20,000 | 12 | Ad prediction |
| Advertiser | `campaign_id` | 100,000 | 12 | Ad prediction |
| User | `profile_type` | ~10 | 16 | Video recommendation |
| User | `territory_code` | ~5,000 | 64 | Video recommendation |
| Context | `device_type` | 11 | 4 | Ad prediction |
| Context | `day_of_week` | 7 | 8 | Video recommendation |
| Context | `slot_position` | 11 | 8 | Ad prediction |

The ad prediction system uses 85+ categorical features with small embeddings (4-16 dims); the video recommendation system uses 34 with larger embeddings (8-256 dims).

---

## 2. Categorical Features at Extreme Scale

Section 1 showed how categorical features become embeddings via direct table lookup. This breaks down when vocabulary reaches billions — a standard embedding table of shape `(1B, 64)` requires ~256 GB of memory, far beyond a single GPU. This is a fundamental infrastructure problem that affects both training (the table must be updated by gradients) and inference (the table must be queried for each request). The industry has developed several approaches.

### Hash Embeddings (compress onto a single machine)

Use $K$ hash functions to map each ID into a small set of tables, then combine the results through a learned MLP:

$$\text{hash\_embed}(id) = \text{MLP}\left(\text{concat}\left[T_1[h_1(id)],\ T_2[h_2(id)],\ T_3[h_3(id)]\right]\right)$$

Here $h_1, h_2, h_3$ are three different hash functions (each with different random seeds), and $T_1, T_2, T_3$ are three separate small embedding tables (each 10,000 rows × 16 dims). The process for a single ID:

```
ID = 500,000,000

Step 1: Hash the ID three ways into table indices (0 to 9,999):
  h_1(ID) = (a_1 × ID + b_1) mod prime mod 10000 = 7,231
  h_2(ID) = (a_2 × ID + b_2) mod prime mod 10000 = 3,894
  h_3(ID) = (a_3 × ID + b_3) mod prime mod 10000 = 9,102

Step 2: Look up each index in its corresponding table:
  T_1[7231] = [0.02, -0.14, ..., 0.08]   (16 dims)
  T_2[3894] = [0.11,  0.03, ..., -0.05]  (16 dims)
  T_3[9102] = [-0.07, 0.21, ..., 0.01]   (16 dims)

Step 3: Concatenate the three vectors:
  → [0.02, -0.14, ..., 0.08, 0.11, 0.03, ..., -0.05, -0.07, 0.21, ..., 0.01]  (48 dims)

Step 4: MLP projects to final output:
  MLP(48 → 64) → final embedding (64 dims)
```

The tables $T_1, T_2, T_3$ are standard `nn.Embedding` layers — initialized with small random values and **trained from scratch** alongside the rest of the model. They're not pretrained or provided externally. The only difference from a normal embedding is how you index into them: instead of using the original ID directly (which would require a 1B-row table), you use the hashed index (which only needs a 10K-row table). During training, gradients flow back through the lookup and update whichever row was accessed. The MLP is also trained from scratch. The entire hash embedding module learns end-to-end with the rest of the model.

This compresses 1B IDs into ~480K parameters (3 tables × 10K entries × 16 dims) versus 256 GB for a direct lookup. The ad prediction system uses this approach for two billion-scale features:

| Feature | Cardinality | Hash functions | Table size | Embedding dim | MLP → Target dim |
|---------|-------------|---------------|------------|---------------|-----------------|
| `customer_id` | 1,000,000,000 | 3 | 10,000 | 16 | 1-layer MLP → 64 |
| `page_asin` (product being viewed) | 1,000,000,000 | 3 | 10,000 | 16 | 1-layer MLP → 64 |

The tradeoff is **hash collisions**: different IDs may map to the same table entries, forcing them to partially share representations. The MLP learns to disambiguate to some extent, but inherently loses per-ID precision compared to a dedicated row.

### Quotient-Remainder Embedding

A variant from Meta (2019): decompose the ID into quotient and remainder, look up each in a separate small table, and combine via element-wise multiply. Memory drops from $V$ to roughly $2 \times \sqrt{V}$.

The ad prediction system experiments with QR embedding applied broadly across features:

| Feature | Cardinality | QR table size ($\approx\sqrt{V}$) | Embedding Dim |
|---------|-------------|----------------------------------|---------------|
| `ad_id` | 3,000,001 | ~1,732 per table | 8 |
| `campaign_id` | 1,000,001 | ~1,000 per table | 12 |
| `site_name` | 1,000,001 | ~1,000 per table | 8 |
| `brand_name` | 1,500,001 | ~1,225 per table | 16 |
| `page_asin_parent` | 10,000,001 | ~3,162 per table | 16 |

Unlike hash embeddings (reserved for billion-scale features), this experiment applies QR compression to *all* categorical features as a uniform memory reduction strategy.

### Distributed Embedding Tables (shard across many machines)

Keep the full 1B-row table but partition it across dozens or hundreds of machines. Each machine holds a slice (e.g., rows 0-10M on machine 1, rows 10M-20M on machine 2). When a training batch needs the embedding for customer #500M, it fetches that row over the network from whichever shard holds it.

This is the approach used by Meta (in their DLRM system), Google, and TikTok for their largest recommendation models. It avoids hash collisions entirely — every ID gets its own dedicated embedding row — but requires:
- A **parameter server** infrastructure to coordinate reads/writes across shards
- High-bandwidth **network communication** during both training and inference
- Significantly more machines and operational complexity

Note: neither production system studied here uses distributed sharding — both fit on single training machines. This approach is documented from public papers (Meta's DLRM, Google's recommendations) as an industry alternative for organizations with the infrastructure to support it.

### Don't Embed At All

For some billion-scale features, the system may decide the per-ID signal isn't worth the cost of a full embedding. This is a deliberate design choice across the industry — it recognizes that for some features, a full embedding table doesn't justify its cost in memory, parameters, and training signal dilution.

This pattern appears in several forms:

- **Exclude from embedding entirely.** The ad prediction system marks certain features as `CategoricalNoCollect` — they never get embedded into the main trunk and are only used as inputs to lightweight processors or auxiliary modules:

  | Feature | Cardinality | How it's consumed instead |
  |---------|-------------|--------------------------|
  | `customer_id` | 1,000,000,000 | Fed to hash embedding module for shortcode generation |
  | `page_asin` | 1,000,000,000 | Fed to hash embedding module for shortcode generation |
  | `ad_asin` | 250,001 | Used by `ElementInList` membership checks |
  | `clicked_asins` (sequence) | 250,001 | Used by `ElementInList` ("is ad in click history?") |
  | `purchased_asins` (sequence) | 250,001 | Used by `ElementInList` ("is ad in purchase history?") |
- **Wide & Deep (Google, 2016).** Splits features into a "wide" path where sparse IDs are used for memorization via simple logistic regression (no learned dense embedding) and a "deep" path where features are embedded.
- **Frequency-based cutoffs.** Only embed IDs that appear more than $N$ times in training data. Everything below the threshold maps to a shared "rare" embedding or is excluded entirely.
- **Feature hashing to small buckets without learned embeddings.** Hash 1B IDs into 10K buckets and feed the bucket index directly to a linear layer as a sparse feature — a crude signal without the expressiveness of a learned embedding, but nearly free in memory and computation.

### When to use which

| Approach | Best when | Tradeoff |
|----------|-----------|----------|
| Hash embedding | Model must fit on a single machine; some collision noise is acceptable | Loses per-ID precision |
| Distributed sharding | Infrastructure budget available; per-ID precision matters (e.g., user personalization) | Network latency, operational complexity |
| Don't embed | Feature has too many values with too few observations each; not worth the cost | Loses direct representation entirely |
| Quotient-Remainder | Middle ground — moderate compression with less collision than hashing | Less expressive than full table, more than hash |

---

## 3. Numerical Features → Scaled/Expanded Vectors

**The problem:** Numbers like `price=29.99` or `watch_time=3600s` already are numbers — unlike categorical features, they don't need an embedding to become numeric. But they have a different problem: wildly different scales and distributions. If one feature ranges from 0 to 10 and another from 0 to 1,000,000, the model's optimization will be dominated by the larger-scale feature. More subtly, many numerical features in recommendation systems (view counts, stream volumes, revenue) follow power-law distributions where most values are small but rare outliers are enormous — raw values would give extreme outliers disproportionate influence on training.

### Approach 1: Direct normalization (transform + scale)

The simplest approach: apply a mathematical transformation to bring the feature into a well-behaved range, then pass the single float directly to the model.

For count-based and volume-based features, the most common approach combines two steps: a **log transform** to compress the power-law distribution, followed by **z-score normalization** to center at zero with unit variance:

```
Raw stream_volume_7d values:      [0, 3, 12, 47, 1580, 892341]
Step 1 — log(x + 1):             [0, 1.4, 2.6, 3.9, 7.4, 13.7]
Step 2 — z-score normalize:      [-0.9, -0.6, -0.4, -0.1, 0.7, 2.2]
```

The `+1` inside the log handles zero values (log(0) is undefined). Step 1 compresses the long tail so that 892,341 and 1,580 are no longer 500x apart. Step 2 brings the result into a standard range so it's comparable to other features.

Log transform is the right choice for power-law features, but not every numerical feature follows a power law. Production systems support a toolkit of transformations, applied in up to two steps:

**Step 1 (optional): Compress the distribution**

| Transform | Formula | When to apply |
|-----------|---------|---------------|
| Log | $\log(x + 1)$ | Power-law features (counts, volumes) — compresses long tail |
| Log10 | $\log_{10}(x + 1)$ | Same purpose, base-10 for interpretability |
| None | $x$ | Features without extreme skew |

**Step 2: Normalize the scale**

| Normalization | Formula | When to apply |
|---------------|---------|---------------|
| Z-score (standard) | $(x - \mu) / \sigma$ | Default — centers at 0, unit variance |
| Min-max | $(x - x_{min}) / (x_{max} - x_{min})$ | Bounded features where output in [0, 1] is desired |
| Robust | $(x - \text{median}) / \text{IQR}$ | Features with extreme outliers where mean/std are unreliable (viral content) |
| Scale factor | $x \times c$ | Manual rescaling by a domain-specific constant |

The most common combination in practice is **log then z-score** — that's what a feature name like `title_stream_volume_log_7d_normalized` encodes. But the two steps are independent: the ad prediction system applies log transform *inside the model* at forward time without a subsequent z-score normalization, while the video recommendation system applies both steps in *preprocessing* before data reaches the model.

The output is a single float per feature per sample: shape `[B, 1]`.

**Examples:**

| Feature | Transform | Normalization | Output | System |
|---------|-----------|---------------|--------|--------|
| `stream_volume_7d` | Log | Z-score | [B, 1] | Video recommendation |
| `cid_volume_365d` | Log | Z-score | [B, 1] | Video recommendation |
| `completion_rate_365d` | None | Z-score | [B, 1] | Video recommendation |
| `runtime_for_completion` | Log | Z-score | [B, 1] | Video recommendation |
| `days_live_on_platform` | Log | Z-score | [B, 1] | Video recommendation |

The video recommendation system uses 23 title numerical features and 25 customer numerical features, nearly all with log + z-score. These are individually passed as single floats — no MLP, no expansion. They are later grouped and projected together (Approach 3).

Notably, the ad prediction system has *no* Approach 1 features — every numerical feature goes through a per-feature MLP (Approach 4). This reflects a design philosophy: since categorical embeddings are 4-16 dimensions wide, numerical features are expanded to similar widths so they have comparable representational capacity in the concatenated vector.

### Approach 2: Bucketing → Embedding

Sometimes the relationship between a number and the prediction isn't smooth. A product priced at \$25 vs \$30 might behave similarly, but the jump from \$30 to \$200 changes user behavior entirely. Similarly, an item that's been live for 3 days vs 7 days is "new" either way, but the jump from 7 days to 365 days crosses a meaningful boundary from "new release" to "catalog title."

Bucketing discretizes the continuous value into a small number of bins, then embeds the bin index exactly like a categorical feature:

```
Raw price:        $29.99
Bucket boundaries: [0, 10, 25, 50, 100, 200, 500, 1000+]
Bucket index:     3  (the $25-50 range)
Embedding:        table[3] → dense vector [B, 8]
```

The ad prediction system uses `BucketedNumerical` for features where these nonlinear thresholds matter. The embedding for bucket 3 (\$25-50) can learn a completely different representation from bucket 5 (\$100-200), without assuming any smooth relationship between them.

The tradeoff: bucketing loses precision within each bin (\$26 and \$49 become identical) and requires choosing boundaries in advance. But it gives the model freedom to learn arbitrary nonlinear effects at each price tier.

**Examples:**

| Feature | Raw value | Bucket boundaries | Embedding Dim | System |
|---------|-----------|-------------------|---------------|--------|
| `hour_of_day` | Unix epoch time | [0,1,2,...,23] (modulo 24) | 4 | Ad prediction |
| `day_of_week` | Unix epoch time | [0,24,48,...,144] (modulo 168) | 10 | Ad prediction |
| `cart_since_activity` | Hours since last cart add | [1,2,4,8,12,24,48,72,168,336,720,1440,2160,4320,8760] | 8 | Ad prediction |
| `purchase_since_activity` | Hours since last purchase | Same as above | 8 | Ad prediction |
| `view_since_activity` | Hours since last page view | Same as above | 8 | Ad prediction |

Note how `hour_of_day` and `day_of_week` are derived from the *same raw input* (epoch time) but bucketed differently. The logarithmic bucket spacing for recency features (1, 2, 4, 8, 12, 24, 48, ... hours) reflects that the difference between "1 hour ago" and "2 hours ago" matters more than "30 days ago" vs "31 days ago."

The video recommendation system uses Approach 1 + 3 for all its numerical features instead. Bucketing is primarily an ad prediction pattern, where discrete decision thresholds (time since last activity, hour of day) are more important than smooth popularity curves.

### Approach 3: Grouped linear projection

When you have many related numerical features — say, 23 different title popularity metrics across multiple time windows — passing each as an independent float creates a wide, sparse input where each individual number carries limited signal. A linear projection compresses the group into a learned dense representation:

```
Input: [stream_vol_1d, stream_vol_7d, stream_vol_28d, ..., runtime, days_live]
       → 23 floats, shape [B, 23]

Linear(23 → 32):
       → 32 floats, shape [B, 32]
```

The video recommendation system projects its 23 title numerical features into a 32-dim vector, and its 25 customer numerical features into another 32-dim vector. The projection layer learns which combinations of time-windowed metrics are informative — for example, it might learn that "high 7-day volume but low 365-day volume" (a recently trending title) is a useful signal worth encoding as a distinct direction in the 32-dim output.

This is more expressive than passing raw floats independently, because the linear layer can compute weighted combinations across the group. But it's simpler than bucketing each feature individually — it assumes the cross-feature relationships are roughly linear, which is reasonable for groups of metrics measuring the same underlying phenomenon at different time scales.

**Examples:**

| Feature group | Input features | Input dim | Output dim | What the group captures | System |
|---------------|---------------|-----------|------------|------------------------|--------|
| Title numerical | stream volumes, cid volumes, popularity metrics, runtime, days_live | 23 | 32 | Compressed title popularity/freshness signal | Video recommendation |
| Customer numerical | session counts, impression counts, playtime, playback counts (across 1d/7d/28d) | 25 | 32 | Compressed user engagement level | Video recommendation |
| Cross numerical | per-user-title playback counts, completion rates, impressions (across time windows) | 26 | 32 | Compressed user-title affinity signal | Video recommendation |

### Approach 4: Per-feature MLP (nonlinear expansion)

The opposite problem from grouped projection: instead of compressing many related numbers into fewer dimensions, you want to *expand* a single important number into a richer representation that captures its nonlinear effects.

The ad prediction system gives each important numerical feature its own small residual MLP:

```python
ad_asin_glance_views = Numerical(
    in_dim=1,              # single raw number (product page view count)
    layers=[8, 8, 8],     # 3-layer residual MLP, width 8
    log_transform=True,    # log(x+1) applied first
    residual_layer=True,   # skip connections between layers
)
```

This takes a single float (e.g., "47,000 page views"), applies log transform, then passes it through a 3-layer residual MLP, outputting an 8-dimensional learned representation.

**Why expand 1 dimension into 8?** The MLP isn't creating information from nothing. It's learning a **nonlinear basis expansion**: converting one number into multiple dimensions that each represent a different "region" or "aspect" of that number's meaning.

```
Input x = log(47000 + 1) = 10.76  (single float)

Projection: Linear(1 → 8)
  = W × 10.76 + b      (W is 8 weights, b is 8 biases)
  → 8 different linear functions of the input
  → e.g., [3.2, -1.4, 0.8, 2.1, -0.3, 1.7, -2.5, 0.4]

ResidualLayer[0]: h = h + ReLU(BatchNorm(Linear(h)))
ResidualLayer[1]: h = h + ReLU(BatchNorm(Linear(h)))
ResidualLayer[2]: h = h + ReLU(BatchNorm(Linear(h)))
  → Each layer adds nonlinear transformations on top
  → After 3 layers, each of the 8 output dims is a different
    nonlinear function of the original number
```

With one float, the downstream layers can only multiply it by one weight per neuron — a single linear relationship. With 8 dimensions, the feature can simultaneously express that "the difference between 100 and 1,000 views matters a lot" (one dimension responds steeply in that range) while "the difference between 100,000 and 101,000 barely matters" (all dimensions are flat there). It's similar to bucketing (Approach 2), but instead of hand-designed bin boundaries, the MLP learns how to partition the number's meaning — and it does so smoothly, without the hard cutoffs that bucketing introduces.

The tradeoff is more parameters per feature (a 3-layer `[8,8,8]` MLP has ~200 parameters), which is why the ad system reserves this treatment for important product-level metrics rather than applying it to every numerical feature.

**Examples:**

| Feature | in_dim | layers | log | Output dim | What it represents |
|---------|--------|--------|-----|-----------|-------------------|
| `glance_views` (product page views) | 1 | [8,8,8] | Yes | 8 | Product traffic level |
| `ordered_units` | 1 | [8,8,8] | Yes | 8 | Purchase velocity |
| `best_price` | 1 | [8,8,8] | Yes | 8 | Price point |
| `avg_review` | 1 | [8,8,8] | No | 8 | Review quality (already bounded ~1-5) |
| `customer_review_count` | 1 | [4,4] | Yes | 4 | Review volume (less important, smaller MLP) |
| `num_offers` | 1 | [4] | No | 4 | Seller competition (single layer suffices) |
| `customer_total_clicks` | 1 | [8,8] | No | 8 | User click history volume |
| `cart_past_day` | 1 | [10,10] | No | 10 | Recent cart activity intensity |

Notice the design choices: features with log transforms are count-based with long tails (page views, ordered units, review counts). Features without log are already bounded or pre-normalized — e.g., `avg_review` is a 1-5 star rating that doesn't need compression. MLP depth scales with feature importance — 3 layers for key product metrics, 1-2 layers for secondary signals.

### Comparing the approaches

| | Approach 1 | Approach 2 | Approach 3 | Approach 4 |
|---|---|---|---|---|
| **Operation** | Normalize | Discretize → embed | Group → linear project | Expand via MLP |
| **Input** | 1 float | 1 float | N related floats | 1 float |
| **Output** | 1 float | embedding dim (4-10) | projection dim (32) | MLP width (4-10) |
| **Direction** | Passthrough | Expansion | Compression | Expansion |
| **Learns** | Nothing (fixed transform) | Per-bin representation | Cross-feature combinations | Per-feature nonlinear response |
| **System** | Video recommendation | Ad prediction | Video recommendation | Ad prediction |

---

## 4. Pretrained Representations

**The problem:** Training representations from scratch requires large amounts of task-specific data for each feature value. For high-cardinality features like product IDs or user IDs, the ranking model may only see each value a handful of times — not enough to learn a good embedding. But a separate, larger model (a product graph embedding, a user representation model, or a content understanding system) may have already trained high-quality representations for these same entities using a different objective and far more data. Throwing away that knowledge and starting from scratch wastes signal.

The solution is to reuse the upstream model's output as a starting point, while attaching a trainable MLP on top that adapts it for the ranking task. This pattern appears in two forms — one for categorical features (where the upstream representation is stored as a frozen embedding table indexed by ID) and one for numerical features (where the upstream representation arrives directly as a pre-computed float vector). Both follow the same principle: freeze the upstream signal, learn an adaptation layer.

### Categorical pretrained: frozen table + projection MLP

For categorical features like product IDs, a frozen embedding table stores the upstream model's representations. The ranking model looks up the ID, gets a fixed vector, and projects it through a trainable MLP:

$$\text{output} = \text{MLP}(\text{frozen\_embedding}[id])$$

The frozen table stays fixed during training (no gradient flows into it) while the projection MLP adjusts freely. At serving time, since the frozen embedding never changes, you can precompute `MLP(embedding[id])` for every ID in the vocabulary and store the result as a new, smaller embedding table — collapsing the two-step operation into a single direct lookup.

### Numerical pretrained: pre-computed vector + same-size MLP

For numerical features where an upstream model has already produced a dense vector (e.g., a 64-dim product view embedding or a 21-dim category affinity score vector), the vector arrives directly as a float tensor. A same-size residual MLP transforms it without changing dimensionality:

```python
view_embedding = Numerical(
    in_dim=64,              # pre-computed 64-dim vector from upstream model
    layers=[64, 64, 64],   # 3-layer residual MLP, same width
    residual_layer=True,    # skip connections preserve original signal
)
```

The residual connections are important here: they ensure the original upstream representation is preserved (through the skip path) while the MLP layers learn task-specific adjustments on top. Without residual connections, the MLP might destructively overwrite useful information from the upstream model.

### Why residual vs. non-residual adaptation

The choice depends on whether the output dimension matches the input:

- **Same-size transformation (64→64):** The upstream model already produced a good representation. The goal is to *refine* it for the ranking task, not replace it. A residual path (`x + F(x)`) guarantees the original upstream signal is preserved — the MLP layers can only add adjustments on top. If the adaptation layers learn nothing useful, the output defaults to the original embedding unchanged. This is conservative by design: don't destroy good upstream signal.

- **Dimension reduction (128→16):** The goal is to *compress* — keeping only the aspects most useful for ranking while discarding the rest. There's no meaningful way to preserve the original when the output is 8x smaller. Residual connections are mechanically impossible here (you can't add a 128-dim vector to a 16-dim vector), but even conceptually, lossy compression is the intent.

### Examples from the ad prediction system

| Feature | Type | Frozen input dim | Trainable adaptation | Output dim | Upstream source |
|---------|------|-----------------|---------------------|-----------|----------------|
| Product ID | Categorical pretrained | 128 (frozen table lookup) | Down-projection MLP 128→16 | 16 | Product graph model |
| `view_embedding` | Numerical pretrained | 64 (pre-computed vector) | Residual MLP [64,64,64] | 64 | Product view co-occurrence model |
| `gl_affinity_scores` | Numerical pretrained | 21 (pre-computed vector) | Residual MLP [21,21] | 21 | Category affinity model |

The video recommendation system does not use pretrained representations in its current production model — all embeddings are trained from scratch end-to-end. It addresses the sparse-data problem through incremental training instead (loading the previous model checkpoint and continuing on fresh data). See the Design Tradeoffs post for a detailed comparison of these two approaches.

---

## 5. Multi-Source Computed Features

Consider the question: "Is the product in this ad something the user has clicked before?" Answering this requires looking at two separate raw inputs simultaneously — the ad's product ID and the user's click history sequence. Neither input alone carries the signal; it only emerges from their *combination*. Similarly, "How many times did this user click on anything in the last 60 minutes?" requires comparing a sequence of click timestamps against the current time — a temporal computation that spans two input fields.

These cross-input signals are too valuable to ignore, but too complex to express as a single raw feature in the input data. The ad prediction system handles them with **multi-source processors**: small learned modules that sit inside the model, consume multiple raw tensors, apply domain-specific logic (set membership, temporal windowing, time-gated masking), and output a dense vector that joins the rest of the features in the main pipeline.

### The two-phase pattern (domain logic → trainable encoding)

Every multi-source processor follows the same structure:

1. **Phase 1 — Domain logic:** Take multiple raw inputs and compute a simple intermediate value using hard-coded rules (membership check → integer, temporal counting → float, time-gated selection → integer)
2. **Phase 2 — Trainable encoding:** Pass that intermediate value through a trainable `nn.Embedding` or MLP to produce a learned dense vector of the configured output dimension

The domain logic is fixed (not learned); the encoding layer is trained end-to-end with the rest of the model.

### Membership features (ElementInList)

"Is this ad's product in the user's recently-clicked products list?" The domain logic compares a single categorical ID against a sequence of IDs and produces a three-way integer: 0 (list is missing/empty), 1 (not in list), or 2 (in list). This integer then indexes into a trainable `nn.Embedding(3, 12)` — a tiny 3-row table that maps each membership status to a learned 12-dim vector.

```
Inputs: ad_asin = 7823, clicked_asins = [42, 7823, 103, ...]
Domain logic: 7823 is in the list → integer 2
  (0 = history missing, 1 = not in list, 2 = in list)

Embedding table nn.Embedding(3, 12):
  row 0 (missing):     [0.01, -0.03, ..., 0.02]   ← used when history data unavailable
  row 1 (not in list): [0.15, -0.22, ..., 0.08]   ← used when product wasn't clicked before
  row 2 (in list):     [-0.31, 0.47, ..., -0.12]  ← used when product WAS clicked before

Lookup: index 2 → row 2 → [-0.31, 0.47, ..., -0.12]  (12 dims)
Output shape: [B, 12]
```

### Temporal activity features (ShopperActivityEstimator)

"How active was this user in the last 60 minutes?" The domain logic takes a sequence of click timestamps and the current bid timestamp, computes the time difference for each, and counts how many fall within the configured window (e.g., 60 minutes). This produces a single float (e.g., 7.0). That float is passed through a `NumericalEmbeddingModule(in_dim=1, layers=[5, 5])` — the same per-feature residual MLP described in Section 3 Approach 4 — producing a learned 5-dim vector.

```
Inputs: click_timestamps = [t1, t2, ..., t25], bid_time = now
Domain logic: count(now - t_i < 60 min) → 7.0
MLP: NumericalEmbeddingModule(1 → [5,5]) → learned 5-dim vector
Output shape: [B, 5]
```

### Time-gated attribute features (DealsFeature)

"What is the deal type for the currently active deal?" The domain logic takes deal start/end dates, the current time, and a sequence of deal attributes. It applies temporal masking (only deals where `start ≤ now ≤ end` are eligible), selects the first active deal, and extracts its attribute as an integer (e.g., deal_type=3). If no deal is active, the value is 0.

```
Inputs: deal_starts = [...], deal_ends = [...], now = current_time, deal_types = [2, 3, 1, ...]
Domain logic: find first deal where start ≤ now ≤ end → deal_type = 3

Embedding table nn.Embedding(7, 4):  (7 possible deal types, 4-dim output)
  row 0 (no active deal): [0.02, -0.01, 0.05, 0.03]
  row 1 (deal type 1):    [0.14, -0.08, 0.22, -0.11]
  row 2 (deal type 2):    [-0.06, 0.19, 0.01, 0.15]
  row 3 (deal type 3):    [0.31, -0.27, 0.09, 0.18]  ← this row is selected
  ...

Lookup: pass integer 3 into the table → returns row 3 → [0.31, -0.27, 0.09, 0.18]
Output shape: [B, 4]
```

### Summary table

| Processor | Inputs | Output Dim | Instances | What each instance encodes |
|-----------|--------|-----------|-----------|---------------------------|
| `ElementInList` | 1 item ID + 1 ID sequence | 12 | 4 | One per combination: product×clicked, product×purchased, brand×clicked, brand×purchased |
| `ShopperActivityEstimator` | Timestamp sequence + current time | 5 | 3 | One per time window: 15 min, 60 min, 24 hours |
| `DealsFeature` | Deal start/end dates + current time + deal attributes | 4 | 5 | One per deal attribute: deal type, Prime access, discount, state, is featured |

These multi-source processors share a common pattern: they encode **domain-specific logic** (temporal windows, set membership, time-gating) that would be difficult for the main network to learn from raw features alone, especially from sparse training signal. They produce fixed-size dense vectors that are then concatenated alongside standard embeddings and fed into the feature interaction stage. The video recommendation system does not use multi-source processors — its equivalent signals (user-item affinity, temporal patterns) are captured through the DIN attention mechanism over rich sequences instead.

---

## 6. Sequence Features

Users are not static profiles — they have histories. A video user might have watched 200 titles over the past year. An ad user might have clicked on 25 products this week. These behavioral histories are often the strongest predictor of future behavior: what you watched yesterday says more about what you'll watch tonight than your demographic profile or the current time of day.

These histories arrive as **lists of items** — variable-length (one user has 5 items, another has 200) and potentially ordered (watching A then B may mean something different from B then A). The model, however, requires a fixed-size vector as input to its downstream layers. So the list must somehow be compressed into a single dense representation — and the choice of *how* determines whether ordering, recency, and per-item relevance are preserved or lost. The design question is: **how much structure from the list do you want to preserve?**

Note: some list-shaped inputs skip sequence encoding entirely and are consumed by multi-source processors (Section 5) — `ElementInList` checks set membership, `DealsFeature` applies temporal filtering. These operations don't encode the list as a sequence; they extract a derived signal from it. This section covers approaches that actually *encode* the list into a representation.

### What each position contains

Orthogonal to how you aggregate a sequence is how much information each position carries. The two production systems take opposite extremes.

#### Simple ID sequences (ad prediction system)

Each position is a single integer — one segment ID or one algorithm ID. There are no additional features per position.

| Feature | Length | Cardinality | Embedding Dim | Content |
|---------|--------|-------------|---------------|---------|
| `user_behavior_segments` | 50 | 20,001 | 12 | Behavioral segment IDs |
| `ad_sourcing_algos` | 50 | 100,001 | 12 | Which algorithms sourced past ads |
| `browse_node_ids` (ad product) | 8 | 50,001 | 16 | Product category hierarchy nodes |
| `browse_node_ids` (page product) | 8 | 50,001 | 16 | Page product's category hierarchy nodes |

All of these are aggregated via mean pooling (Level 1 below). The ad prediction system does not capture sequential ordering from any of its list features.

#### Rich multi-feature sequences (video recommendation system)

The user's viewing history is stored as a sequence of up to 128 positions, where each position represents one title the user watched (position 1 = most recent, position 128 = oldest). Unlike the ad system's simple ID lists, each position carries ~52 features describing both the title and the context of viewing:

```
Position 47 in history:
  title_id = "The Grand Tour S3"
  genre = "automotive"
  language = "English"
  maturity_rating = "16+"
  stream_volume_7d = 4.2  (log-normalized popularity)
  device = "FireTV"
  hour_type = "evening"
  ... (~52 features total)
```

Each position gets its categorical features embedded and its numerical features projected (reusing the same techniques and embedding tables from Sections 1 and 3), then all are concatenated into a single per-position vector:

```
Per position: [title_id_emb(256) | genre_emb(32) | language_emb(32) | ... | num_proj(32)]
  → shape [B, 128, ~1200] (128 positions × ~1200 dims each)
```

**What each position carries:**

| Column group | Count | What it describes | How encoded per position |
|---|---|---|---|
| Title metadata (categorical) | 21 | What the user watched | Each column → embedding lookup (Section 1) |
| Viewing context (categorical) | 8 | When/where/how they watched it | Each column → embedding lookup (Section 1) |
| Title popularity (numerical) | 23 | How popular the title was at viewing time | All 23 → grouped linear projection (Section 3 Approach 3) |

This richness is what makes the higher-level aggregation strategies (DIN attention, Transformer) so expressive — when scoring "how relevant is history position #47 to the current candidate?", the model can compare across genre, language, popularity tier, device context, and time-of-day pattern simultaneously.

### The aggregation spectrum

Regardless of per-position richness, the sequence must become a fixed-size vector. The approaches form a spectrum from least to most structure-preserving:

| Level | Approach | Preserves order? | Involves candidate? | Pipeline stage |
|-------|----------|-----------------|--------------------:|----------------|
| 1 | Mean pooling | No | No — averages all items regardless | Input Representation |
| 2 | DIN attention | Partially (via temporal features per position) | **Yes** — candidate queries history | Feature Interaction |
| 3a | Transformer encoder | **Yes** — positional encoding + self-attention | No — history items attend to each other only | Input Representation |
| 3b | DIN attention (on enriched history) | **Yes** (inherits from 3a) | **Yes** — candidate queries enriched history | Feature Interaction |

The two production systems sit at opposite ends: the ad prediction system uses Level 1; the video recommendation system uses Level 3a+3b (or Level 2 alone when Transformer is disabled).

#### Level 1: Mean pooling (history only, no candidate)

Embed each item in the list, then average all embeddings into a single vector. This captures *what kinds of items* are in the list (the average "flavor" of the user's history) but loses ordering entirely — `[A, B, C]` and `[C, A, B]` produce the same output. The candidate is not involved; the same pooled vector is produced regardless of what's being scored.

The ad prediction system uses this for features where the *composition* of the set matters (what segments does this user belong to?) but the *order* does not.

#### Level 2: DIN attention (candidate × history)

Embed each item in the list, then weight each by its **relevance to the current candidate** before summing. This captures which specific history items matter for *this particular prediction* — unlike mean pooling where all items contribute equally regardless of what's being scored.

This is the video recommendation system's primary approach. The candidate title's embedding serves as a "query" that scores each history position. Crime thrillers in the history get high weight when scoring a new detective series; cooking shows get high weight when scoring a food documentary. The same user gets a different representation for each candidate.

DIN attention alone does not capture **positional order** — it doesn't inherently know whether an item was watched yesterday or six months ago. The video system compensates by including temporal context features (like `hour_type` and `day_of_week`) per position, giving the attention mechanism indirect access to recency information.

The mechanics of DIN attention (scoring function, softmax, weighted sum) are covered in the Feature Interaction deep-dive post.

#### Level 3: Transformer → DIN attention (full sequential structure)

Two separate steps chained together, each doing a different job:

**Step 3a: Transformer encoder (history only — no candidate involved).** Each history position attends to every other position, enriching its representation with context from surrounding items. This captures sequential patterns that DIN alone cannot: "users who watch A then B behave differently than those who watch B then A," or "a burst of similar titles followed by a genre switch signals exploration." Positional encoding gives the model explicit awareness of where each item sits in the sequence. The output has the same shape as the input: `[B, 128, D]` — one enriched vector per position. The candidate is not involved.

**Step 3b: DIN attention (candidate × enriched history).** The same Level 2 operation, applied to the Transformer-enriched positions. The candidate scores each enriched position by relevance, and the weighted sum produces the final fixed-size vector.

```
History [B, 128, D] → Step 3a: Transformer (self-attention, no candidate)
                     → Enriched history [B, 128, D]
                     → Step 3b: DIN Attention (candidate queries history)
                     → Single vector [B, D]
```

The video recommendation system supports Transformer encoding as an optional configuration (`use_transformer: true`), stacking it before DIN attention for maximum expressiveness at the cost of significantly more computation.

---

## 7. Putting It All Together: The Uniform Output

Every path through Sections 1-6 produces the same thing: a dense float tensor of shape `[B, dim]` per feature. After input representation, the model holds a collection of these vectors — one per feature or feature group:

```python
features_dict = {
    "advertiser_id":         tensor([B, 12]),   # Section 1: categorical embedding
    "ad_id":                 tensor([B, 8]),    # Section 1: categorical embedding
    "campaign_id":           tensor([B, 12]),   # Section 1: categorical embedding
    "hour_of_day":           tensor([B, 4]),    # Section 3: bucketed numerical
    "glance_views":          tensor([B, 8]),    # Section 3: per-feature MLP
    "best_price":            tensor([B, 8]),    # Section 3: per-feature MLP
    "view_embedding":        tensor([B, 64]),   # Section 4: pretrained numerical
    "asin_in_clicked":       tensor([B, 12]),   # Section 5: ElementInList
    "activity_60min":        tensor([B, 5]),    # Section 5: ShopperActivity
    "deal_type":             tensor([B, 4]),    # Section 5: DealsFeature
    "user_behavior_avg":     tensor([B, 12]),   # Section 6: mean-pooled sequence
    "customer_shortcode":    tensor([B, 64]),   # compressed user representation
    ...
}
```

Every entry is a float tensor with the same batch dimension `B` but a feature-specific width. The subsequent stage of the pipeline assembles these into a combined input, typically through one of two strategies:

**Concatenate all into one flat vector** (ad prediction system):
```python
main_input = torch.cat(list(features_dict.values()), dim=1)
# → [B, ~800-1200]  (sum of all feature dims)
```

At this point, feature identity is lost. The DCN and MLP layers operate on a flat vector without knowing which positions came from which feature.

**Group into separate towers, concatenate later** (video recommendation system):
```python
seq_repr = attention_pool(sequence_features)    # [B, ~1200]
imp_repr = concat(impression_features)          # [B, ~1000]
cust_repr = concat(customer_features)           # [B, ~180]
final_input = torch.cat([seq_repr, imp_repr, cust_repr], dim=1)  # [B, ~2400]
```

Features retain their group identity through the attention step (where sequence and impression towers interact), then merge for the final MLP.

Both paths deliver the same thing to the next stage: a single dense float tensor per sample, ready for feature interaction and prediction.

---

## 8. Design Tradeoffs

The input representation stage involves several architectural decisions where the two production systems made different choices driven by their different constraints.

### Pretrained Representations vs. Incremental Training

When embeddings for rare features lack sufficient training signal, there are two strategies:

| | Pretrained (ad prediction) | Incremental (video recommendation) |
|---|---|---|
| **Cold-start** | New items get meaningful embeddings immediately from upstream model | New items start near-random, improve over successive training cycles |
| **Signal source** | Upstream model may leverage billions of signals the ranking model never sees (search queries, browse patterns, product relationships) | Only learns from ranking-task data, but accumulates over time |
| **End-to-end optimization** | Frozen embedding wasn't optimized for ranking — adaptation MLP bridges the gap imperfectly | Embeddings are always directly optimized for the prediction task |
| **Infrastructure** | Depends on upstream model's pipeline, output format, and update cadence | Self-contained — no external model dependency |
| **Architecture changes** | Can swap upstream embeddings independently of ranking model | Checkpoint loading requires architecture compatibility |

The choice depends on data sparsity. The ad prediction system *needs* pretrained embeddings because it must bid on products with very sparse click history — a new product might get only a handful of impressions, far too few to learn a meaningful embedding from scratch. The video recommendation system can afford incremental training because most titles in the catalog accumulate substantial viewing data over time, making the cold-start window shorter and less costly.

### In-Model Computed Features vs. Pre-Computed Pipeline Features

Both systems need cross-entity signals (e.g., "how has this user interacted with this item before?"). They compute them at different stages:

The ad prediction system computes these **inside the model** at forward time through learned modules (`ElementInList`, `ShopperActivityEstimator`, `DealsFeature`). The video recommendation system **pre-computes** them in the data pipeline as plain numerical features (e.g., `num_playbacks_cross_feat_cid_gti_7d = 3`).

**Why compute inside the model?**

The ad system's `ElementInList` module doesn't just output "yes/no" — it produces a *learned embedding* with trainable parameters. The module can learn nuances that a pre-computed binary flag cannot: that being the 2nd item in click history differs from being the 25th (recency), or that membership in the purchase list means something different from membership in the click list. Similarly, `ShopperActivityEstimator` passes temporal counts through an MLP that learns *how* the raw count translates into a behaviorally meaningful signal. These representations are trained end-to-end with the prediction task — if you pre-computed them in a pipeline, you'd have to decide the output representation (binary? count? bucket?) before training, losing the ability to learn the most useful encoding.

**Why pre-compute in the pipeline?**

The video system's cross features are straightforward aggregations — count events in a time window. There's no ambiguity about what the "right" computation is (it's just a count), and a log-normalized count is a perfectly adequate representation. Making this computation learned inside the model would add parameters and latency for minimal benefit. Furthermore, the video system already has DIN attention operating over the full 128-item viewing history. The attention mechanism can *implicitly* learn "this user has watched this title 3 times recently" by seeing the same title_id appear at multiple positions with recent contextual timestamps. The pre-computed cross feature is a convenience shortcut — an explicit count — rather than the only way to access that signal.

**When to use which:**

| Compute inside the model when... | Pre-compute in pipeline when... |
|---|---|
| The "right" encoding isn't obvious and should be learned end-to-end | The computation is a simple aggregation with an obvious representation |
| The signal benefits from gradient-optimized representation | A normalized count or ratio is sufficient |
| Temporal logic (masking, windowing) interacts with model behavior | The model already captures the signal through other mechanisms (e.g., attention) |
| The in-model computation is cheap relative to the quality gain | Latency is extremely tight and offline pre-computation is cheaper |

---

## What Happens Next

The output of input representation — a collection of dense vectors, potentially concatenated — feeds into the **Feature Interaction** stage. This is where the model learns how features combine: explicit polynomial crosses (DCN), pairwise latent interactions (FM), or candidate-aware attention (DIN). That's the subject of the next post in this series.

---

## Appendix: Hash Embedding Source Code Walk-Through

The following is the production PyTorch implementation of hash embedding, annotated to show how each line corresponds to the conceptual steps described in Section 2.

```python
class HashEmbedding(nn.Module):
    def __init__(self, config: HashEmbeddingConfig):
        super().__init__()

        # ─── Fixed hash function coefficients (NOT trained) ───
        # These random numbers define the hash functions. They're registered
        # as "buffers" — saved with the model but never updated by gradients.
        gen = torch.Generator().manual_seed(config.hash_seed)
        a = torch.randint(1, config.hash_prime, (config.num_hash_functions,), generator=gen)
        b = torch.randint(0, config.hash_prime, (config.num_hash_functions,), generator=gen)
        self.register_buffer("hash_a", a)   # e.g., [742391, 318207, 591843]
        self.register_buffer("hash_b", b)   # e.g., [103847, 876512, 445629]
        self.register_buffer("hash_prime", torch.tensor(config.hash_prime))   # 999983
        self.register_buffer("table_size", torch.tensor(config.table_size))   # 10000

        # ─── Trainable embedding table ───
        # Instead of 3 separate tables, store one big table with 3 slices.
        # Shape: (3 × 10,000) rows × 16 dims = 30,000 rows total.
        # This IS trained — gradients update the rows that get looked up.
        self.embedding = nn.Embedding(
            config.num_hash_functions * config.table_size,  # 30,000
            config.embedding_dim                            # 16
        )

        # Offsets to address each hash function's slice of the table:
        # [0, 10000, 20000] — hash_1 uses rows 0-9999, hash_2 uses 10000-19999, etc.
        self.register_buffer(
            "offsets",
            torch.arange(config.num_hash_functions).unsqueeze(0) * config.table_size
        )

        # ─── Trainable MLP ───
        # Concatenated embeddings (3 × 16 = 48 dims) → target_dim (64 dims).
        # This IS trained — learns how to combine the 3 hash lookups.
        mlp_in = config.num_hash_functions * config.embedding_dim  # 48
        layers = []
        in_dim = mlp_in
        for _ in range(config.mlp_layers):
            layers.append(nn.Linear(in_dim, config.mlp_hidden_dim))
            layers.append(nn.PReLU(num_parameters=config.mlp_hidden_dim))
            in_dim = config.mlp_hidden_dim
        layers.append(nn.Linear(in_dim, config.target_dim))        # → 64
        self.mlp = nn.Sequential(*layers)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        ids: [batch_size] — raw IDs (e.g., customer_id = 500,000,000)
        returns: [batch_size, 64] — learned embedding for each ID
        """
        # Safety: wrap IDs to valid range
        ids = ids % self.cardinality                      # [B]

        # ─── Step 1: Hash the ID three ways into table indices ───
        ids_expanded = ids.unsqueeze(1)                   # [B, 1]
        indices = (
            (self.hash_a * ids_expanded + self.hash_b)    # universal hash: a*id + b
            % self.hash_prime                             # mod prime (reduces collisions)
        ) % self.table_size                               # mod 10,000 (fits in table)
        # Result: [B, 3] — three indices per sample, each in range [0, 9999]

        # ─── Step 2: Look up embeddings from table slices ───
        # Add offsets so hash_1 looks up rows 0-9999, hash_2 rows 10000-19999, etc.
        embeddings = self.embedding(indices + self.offsets)  # [B, 3, 16]

        # ─── Step 3: Concatenate the three vectors ───
        concatenated = embeddings.view(embeddings.size(0), -1)  # [B, 48]

        # ─── Step 4: MLP projects to final output ───
        return self.mlp(concatenated)                     # [B, 64]
```

**What's trainable vs. fixed:**

| Component | Trainable? | Shape | Purpose |
|-----------|-----------|-------|---------|
| `hash_a`, `hash_b` | No (buffer) | [3] each | Define the hash functions — fixed random numbers |
| `hash_prime`, `table_size` | No (buffer) | scalar | Hash function constants |
| `self.embedding` | **Yes** (parameter) | [30000, 16] | The actual learned representations — updated by gradients |
| `self.mlp` | **Yes** (parameter) | 48→64→64 | Learns how to combine the 3 hash lookups |
| `offsets` | No (buffer) | [1, 3] | Routing indices to correct table slices |

**Why one big table instead of three separate ones?** A single `nn.Embedding(30000, 16)` is more memory-efficient than three `nn.Embedding(10000, 16)` objects — fewer Python objects, one contiguous memory allocation, and simpler gradient accumulation. The `offsets` buffer handles the routing: hash function $k$ accesses rows $[k \times 10000, (k+1) \times 10000)$.
