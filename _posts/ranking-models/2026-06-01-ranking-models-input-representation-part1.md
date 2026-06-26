---
title: "Input Representation I: Encoding Individual Features"
date: 2026-06-01 19:00:00 +0000
categories: [Ranking Models]
tags: [Deep Learning, Recommendation Systems, Feature Engineering, Embeddings]
mermaid: true
math: true
---

This is Part 2a of a series on deep learning for ranking models. [Part 1 (the overview)](/posts/ranking-models-overview) presents the full pipeline and taxonomy. This post covers how individual raw features — categorical labels, numerical values, and pretrained vectors — become dense vectors the model can process. [Part 2b](/posts/ranking-models-input-representation-part2) continues with sequences, computed features, and how everything assembles into the model's input.

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

The sections that follow explain how each feature type reaches this uniform format. This post covers the first four (individual features); [Part 2b](/posts/ranking-models-input-representation-part2) covers sequences, computed features, and assembly.

---
## 1. Categorical Features → Dense Vectors

**The problem:** Features like `genre=action` or `advertiser_id=38271` are discrete symbols with no inherent numeric meaning. A neural network operates entirely through arithmetic — addition, multiplication, comparison of numbers. A discrete label like "action" or "advertiser #38271" has no numeric form the network can compute with. Before the model can process these features at all, they must be converted into numbers — and the choice of *how* to convert them has major consequences for what the model can learn.

### The abandoned approach: one-hot encoding

For a feature with vocabulary size $V$, one-hot produces a binary vector of length $V$ with a single 1 at the feature's index. For example, if there are 20,000 unique advertisers in the system:

$$\text{one}\_\text{hot}(\text{advertiser}\_\text{id}=7823) = [0, 0, \ldots, \underbrace{1}_{\text{index 7823}}, \ldots, 0] \in \mathbb{R}^{20000}$$

With 100 such features, the input vector reaches millions of dimensions, almost entirely zeros. The first layer would need millions of parameters, most learning nothing because their corresponding input is permanently zero.

### The standard approach: learned embeddings

Replace the sparse lookup with a dense one. Store a trainable matrix $E \in \mathbb{R}^{V \times d}$, where $V$ is the vocabulary size (the number of unique IDs) and $d \ll V$ is the embedding dimension (how many numbers represent each ID). For the advertiser feature, $V = 20{,}000$ and $d = 12$, so $E \in \mathbb{R}^{20000 \times 12}$. The forward pass is a simple index operation:

$$\text{embed}(\text{advertiser}\_\text{id}=7823) = E[7823] \in \mathbb{R}^{12}$$

Note that 7823 is *not* the vocabulary size — it is the *index* of the particular advertiser being looked up, i.e. which of the $V = 20{,}000$ rows of $E$ to select. The lookup returns that single row, a $d = 12$ dimensional vector. This is mathematically equivalent to one-hot times a weight matrix — selecting row 7823 — but without constructing the sparse vector or performing the multiply-by-zero.

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

How many numbers should represent each ID? There is no universal formula, but the heuristics that dominate practice all share one property: the embedding dimension $d$ grows *sublinearly* with the cardinality $n$ — far slower than the vocabulary itself. Doubling the vocabulary does not double the dimension; in fact, you hit sharply diminishing returns, where a larger table buys little additional accuracy.

Three heuristics are widely used (let $n$ = cardinality, $d$ = embedding dimension):

1. **Fourth-root rule** $d \approx n^{0.25}$ (TensorFlow / Google guidance). The most conservative; yields very small dimensions.
2. **Scaled fourth-root** $d \approx 6 \cdot n^{0.25}$ (common practitioner variant). The same curve scaled up ~6×; this is the one most published category-embedding tables track.
3. **fastai rule** $d = \min\!\left(600,\ \text{round}(1.6 \cdot n^{0.56})\right)$. Grows faster (exponent 0.56) but is hard-capped at 600.

Across a range of cardinalities, the formulas give very different numbers — note how slowly all of them grow relative to $n$. In practice, teams round to a power of two (8, 16, 32, …) for kernel efficiency and collapse the formula into a few cardinality buckets. The **Bucketed pick** column below is a typical such starting point, which closely tracks the $6\,n^{0.25}$ curve:

| Cardinality $n$ | $n^{0.25}$ | $6\,n^{0.25}$ | fastai $1.6\,n^{0.56}$ | Bucketed pick | Rationale                                    |
| --------------- | ---------- | ------------- | ---------------------- | ------------- | -------------------------------------------- |
| 10              | 2          | 11            | 6                      | 8             | Very few values, minimal information content |
| 50              | 3          | 16            | 14                     | 16            | Small set, moderate capacity needed          |
| 100             | 3          | 19            | 21                     | 32            | Standard categorical features                |
| 1,000           | 6          | 34            | 77                     | 32            | Standard categorical features                |
| 10,000          | 10         | 60            | 278                    | 64            | Medium-cardinality features                  |
| 20,000          | 12         | 71            | 410                    | 64            | Medium-cardinality features                  |
| 100,000         | 18         | 107           | 600 (cap)              | 128           | High-cardinality (title IDs, product IDs)    |
| 1,000,000       | 32         | 190           | 600 (cap)              | 256           | High-cardinality (title IDs, product IDs)    |

**Caveats that matter more than the exact formula.** The number you pick from any of these heuristics is a starting point, not an optimum:

- **They are starting points, not optima.** The right $d$ depends on how much signal the feature actually carries and how much training data you have to populate the table. Treat it as a hyperparameter to tune, not a constant to read off a table.
- **Total embedding budget dominates.** A formula sizes *one* embedding in isolation. Real models concatenate dozens of them into a single input vector, so the per-feature dimension is really a function of *how many features there are*. With many features you deliberately undershoot these numbers; with few, you can overshoot. (This is exactly why the two systems below diverge — see the next section.)
- **Use powers of two.** 8, 16, 32, 64, … — hardware/kernel efficiency, with no accuracy cost.
- **Diminishing returns are real.** Going 64 → 256 on a high-cardinality feature roughly 4× the parameters of that table while often adding little, because the formulas grow sublinearly for a reason.

### Examples from production systems

The same heuristics, two opposite design points. In the table below, the **Recommended Dim** column applies the bucketed heuristic above to each feature's cardinality, while **Actual Dim** is what the system ships — and the gap between them is the story. The two systems *bracket* the practical recommendation: the video recommendation model rides the top edge of the recommended ranges, while the ad prediction model sits well below it, with total feature count as the deciding factor.

| System               | Role       | Feature                 | Cardinality | Recommended Dim | Actual Dim |
| -------------------- | ---------- | ----------------------- | ----------- | --------------- | ---------- |
| Ad prediction        | Item       | `ad_id`                 | 300,000     | 128–256         | 8          |
| Ad prediction        | Advertiser | `advertiser_id`         | 20,000      | 64              | 12         |
| Ad prediction        | Advertiser | `campaign_id`           | 100,000     | 128–256         | 12         |
| Ad prediction        | Context    | `device_type`           | 201         | 32              | 8          |
| Ad prediction        | Context    | `slot_position`         | 11          | 16              | 8          |
| Video recommendation | Item       | `title_id` (content ID) | ~2,000,000  | 128–256         | 256        |
| Video recommendation | Item       | `genre`                 | ~500        | 32              | 32         |
| Video recommendation | User       | `profile_type`          | ~30         | 16              | 16         |
| Video recommendation | User       | `territory_code`        | ~5,000      | 64              | 64         |
| Video recommendation | Context    | `day_of_week`           | 7           | 8               | 8          |

**Video recommendation (~32 categorical features) follows the heuristic almost exactly.** Every feature lands at the top edge of its recommended range — `title_id` at 256 for millions of IDs, `territory_code` at 64 for ~5k, `genre` at 32 for ~500, `day_of_week` at 8 for 7. With relatively few features, the model can afford a rich per-feature representation, so it sizes each embedding directly from cardinality.

**Ad prediction (~58 categorical features) deliberately ignores the heuristic for high-cardinality features.** `ad_id` (300k) gets 8 dimensions where the formula suggests 128–256; `campaign_id` (100k) and `advertiser_id` (20k) get 12 where the formula suggests 64+. The reason is the budget caveat: with ~58 features concatenated, sizing each one "correctly" would blow up the input vector, so the model caps per-feature dimensions at 8–12 and pushes expressiveness into feature-interaction modules (FM, DCN) that build signal from *combinations* rather than from large individual embeddings. Only the lowest-cardinality features (`slot_position`, `device_type`) land near the generic recommendation.

---

## 2. Categorical Features at Extreme Scale

Section 1 showed how categorical features become embeddings via direct table lookup. This breaks down when vocabulary reaches billions — a standard embedding table of shape `(1B, 64)` requires ~256 GB of memory, far beyond a single GPU. This is a fundamental infrastructure problem that affects both training (the table must be updated by gradients) and inference (the table must be queried for each request). The industry has developed several approaches.

### Hash Embeddings (compress onto a single machine)

Use $K$ hash functions to map each ID into a small set of tables, then combine the results through a learned MLP:

$$\text{hash}\_\text{embed}(id) = \text{MLP}\left(\text{concat}\left[T_1[h_1(id)],\ T_2[h_2(id)],\ T_3[h_3(id)]\right]\right)$$

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

| Feature                            | Cardinality   | Hash functions | Table size | Embedding dim | MLP → Target dim |
| ---------------------------------- | ------------- | -------------- | ---------- | ------------- | ---------------- |
| `customer_id`                      | 1,000,000,000 | 3              | 10,000     | 16            | 1-layer MLP → 64 |
| `page_asin` (product being viewed) | 1,000,000,000 | 3              | 10,000     | 16            | 1-layer MLP → 64 |

The tradeoff is **hash collisions**: different IDs may map to the same table entries, forcing them to partially share representations. The MLP learns to disambiguate to some extent, but inherently loses per-ID precision compared to a dedicated row.

### Quotient-Remainder Embedding

A variant from Meta (2019): decompose the ID into quotient and remainder, look up each in a separate small table, and combine via element-wise multiply. Memory drops from $V$ to roughly $2 \times \sqrt{V}$.

The ad prediction system experiments with QR embedding applied broadly across features:

| Feature            | Cardinality | QR table size ($\approx\sqrt{V}$) | Embedding Dim |
| ------------------ | ----------- | --------------------------------- | ------------- |
| `ad_id`            | 3,000,001   | ~1,732 per table                  | 8             |
| `campaign_id`      | 1,000,001   | ~1,000 per table                  | 12            |
| `site_name`        | 1,000,001   | ~1,000 per table                  | 8             |
| `brand_name`       | 1,500,001   | ~1,225 per table                  | 16            |
| `page_asin_parent` | 10,000,001  | ~3,162 per table                  | 16            |

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

- **Exclude from embedding entirely.** Some features are never embedded — they stay as raw integers and are only consumed by lightweight processors that perform simple comparisons. For example, in the ad prediction system, `ad_asin` (cardinality 250,001) is never turned into a dense vector. It exists solely as a raw integer that gets compared against the user's click history to check membership (covered in Section 5). The membership-check processor produces the dense vector output — the raw ID itself never becomes one.
- **Wide & Deep (Google, 2016).** Splits features into a "wide" path where sparse IDs are used for memorization via simple logistic regression (no learned dense embedding) and a "deep" path where features are embedded.
- **Frequency-based cutoffs.** Only embed IDs that appear more than $N$ times in training data. Everything below the threshold maps to a shared "rare" embedding or is excluded entirely.
- **Feature hashing to small buckets without learned embeddings.** Hash 1B IDs into 10K buckets and feed the bucket index directly to a linear layer as a sparse feature — a crude signal without the expressiveness of a learned embedding, but nearly free in memory and computation.

### When to use which

| Approach             | Best when                                                                              | Tradeoff                                        |
| -------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Hash embedding       | Model must fit on a single machine; some collision noise is acceptable                 | Loses per-ID precision                          |
| Distributed sharding | Infrastructure budget available; per-ID precision matters (e.g., user personalization) | Network latency, operational complexity         |
| Don't embed          | Feature has too many values with too few observations each; not worth the cost         | Loses direct representation entirely            |
| Quotient-Remainder   | Middle ground — moderate compression with less collision than hashing                  | Less expressive than full table, more than hash |

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

| Transform | Formula            | When to apply                                               |
| --------- | ------------------ | ----------------------------------------------------------- |
| Log       | $\log(x + 1)$      | Power-law features (counts, volumes) — compresses long tail |
| Log10     | $\log_{10}(x + 1)$ | Same purpose, base-10 for interpretability                  |
| None      | $x$                | Features without extreme skew                               |

**Step 2: Normalize the scale**

| Normalization      | Formula                               | When to apply                                                                |
| ------------------ | ------------------------------------- | ---------------------------------------------------------------------------- |
| Z-score (standard) | $(x - \mu) / \sigma$                  | Default — centers at 0, unit variance                                        |
| Min-max            | $(x - x_{min}) / (x_{max} - x_{min})$ | Bounded features where output in [0, 1] is desired                           |
| Robust             | $(x - \text{median}) / \text{IQR}$    | Features with extreme outliers where mean/std are unreliable (viral content) |
| Scale factor       | $x \times c$                          | Manual rescaling by a domain-specific constant                               |

The output is a single float per feature per sample: shape `[B, 1]`.

**Examples (video recommendation system):**

The most common combination in practice is log then z-score. The video recommendation system applies both steps in offline data preprocessing (Spark/EMR jobs that run before training). The training data arrives with features already transformed and normalized — the model receives clean floats and doesn't need to do any transformation at forward time:

| Feature                                    | Description                          | Applied in preprocessing                    | What the model receives        |
| ------------------------------------------ | ------------------------------------ | ------------------------------------------- | ------------------------------ |
| `title_stream_volume_log_7d_normalized`    | How many times this title was streamed in the last 7 days | Log + Z-score              | Single float, centered near 0  |
| `title_cid_volume_log_365d_normalized`     | How many unique viewers over the past year | Log + Z-score              | Single float, centered near 0  |
| `completion_rate_365d_normalized`          | Fraction of viewers who finished the title | Z-score only (no log — already bounded 0-1) | Single float, centered near 0  |
| `runtime_for_completion_log_normalized`    | Episode/movie length in seconds      | Log + Z-score                               | Single float, centered near 0  |
| `days_live_pv_num_log_normalized`          | How many days since the title launched on platform | Log + Z-score              | Single float, centered near 0  |

The video recommendation system uses 23 numerical features describing the content item (popularity, stream volume, runtime, freshness) and 25 describing the user (session counts, playtime, engagement metrics), nearly all preprocessed with log + z-score. The model passes them through as single floats — no further transformation, no MLP, no expansion. They are later grouped and projected together (Approach 3).

Notably, the ad prediction system has *no* Approach 1 features — it stores raw untransformed values in the training data and applies log transform inside the model as the first step of a per-feature residual MLP (Approach 4). There is no standalone "normalize and pass through" path. See Approach 4 for why.

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

**Examples (ad prediction system):**

Bucketing is primarily an ad prediction pattern, where discrete decision thresholds (time since last activity, hour of day) are more important than smooth popularity curves. The video recommendation system does not use bucketing — it uses Approach 1 + 3 for all its numerical features instead.

| Feature                   | Raw value                  | Bucket boundaries                                     | Embedding Dim |
| ------------------------- | -------------------------- | ----------------------------------------------------- | ------------- |
| `hour_of_day`             | Unix epoch time            | [0,1,2,...,23] (modulo 24)                            | 4             |
| `day_of_week`             | Unix epoch time            | [0,24,48,...,144] (modulo 168)                        | 10            |
| `cart_since_activity`     | Hours since last cart add  | [1,2,4,8,12,24,48,72,168,336,720,1440,2160,4320,8760] | 8             |
| `purchase_since_activity` | Hours since last purchase  | Same as above                                         | 8             |
| `view_since_activity`     | Hours since last page view | Same as above                                         | 8             |

Note how `hour_of_day` and `day_of_week` are derived from the *same raw input* (epoch time) but bucketed differently. The logarithmic bucket spacing for recency features (1, 2, 4, 8, 12, 24, 48, ... hours) reflects that the difference between "1 hour ago" and "2 hours ago" matters more than "30 days ago" vs "31 days ago."

### Approach 3: Grouped MLP projection

Some features naturally come in groups. The video recommendation system has 23 numerical features describing a title's popularity, all measuring the same underlying phenomenon (how popular is this content?) at different time scales (1 day, 7 days, 28 days, 91 days, 365 days) and through different lenses (stream count, unique viewer count, popularity rank). Passing each as an independent float would scatter related signals across 23 separate positions in the concatenated vector, making it harder for downstream layers to recognize that they form a coherent group. A grouped projection addresses this by compressing the entire group into a single learned dense representation that captures the most informative combinations.

The projection is implemented as a 2-layer MLP with nonlinear activation — not a single linear layer:

```python
self.title_num_proj = nn.Sequential(
    nn.Linear(23, 64),      # 23 input features → expand to 2× target width
    nn.ReLU(),              # nonlinear activation
    nn.Dropout(0.2),        # regularization
    nn.Linear(64, 32),      # compress to target width (32)
)
```

The intermediate expansion (23→64) followed by compression (64→32) allows the network to learn nonlinear combinations of the input features before distilling them. For example, it might learn that "high 7-day volume but low 365-day volume" (a recently trending title) is a useful signal worth encoding as a distinct direction in the 32-dim output — something a single linear layer could also capture, but the ReLU nonlinearity allows sharper thresholds and more complex interaction patterns within the group.

**Examples (video recommendation system):**

| Feature group      | Input features                                                                      | Input dim | Output dim | What the group captures                      |
| ------------------ | ----------------------------------------------------------------------------------- | --------- | ---------- | -------------------------------------------- |
| Title numerical    | stream volumes, cid volumes, popularity metrics, runtime, days_live                 | 23        | 32         | Compressed title popularity/freshness signal |
| Customer numerical | session counts, impression counts, playtime, playback counts (across 1d/7d/28d)     | 25        | 32         | Compressed user engagement level             |
| Cross numerical    | per-user-title playback counts, completion rates, impressions (across time windows) | 26        | 32         | Compressed user-title affinity signal        |

### Approach 4: Per-feature residual MLP (single feature → multi-dim expansion)

**The problem:** In the ad prediction system, all features are concatenated into one flat vector before entering the downstream layers. Categorical features contribute 4-16 dimensions each (their embedding width). If a numerical feature contributes just 1 dimension, it's structurally disadvantaged — it connects to each downstream neuron through only one weight, while a categorical embedding connects through 8 or 12. The numerical feature gets proportionally less influence on the prediction simply because it's narrower in the concatenated vector.

**The solution:** Expand each important numerical feature from 1 dimension into 4-10 dimensions via a per-feature residual MLP. This gives numerical features comparable "width" to the categorical embeddings around them, ensuring they aren't drowned out. The raw value (e.g., 47,000 page views) is first log-compressed (`log(47000+1) = 10.76`), then a projection layer (`Linear(1→8)`) creates 8 different linear functions of that single number (8 slopes and intercepts). Each subsequent residual layer adds nonlinear transformations, mixing the 8 dimensions together. After 3 layers, the 8 output dimensions encode different nonlinear responses to the original number — effectively a learned multi-dimensional response curve.

```python
ad_asin_glance_views = Numerical(
    in_dim=1,              # single raw number (product page view count)
    layers=[8, 8, 8],     # 3-layer residual MLP, width 8
    log_transform=True,    # log(x+1) applied first
    residual_layer=True,   # skip connections between layers
)
```

**Why this is better than a single float or bucketing:** With 8 dimensions, the feature can simultaneously express that "the difference between 100 and 1,000 views matters a lot" (one dimension responds steeply in that range) while "the difference between 100,000 and 101,000 barely matters" (all dimensions are flat there). It achieves what bucketing (Approach 2) does — different responses in different value ranges — but learned smoothly from data rather than requiring hand-designed bin boundaries.

**The tradeoff:** More parameters per feature (a 3-layer `[8,8,8]` MLP has ~200 parameters), which is why the ad system reserves this treatment for important product-level metrics rather than applying it to every numerical feature.

**Examples (ad prediction system):**

Features with log transforms are count-based with long tails (page views, ordered units, review counts). Features without log are already bounded or pre-normalized — e.g., `avg_review` is a 1-5 star rating that doesn't need compression. MLP depth scales with feature importance — 3 layers for key product metrics, 1-2 layers for secondary signals.

| Feature                             | in_dim | layers  | log | Output dim | What it represents                          |
| ----------------------------------- | ------ | ------- | --- | ---------- | ------------------------------------------- |
| `glance_views` (product page views) | 1      | [8,8,8] | Yes | 8          | Product traffic level                       |
| `ordered_units`                     | 1      | [8,8,8] | Yes | 8          | Purchase velocity                           |
| `best_price`                        | 1      | [8,8,8] | Yes | 8          | Price point                                 |
| `avg_review`                        | 1      | [8,8,8] | No  | 8          | Review quality (already bounded ~1-5)       |
| `customer_review_count`             | 1      | [4,4]   | Yes | 4          | Review volume (less important, smaller MLP) |
| `num_offers`                        | 1      | [4]     | No  | 4          | Seller competition (single layer suffices)  |
| `customer_total_clicks`             | 1      | [8,8]   | No  | 8          | User click history volume                   |
| `cart_past_day`                     | 1      | [10,10] | No  | 10         | Recent cart activity intensity              |

### Comparing the approaches

The four approaches represent a spectrum from simplest (no learned parameters) to most expressive (per-feature nonlinear expansion). The choice depends on how important the feature is, whether its relationship to the prediction is smooth or threshold-based, and whether you need features to have comparable width in the concatenated vector.

|               | Approach 1                | Approach 2             | Approach 3                 | Approach 4                     |
| ------------- | ------------------------- | ---------------------- | -------------------------- | ------------------------------ |
| **Operation** | Normalize                 | Discretize → embed     | Group → 2-layer MLP project | Expand via residual MLP        |
| **Input**     | 1 float                   | 1 float                | N related floats           | 1 float                        |
| **Output**    | 1 float                   | embedding dim (4-10)   | projection dim (32)        | MLP width (4-10)               |
| **Direction** | Passthrough               | Expansion              | Compression                | Expansion                      |
| **Learns**    | Nothing (fixed transform) | Per-bin representation | Cross-feature nonlinear combinations | Per-feature nonlinear response |
| **System**    | Video recommendation      | Ad prediction          | Video recommendation       | Ad prediction                  |

**When to use which:**

- **Approach 1** when the feature will be grouped with others (Approach 3) downstream, or when the system preprocesses data offline and the model doesn't need to transform anything at forward time.
- **Approach 2** when the feature has meaningful thresholds that define distinct behavioral regimes (recency buckets, time-of-day categories) and you want the model to learn completely independent representations per regime.
- **Approach 3** when you have many related features measuring the same phenomenon at different granularities, and the useful signal lives in their cross-feature combinations (trending = high short-term, low long-term).
- **Approach 4** when the feature is individually important, its effect is nonlinear, and it must compete for influence against multi-dimensional categorical embeddings in a concatenated vector.

---

## 4. Pretrained Representations

**The problem:** Training representations from scratch requires large amounts of task-specific data for each feature value. For high-cardinality features like product IDs or user IDs, the ranking model may only see each value a handful of times — not enough to learn a good embedding. But a separate, larger model (a product graph embedding, a user representation model, or a content understanding system) may have already trained high-quality representations for these same entities using a different objective and far more data. Throwing away that knowledge and starting from scratch wastes signal.

The solution is to reuse the upstream model's output as a starting point, while attaching a trainable MLP on top that adapts it for the ranking task. This pattern appears in two forms — one for categorical features (where the upstream representation is stored as a frozen embedding table indexed by ID) and one for numerical features (where the upstream representation arrives directly as a pre-computed float vector). Both follow the same principle: freeze the upstream signal, learn an adaptation layer.

### Categorical pretrained: frozen table + projection MLP

For categorical features like product IDs, a frozen embedding table stores the upstream model's representations. The ranking model looks up the ID, gets a fixed vector, and projects it through a trainable MLP:

$$\text{output} = \text{MLP}(\text{frozen}\_\text{embedding}[id])$$

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

| Feature              | Type                   | Frozen input dim          | Trainable adaptation       | Output dim | Upstream source                  |
| -------------------- | ---------------------- | ------------------------- | -------------------------- | ---------- | -------------------------------- |
| Product ID           | Categorical pretrained | 128 (frozen table lookup) | Down-projection MLP 128→16 | 16         | Product graph model              |
| `view_embedding`     | Numerical pretrained   | 64 (pre-computed vector)  | Residual MLP [64,64,64]    | 64         | Product view co-occurrence model |
| `gl_affinity_scores` | Numerical pretrained   | 21 (pre-computed vector)  | Residual MLP [21,21]       | 21         | Category affinity model          |

The video recommendation system does not use pretrained representations — all its embeddings are trained from scratch. See the Design Tradeoffs section in [Part 2b](/posts/ranking-models-input-representation-part2) for a comparison of the two systems' approaches.

---

---

## What's Next

This post covered encoding individual features: categorical embeddings, extreme-scale solutions, numerical approaches, and pretrained representations. But ranking models also deal with multi-value inputs (user histories), derived signals that require combining multiple raw features, and the question of how all these vectors assemble into a single model input.

[Part 2b: Sequences, Computed Features, and Assembly](/posts/ranking-models-input-representation-part2) continues with those topics.

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

| Component                  | Trainable?          | Shape       | Purpose                                                   |
| -------------------------- | ------------------- | ----------- | --------------------------------------------------------- |
| `hash_a`, `hash_b`         | No (buffer)         | [3] each    | Define the hash functions — fixed random numbers          |
| `hash_prime`, `table_size` | No (buffer)         | scalar      | Hash function constants                                   |
| `self.embedding`           | **Yes** (parameter) | [30000, 16] | The actual learned representations — updated by gradients |
| `self.mlp`                 | **Yes** (parameter) | 48→64→64    | Learns how to combine the 3 hash lookups                  |
| `offsets`                  | No (buffer)         | [1, 3]      | Routing indices to correct table slices                   |

**Why one big table instead of three separate ones?** A single `nn.Embedding(30000, 16)` is more memory-efficient than three `nn.Embedding(10000, 16)` objects — fewer Python objects, one contiguous memory allocation, and simpler gradient accumulation. The `offsets` buffer handles the routing: hash function $k$ accesses rows $[k \times 10000, (k+1) \times 10000)$.
