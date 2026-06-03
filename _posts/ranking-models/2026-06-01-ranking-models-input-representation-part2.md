---
title: "Input Representation II: Sequences, Computed Features, and Assembly"
date: 2026-06-01 20:00:00 +0000
categories: [Ranking Models]
tags: [Deep Learning, Recommendation Systems, Sequence Encoding, DIN, Attention]
mermaid: true
math: true
---

This is Part 2b of a series on deep learning for ranking models. [Part 2a](/posts/ranking-models-input-representation-part1) covered encoding individual features (categorical embeddings, extreme-scale solutions, numerical approaches, pretrained representations). This post continues with multi-source computed features, sequence encoding, and how everything assembles into the model's input.

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

"How active was this user in the last 60 minutes?" The domain logic takes a sequence of click timestamps and the current bid timestamp, computes the time difference for each, and counts how many fall within the configured window (e.g., 60 minutes). This produces a single float (e.g., 7.0). That float is passed through a `NumericalEmbeddingModule(in_dim=1, layers=[5, 5])` — the same per-feature residual MLP described in Part 2a, Section 3 Approach 4 — producing a learned 5-dim vector.

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

| Processor                  | Inputs                                                | Output Dim | Instances | What each instance encodes                                                              |
| -------------------------- | ----------------------------------------------------- | ---------- | --------- | --------------------------------------------------------------------------------------- |
| `ElementInList`            | 1 item ID + 1 ID sequence                             | 12         | 4         | One per combination: product×clicked, product×purchased, brand×clicked, brand×purchased |
| `ShopperActivityEstimator` | Timestamp sequence + current time                     | 5          | 3         | One per time window: 15 min, 60 min, 24 hours                                           |
| `DealsFeature`             | Deal start/end dates + current time + deal attributes | 4          | 5         | One per deal attribute: deal type, Prime access, discount, state, is featured           |

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

| Feature                          | Length | Cardinality | Embedding Dim | Content                                 |
| -------------------------------- | ------ | ----------- | ------------- | --------------------------------------- |
| `user_behavior_segments`         | 50     | 20,001      | 12            | Behavioral segment IDs                  |
| `ad_sourcing_algos`              | 50     | 100,001     | 12            | Which algorithms sourced past ads       |
| `browse_node_ids` (ad product)   | 8      | 50,001      | 16            | Product category hierarchy nodes        |
| `browse_node_ids` (page product) | 8      | 50,001      | 16            | Page product's category hierarchy nodes |

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

Each position gets its categorical features embedded and its numerical features projected (reusing the same techniques and embedding tables from [Part 2a](/posts/ranking-models-input-representation-part1)), then all are concatenated into a single per-position vector:

```
Per position: [title_id_emb(256) | genre_emb(32) | language_emb(32) | ... | num_proj(32)]
  → shape [B, 128, ~1200] (128 positions × ~1200 dims each)
```

**What each position carries:**

| Column group                  | Count | What it describes                         | How encoded per position                                  |
| ----------------------------- | ----- | ----------------------------------------- | --------------------------------------------------------- |
| Title metadata (categorical)  | 21    | What the user watched                     | Each column → embedding lookup (Part 2a, Section 1)                |
| Viewing context (categorical) | 8     | When/where/how they watched it            | Each column → embedding lookup (Part 2a, Section 1)                |
| Title popularity (numerical)  | 23    | How popular the title was at viewing time | All 23 → grouped MLP projection (Part 2a, Section 3 Approach 3) |

This richness is what makes the higher-level aggregation strategies (DIN attention, Transformer) so expressive — when scoring "how relevant is history position #47 to the current candidate?", the model can compare across genre, language, popularity tier, device context, and time-of-day pattern simultaneously.

### The aggregation spectrum

Regardless of per-position richness, the sequence must become a fixed-size vector. The approaches form a spectrum from least to most structure-preserving:

| Level | Approach                            | Preserves order?                               |                          Involves candidate? | Pipeline stage       |
| ----- | ----------------------------------- | ---------------------------------------------- | -------------------------------------------: | -------------------- |
| 1     | Mean pooling                        | No                                             |           No — averages all items regardless | Input Representation |
| 2     | DIN attention                       | Partially (via temporal features per position) |          **Yes** — candidate queries history | Feature Interaction  |
| 3a    | Transformer encoder                 | **Yes** — positional encoding + self-attention | No — history items attend to each other only | Input Representation |
| 3b    | DIN attention (on enriched history) | **Yes** (inherits from 3a)                     | **Yes** — candidate queries enriched history | Feature Interaction  |

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

Every path through the input representation stage (Sections 1-6 across [Part 2a](/posts/ranking-models-input-representation-part1) and this post) produces the same thing: a dense float tensor of shape `[B, dim]` per feature. After input representation, the model holds a collection of these vectors — one per feature or feature group:

```python
features_dict = {
    "advertiser_id":         tensor([B, 12]),   # Section 1: categorical embedding
    "ad_id":                 tensor([B, 8]),    # Section 1: categorical embedding
    "campaign_id":           tensor([B, 12]),   # Section 1: categorical embedding
    "hour_of_day":           tensor([B, 4]),    # Part 2a Section 3: bucketed numerical
    "glance_views":          tensor([B, 8]),    # Part 2a Section 3: per-feature MLP
    "best_price":            tensor([B, 8]),    # Part 2a Section 3: per-feature MLP
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

|                             | Pretrained (ad prediction)                                                                                                            | Incremental (video recommendation)                                   |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Cold-start**              | New items get meaningful embeddings immediately from upstream model                                                                   | New items start near-random, improve over successive training cycles |
| **Signal source**           | Upstream model may leverage billions of signals the ranking model never sees (search queries, browse patterns, product relationships) | Only learns from ranking-task data, but accumulates over time        |
| **End-to-end optimization** | Frozen embedding wasn't optimized for ranking — adaptation MLP bridges the gap imperfectly                                            | Embeddings are always directly optimized for the prediction task     |
| **Infrastructure**          | Depends on upstream model's pipeline, output format, and update cadence                                                               | Self-contained — no external model dependency                        |
| **Architecture changes**    | Can swap upstream embeddings independently of ranking model                                                                           | Checkpoint loading requires architecture compatibility               |

The choice depends on data sparsity. The ad prediction system *needs* pretrained embeddings because it must bid on products with very sparse click history — a new product might get only a handful of impressions, far too few to learn a meaningful embedding from scratch. The video recommendation system can afford incremental training because most titles in the catalog accumulate substantial viewing data over time, making the cold-start window shorter and less costly.

### In-Model Computed Features vs. Pre-Computed Pipeline Features

Both systems need cross-entity signals (e.g., "how has this user interacted with this item before?"). They compute them at different stages:

The ad prediction system computes these **inside the model** at forward time through learned modules (`ElementInList`, `ShopperActivityEstimator`, `DealsFeature`). The video recommendation system **pre-computes** them in the data pipeline as plain numerical features (e.g., `num_playbacks_cross_feat_cid_gti_7d = 3`).

**Why compute inside the model?**

The ad system's `ElementInList` module doesn't just output "yes/no" — it produces a *learned embedding* with trainable parameters. The module can learn nuances that a pre-computed binary flag cannot: that being the 2nd item in click history differs from being the 25th (recency), or that membership in the purchase list means something different from membership in the click list. Similarly, `ShopperActivityEstimator` passes temporal counts through an MLP that learns *how* the raw count translates into a behaviorally meaningful signal. These representations are trained end-to-end with the prediction task — if you pre-computed them in a pipeline, you'd have to decide the output representation (binary? count? bucket?) before training, losing the ability to learn the most useful encoding.

**Why pre-compute in the pipeline?**

The video system's cross features are straightforward aggregations — count events in a time window. There's no ambiguity about what the "right" computation is (it's just a count), and a log-normalized count is a perfectly adequate representation. Making this computation learned inside the model would add parameters and latency for minimal benefit. Furthermore, the video system already has DIN attention operating over the full 128-item viewing history. The attention mechanism can *implicitly* learn "this user has watched this title 3 times recently" by seeing the same title_id appear at multiple positions with recent contextual timestamps. The pre-computed cross feature is a convenience shortcut — an explicit count — rather than the only way to access that signal.

**When to use which:**

| Compute inside the model when...                                    | Pre-compute in pipeline when...                                                  |
| ------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| The "right" encoding isn't obvious and should be learned end-to-end | The computation is a simple aggregation with an obvious representation           |
| The signal benefits from gradient-optimized representation          | A normalized count or ratio is sufficient                                        |
| Temporal logic (masking, windowing) interacts with model behavior   | The model already captures the signal through other mechanisms (e.g., attention) |
| The in-model computation is cheap relative to the quality gain      | Latency is extremely tight and offline pre-computation is cheaper                |

---

## What Happens Next

The output of input representation — a collection of dense vectors, potentially concatenated — feeds into the **Feature Interaction** stage. This is where the model learns how features combine: explicit polynomial crosses (DCN), pairwise latent interactions (FM), or candidate-aware attention (DIN). That's the subject of the next post in this series.

