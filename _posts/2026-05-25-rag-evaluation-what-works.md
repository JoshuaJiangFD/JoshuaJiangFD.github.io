---
title: "RAG Evaluation: What to Measure and Where Metrics Fail"
date: 2026-05-25 10:00:00 +0000
categories: [RAG Evaluation]
tags: [RAG, Evaluation, RAGAS, Faithfulness, Retrieval Metrics, NLI]
mermaid: true
math: true
---

Everyone building RAG systems eventually hits the same wall: how do you know if it's working? The retrieval looks reasonable, the answers sound fluent, users aren't complaining loudly. But you have no quantitative grip on quality. You can't tell if your last change made things better or worse.

Getting that grip requires understanding where quality can break down. A RAG pipeline has a retriever and a generator, and quality can degrade at either stage or in the relationship between them. This creates three evaluation surfaces (Section 1). For each surface, there are metrics with different tradeoffs in annotation cost, correlation with downstream quality, and production feasibility (Sections 2-5). Even with the right metrics in place, certain failure modes remain invisible to all current automated approaches (Section 6). 

---

## 1. The RAG Pipeline and Its Evaluation Surfaces

A Retrieval-Augmented Generation (RAG) system augments a language model with external knowledge at inference time. Rather than relying solely on what the model memorized during pretraining, it retrieves relevant documents from a corpus and feeds them to the generator as context. The pipeline has two core stages:

1. **Retrieval.** Given a user query, an embedding model encodes the query into a vector. A search index (dense, sparse, or hybrid) returns the top-K most similar document chunks from the corpus.
2. **Generation.** The retrieved chunks are concatenated into the LLM's context window alongside the original query. The LLM generates a response conditioned on both the query and the retrieved evidence.

```mermaid
flowchart LR
    Q[User Query] --> R
    subgraph R[Retriever]
        direction LR
        E[Embedding Model] --> I[Search Index]
    end
    R --> C[Retrieved Context]
    Q --> G[LLM Generator]
    C --> G
    G --> A[Response]
```

This two-stage structure creates three relationships that can independently fail:

| Edge | Question | Failure Mode |
|------|----------|--------------|
| Query → Retrieved Context | Did the retriever find relevant passages? | Irrelevant or missing context |
| Retrieved Context → Response | Did the generator stay faithful to the evidence? | Hallucination, fabrication |
| Query → Response | Does the final output address the user's need? | Correct retrieval + faithful generation, but wrong question answered |

```mermaid
flowchart LR
    Q[User Query] --> R[Retriever]
    R --> C[Retrieved Context]
    C --> G[LLM Generator]
    G --> A[Response]

    Q -.->|"Retrieval quality"| C
    C -.->|"Faithfulness"| A
    Q -.->|"End-to-end relevance"| A
```

Multiple evaluation frameworks have independently operationalized these three edges. The following table maps each framework's metrics to the edge they address:

| Framework | Edge 1: Retrieval Quality | Edge 2: Faithfulness | Edge 3: End-to-End Relevance | Notes |
|---|---|---|---|---|
| **[TruLens](https://www.trulens.org/getting_started/core_concepts/rag_triad/)** (2023) | [Context Relevance](https://www.trulens.org/getting_started/core_concepts/rag_triad/#context-relevance) | [Groundedness](https://www.trulens.org/getting_started/core_concepts/rag_triad/#groundedness) | [Answer Relevance](https://www.trulens.org/getting_started/core_concepts/rag_triad/#answer-relevance) | Branded as "RAG Triad" |
| **[RAGAS](https://docs.ragas.io)** (Es et al., 2023) | [Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/context_precision/), [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/context_recall/) | [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/faithfulness/) | [Answer Relevance](https://docs.ragas.io/en/stable/concepts/metrics/answer_relevance/) | Also includes Noise Sensitivity (edge 2 diagnostic) |
| **[DeepEval](https://deepeval.com/docs)** | [Contextual Relevancy](https://deepeval.com/docs/metrics-contextual-relevancy), [Contextual Precision](https://deepeval.com/docs/metrics-contextual-precision), [Contextual Recall](https://deepeval.com/docs/metrics-contextual-recall) | [Faithfulness](https://deepeval.com/docs/metrics-faithfulness) | [Answer Relevancy](https://deepeval.com/docs/metrics-answer-relevancy) | Separates retriever vs. generator metric groups |
| **[Google Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/models/metrics-templates)** | — | [Groundedness](https://cloud.google.com/vertex-ai/generative-ai/docs/models/metrics-templates) | [Text Quality](https://cloud.google.com/vertex-ai/generative-ai/docs/models/metrics-templates), [Instruction Following](https://cloud.google.com/vertex-ai/generative-ai/docs/models/metrics-templates), Fluency | Adds Safety (no edge equivalent); no explicit retrieval metric |
| **[Databricks](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG)** | — | — | [Correctness](https://docs.databricks.com/en/generative-ai/agent-evaluation/llm-judge-metrics.html) (60%), Comprehensiveness (20%), Readability (20%) | Evaluates answer only; retrieval and faithfulness not separately measured |

The convergence across frameworks is natural. It follows from the pipeline topology. The differences are in emphasis: evaluation tool vendors (TruLens, RAGAS, DeepEval) measure all three edges explicitly, while platform companies (Google, Databricks) collapse edges 1 and 2 into the answer-level evaluation and add dimensions the triad does not cover (safety, fluency, instruction-following).

The remainder of this post uses RAGAS as the primary example when explaining metric mechanics, because it has an accompanying research paper (arXiv:2309.15217) that formally defines each metric with formulas and validation. [TruLens](https://github.com/truera/trulens) (open-source, maintained by Snowflake) and [DeepEval](https://github.com/confident-ai/deepeval) (open-source, by Confident AI) cover the same three edges but differ in implementation:

| Dimension | RAGAS | TruLens | DeepEval |
|-----------|-------|---------|----------|
| Claim extraction for faithfulness | LLM decomposes into atomic claims with pronoun resolution | NLTK sentence splitting | LLM-based claim extraction |
| Faithfulness scoring | Binary verdict per claim (supported or not) | Graded 0-3 Likert per sentence | Binary verdict per claim |
| Non-LLM option for faithfulness | Vectara HHEM | DeBERTa-v3 NLI model | None (LLM-only) |
| Context relevance target | Relevance to the **reference answer** | Relevance to the **query** | Relevance to the **query** |
| Answer relevance method | Reverse question generation + embedding cosine similarity | Direct LLM Likert scoring | LLM-as-judge (G-Eval style) |


---

## 2. Classical Retrieval Metrics

Retrieval quality ultimately determines what the LLM sees. A bad retrieval means the generator either works with irrelevant context (leading to distraction or hallucination) or misses the evidence it needs (leading to incomplete or wrong answers). But "did we retrieve the right things" can be sliced multiple ways, and each way corresponds to a different metric:

- Did we find *anything* relevant? (Hit Rate)
- How much noise is mixed in with the relevant results? (Precision)
- Did we miss important sources? (Recall)
- How quickly does the first useful result appear? (MRR)
- Are the best results ranked above the mediocre ones? (NDCG)

Each metric answers a different question and requires a different level of annotation effort. This tradeoff between diagnostic power and annotation cost determines which metrics are feasible in practice. These metrics predate RAG. They were developed for web search ranking and document retrieval, where the same problem exists: given a query, surface the most relevant results. RAG inherits both the metrics and their limitations from that lineage.

All retrieval metrics are evaluated over a set of queries $Q$. Each metric is first computed per-query, then the final evaluation score is the mean across all queries in the set. Throughout the examples below, assume a retriever returns K=5 chunks, and the corpus contains 4 total relevant chunks for the query.

### 2.1 Hit Rate@K

Hit Rate@K (also called Hit@K) measures whether *any* relevant document appears in the top-K results.

$$\text{Hit Rate@K} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{1}[\text{at least one relevant doc in top-K for query } i]$$

Example: the retriever returns 5 chunks. At least one of them is relevant, so the hit for this query is 1. Averaged across a set of 100 evaluation queries where 85 have at least one relevant chunk in top-5:

$$\text{Hit Rate@5} = \frac{85}{100} = 0.85$$

Strength: the simplest retrieval metric. Answers the most basic question: did the retriever find *anything* useful?

Weakness: does not distinguish between retrieving 1 relevant chunk and retrieving 5. A system that returns 5 highly relevant passages scores the same as one that returns 1 relevant passage buried among 4 irrelevant ones. Also insensitive to rank position.

Annotation cost: requires only one known-relevant document per query. This can be generated synthetically (ask an LLM to produce a question from a chunk; that chunk is the relevant document).

### 2.2 Precision@K

Precision@K measures what fraction of the retrieved chunks are relevant.

Per-query:

$$\text{Precision@K}_i = \frac{\text{Number of relevant documents in top-K for query } i}{K}$$

Averaged across the evaluation set:

$$\text{Precision@K} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \text{Precision@K}_i$$

Example: the retriever returns 5 chunks, of which 3 are relevant.

$$\text{Precision@5} = \frac{3}{5} = 0.6$$

Strength: simple and interpretable. Directly answers "how much noise is in the retrieval?"

Weakness: treats all positions equally. A relevant chunk buried at rank 5 counts the same as one at rank 1. In RAG, this matters because LLMs attend more strongly to context at the beginning and end of the window ("lost in the middle" phenomenon), so rank position affects downstream generation quality.

Annotation cost: binary relevance labels on the K retrieved chunks only. No corpus-level annotation needed.

### 2.3 Recall@K

Recall@K measures what fraction of all relevant chunks were retrieved.

Per-query:

$$\text{Recall@K}_i = \frac{\text{Number of relevant documents in top-K for query } i}{\text{Total relevant documents in corpus for query } i}$$

Averaged across the evaluation set:

$$\text{Recall@K} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \text{Recall@K}_i$$

Example: same retrieval (3 relevant in top-5), but the corpus contains 4 relevant chunks total.

$$\text{Recall@5} = \frac{3}{4} = 0.75$$

Strength: ensures the retriever does not miss important information. For multi-hop questions that require evidence from multiple documents, high recall is essential because missing even one source can make the question unanswerable.

Weakness: the denominator requires knowing every relevant document in the entire corpus. This makes the metric impossible to compute exactly without exhaustive annotation.

Annotation cost: exhaustive corpus-level annotation. Every document must be labeled for relevance against each query. For a corpus of 100K chunks and 500 evaluation queries, this means 50M judgments in the worst case. In practice, pooling strategies reduce this (e.g., only label documents surfaced by any retrieval method), but the cost remains fundamentally higher than per-retrieval metrics.

### 2.4 MRR (Mean Reciprocal Rank)

MRR measures how quickly the first relevant chunk appears.

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

where $\text{rank}_i$ is the position of the first relevant document for query $i$.

Example: for a single query, the first relevant chunk appears at position 2.

$$\text{RR} = \frac{1}{2} = 0.5$$

Strength: captures whether the retriever surfaces relevant material immediately. In RAG, the first relevant chunk often determines whether the LLM can begin generating a correct answer.

Weakness: only considers the first relevant document. A retriever that places one relevant chunk at rank 1 but misses the other three still scores perfectly. For questions requiring synthesis across multiple sources, MRR gives no signal about whether all necessary evidence was retrieved.

Annotation cost: minimal. Requires identifying at least one relevant document per query. An annotator can stop labeling as soon as the first relevant document is found in the ranked list.

### 2.5 NDCG (Normalized Discounted Cumulative Gain)

NDCG measures whether highly relevant chunks are ranked above marginally relevant ones.

Per-query:

$$\text{DCG@K}_i = \sum_{k=1}^{K} \frac{2^{rel_k} - 1}{\log_2(k+1)}$$

$$\text{NDCG@K}_i = \frac{\text{DCG@K}_i}{\text{IDCG@K}_i}$$

where $rel_k$ is the graded relevance of the document at rank $k$ (e.g., 0=irrelevant, 1=partial, 2=highly relevant). IDCG (Ideal DCG) is the DCG computed on the ideal ranking, where all relevant documents are sorted by descending relevance at the top:

$$\text{IDCG@K}_i = \sum_{k=1}^{K} \frac{2^{rel_k^*} - 1}{\log_2(k+1)}$$

where $rel_k^*$ is the relevance of the document at rank $k$ in the ideal ordering.

Averaged across the evaluation set:

$$\text{NDCG@K} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \text{NDCG@K}_i$$

Example: retriever returns 5 chunks with relevance grades [2, 0, 1, 0, 1]:

$$\text{DCG@5} = \frac{2^2-1}{\log_2 2} + \frac{2^0-1}{\log_2 3} + \frac{2^1-1}{\log_2 4} + \frac{2^0-1}{\log_2 5} + \frac{2^1-1}{\log_2 6} = 3.0 + 0 + 0.5 + 0 + 0.39 = 3.89$$

The ideal ranking sorts all documents by descending relevance: [2, 2, 1, 1, 0].

$$\text{IDCG@5} = \frac{2^2-1}{\log_2 2} + \frac{2^2-1}{\log_2 3} + \frac{2^1-1}{\log_2 4} + \frac{2^1-1}{\log_2 5} + \frac{2^0-1}{\log_2 6} = 3.0 + 1.89 + 0.5 + 0.43 + 0 = 5.82$$

$$\text{NDCG@5} = \frac{3.89}{5.82} = 0.67$$

Strength: the most informative single metric. Respects rank position (via log discount) and relevance magnitude (via grading). A highly relevant document at rank 1 contributes far more than a marginally relevant document at rank 5. This aligns well with RAG, where the most relevant passage should appear first in the context window.

Weakness: the IDCG normalization requires knowing the ideal ranking, which means all relevant documents in the corpus must be identified and graded. Without exhaustive annotation, IDCG cannot be computed exactly, and the metric is undefined.

Annotation cost: the highest of the five metrics. Requires (1) exhaustive identification of all relevant documents in the corpus (same as Recall@K), and (2) graded relevance labels rather than binary (e.g., distinguishing "partially relevant" from "highly relevant"). This demands more annotator cognitive effort per judgment.

However, practical approximations exist that make NDCG feasible without exhaustive annotation:

- **Shallow pooling.** Instead of labeling the entire corpus, pool only the top-10 results from 2-5 candidate retrieval configurations, then judge that combined set. For NDCG@10, this works well because the top positions are well-covered by the pool. Buckley et al. (2007) showed that pooling error is less severe for NDCG@K with small K.
- **LLM-as-judge for graded relevance.** Use an LLM to assign relevance grades (0-3 scale) to pooled documents instead of human annotators. Thomas et al. (SIGIR 2024) showed LLM-based NDCG correlates with human-NDCG at Kendall's tau > 0.9. The UMBRELA benchmark (Upadhyay et al., 2024) confirmed this at scale on the TREC Deep Learning track. This is now the dominant approach for reducing evaluation cost.
- **Inferred NDCG (Yilmaz & Aslam, 2006).** Sample documents at different rank strata, judge only the sample, then statistically infer the full metric. Still used in some TREC tracks (Product Search 2023, Deep Learning document task), but the field's attention has shifted toward LLM-as-judge as the primary cost-reduction method.

The common production pattern combines the first two: pool top-10 from the current system plus a baseline, have an LLM assign graded relevance to all pooled documents, compute NDCG@10. This costs a few cents per query and produces system rankings that correlate strongly with full human evaluation.

### 2.6 Production vs. Benchmark Usage

The annotation cost of each metric largely determines where it is used in practice:

| Metric | Annotation Scope | Used in Production? | Used in Benchmarks? |
|--------|-----------------|--------------------|--------------------|
| Hit Rate@K | 1 relevant doc per query | Yes (dominant) | Rarely |
| MRR | 1 relevant doc per query | Yes (common) | Yes (BEIR, MTEB) |
| Precision@K | Binary labels on K retrieved docs | Sometimes | Yes |
| Recall@K | All relevant docs in corpus | Rarely (too expensive) | Yes (BEIR) |
| NDCG@K | Graded labels on all relevant docs | Rarely (too expensive) | Yes (primary metric in BEIR, MTEB) |

Academic benchmarks (BEIR, MTEB) can use NDCG@10 as their primary metric because they build on existing exhaustively-annotated datasets (TREC, MS MARCO) where the annotation work was done once and reused. Production teams cannot afford this for their proprietary corpora.

The common production pattern for retrieval evaluation:

1. **Offline evaluation (low annotation).** Generate synthetic question-answer pairs from the corpus using an LLM. The source chunk becomes the known-relevant document. Compute Hit Rate@K and MRR against these synthetic pairs. This requires zero human annotation and is sufficient for comparing embedding models, chunk sizes, and retrieval strategies.

2. **Online monitoring (zero annotation).** Use LLM-as-judge to score context relevance per query at inference time. TruLens, RAGAS, and DeepEval all support this. No pre-labeled dataset needed.

3. **Higher investment (small labeled set).** Create 50–100 human-labeled (query, relevant passages) pairs. Compute Precision@K or NDCG@K on this set. Calibrate an LLM judge against the human labels, then use the LLM judge to scale evaluation to thousands of queries.

---

## 3. RAGAS Retrieval Metrics

RAGAS (Retrieval Augmented Generation Assessment; Es et al., arXiv:2309.15217) is an open-source evaluation framework designed to reduce the annotation cost of RAG evaluation. Classical IR metrics require human annotators to label relevance. RAGAS replaces human annotators with LLM judges, enabling automated evaluation that requires at most a reference answer per query rather than exhaustive corpus-level annotation.

RAGAS introduces a conceptual shift from classical IR metrics: instead of measuring relevance to **the query**, it measures relevance to **the known correct answer**. It has two retrieval metrics, Context Precision and Context Recall, to evaluate edge 1 (retrieval quality). Both require a reference (ground truth) answer as input.

### 3.1 Context Precision

Context Precision measures whether the *useful* chunks are ranked near the top of the retrieval results.

**How relevance is determined:** For each retrieved chunk at position $k$, an LLM is prompted: "Given this question, this reference answer, and this context chunk, was the chunk useful in arriving at the given answer?" The LLM returns a binary verdict $v_k \in \{0, 1\}$.

This is subtly different from classical relevance. A chunk might be topically related to the query but receive $v_k = 0$ if it was not useful for producing the specific reference answer.

**Score computation (Average Precision):**

$$\text{Context Precision@K} = \frac{\sum_{k=1}^{K} \text{Precision@k} \times v_k}{\sum_{k=1}^{K} v_k}$$

where $\text{Precision@k} = \frac{\sum_{j=1}^{k} v_j}{k}$ is the cumulative precision at position $k$.

**Example:**

Query: "Where is the Eiffel Tower?"
Reference answer: "The Eiffel Tower is located in Paris, France."
Retrieved chunks (K=3):
- Chunk 1: "The Eiffel Tower is a wrought-iron lattice tower in Paris, France." → $v_1 = 1$
- Chunk 2: "The Statue of Liberty is in New York City." → $v_2 = 0$
- Chunk 3: "Paris is the capital of France." → $v_3 = 1$

Computation:
- $k=1$: $v_1=1$, Precision@1 = 1/1 = 1.0, contributes 1.0
- $k=2$: $v_2=0$, skip (only sum where $v_k=1$)
- $k=3$: $v_3=1$, Precision@3 = 2/3 = 0.667, contributes 0.667

$$\text{Context Precision@3} = \frac{1.0 + 0.667}{2} = 0.833$$

If the irrelevant chunk were ranked first ([Chunk 2, Chunk 1, Chunk 3]):
- $k=1$: $v_1=0$, skip
- $k=2$: $v_2=1$, Precision@2 = 1/2 = 0.5, contributes 0.5
- $k=3$: $v_3=1$, Precision@3 = 2/3 = 0.667, contributes 0.667

Score = (0.5 + 0.667) / 2 = 0.583 (lower because useful chunks are ranked lower).

**Comparison with classical Precision@K:**

| | Classical Precision@K | RAGAS Context Precision |
|---|---|---|
| Who judges relevance? | Human annotators | LLM |
| Relevant to what? | The query | The reference answer |
| Position-sensitive? | No (simple ratio) | Yes (Average Precision rewards top-ranked relevant chunks) |
| Annotation required | Binary labels on K chunks | Reference answer per query |

### 3.2 Context Recall

Context Recall measures whether the retrieved context covers all the information needed to produce the ground truth answer.

**How it works:** The LLM receives all retrieved contexts (concatenated) and the reference answer. It decomposes the reference answer into **sentences** (called "claims" in RAGAS terminology), then for each sentence judges: "Can this sentence be attributed to the retrieved context?" The verdict is binary: attributed (1) or not attributed (0).

$$\text{Context Recall} = \frac{\text{Number of reference sentences attributed to context}}{\text{Total sentences in reference answer}}$$

**The "claim" concept:** In Context Recall, a claim is operationally a sentence in the reference answer. The LLM identifies sentence boundaries and evaluates each one independently. There is no sub-sentence decomposition.

**Example:**

Query: "What is the capital of France and its population?"
Reference answer: "Paris is the capital of France. It has a population of approximately 2.1 million."
Retrieved contexts: ["Paris is the capital city of France, located on the Seine river."]

LLM evaluates each sentence in the reference:
- "Paris is the capital of France." → attributed = 1 (context confirms this)
- "It has a population of approximately 2.1 million." → attributed = 0 (no population data in context)

$$\text{Context Recall} = \frac{1}{2} = 0.5$$

**Comparison with classical Recall@K:**

| | Classical Recall@K | RAGAS Context Recall |
|---|---|---|
| Denominator | Total relevant documents in corpus | Total sentences in reference answer |
| What it measures | Coverage of relevant corpus | Coverage of reference answer information |
| Requires corpus annotation? | Yes (must identify all relevant docs) | No (only needs reference answer) |
| Can detect missed documents? | Yes | No (only knows what the reference says) |
| Annotation required | Exhaustive corpus-level labels | Reference answer per query |

The denominators are fundamentally different. Classical recall measures "what fraction of all relevant documents did we retrieve?" RAGAS Context Recall measures "what fraction of the ground truth information is supported by what we retrieved?" A system can score 1.0 on RAGAS Context Recall while missing many relevant documents in the corpus, as long as the retrieved chunks happen to cover all claims in the reference answer.

**Implication for annotation cost:** Both RAGAS context metrics require a reference answer per query, which is cheaper than corpus-level annotation but more expensive than the zero-annotation approaches (Hit Rate on synthetic data, LLM-as-judge without reference). They are suitable for offline evaluation on curated test sets, not for online monitoring of production traffic.


---

## 4. Faithfulness / Groundedness Metrics

This is the hallucination check, and arguably the most critical dimension for trust. Faithfulness metrics answer: "Is the generated response supported by the retrieved context, or did the LLM fabricate information?"

Two families of approaches exist: LLM-based (prompt a general-purpose LLM to verify claims) and classifier-based (use a trained model specifically for entailment detection). They trade off flexibility against cost and determinism.

### 4.1 LLM-Based: Claim Decomposition (RAGAS)

$$\text{Faithfulness} = \frac{\text{Claims in response supported by context}}{\text{Total claims in response}}$$

Two-step process:
1. An LLM decomposes the generated answer into atomic claims. For example, "Paris is the capital of France and has 2.1 million people" becomes two claims: "Paris is the capital of France" and "Paris has 2.1 million people."
2. For each claim, the LLM judges whether it is supported by the retrieved context.

If an answer produces 5 claims and 3 are grounded in the passages, the faithfulness score is 0.6.

Strength: fine-grained. Identifies exactly which claims are unsupported. Flexible across domains because the LLM generalizes.

Weakness: expensive (multiple LLM calls per response). Non-deterministic. The claim decomposition step itself can introduce errors if the LLM splits claims poorly.

### 4.2 Classifier-Based: NLI and Trained Models

Natural Language Inference (NLI) is a classification task where a model determines the relationship between two texts: given a *premise* and a *hypothesis*, the model predicts whether the premise **entails** the hypothesis, **contradicts** it, or is **neutral**. NLI models are trained on large datasets of human-annotated text pairs (e.g., SNLI, MultiNLI).

Applied to faithfulness: the retrieved context is the premise, the generated response (or individual sentences from it) is the hypothesis. If the context entails the response, the response is grounded. If not, it may be hallucinated.

The TRUE benchmark (Honovich et al., arXiv:2204.04991) validated this approach, finding that "large-scale NLI and question generation-and-answering-based approaches achieve strong and complementary results."

Production-oriented classifiers in this category:

| Model | Type | Key Property |
|-------|------|-------------|
| **Vectara HHEM-2.1-Open** | Small binary classifier | Free, fast, deterministic. Replaces LLM-based verification for production use. |
| **Fine-tuned DeBERTa** (as in ARES) | NLI model fine-tuned on domain data | ROC-AUC improved from 0.56 to 0.85 with ~1000 task-specific training examples |

Strength: deterministic, fast, cheap (no LLM API calls). Can run on CPU or small GPU.

Weakness: less flexible than LLM-based approaches. NLI models trained on general entailment data may not transfer well to domain-specific language. Requires fine-tuning on task-specific data to achieve strong performance.

### 4.3 How Accurate Is Faithfulness Detection?

Detecting hallucination is itself an unsolved problem. Current accuracy of different approaches:

| Approach | Accuracy / Performance | Source |
|----------|----------------------|--------|
| ChatGPT (zero-shot) | 53.8–58.5% accuracy on HaluEval | HaluEval benchmark |
| Claude 2 (zero-shot) | 53.8–58.5% accuracy on HaluEval | HaluEval benchmark |
| Off-the-shelf NLI model | ROC-AUC 0.56 | TRUE benchmark |
| NLI fine-tuned on ~1000 domain examples | ROC-AUC 0.85 | TRUE benchmark follow-up |
| ChainPoll (Galileo) | AUROC 0.781 | arXiv:2310.18344 |

The gap between zero-shot LLM detection (~55% accuracy, barely above random) and fine-tuned classifiers (ROC-AUC 0.85) is striking. It suggests that faithfulness detection requires task-specific training to be reliable. General-purpose LLMs are surprisingly poor at detecting hallucinations in their own outputs without specialized prompting or fine-tuning.

### 4.4 What Faithfulness Metrics Cannot Catch

A response can be perfectly faithful (every claim supported by context) while being *wrong*, because the reasoning that connects those facts is incorrect. Faithfulness checks attribution, not inference quality.

Example: if the context says "Company X revenue was $10M in 2023" and "Company X revenue was $8M in 2022", a response stating "Company X revenue declined by 50%" is faithful (both numbers are in the context) but the arithmetic is wrong. No faithfulness metric catches this because each individual claim ("revenue was $10M", "revenue was $8M", "revenue declined") can be attributed to the context. The error is in the reasoning between claims, not in the claims themselves.

---

## 5. Answer Relevance Metrics

### 5.1 RAGAS Answer Relevancy

$$\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \cos(E_{q_i}, E_{q_{orig}})$$

The approach generates $N$ artificial questions from the response, then computes cosine similarity between these generated questions and the original question. If the response correctly addresses a question, the original question should be reconstructable from the answer alone. Penalizes both incompleteness and verbosity.

### 5.2 Weighted Composite Scoring

Databricks published a methodology using a different decomposition: Correctness (60%), Comprehensiveness (20%), Readability (20%), scored on a 0–3 integer scale with LLM-as-judge. Their key findings:

- LLM-human exact agreement: >80%
- LLM-human within 1-score distance: >95%
- GPT-3.5 with few-shot examples: 10x cheaper, 3x faster than GPT-4
- GPT-3.5 without grading rubric: "completely unusable"
- "Evaluation results can't be transferred between use cases"

This illustrates a general pattern: answer-level evaluation benefits from task-specific decomposition rather than generic metrics.

### 5.3 The Problem with Answer Relevance as a Separate Dimension

In practice, answer relevance is hard to disentangle from correctness. An answer that is relevant but wrong (addresses the right topic, gives incorrect information) scores well on answer relevance and fails on faithfulness. Users experience it as simply "wrong." The decomposition creates a clean analytical framework that does not always map to user experience.

---

## 6. Failure Modes and Metric Coverage

Sections 2-5 introduced metrics for each evaluation surface of a RAG pipeline. A natural question follows: if I deploy these metrics, what failures will I catch, and what will slip through undetected? The following table maps common RAG failure modes to the metrics that detect them and, more importantly, the ones that do not.

**Failures that current metrics detect:**

| Failure Mode | Description | Detected By |
|---|---|---|
| **Irrelevant retrieval** | Retrieved chunks unrelated to query | Context Relevance, Precision@K |
| **Missed retrieval** | Relevant docs not retrieved | Recall@K, Context Recall |
| **Hallucination** | LLM invents facts not in context | Faithfulness, NLI |
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

Beyond blind spots, metrics can also be actively gamed. A system that copies context verbatim scores perfect faithfulness but produces useless answers. LLM judges prefer longer responses (Dubois et al., COLM 2024), so systems optimized against them drift toward verbosity. Optimizing for comprehensiveness produces padded answers that technically cover everything but bury the useful signal. Evaluation scores should be used as diagnostics for identifying where quality degrades, not as optimization targets.


---

## References

| Paper | Venue/Year | Key Contribution |
|---|---|---|
| Es et al., "RAGAS" | arXiv:2309.15217, 2023 | Reference-free RAG evaluation framework |
| Honovich et al., "TRUE" | arXiv:2204.04991, 2022 | NLI-based faithfulness benchmark |
| Thomas et al., "LLMs can Accurately Predict Searcher Preferences" | SIGIR 2024 | LLM-based NDCG correlates with human at tau > 0.9 |
| Upadhyay et al., "UMBRELA" | 2024 | Large-scale LLM assessor benchmark for TREC |
| Buckley et al., "Bias and the Limits of Pooling" | 2007 | Shallow pooling error analysis for NDCG |
| Dubois et al., "Length-Controlled AlpacaEval" | COLM 2024 | Verbosity debiasing (referenced in Goodhart section) |
