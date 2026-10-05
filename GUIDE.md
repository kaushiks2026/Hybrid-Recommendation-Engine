# 📘 Comprehensive User & Testing Guide: Hybrid Recommendation Engine

> **Project**: AIML-01 · Cold-Start–Resilient Hybrid Recommender  
> **Dataset**: MovieLens 1M (1,000,209 ratings · 6,040 users · 3,706 movies)  
> **Key Result**: Blended hybrid achieves **+89% NDCG@10** and **+115% Recall@10** over pure Collaborative Filtering on cold-start users.

---

## 🧭 Table of Contents
1. [Plain-English Overview (What is this?)](#1-plain-english-overview)
2. [What is Happening Behind the Scenes?](#2-what-is-happening-behind-the-scenes)
3. [How to Test & Use the Project](#3-how-to-test--use-the-project)
4. [Understanding the Evaluation Results](#4-understanding-the-evaluation-results)
5. [Failure Cases & Limitations](#5-failure-cases--limitations)
6. [Project File Map](#6-project-file-map)

---

## 1. Plain-English Overview

### The Streaming Platform Problem
Imagine you run a movie streaming service like Netflix.
- **Collaborative Filtering (CF)**: Learns from crowds ("People who loved *The Matrix* and *Inception* also loved *Interstellar*"). This is powerful when users have watched dozens of movies.
- **The Cold-Start Crash**: When a brand-new user signs up (or a new movie is added), CF has zero ratings to work with. It either crashes or guesses the generic average.
- **Content-Based Filtering (CB)**: Ignores other users and looks at movie DNA (genres like "Sci-Fi | Action"). It works with just 1 rating, but misses out on crowd wisdom.

### Our Solution: The Adaptive Hybrid
Instead of a rigid 50/50 mix, our engine uses an **adaptive dial** $\alpha(u)$:

$$\alpha(u) = \min\left(\frac{N_{\text{ratings}}(u)}{20}, 1.0\right)$$

$$\text{Hybrid Score} = \alpha(u) \cdot \text{CF}_{\text{norm}} + (1 - \alpha(u)) \cdot \text{CB}_{\text{norm}}$$

- **Cold User** (e.g., 2 ratings): $\alpha = 0.10$ → **90% Content-Based**, 10% Collaborative.
- **Medium User** (e.g., 10 ratings): $\alpha = 0.50$ → **50/50 balanced blend**.
- **Warm User** (e.g., 20+ ratings): $\alpha = 1.00$ → **100% Collaborative Filtering**.

---

## 2. What is Happening Behind the Scenes?

When you run the pipeline, it executes 8 distinct phases:

```
[Phase 1: Data Loading & Slicing]
  ├─ 1,000,209 ratings loaded from MovieLens 1M
  ├─ Leave-last-out split: Hold out the latest rating per user for test
  └─ Explicit Cold-Start Slice: 20% of users (1,208 users) masked to <5 ratings

[Phase 2: Collaborative Filtering (SVD)]
  ├─ Simon Funk SVD with user and item biases trained via regularized SGD
  └─ Learns 100-dimensional latent taste vectors for users and movies

[Phase 3: Content-Based Component (TF-IDF)]
  ├─ Converts 18 movie genres into TF-IDF vectors
  └─ Vectorized user profiles: R @ item_vectors / weights (instant sparse BLAS)

[Phase 4: Adaptive Hybrid Blending]
  ├─ Min-Max normalizes CF (1-5 scale) and CB (0-1 cosine similarity)
  └─ Dynamically weights scores according to user's interaction count

[Phase 5: Recommendation Generation]
  └─ Fast vectorized top-20 ranking across Popularity, CF, CB, and Hybrid

[Phase 6: Ranking Evaluation]
  └─ Precision@K, Recall@K, NDCG@K (K=5, 10, 20) across All, Cold, and Warm slices

[Phase 7: Failure Analysis]
  └─ Analyzes complete misses (NDCG=0), unrecommended niche items, popularity bias

[Phase 8: Diversity Re-Ranking (MMR Bonus)]
  └─ Maximal Marginal Relevance (λ=0.7) penalizes redundant recommendations
  └─ Measures Intra-List Diversity (ILD) improvement
```

---

## 3. How to Test & Use the Project

### A. Quick Interactive Demo (Inspect Any User)
We built an interactive test tool [`scripts/interactive_demo.py`](scripts/interactive_demo.py) that loads the models and inspects recommendations in real time.

#### 1. Compare a Cold-Start User vs. a Warm-Start User Side-by-Side:
```bash
python scripts/interactive_demo.py --cold-demo
```
**What you will see**:
- **Cold User (User 26, 2 ratings)**: $\alpha = 0.10$ → CB dominates (recommends Comedy/Drama matching their exact 2 ratings).
- **Warm User (User 1, 52 ratings)**: $\alpha = 1.00$ → CF takes over (recommends classic masterworks like *The Godfather* and *Rear Window* based on community patterns).

#### 2. Query Recommendations for Any Specific User (1 to 6040):
```bash
python scripts/interactive_demo.py --user 42 -k 10
```
Outputs:
- User training history count and cold-start classification
- Top-5 Collaborative Filtering picks
- Top-5 Content-Based picks
- Top-10 Adaptive Blended Hybrid picks
- Top-10 Diversity Re-Ranked picks via MMR
- Intra-List Diversity (ILD) metric lift

---

### B. Run the Full End-to-End Pipeline
To re-run training, evaluation across all 6,040 users, failure analysis, and chart generation:
```bash
python scripts/run_pipeline.py
```
This produces all outputs in [`results/`](results):
- `final_evaluation.csv`: Full metric table
- `model_comparison.png`: Comparison bar charts
- `alpha_distribution.png`: Blending weight histograms
- `failure_analysis.txt`: Breakdown of failure patterns
- `diversity_metrics.json`: Intra-List Diversity before and after MMR

---

### C. View the Notebook Write-Up
The project includes a formatted Jupyter notebook write-up:
[`notebooks/writeup.ipynb`](notebooks/writeup.ipynb)

To open in Jupyter:
```bash
jupyter notebook notebooks/writeup.ipynb
```
*(All plots, tables, and metric breakdowns are pre-executed and visible immediately).*

---

## 4. Understanding the Evaluation Results

Here are the benchmark results evaluated across all 6,040 users on the held-out test ratings:

### Primary Ranking Metrics @ K = 10

| Model | Slice | P@10 | Recall@10 | NDCG@10 | Cold Lift over CF |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Popularity Baseline** | All | 0.0031 | 0.0306 | 0.0150 | — |
| **CF-only (SVD)** | All | 0.0017 | 0.0171 | 0.0092 | — |
| **CB-only (TF-IDF)** | All | 0.0008 | 0.0079 | 0.0038 | — |
| **Hybrid (Adaptive)** | **All** | **0.0020** | **0.0195** | **0.0103** | — |
| | | | | | |
| **CF-only (SVD)** | **Cold (<5)** | 0.0011 | 0.0108 | 0.0062 | Baseline |
| **CB-only (TF-IDF)** | **Cold (<5)** | 0.0016 | 0.0157 | 0.0068 | +9.7% |
| **Hybrid (Adaptive)** | **Cold (<5)** | **0.0023** | **0.0232** | **0.0117** | **+88.7% NDCG** |
| | | | | | |
| **CF-only (SVD)** | **Warm (≥5)** | 0.0019 | 0.0186 | 0.0099 | — |
| **CB-only (TF-IDF)** | **Warm (≥5)** | 0.0006 | 0.0060 | 0.0031 | — |
| **Hybrid (Adaptive)** | **Warm (≥5)** | **0.0019** | **0.0186** | **0.0099** | Maintains CF accuracy |

### Key Takeaways
1. **Cold-Start Lift**: On the 1,208 cold-start users, the adaptive hybrid achieves **0.0117 NDCG@10** vs **0.0062 for CF-only** (+88.7% lift) and **0.0232 Recall@10** vs **0.0108 for CF-only** (+114.8% lift).
2. **Zero Degradation on Warm Users**: On warm users, the hybrid achieves the exact same NDCG as pure CF (0.0099) because $\alpha$ shifts smoothly to 1.0, preserving peak performance.
3. **Diversity Lift (Bonus)**: Maximal Marginal Relevance (MMR with $\lambda=0.7$) increased Intra-List Diversity (ILD) from **0.6415 to 0.7180 (+7.65%)**, preventing lists dominated by single genres.

---

## 5. Failure Cases & Limitations

Extracted from the automated failure analysis in [`results/failure_analysis.txt`](results/failure_analysis.txt):

1. **Complete Misses (NDCG=0)**: In leave-last-out evaluation on a 3,700-item catalog, predicting the exact single hidden movie in the top-10 is challenging. 19.9% of misses are cold users who provided too few signals.
2. **Unrecommended Niche Items**: 554 items never appeared in any user's top-20. Movies like *Full Tilt Boogie (1997)* had only 4 training ratings, giving SVD little opportunity to learn their latent factors.
3. **Popularity Bias**: 27.8% of hybrid recommendations come from the top 100 popular movies. The MMR re-ranking step directly mitigates this.
4. **Genre Monotonicity**: 633 users received top-10 lists spanning $\le 2$ genres before MMR re-ranking was applied.
5. **Contradictory Taste Blur**: Users who rated 10+ divergent genres highly caused TF-IDF profile vectors to average out into diffuse, non-specific blobs.

---

## 6. Project File Map

```
HMR/
├── README.md                      <- Project documentation & reproduction instructions
├── GUIDE.md                       <- THIS FILE (User & Testing Guide)
├── requirements.txt               <- Python dependencies (Windows-safe)
├── notebooks/
│   └── writeup.ipynb              <- Executed write-up with charts & tables
├── src/
│   ├── data_loader.py             <- Download & parse MovieLens 1M .dat files
│   ├── preprocessing.py           <- Leave-last-out split & cold-start slice construction
│   ├── collaborative.py           <- SVD matrix factorization with fast vectorization
│   ├── content_based.py           <- TF-IDF item representations & sparse user profiles
│   ├── hybrid.py                  <- Adaptive blending function α(u)
│   ├── reranker.py                <- Maximal Marginal Relevance (MMR) & ILD
│   ├── evaluate.py                <- Ranking metrics (Precision, Recall, NDCG)
│   └── utils.py                   <- Configuration, logging, paths
├── scripts/
│   ├── download_data.py           <- Standalone downloader
│   ├── run_pipeline.py            <- End-to-end training & evaluation
│   └── interactive_demo.py        <- Interactive user query & cold-demo CLI
└── results/
    ├── evaluation_metrics.csv     <- Baseline and hybrid metric numbers
    ├── final_evaluation.csv       <- Complete metrics including MMR
    ├── model_comparison.png       <- Visual comparison chart
    ├── alpha_distribution.png     <- Blending weight distribution
    ├── failure_analysis.txt       <- Concrete failure pattern analysis
    └── diversity_metrics.json     <- Intra-list diversity metrics
```
