# 🎬 Hybrid Recommendation Engine with Cold-Start Handling

A hybrid recommender that blends **Collaborative Filtering** (SVD matrix factorization) with **Content-Based Filtering** (TF-IDF on genre metadata), featuring an adaptive blending strategy that gracefully handles cold-start users and items.

Built for the **AIML-01** project: demonstrating that a hybrid approach degrades gracefully when interaction history is sparse, rather than breaking entirely.

---

## 🏗️ Architecture

```
MovieLens 1M Data
       │
       ├──► Collaborative Filter (SVD) ──► CF Scores ─┐
       │                                               │    α(u) = f(n_ratings)
       └──► Content-Based (TF-IDF)     ──► CB Scores ─┼──► Hybrid Score ──► [MMR Re-Rank] ──► Top-K Recs
                                                       │
                                              Adaptive Blending
                                         (α→1 for warm, α→0 for cold)
```

### Key Innovation: Adaptive Blending

Instead of a fixed 50/50 mix, the blending weight `α` is a function of each user's interaction count:

```
α(u) = min(n_ratings(u) / threshold, 1.0)
hybrid_score = α · CF_normalized + (1 - α) · CB_normalized
```

- **Cold users (0 ratings)**: α = 0 → 100% content-based
- **Medium users (10 ratings, threshold=20)**: α = 0.5 → 50/50
- **Warm users (20+ ratings)**: α = 1.0 → 100% collaborative

---

## 📁 Project Structure

```
HMR/
├── README.md                     ← High-level summary & reproduction
├── GUIDE.md                      ← Detailed User & Testing Guide (with analogies & metrics)
├── requirements.txt              ← Python dependencies
├── data/                         ← MovieLens data (downloaded at runtime)
├── notebooks/
│   └── writeup.ipynb             ← Project write-up with results
├── src/
│   ├── __init__.py
│   ├── data_loader.py            ← Download & load MovieLens 1M
│   ├── preprocessing.py          ← Train/test split, cold-start slicing
│   ├── collaborative.py          ← SVD model (Surprise library)
│   ├── content_based.py          ← TF-IDF genre-based recommender
│   ├── hybrid.py                 ← Adaptive blending logic
│   ├── reranker.py               ← (Bonus) MMR diversity re-ranker
│   ├── evaluate.py               ← Precision@K, Recall@K, NDCG@K
│   └── utils.py                  ← Config, logging, helpers
├── scripts/
│   ├── download_data.py          ← Standalone data download
│   ├── run_pipeline.py           ← End-to-end pipeline
│   └── interactive_demo.py       ← Interactive query & cold vs warm demo
└── results/                      ← Saved metrics, plots, reports
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/HMR.git
cd HMR

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download Data

The dataset is automatically downloaded when you run the pipeline. To download manually:

```bash
python scripts/download_data.py
```

This fetches **MovieLens 1M** (~6MB compressed) from GroupLens.

---

## ▶️ How to Reproduce Results

### Run the Full Pipeline

```bash
python scripts/run_pipeline.py
```

This single command runs the **entire pipeline**:
1. **Downloads** MovieLens 1M (if not already present)
2. **Loads & preprocesses** data (leave-last-out split, cold-start slicing)
3. **Trains** SVD collaborative filter
4. **Builds** TF-IDF content-based model
5. **Generates** hybrid recommendations with adaptive blending
6. **Evaluates** using Precision@K, Recall@K, NDCG@K for All/Cold/Warm slices
7. **Compares** against baselines (Popularity, CF-only, CB-only)
8. **Analyzes** failure cases
9. **Re-ranks** with MMR for diversity (bonus)

### Quick Interactive Demo (Test Any User)

Test how the hybrid engine behaves for cold vs. warm users or query specific user IDs:

```bash
# Compare a cold user (<5 ratings) with a warm user (>=5 ratings) side-by-side
python scripts/interactive_demo.py --cold-demo

# Query recommendations for a specific user ID
python scripts/interactive_demo.py --user 42 -k 10
```

### Output

Results are saved to `results/`:

| File | Description |
|---|---|
| `evaluation_metrics.csv` | All model metrics (P@K, R@K, NDCG@K) |
| `final_evaluation.csv` | Including Hybrid+MMR results |
| `model_comparison.png` | Bar chart comparing all models |
| `alpha_distribution.png` | Blending weight analysis |
| `failure_analysis.txt` | Where the hybrid still fails |
| `diversity_metrics.json` | ILD before/after MMR |

---

## 📊 Evaluation Methodology

### Metrics

| Metric | What it measures |
|---|---|
| **Precision@K** | Of the K items recommended, how many were relevant? |
| **Recall@K** | Of all relevant items, how many appeared in top-K? |
| **NDCG@K** | Were the best items near the *top* of the list? |

### Slicing

- **Warm users**: ≥5 training interactions (mainstream case)
- **Cold users**: <5 training interactions (cold-start case)
- An item rated ≥4 stars is considered "relevant"

### Train/Test Split

**Leave-last-out**: For each user, their most recent rating is the test item; all prior ratings are training. This mimics real-world sequential prediction.

---

## 🎯 Models Compared

| Model | Description |
|---|---|
| **Popularity** | Baseline — recommends most-watched movies |
| **CF-only (SVD)** | Matrix factorization, no content info |
| **CB-only (TF-IDF)** | Genre-based similarity, no collaborative signal |
| **Hybrid** | Adaptive blend of CF + CB |
| **Hybrid + MMR** | Hybrid with diversity re-ranking |

---

## 🔧 Configuration

Key parameters can be adjusted in `src/utils.py` → `CONFIG` dict:

| Parameter | Default | Description |
|---|---|---|
| `cold_start_threshold` | 5 | Users/items with < this many interactions are "cold" |
| `svd_n_factors` | 100 | Latent dimensions for SVD |
| `blend_threshold` | 20 | # ratings at which CF gets full weight |
| `relevance_threshold` | 4.0 | Rating ≥ this is "relevant" for metrics |
| `mmr_lambda` | 0.7 | Relevance vs. diversity trade-off |

---

## 📝 License

This project is for educational purposes (AIML-01 assignment).

## 📖 References

- [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/) — GroupLens Research
- [Surprise Library](https://surpriselib.com/) — SVD implementation
- Harper, F.M. & Konstan, J.A. (2015). The MovieLens Datasets. *ACM Transactions on Interactive Intelligent Systems*.
>>>>>>> eb24169 (Initial commit: Cold-Start-Resilient Hybrid Recommendation Engine)
