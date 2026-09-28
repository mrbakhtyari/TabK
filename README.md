<p align="center">
  <h1 align="center">TabK: Amortized Bayesian Estimation of the<br/>Number of Clusters in Tabular Data</h1>
</p>

<p align="center">
  <a href="https://github.com/mrbakhtyari/TabK/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/mrbakhtyari/TabK/ci.yml?branch=main&style=flat-square&label=CI"/></a>
  <a href="LICENSE"><img alt="License: CC BY-NC 4.0" src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg?style=flat-square"/></a>
  <img alt="Python 3.13+" src="https://img.shields.io/badge/Python-3.13%2B-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img alt="PyTorch 2.9+" src="https://img.shields.io/badge/PyTorch-2.9%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img alt="Conference" src="https://img.shields.io/badge/NeurIPS_2026-Accepted-8B5CF6?style=flat-square"/>
</p>

<p align="center">
  <em>A permutation-invariant Transformer architecture for zero-shot cluster cardinality estimation in tabular data.</em>
</p>

---

## Overview

**TabK** is a set-of-sets Transformer architecture that estimates the number of clusters *k* in a single forward pass, without ever running a clustering algorithm. By amortizing Bayesian inference over a diverse synthetic prior and framing the prediction as an ordinal-aware distribution learning task (DLDL), TabK replaces the expensive iterative grid searches of traditional methods with sub-second inference.

### Key Results (50 OpenML Benchmarks)

| Metric | TabK | Best Competitor |
|:---|:---:|:---:|
| **MAE ↓** | **1.04** | 2.24 (TabClustPFN) |
| **Exact Match ↑** | **50.0%** | 42.0% (Silhouette) |
| **±1 Tolerance ↑** | **80.0%** | 62.0% (Calinski-Harabasz) |
| **ARI ↑** | **0.232** | 0.228 (X-Means) |
| **Latency ↓** | **0.67s** | 1.50s (TabClustPFN) |

> TabK achieves the best position on the Pareto frontier, providing predictions of *k* in less than one second — at least **2× faster** than TabClustPFN and **25× faster** than classical CVIs.

---

## Installation

### Requirements

- **Python** ≥ 3.13
- **CUDA** ≥ 12.6 (optional, for GPU acceleration)
- **[uv](https://docs.astral.sh/uv/)** package manager

### Setup

```bash
git clone https://github.com/mrbakhtyari/TabK.git
cd TabK
uv sync
```

---

## Inference

The pretrained model (a 5-fold checkpoint ensemble) ships with the repository in `models/TabK`.

```python
from sklearn.datasets import load_iris
from tabk.architecture import load_inference_context, predict_single
from tabk.utils import apply_standard_scaling

# Load the Iris dataset
X, _ = load_iris(return_X_y=True)
X_scaled = apply_standard_scaling(X)

# Load ensemble (5-fold checkpoints)
ctx = load_inference_context("models/TabK")

# Predict k in a single forward pass
predicted_k = predict_single(ctx, X_scaled)
print(f"Predicted number of clusters: {predicted_k}")  # expected: 3
```

The same example runs with `uv run python -m tabk.main`.

The pretrained model predicts *k* ∈ {2, …, 15} and was trained on tables with 100–2,500 rows and 2–200 features.

---

## Training

### 1. Generate Synthetic Training Data

```bash
uv run scripts/generate_datasets.py \
    --n-configs 40000 \
    --k-min 2 --k-max 15 \
    --n-low 100 --n-high 2500 \
    --d-low 2 --d-high 200 \
    --output-dir datasets/synthetic_n40000_r1
```

### 2. Build the HDF5 DataLake

```bash
uv run scripts/build_h5_from_raw.py \
    --data-dir datasets/synthetic_n40000_r1 \
    --test-ratio 0.1 -vv
```

### 3. Train TabK

```bash
uv run scripts/train_TabK.py \
    --d-model 256 \
    --n-head 8 \
    --n-layers 4 \
    --num-bins 50 \
    --k-head-mode distribution \
    --sigma 0.5 \
    --lr 2e-4 \
    --epochs 20 \
    --k-folds 5 \
    --data-dir datasets \
    --h5-filename synthetic_n40000_r1.h5 \
    --output-dir models/TabK_retrained \
    -v
```

The trained ensemble can then be loaded with `load_inference_context("models/TabK_retrained")`.

### Reproducibility

- All experiments use a fixed random seed (42) with full deterministic settings
- Training uses 5-fold cross-validation
- The training run completes in approximately 37.6 hours on a single NVIDIA A100-SXM4-40GB
- Estimated carbon footprint: ~6.4 kg CO₂

---

## Project Structure

```
TabK/
├── src/tabk/
│   ├── architecture/            # Model, training, inference, heads
│   │   ├── model.py             # DoubleInvariantTransformer backbone
│   │   ├── heads/               # Prediction heads (DLDL, classification, focal, ordinal, regression)
│   │   ├── ablation_models.py   # Ablation variants (w/o QFE, PMA, col-attn)
│   │   ├── train.py             # K-fold training loop with AMP & gradient accumulation
│   │   ├── inference.py         # Ensemble inference (5-fold checkpoint averaging)
│   │   ├── dataset.py           # HDF5 DataLake → in-memory PyTorch Dataset
│   │   ├── config.py            # Dataclass-based configuration
│   │   └── pipeline.py          # End-to-end training orchestration
│   │
│   ├── synthesis/               # Synthetic data generation (generative prior)
│   │   ├── strategies.py        # Geometric generators (see paper Section B)
│   │   ├── pipeline.py          # Parallel generation with timeout safety
│   │   ├── h5_builder.py        # Raw NPZ → unified HDF5 DataLake
│   │   └── registry.py          # Strategy registry & hyperparameter samplers
│   │
│   └── utils/                   # Preprocessing, logging and progress helpers
│
├── scripts/                     # Command-line entry points
│   ├── generate_datasets.py     # Synthetic prior generation
│   ├── build_h5_from_raw.py     # NPZ → HDF5 conversion
│   ├── train_TabK.py            # Training
│   └── create_subsampled_h5.py  # Training-set size ablation helper
│
├── models/TabK/                 # Pretrained checkpoints
└── tests/                       # Unit tests (pytest)
```

---

## Citation

TabK has been accepted at NeurIPS 2026. The paper link and citation will be added here once the paper is published.

---

## License

This project is released under the [CC BY-NC 4.0](LICENSE) license. It is intended for **educational and research purposes only**; commercial use is not permitted.
