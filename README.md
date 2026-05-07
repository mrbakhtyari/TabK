<p align="center">
  <h1 align="center">TabK: Amortized Bayesian Estimation of the<br/>Number of Clusters in Tabular Data</h1>
</p>

<p align="center">
  <a href="https://anonymous.4open.science/r/TabK"><img alt="Anonymous Code" src="https://img.shields.io/badge/Code-Anonymous_4open-blue?style=flat-square"/></a>
  <a href="https://creativecommons.org/licenses/by-nc/4.0/"><img alt="License: CC BY-NC 4.0" src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg?style=flat-square"/></a>
  <img alt="Python 3.13+" src="https://img.shields.io/badge/Python-3.13%2B-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img alt="PyTorch 2.9+" src="https://img.shields.io/badge/PyTorch-2.9%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img alt="Conference" src="https://img.shields.io/badge/NeurIPS-2026_(Under Review)-8B5CF6?style=flat-square"/>
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

## Project Structure

```
TabK/
├── src/tabk/              # Core library
│   ├── architecture/            # Model, training, inference, heads
│   │   ├── model.py             # DoubleInvariantTransformer backbone
│   │   ├── heads/               # Task-specific prediction heads
│   │   │   ├── k_estimator.py   # DLDL, classification, focal, ordinal, regression
│   │   │   └── base.py          # Abstract head interface
│   │   ├── ablation_models.py   # Ablation variants (w/o QFE, PMA, col-attn)
│   │   ├── train.py             # K-fold training loop with AMP & gradient accumulation
│   │   ├── inference.py         # Ensemble inference (5-fold checkpoint averaging)
│   │   ├── dataset.py           # HDF5 DataLake → in-memory PyTorch Dataset
│   │   ├── config.py            # Dataclass-based configuration system
│   │   └── pipeline.py          # End-to-end training orchestration
│   │
│   ├── synthesis/               # Synthetic data generation (generative prior)
│   │   ├── strategies.py        # 5 geometric generators (see paper Section B)
│   │   ├── pipeline.py          # Parallel generation with timeout safety
│   │   ├── h5_builder.py        # Raw NPZ → unified HDF5 DataLake
│   │   └── registry.py          # Strategy registry & hyperparameter samplers
│   │
│   ├── baseline/                # Baseline method implementations
│   │   ├── k_estimation.py      # Unified k-selection evaluation pipeline
│   │   ├── base_algorithm.py    # K-Means, GMM, Spectral execution engine
│   │   ├── cvi_methods.py       # Silhouette, CH, DB, Dunn wrappers
│   │   ├── gap_statistic.py     # Gap statistic implementation
│   │   ├── imwkmeans_k.py       # Intelligent MWK-Means
│   │   └── x_means.py           # X-Means via pyclustering
│   │
│   └── utils/                   # Shared utilities
│       ├── plotting.py          # NeurIPS-style plotting & method registry
│       ├── preprocessing.py     # StandardScaler normalization
│       ├── metrics.py           # Dunn Index implementation
│       └── progress.py          # Terminal progress utilities
│
├── scripts/                     # Executable entry points
│   ├── train_TabK.py            # Training with full CLI (argparse)
│   ├── generate_datasets.py     # Synthetic prior generation
│   ├── build_h5_from_raw.py     # NPZ → HDF5 conversion
│   └── create_subsampled_h5.py  # Dataset scaling ablation helper
│
└── tests/                       # Unit tests (pytest)

```

---

## Installation

### Requirements

- **Python** ≥ 3.13
- **CUDA** ≥ 12.6 (optional, for GPU acceleration)
- **[uv](https://docs.astral.sh/uv/)** package manager

### Setup

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/TabK
cd TabK

# Install all dependencies
uv sync
```

---

## Run Inference

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
    --output-dir models/TabK \
    -v
```

---

## Reproducibility

- All experiments use a fixed random seed (42) with full deterministic settings
- Training uses 5-fold cross-validation
- The training run completes in approximately 37.6 hours on a single NVIDIA A100-SXM4-40GB
- Estimated carbon footprint: ~6.4 kg CO₂

---

## Citation

This section will be updated upon paper acceptance.

---

## License

This project is released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. It is intended for **educational and research purposes only**. Commercial use is strictly prohibited.
