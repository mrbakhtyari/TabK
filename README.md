<p align="center">
  <h1 align="center">TabK: Amortized Bayesian Estimation of the<br/>Number of Clusters in Tabular Data</h1>
</p>

<p align="center">
  <a href="https://huggingface.co/mrbakhtyari/TabK"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-mrbakhtyari%2FTabK-FFD21E?style=flat-square"/></a>
  <a href="https://github.com/mrbakhtyari/TabK/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/mrbakhtyari/TabK/ci.yml?branch=main&style=flat-square&label=CI"/></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square"/></a>
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
git clone --depth 1 https://github.com/mrbakhtyari/TabK.git
cd TabK
uv sync
```

---

## Inference

The pretrained model (a 5-fold checkpoint ensemble) is hosted on the [Hugging Face Hub](https://huggingface.co/mrbakhtyari/TabK) and is downloaded automatically on first use.

```python
from sklearn.datasets import load_iris
from tabk import TabK

X, _ = load_iris(return_X_y=True)

model = TabK.from_pretrained()
print(model.predict(X))        # 3
print(model.predict_proba(X))  # probability of each k in model.k_values
```

Or from the command line, for a CSV file with one row per sample:

```bash
uv run tabk predict data.csv
```

Features are standardized automatically. The pretrained model predicts *k* ∈ {2, …, 15} and was trained on tables with 100–2,500 rows and 2–200 features. Inference uses all input rows.

---

## Training

```bash
uv run tabk generate   # writes datasets/synthetic/datalake.h5
uv run tabk train      # 5-fold ensemble           -> models/TabK_retrained
```

The defaults reproduce the paper's setup; run `uv run tabk <command> --help` for all options. Use the trained model with `TabK.from_pretrained("models/TabK_retrained")` or `tabk predict data.csv --model models/TabK_retrained`.

To publish it in the Hugging Face Hub format (safetensors weights plus `config.json`), run `uv run tabk export`.

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
│   ├── cli.py                   # `tabk` command: predict, generate, train, export
│   ├── architecture/            # Model, training, inference
│   │   ├── model.py             # DoubleInvariantTransformer backbone
│   │   ├── head.py              # DLDL head: distribution over k
│   │   ├── inference.py         # TabK: ensemble loading and prediction
│   │   ├── train.py             # K-fold training loop with AMP & gradient accumulation
│   │   ├── pipeline.py          # End-to-end training orchestration
│   │   ├── dataset.py           # HDF5 DataLake → in-memory PyTorch Dataset
│   │   ├── config.py            # Dataclass-based configuration
│   │   └── export.py            # Conversion to the Hugging Face Hub format
│   │
│   ├── synthesis/               # Synthetic data generation (generative prior)
│   │   ├── config.py            # Typed sampling priors and validation
│   │   ├── sampling.py          # Hyperparameter samplers
│   │   ├── strategies.py        # Geometric generators (see paper Section B)
│   │   ├── pipeline.py          # Parallel generation with timeout safety
│   │   ├── writer.py            # HDF5 dataset writer
│   │   └── registry.py          # Strategy registry & hyperparameter samplers
│   │
│   └── utils/                   # Preprocessing, logging and progress helpers
│
└── tests/                       # Unit tests (pytest)
```

---

## Citation

TabK has been accepted at NeurIPS 2026. The paper link and citation will be added here once the paper is published.

---

## License

This project is released under the [MIT License](LICENSE).
