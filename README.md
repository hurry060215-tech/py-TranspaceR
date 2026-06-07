# py-TranspaceR

[![PyPI version](https://img.shields.io/pypi/v/py-TranspaceR.svg)](https://pypi.org/project/py-TranspaceR/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-29%20passed-brightgreen.svg)](#testing)
[![Speedup](https://img.shields.io/badge/Speedup-17.7x-orange.svg)](#speed-benchmark)

Python port of [TranspaceR](https://github.com/TranspaceR/TranspaceR) — Statistical analysis of Spatial transcriptomic data.

## Correlation Benchmark (Python vs R)

| Function | Pearson r | Max Abs Error |
|---|---|---|
| `C_normalisation` | 1.000000 | 4.10e-05 |
| `Otsu_thresholding` | — | 3.54e-05 |
| `colvars_sparse` | 1.000000 | 4.40e-05 |
| `Get_variogram_map` | Deterministic match | 0 |
| `Get_isotropic_vario` | 1.000000 | 0 |

All outputs are highly consistent with R references, with errors within floating-point precision.

## Speed Benchmark (39,047 cells x 539 genes)

| Function | R Time | Python Time | Speedup |
|---|---|---|---|
| `C_normalisation` | 1.47s | 0.157s | 9.4x |
| `Otsu_thresholding` | 0.31s | 0.031s | 10.0x |
| `colvars_sparse` | 1.72s | 0.011s | 156x |
| `Get_variogram_map` | 0.02s | 0.0004s | 50x |
| `Get_isotropic_vario` | 0.01s | 0.0004s | 25x |
| **Total** | **3.53s** | **0.20s** | **17.7x** |

### Why faster

- NumPy/SciPy compiled C backend vs R interpreted execution
- Direct CSC sparse matrix memory layout access
- Broadcasting replaces R's row-wise `apply` loops

## Installation

```bash
pip install py-TranspaceR
```

With optional dependencies:

```bash
pip install "py-TranspaceR[all]"
```

## Quick Start

```python
import transspacer as ts
import numpy as np
import pandas as pd

# Load data
expr = pd.read_csv("Expression_file.csv.gz", index_col=0)
meta = pd.read_csv("Meta_data.csv", index_col=0)

# Cell-size normalisation
normed = ts.c_normalisation(expr.values.astype(float), meta["Area"].values)

# Otsu thresholding
threshold = ts.otsu_thresholding(np.log10(expr.values.sum(axis=1) + 1))

# Variogram analysis
result = ts.compute_variogram(normed, meta["cell_centroid_x"].values,
                               meta["cell_centroid_y"].values)

# Geary's C spatial autocorrelation
gc = ts.geary_c_score(normed, coords, pvalue_threshold=0.01)

# Clustering
labels = ts.cell_clustering_function(pca_data, K=10, resolution=1.0)
```

## Modules

| Module | Description |
|---|---|
| `fft_utils` | `fftshift`, `ifftshift`, `pad_definitor` |
| `normalization` | `C_normalisation` cell-size normalisation |
| `sparse_utils` | Sparse matrix column variance, group aggregation |
| `variogram` | FFT variogram map, variogram model fitting |
| `spatial_stats` | Geary's C, NB excess variance / excess zero score |
| `clustering` | KNN + Leiden/Louvain clustering, UMAP |
| `gene_selection` | `log2FC`, gene set union |
| `qc` | Otsu thresholding, QC gene filtering |
| `plotting` | Spatial visualization, heatmaps, UMAP plots |

## Testing

```bash
pytest tests/ -q
# 29 passed
```

## Dependencies

**Core:** `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`

**Optional:** `umap-learn` (UMAP), `python-igraph` + `leidenalg` (Leiden clustering), `statsmodels` (FDR correction)

## License

MIT
