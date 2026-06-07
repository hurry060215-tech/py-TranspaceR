# %% [markdown]
# # Notebook 2: Variogram Pipeline
#
# Demonstrates the FFT-based variogram computation for spatial gene selection.

# %%
import numpy as np
import pandas as pd
import transspacer as ts
from transspacer.variogram import get_variogram_map, get_isotropic_vario, process_gene

# %%
expr = pd.read_csv(
    "D:/test/TranspaceR-main/TranspaceR-main/Example_data/Expression_file_example.csv.gz",
    index_col=0
)
meta = pd.read_csv(
    "D:/test/TranspaceR-main/TranspaceR-main/Example_data/Meta_data_example.csv",
    index_col=0
)

# Use a small subset for speed
np.random.seed(42)
n_sub = 2000
idx = np.random.choice(expr.shape[0], n_sub, replace=False)
expr_sub = expr.values[idx].astype(float)
x = meta["cell_centroid_x"].values[idx]
y = meta["cell_centroid_y"].values[idx]

print(f"Subset: {n_sub} cells, {expr_sub.shape[1]} genes")

# %% [markdown]
# ## 1. Compute padding

# %%
n_pad = ts.pad_definitor(x, y)
print(f"Padding: {n_pad}")

# %% [markdown]
# ## 2. Variogram map for a single gene

# %%
# Rasterize gene expression into a grid
gene_expr = expr_sub[:, 0]  # first gene
breaks_x = np.arange(x.min(), x.max() + n_pad, n_pad)
breaks_y = np.arange(y.min(), y.max() + n_pad, n_pad)
ix = np.digitize(x, breaks_x) - 1
iy = np.digitize(y, breaks_y) - 1
nx = len(breaks_x) - 1
ny = len(breaks_y) - 1

raster = np.full((ny, nx), 0.0)
for k in range(len(gene_expr)):
    xi, yi = ix[k], iy[k]
    if 0 <= xi < nx and 0 <= yi < ny:
        raster[yi, xi] += gene_expr[k]

print(f"Raster shape: {raster.shape}")

# %%
# Compute variogram map
vmap = get_variogram_map(raster)
print(f"Variogram map shape: {vmap.shape}")

# %%
# Get isotropic variogram
r, v = get_isotropic_vario(vmap)
print(f"Isotropic variogram: {len(r)} distance bins")
print(f"Distances (first 10): {r[:10]}")
print(f"Variogram values (first 10): {v[:10]}")

# %% [markdown]
# ## 3. Process a single gene (variogram fitting)

# %%
result = process_gene(expr_sub[:, 0], n_pad, x, y)
print(f"Model: {result['Model']}")
print(f"Sill: {result['Sill']:.4f}")
print(f"Range: {result['Range']}")
print(f"R2: {result['R2']:.4f}")
print(f"AIC: {result['AIC']:.4f}")

# %% [markdown]
# ## 4. Full variogram computation (subset)

# %%
# Run on a small gene set for speed
gene_indices = list(range(10))
expr_small = expr_sub[:, gene_indices]
result = ts.compute_variogram(expr_small, x, y, automatic_pad=True, ncores=1)

print(f"Selected genes: {result['Selected_genes']}")
print(f"Parameters keys: {list(result['Parameters'].keys())}")
