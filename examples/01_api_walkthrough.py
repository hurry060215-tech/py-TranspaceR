# %% [markdown]
# # Notebook 1: py-TranspaceR API Walkthrough
#
# This notebook demonstrates the core API of py-TranspaceR, the Python port of TranspaceR.

# %%
import numpy as np
import pandas as pd
import transspacer as ts

print(f"py-TranspaceR version: {ts.__version__}")

# %% [markdown]
# ## Load example data

# %%
expr = pd.read_csv(
    "D:/test/TranspaceR-main/TranspaceR-main/Example_data/Expression_file_example.csv.gz",
    index_col=0
)
meta = pd.read_csv(
    "D:/test/TranspaceR-main/TranspaceR-main/Example_data/Meta_data_example.csv",
    index_col=0
)

print(f"Expression: {expr.shape[0]} cells x {expr.shape[1]} genes")
print(f"Metadata columns: {list(meta.columns)}")

# %% [markdown]
# ## 1. Cell-size normalization

# %%
area = meta["Area"].values
expr_arr = expr.values.astype(float)
normed = ts.c_normalisation(expr_arr, area)
print(f"Normalized shape: {normed.shape}")
print(f"Col sums (first 5): {np.nansum(normed, axis=0)[:5]}")

# %% [markdown]
# ## 2. Otsu thresholding

# %%
lib_size = expr_arr.sum(axis=1)
threshold = ts.otsu_thresholding(np.log10(lib_size + 1))
print(f"Otsu threshold (log10): {threshold:.4f}")
print(f"Otsu threshold (raw): {10**threshold:.0f}")

# %% [markdown]
# ## 3. FFT utilities

# %%
n_pad = ts.pad_definitor(meta["cell_centroid_x"].values, meta["cell_centroid_y"].values)
print(f"Auto padding: {n_pad}")

# Demonstrate fftshift roundtrip
m = np.random.rand(4, 4)
shifted = ts.fftshift(m)
unshifted = ts.ifftshift(shifted)
print(f"fftshift roundtrip OK: {np.allclose(m, unshifted)}")

# %% [markdown]
# ## 4. Sparse matrix utilities

# %%
from scipy.sparse import csc_matrix
sp = csc_matrix(expr_arr[:1000])
cv = ts.colvars_sparse(sp)
print(f"Column variance shape: {cv.shape}")
print(f"Top 5 most variable genes: {np.argsort(-cv)[:5]}")

# %% [markdown]
# ## 5. Gene selection utilities

# %%
# calculate_log2fc with synthetic clusters
np.random.seed(42)
clusters = np.random.choice(["A", "B", "C"], size=expr_arr.shape[0])
lfc = ts.calculate_log2fc(expr_arr, 0, clusters)
print(f"Log2FC for gene 0: {lfc}")

# select_genes
gene_sets = [["GENE1", "GENE2", "GENE3"], ["GENE2", "GENE4"]]
shared = ts.select_genes(gene_sets)
print(f"Shared genes: {shared}")
