# %% [markdown]
# # Notebook 3: Spatial Statistics
#
# Demonstrates Geary's C, excess variance ratio, and excess zero score.

# %%
import numpy as np
import pandas as pd
import transspacer as ts
from transspacer.spatial_stats import geary_c_score, excess_variance_ratio_nb, excess_zero_score_nb

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
n_sub = 500
idx = np.random.choice(expr.shape[0], n_sub, replace=False)
expr_sub = expr.values[idx].astype(float)
coords = np.column_stack([
    meta["cell_centroid_x"].values[idx],
    meta["cell_centroid_y"].values[idx]
])

print(f"Subset: {n_sub} cells, {expr_sub.shape[1]} genes")

# %% [markdown]
# ## 1. Geary's C Score
#
# Geary's C measures spatial autocorrelation. Values < 1 indicate positive spatial autocorrelation.

# %%
# Run on first 20 genes for speed
result = geary_c_score(expr_sub[:, :20], coords, pvalue_threshold=0.05)
print(f"Geary C values (1/C, first 5): {result['Values'][:5]}")
print(f"Selected genes: {len(result['Selected_genes'])}")
print(f"P-values (first 5): {result['P_values'][:5]}")

# %% [markdown]
# ## 2. Excess Variance Ratio (NB model)
#
# Identifies genes with variance exceeding the NB null model expectation.

# %%
result_var = excess_variance_ratio_nb(expr_sub[:, :20], p_value_threshold=0.01, ratio_threshold=1.5)
print(f"Selected genes (excess variance): {len(result_var['Selected_genes'])}")
print(f"Excess variance ratios (first 5): {result_var['Excess_variance_ratio'][:5]}")

# %% [markdown]
# ## 3. Excess Zero Score (NB model)
#
# Identifies genes with more zeros than expected under the NB model.

# %%
result_zero = excess_zero_score_nb(expr_sub[:, :20], p_value_threshold=0.01, delta_threshold=0.01)
print(f"Selected genes (excess zeros): {len(result_zero['Selected_genes'])}")
print(f"Excess zero scores (first 5): {result_zero['Excess_zero_score'][:5]}")
