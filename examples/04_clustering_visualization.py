# %% [markdown]
# # Notebook 4: Clustering and Visualization
#
# Demonstrates clustering pipeline and spatial visualization.

# %%
import numpy as np
import pandas as pd
import transspacer as ts
from transspacer.clustering import cell_clustering_function, clusters_maker
from transspacer.plotting import plot_fov, save_umap

# %%
expr = pd.read_csv(
    "D:/test/TranspaceR-main/TranspaceR-main/Example_data/Expression_file_example.csv.gz",
    index_col=0
)
meta = pd.read_csv(
    "D:/test/TranspaceR-main/TranspaceR-main/Example_data/Meta_data_example.csv",
    index_col=0
)

# Use a small subset
np.random.seed(42)
n_sub = 1000
idx = np.random.choice(expr.shape[0], n_sub, replace=False)
expr_sub = expr.values[idx].astype(float)
x = meta["cell_centroid_x"].values[idx]
y = meta["cell_centroid_y"].values[idx]

print(f"Subset: {n_sub} cells, {expr_sub.shape[1]} genes")

# %% [markdown]
# ## 1. PCA + Clustering

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=20, random_state=42)
pca_data = pca.fit_transform(expr_sub)
print(f"PCA shape: {pca_data.shape}")

# %%
# KNN + Louvain clustering
labels = ts.cell_clustering_function(pca_data, K=10, resolution=1.0)
n_clusters = len(np.unique(labels))
print(f"Number of clusters: {n_clusters}")
print(f"Cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")

# %% [markdown]
# ## 2. UMAP embedding

# %%
umap_coords = ts.umap_maker(pca_data, n_components=2, random_state=42)
print(f"UMAP shape: {umap_coords.shape}")

# %% [markdown]
# ## 3. Spatial visualization

# %%
# Plot cells colored by cluster
plot_fov(x, y, labels, output_path="examples/fov_clusters.png", title="Spatial Clusters")
print("Saved fov_clusters.png")

# %% [markdown]
# ## 4. UMAP visualization

# %%
save_umap(umap_coords, labels, output_path="examples/umap_clusters.png", title="UMAP Clusters")
print("Saved umap_clusters.png")

# %% [markdown]
# ## 5. Log2FC marker analysis

# %%
# Compute log2FC for each gene per cluster
genes_of_interest = [0, 1, 2, 3, 4]  # first 5 genes
for g in genes_of_interest:
    lfc = ts.calculate_log2fc(expr_sub, g, labels)
    gene_name = expr.columns[g]
    print(f"{gene_name}: {lfc}")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated:
# - PCA dimensionality reduction
# - KNN + Louvain clustering
# - UMAP embedding
# - Spatial field-of-view visualization
# - Log2FC marker gene analysis
