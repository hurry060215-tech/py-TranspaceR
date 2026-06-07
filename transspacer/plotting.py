"""Plotting functions: spatial visualization, heatmaps, UMAP, etc."""

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_fov(x, y, cluster_labels, output_path=None, title="FoV"):
    """Scatter plot of cell centroids colored by cluster.

    Parameters
    ----------
    x, y : np.ndarray
        Cell coordinates.
    cluster_labels : array-like
        Cluster labels.
    output_path : str, optional
        Save path.
    title : str
        Plot title.
    """
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    unique_labels = np.unique(cluster_labels)
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        ax.scatter(x[mask], y[mask], s=1, c=[cmap(i)], label=str(label), alpha=0.7)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fov_gene(x, y, expression_values, gene_name, output_path=None):
    """Scatter plot colored by gene expression (white-yellow-orange-red)."""
    if not HAS_MPL:
        return
    from .utils import color_convertion
    colors = color_convertion(expression_values)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x, y, s=1, c=colors, alpha=0.7)
    ax.set_title(gene_name)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_variogram(sill, range_val, alpha, nugget, model, r_data, v_data, r2, output_path=None):
    """Plot fitted variogram curve over data points."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(r_data, v_data, s=10, alpha=0.5, label="Data")

    r_fit = np.linspace(0, max(r_data) * 1.1, 100)
    if model == "Constant":
        ax.axhline(sill, color="red", label=f"Constant (C={sill:.3f})")
    elif model == "Exponential":
        v_fit = sill * (1 - np.exp(-r_fit / range_val)) + nugget
        ax.plot(r_fit, v_fit, "r-", label=f"Exp (C={sill:.3f}, tau={range_val:.3f})")
    elif model == "Finetuned exponential":
        v_fit = sill * (1 - np.exp(-r_fit**alpha / range_val)) + nugget
        ax.plot(r_fit, v_fit, "r-", label=f"FineExp (alpha={alpha:.3f})")

    ax.set_xlabel("Distance")
    ax.set_ylabel("Variogram")
    ax.set_title(f"R2={r2:.3f}, Model={model}")
    ax.legend()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_umap(umap_coords, labels, output_path, title="UMAP"):
    """Save UMAP scatter plot."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    unique = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", len(unique))
    for i, l in enumerate(unique):
        mask = labels == l
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1], s=1, c=[cmap(i)], label=str(l), alpha=0.5)
    ax.set_title(title)
    ax.legend(markerscale=10, fontsize=6)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_heatmap_markers(expression, gene_names, cluster_labels, output_path,
                          top_n: int = 5):
    """Save marker gene heatmap per cluster."""
    if not HAS_MPL:
        return
    groups = np.unique(cluster_labels)
    markers = []
    for g in groups:
        mask = cluster_labels == g
        means_in = expression[mask, :].mean(axis=0)
        means_out = expression[~mask, :].mean(axis=0) if np.any(~mask) else np.zeros_like(means_in)
        fc = means_in - means_out
        top_idx = np.argsort(-fc)[:top_n]
        markers.extend(top_idx.tolist())
    markers = sorted(set(markers))

    fig, ax = plt.subplots(figsize=(12, 8))
    data = expression[:, markers]
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    plt.colorbar(im)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_annotation_plot(cluster_labels, cell_types, output_path):
    """Bar plot of cell type proportions per cluster."""
    if not HAS_MPL:
        return
    import pandas as pd
    df = pd.DataFrame({"cluster": cluster_labels, "type": cell_types})
    ct = pd.crosstab(df["cluster"], df["type"], normalize="index")
    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Proportion")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_boxplot(expression, gene_idx, gene_name, cluster_labels, output_path):
    """Gene expression boxplot by cluster."""
    if not HAS_MPL:
        return
    groups = np.unique(cluster_labels)
    data = [expression[cluster_labels == g, gene_idx] for g in groups]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, labels=groups)
    ax.set_title(gene_name)
    ax.set_ylabel("Expression")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_dendogram(data, labels, output_path):
    """Hierarchical clustering dendrogram."""
    if not HAS_MPL:
        return
    from scipy.cluster.hierarchy import linkage, dendrogram
    Z = linkage(data, method="ward")
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=6)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_geary_variance_plot(geary_values, variance_values, selected_mask,
                              output_path, title="Geary vs Variance"):
    """Scatter of variance vs Geary score."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(geary_values, variance_values, s=5, alpha=0.3, c="black")
    if selected_mask is not None and np.any(selected_mask):
        ax.scatter(geary_values[selected_mask], variance_values[selected_mask],
                   s=10, c="red", label="Selected")
    ax.set_xlabel("Geary Score")
    ax.set_ylabel("Variance")
    ax.set_title(title)
    ax.legend()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_tissue_visualization(x, y, values, output_path, title="Tissue"):
    """Spatial tissue visualization."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(x, y, s=1, c=values, cmap="viridis", alpha=0.7)
    plt.colorbar(sc)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
