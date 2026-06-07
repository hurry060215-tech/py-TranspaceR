"""Gene selection utilities: log2FC, shared gene selection."""

import numpy as np
from collections import Counter


def calculate_log2fc(expression: np.ndarray, gene_idx: int, clustering: np.ndarray) -> dict:
    """Per-gene log2FC: mean in cluster vs weighted mean in other clusters.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (cells x genes).
    gene_idx : int
        Column index of the gene.
    clustering : np.ndarray
        Cluster labels per cell.

    Returns
    -------
    dict
        {cluster_label: log2fc_value} for each cluster.
    """
    x = expression[:, gene_idx]
    groups = np.unique(clustering)
    mean_expression = np.array([x[clustering == g].mean() for g in groups])
    counts = Counter(clustering)
    total = len(clustering)
    proportions = {g: counts[g] / total for g in groups}

    # Build proportion matrix (off-diagonal, row-normalized)
    result = {}
    for i, g in enumerate(groups):
        weighted_other = 0.0
        weight_sum = 0.0
        for j, g2 in enumerate(groups):
            if i != j:
                weighted_other += proportions[g2] * mean_expression[j]
                weight_sum += proportions[g2]
        if weight_sum > 0:
            weighted_other /= weight_sum
        if weighted_other > 0:
            result[g] = np.log2(mean_expression[i] / weighted_other)
        else:
            result[g] = np.nan
    return result


def select_genes(selected_objects: list, selected_names: list = None) -> list:
    """Union of gene lists.

    Parameters
    ----------
    selected_objects : list of list
        Each element is a list/set of gene names.
    selected_names : list of str, optional
        Names for each gene set.

    Returns
    -------
    list
        Unique shared genes across all sets.
    """
    shared = set()
    for obj in selected_objects:
        shared.update(obj)
    return sorted(shared)
