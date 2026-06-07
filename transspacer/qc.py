"""Quality control: Otsu thresholding and QC gene/RNA thresholds."""

import numpy as np


def otsu_thresholding(x: np.ndarray, number_bins: int = 100) -> float:
    """Otsu's method to find optimal threshold minimizing intra-class variance.

    Parameters
    ----------
    x : np.ndarray
        1D data vector.
    number_bins : int
        Number of quantile-based bins.

    Returns
    -------
    float
        Selected threshold value.
    """
    list_bin = np.quantile(x, np.linspace(0, 1, number_bins))
    intra_variance = np.empty(number_bins)
    for k in range(number_bins):
        threshold = list_bin[k]
        below = x[x < threshold]
        above = x[x > threshold]
        s = 0.0
        if len(below) > 1:
            s += len(below) * np.var(below, ddof=1)
        if len(above) > 1:
            s += len(above) * np.var(above, ddof=1)
        intra_variance[k] = s
    return list_bin[np.argmin(intra_variance)]


def qc_gene_threshold(expression, meta_data_radius, gene_names):
    """QC gene threshold using Otsu on log10 gene counts.

    Parameters
    ----------
    expression : sparse or dense matrix
        Gene expression (cells x genes).
    meta_data_radius : np.ndarray
        Cell radii.
    gene_names : list of str
        Gene names.

    Returns
    -------
    dict
        Otsu_threshold, Separability_score.
    """
    from scipy.sparse import issparse

    if issparse(expression):
        gene_size = np.array(expression.sum(axis=0)).flatten()
    else:
        gene_size = expression.sum(axis=0)

    neg_mask = np.array([
        any(pat in str(g) for pat in ["Neg", "False", "Blank"])
        for g in gene_names
    ])
    neg_counts = np.log10(gene_size[neg_mask])
    neg_counts = neg_counts[np.isfinite(neg_counts)]

    x = np.log10(gene_size)
    x = x[np.isfinite(x)]

    threshold = otsu_thresholding(x)
    neg_before = np.sum(neg_counts < threshold)
    score = neg_before / len(neg_counts) if len(neg_counts) > 0 else 0.0

    return {"Otsu_threshold": 10**threshold, "Separability_score": score}
