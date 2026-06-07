"""Sparse matrix utilities: column variance, group-wise aggregation."""

import numpy as np
from scipy.sparse import issparse, csc_matrix


def colvars_sparse(mat) -> np.ndarray:
    """Column variance of a sparse matrix (CSC format).

    Parameters
    ----------
    mat : scipy.sparse.csc_matrix or np.ndarray
        Input matrix.

    Returns
    -------
    np.ndarray
        Variance of each column.
    """
    if issparse(mat):
        mat = csc_matrix(mat)
        n_cells = mat.shape[0]
        means = np.array(mat.mean(axis=0)).flatten()
        variances = np.zeros(mat.shape[1])
        for j in range(mat.shape[1]):
            start = mat.indptr[j]
            end = mat.indptr[j + 1]
            if start < end:
                diff_sq = (mat.data[start:end] - means[j]) ** 2
                sum_sq = np.sum(diff_sq)
            else:
                sum_sq = 0.0
            variances[j] = (sum_sq + (n_cells - (end - start)) * means[j] ** 2) / n_cells
        return variances
    else:
        return np.var(mat, axis=0)


def aggregate_sparse(expression, grouping) -> np.ndarray:
    """Group-wise column means on a matrix.

    Parameters
    ----------
    expression : sparse or dense matrix
        Expression matrix (cells x genes).
    grouping : array-like
        Cluster/group labels per cell.

    Returns
    -------
    np.ndarray
        Mean expression per group (genes x groups), matching R's sapply output.
    """
    groups = np.unique(grouping)
    if issparse(expression):
        result = np.zeros((expression.shape[1], len(groups)))
        for i, g in enumerate(groups):
            idx = np.where(grouping == g)[0]
            if len(idx) > 0:
                result[:, i] = np.array(expression[idx, :].mean(axis=0)).flatten()
    else:
        result = np.zeros((expression.shape[1], len(groups)))
        for i, g in enumerate(groups):
            idx = np.where(grouping == g)[0]
            if len(idx) > 0:
                result[:, i] = expression[idx, :].mean(axis=0)
    return result
