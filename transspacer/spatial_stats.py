"""Spatial statistics: Geary's C, excess variance ratio, excess zero score."""

import numpy as np
from scipy.sparse import issparse, csc_matrix
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm, kurtosis as scipy_kurtosis
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


def _build_spatial_weight_matrix(coords: np.ndarray):
    """Build spatial weight matrix from Delaunay triangulation."""
    from scipy.sparse import lil_matrix

    tri = Delaunay(coords)
    n = len(coords)
    W = lil_matrix((n, n), dtype=float)

    # Extract edges from simplices
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                edges.add(edge)

    for i, j in edges:
        W[i, j] = 1.0
        W[j, i] = 1.0

    W = csc_matrix(W)
    # Normalize: W[i,j] = 1/degree for neighbors
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    # Symmetric normalization
    W = W + W.T
    W.data = np.ones_like(W.data)  # binary adjacency
    return csc_matrix(W)


def geary_c_score(expression: np.ndarray, coords: np.ndarray,
                  pvalue_threshold: float = 0.01) -> dict:
    """Compute Geary's C score for each gene.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (cells x genes).
    coords : np.ndarray
        Spatial coordinates (cells x 2).
    pvalue_threshold : float
        FDR-adjusted p-value threshold for gene selection.

    Returns
    -------
    dict
        Values (1/C), Selected_genes, P_values.
    """
    n_cells, n_genes = expression.shape

    # Build Delaunay-based spatial weight matrix
    W = _build_spatial_weight_matrix(coords)
    W_dense = W.toarray()

    S_0 = W_dense.sum()
    S_1 = 0.5 * np.sum((W_dense + W_dense.T) ** 2)
    x_sum = W_dense.sum(axis=0)
    y_sum = W_dense.sum(axis=1)
    S_2 = np.sum((x_sum + y_sum) ** 2)
    N = n_cells
    W_sum = W_dense.sum()

    geary_values = np.zeros(n_genes)
    p_values = np.zeros(n_genes)

    for i in range(n_genes):
        X = expression[:, i].reshape(-1, 1)
        X_sq = X ** 2

        product_1 = 2 * np.sum(W_dense @ X_sq)
        product_2 = 2 * float((X.T @ W_dense @ X).item())
        var_X = np.var(X, ddof=1) * N

        if var_X == 0:
            geary_values[i] = 1.0
            p_values[i] = 1.0
            continue

        geary_c = (N - 1) * (product_1 - product_2) / (2 * var_X * W_sum)
        geary_values[i] = geary_c

        # Kurtosis for variance under randomness
        b_2 = scipy_kurtosis(expression[:, i], fisher=False)

        var_random = (
            (N - 1) * S_1 * (N**2 - 3*N + 3 - (N-1)*b_2)
            - 0.25 * (N - 1) * S_2 * (N**2 + 3*N - 6 - (N**2 - N + 2)*b_2)
            + S_0**2 * (N**2 - 3 - (N-1)**2 * b_2)
        ) / (N * (N-2) * (N-3) * S_0**2)

        p_values[i] = norm.cdf(geary_c, loc=1, scale=np.sqrt(max(var_random, 1e-300)))

    # Invert C (higher = more spatial)
    inv_geary = 1.0 / geary_values
    inv_geary[~np.isfinite(inv_geary)] = 0

    # FDR correction (Benjamini-Hochberg)
    from statsmodels.stats.multitest import multipletests
    try:
        _, adj_p, _, _ = multipletests(p_values, method="fdr_bh")
    except ImportError:
        # Simple BH fallback
        order = np.argsort(p_values)
        ranked = np.arange(1, len(p_values) + 1)
        adj_p = np.zeros_like(p_values)
        adj_p[order] = p_values[order] * len(p_values) / ranked
        adj_p = np.minimum(adj_p, 1.0)
        # Enforce monotonicity
        for i in range(len(adj_p) - 2, -1, -1):
            adj_p[order[i]] = min(adj_p[order[i]], adj_p[order[i + 1]])

    # Sort by geary score descending
    sort_idx = np.argsort(-inv_geary)
    inv_geary_sorted = inv_geary[sort_idx]
    adj_p_sorted = adj_p[sort_idx]

    selected_mask = adj_p_sorted <= pvalue_threshold
    selected_genes = np.where(selected_mask)[0].tolist()

    # Clip extreme p-values
    adj_p_clipped = adj_p.copy()
    adj_p_clipped[adj_p_clipped < 1e-300] = 1e-300

    return {
        "Values": inv_geary_sorted,
        "Selected_genes": selected_genes,
        "P_values": adj_p_clipped,
        "sort_idx": sort_idx
    }


def excess_variance_ratio_nb(expression: np.ndarray,
                              p_value_threshold: float = 0.01,
                              ratio_threshold: float = 1.5) -> dict:
    """Excess variance ratio using Negative Binomial model.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (cells x genes).
    p_value_threshold : float
        P-value threshold for significance.
    ratio_threshold : float
        Variance ratio threshold.

    Returns
    -------
    dict
        Selected_genes, Excess_variance_ratio.
    """
    if issparse(expression):
        mean_expr = np.array(expression.mean(axis=0)).flatten()
        var_gene = np.array([
            np.var(expression[:, j].toarray().flatten(), ddof=1)
            if issparse(expression) else np.var(expression[:, j], ddof=1)
            for j in range(expression.shape[1])
        ])
    else:
        mean_expr = np.mean(expression, axis=0)
        var_gene = np.var(expression, axis=0, ddof=1)

    mean_sq = mean_expr ** 2

    # Fit NB null model: var ~ mu + mu^2 (forced through origin)
    # var = s * mu^2 + mu => var - mu = s * mu^2
    valid = mean_sq > 0
    y_fit = var_gene[valid] - mean_expr[valid]
    X_fit = mean_sq[valid].reshape(-1, 1)

    # Simple OLS: y = s * X
    s_est = float(np.sum(y_fit * X_fit.flatten()) / np.sum(X_fit.flatten() ** 2))
    s_est = max(s_est, 1e-10)

    fitted_var = mean_expr + s_est * mean_sq
    excess_ratio = var_gene / fitted_var
    excess_ratio[~np.isfinite(excess_ratio)] = 0

    # Monte Carlo significance
    n_cells = expression.shape[0]
    n_sim = 500
    mu_values = np.quantile(mean_expr, np.linspace(0, 1, 30))

    # Estimate variance-of-variance at each mu
    table_var = np.zeros(len(mu_values))
    for idx, mu in enumerate(mu_values):
        sims_var = np.zeros(n_sim)
        for s in range(n_sim):
            # NB: size = 1/s, mu = mu
            samples = np.random.negative_binomial(1 / s_est, 1 / s_est / (mu + 1 / s_est), size=n_cells)
            sims_var[s] = np.var(samples, ddof=1)
        table_var[idx] = np.var(sims_var)

    # Interpolate
    log_mu = np.log10(mu_values + 1e-300)
    log_var = np.log10(table_var + 1e-300)
    est_var = 10 ** np.interp(np.log10(mean_expr + 1e-300), log_mu, log_var)

    p_vals = 1 - norm.cdf(var_gene, loc=fitted_var, scale=np.sqrt(est_var))

    # BH FDR
    from statsmodels.stats.multitest import multipletests
    try:
        _, adj_p, _, _ = multipletests(p_vals, method="fdr_bh")
    except ImportError:
        order = np.argsort(p_vals)
        ranked = np.arange(1, len(p_vals) + 1)
        adj_p = np.zeros_like(p_vals)
        adj_p[order] = p_vals[order] * len(p_vals) / ranked
        adj_p = np.minimum(adj_p, 1.0)

    selected = np.where((adj_p < p_value_threshold) & (excess_ratio > ratio_threshold))[0]

    return {
        "Selected_genes": selected.tolist(),
        "Excess_variance_ratio": excess_ratio
    }


def excess_zero_score_nb(expression: np.ndarray,
                         p_value_threshold: float = 0.01,
                         delta_threshold: float = 0.01) -> dict:
    """Excess zero score using Negative Binomial model.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (cells x genes).
    p_value_threshold : float
        P-value threshold.
    delta_threshold : float
        Excess zero score threshold.

    Returns
    -------
    dict
        Selected_genes, Excess_zero_score.
    """
    n_cells = expression.shape[0]

    if issparse(expression):
        mean_expr = np.array(expression.mean(axis=0)).flatten()
        total_expr = np.array(expression.sum(axis=0)).flatten()
        prop_zero = np.array((expression == 0).sum(axis=0)).flatten() / n_cells
        # Variance via colvars_sparse
        from .sparse_utils import colvars_sparse
        var_expr = colvars_sparse(expression)
    else:
        mean_expr = np.mean(expression, axis=0)
        total_expr = np.sum(expression, axis=0)
        prop_zero = np.mean(expression == 0, axis=0)
        var_expr = np.var(expression, axis=0, ddof=1)

    # Fit NB model for expected zero proportion
    # theta_init = median(1/((var-mean)/mean^2))
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_raw = (var_expr - mean_expr) / mean_expr ** 2
    theta_raw = theta_raw[theta_raw > 0]
    theta_init = np.median(theta_raw) if len(theta_raw) > 0 else 1.0

    # Fit: log(prop_zero) ~ theta*log(theta) - theta*log(theta+mean)
    valid = (prop_zero > 0) & (prop_zero < 1) & (mean_expr > 0)
    if np.sum(valid) < 3:
        return {"Selected_genes": [], "Excess_zero_score": np.zeros(expression.shape[1])}

    log_prop = np.log(prop_zero[valid])
    mu_v = mean_expr[valid]

    def nb_zero_model(mu, theta):
        return theta * np.log(theta) - theta * np.log(theta + mu)

    try:
        popt, _ = curve_fit(nb_zero_model, mu_v, log_prop, p0=[theta_init],
                            bounds=([1e-10], [np.inf]), method="trf")
        theta = popt[0]
    except Exception:
        theta = theta_init

    expected_prop_zero = (theta / (theta + mean_expr)) ** theta
    delta = prop_zero - expected_prop_zero

    # Monte Carlo for variance estimation
    n_sim = 500
    mu_values = np.quantile(mean_expr, np.linspace(0, 1, 30))

    table_var = np.zeros(len(mu_values))
    for idx, mu in enumerate(mu_values):
        sims_pz = np.zeros(n_sim)
        for s in range(n_sim):
            samples = np.random.negative_binomial(1 / theta, 1 / theta / (mu + 1 / theta), size=n_cells)
            sims_pz[s] = np.mean(samples == 0)
        table_var[idx] = np.var(sims_pz)

    log_mu = np.log10(mu_values + 1e-300)
    log_var = np.log10(table_var + 1e-300)
    est_var = 10 ** np.interp(np.log10(mean_expr + 1e-300), log_mu, log_var)

    p_vals = 1 - norm.cdf(prop_zero, loc=expected_prop_zero, scale=np.sqrt(est_var))

    from statsmodels.stats.multitest import multipletests
    try:
        _, adj_p, _, _ = multipletests(p_vals, method="fdr_bh")
    except ImportError:
        order = np.argsort(p_vals)
        ranked = np.arange(1, len(p_vals) + 1)
        adj_p = np.zeros_like(p_vals)
        adj_p[order] = p_vals[order] * len(p_vals) / ranked
        adj_p = np.minimum(adj_p, 1.0)

    selected = np.where((adj_p < p_value_threshold) & (delta > delta_threshold))[0]

    return {
        "Selected_genes": selected.tolist(),
        "Excess_zero_score": delta
    }
