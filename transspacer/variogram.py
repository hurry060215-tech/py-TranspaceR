"""FFT-based variogram computation pipeline."""

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.optimize import curve_fit
from .fft_utils import fftshift, pad_definitor


def get_variogram_map(im: np.ndarray) -> np.ndarray:
    """Compute variogram map using FFT.

    Parameters
    ----------
    im : np.ndarray
        2D image matrix (rasterized gene expression).

    Returns
    -------
    np.ndarray
        Variogram map (centered via fftshift).
    """
    im_index = np.ones_like(im)
    im_index[np.isnan(im)] = 0
    im_clean = im.copy()
    im_clean[np.isnan(im_clean)] = 0

    im_ft = fft2(im_clean)
    im_index_ft = fft2(im_index)
    im_2_ft = fft2(im_clean ** 2)
    N = im.shape[0] * im.shape[1]

    numerator = np.real(
        ifft2(
            im_2_ft * np.conj(im_index_ft)
            + np.conj(im_2_ft) * im_index_ft
            - 2 * im_ft * np.conj(im_ft)
        )
    )
    denominator = 2 * np.real(ifft2(im_index_ft * np.conj(im_index_ft)))
    denominator[denominator == 0] = 1  # avoid division by zero

    variogram_map = numerator / denominator
    return fftshift(variogram_map)


def get_isotropic_vario(variogram: np.ndarray) -> tuple:
    """Radial averaging of 2D variogram map.

    Parameters
    ----------
    variogram : np.ndarray
        2D variogram map.

    Returns
    -------
    tuple
        (r_values, variogram_values) arrays.
    """
    n_x = variogram.shape[1]
    n_y = variogram.shape[0]

    center_x = n_x // 2
    center_y = n_y // 2

    m_x = np.tile(np.arange(-center_x, center_x), (n_y, 1))
    m_y = np.tile(np.arange(-center_y, center_y).reshape(-1, 1), (1, n_x))

    # Adjust dimensions to match variogram shape
    if m_x.shape[0] != variogram.shape[0]:
        reps = int(np.ceil(variogram.shape[0] / m_x.shape[0]))
        m_x = np.tile(m_x, (reps, 1))[:variogram.shape[0], :]
        m_y = np.tile(m_y, (reps, 1))[:variogram.shape[0], :]
    if m_x.shape[1] != variogram.shape[1]:
        reps = int(np.ceil(variogram.shape[1] / m_x.shape[1]))
        m_x = np.tile(m_x, (1, reps))[:, :variogram.shape[1]]
        m_y = np.tile(m_y, (1, reps))[:, :variogram.shape[1]]

    m_dist = np.sqrt(m_x**2 + m_y**2)
    dist_flat = m_dist.ravel()
    var_flat = variogram.ravel()

    # Bin by distance and average
    unique_dists = np.unique(np.round(dist_flat, 6))
    r_values = []
    var_values = []
    for d in unique_dists:
        mask = np.abs(dist_flat - d) < 1e-6
        if np.any(mask):
            r_values.append(d)
            var_values.append(np.mean(var_flat[mask]))

    return np.array(r_values), np.array(var_values)


def process_gene(gene_expression: np.ndarray, n_pad: int,
                 x_coords: np.ndarray, y_coords: np.ndarray) -> dict:
    """Compute variogram for one gene and fit models.

    Parameters
    ----------
    gene_expression : np.ndarray
        Expression values for one gene per cell.
    n_pad : int
        Padding/bin size for rasterization.
    x_coords, y_coords : np.ndarray
        Spatial coordinates.

    Returns
    -------
    dict
        Fitted variogram parameters.
    """
    # Rasterize: bin cells into grid
    breaks_x = np.arange(np.min(x_coords), np.max(x_coords) + n_pad, n_pad)
    breaks_y = np.arange(np.min(y_coords), np.max(y_coords) + n_pad, n_pad)

    ix = np.digitize(x_coords, breaks_x) - 1
    iy = np.digitize(y_coords, breaks_y) - 1

    nx = len(breaks_x) - 1
    ny = len(breaks_y) - 1

    # Build raster matrix
    raster = np.full((ny, nx), np.nan)
    count = np.zeros((ny, nx))
    for k in range(len(gene_expression)):
        xi, yi = ix[k], iy[k]
        if 0 <= xi < nx and 0 <= yi < ny:
            if np.isnan(raster[yi, xi]):
                raster[yi, xi] = gene_expression[k]
            else:
                raster[yi, xi] += gene_expression[k]
            count[yi, xi] += 1

    # Replace NaN with 0 for FFT
    raster_clean = raster.copy()
    raster_clean[np.isnan(raster_clean)] = 0

    # Compute variogram
    vmap = get_variogram_map(raster_clean)
    r, v = get_isotropic_vario(vmap)

    # Filter r > 0 and r < 30
    mask = (r > 0) & (r < 30)
    r_filt = r[mask]
    v_filt = v[mask]

    if len(r_filt) == 0:
        return {
            "Sill": np.nan, "Range": np.nan, "Nugget": np.nan,
            "Alpha": np.nan, "Total_variance": np.nan, "R2": np.nan,
            "Prop": np.nan, "AIC": np.nan, "Model": "Constant",
            "Vario_values": []
        }

    # Group by 10 and average
    n_groups = len(r_filt) // 10
    if n_groups == 0:
        n_groups = 1
    group_idx = np.arange(len(r_filt)) // 10
    r_grouped = np.array([r_filt[group_idx == g].max() for g in range(n_groups)])
    v_grouped = np.array([v_filt[group_idx == g].mean() for g in range(n_groups)])
    r_grouped = r_grouped * n_pad

    a = r_grouped
    b = v_grouped

    # Fit models
    AIC_values = [np.inf, np.inf, np.inf]
    models = [None, None, None]
    params = [None, None, None]

    # Model 0: Constant
    try:
        p0 = [np.mean(b)]
        popt, _ = curve_fit(lambda x, C: C * np.ones_like(x), a, b, p0=p0)
        predicted = np.full_like(b, popt[0])
        error2 = np.sum((b - predicted) ** 2)
        AIC_values[0] = len(b) * np.log(error2 / len(b)) + 2 * 1
        models[0] = ("Constant", popt)
    except Exception:
        pass

    # Model 1: Exponential: C*(1 - exp(-a/tau)) + n
    try:
        tau0 = r_grouped[np.argmax(v_grouped)]
        n0 = np.min(v)
        C0 = np.max(v) - n0
        popt, _ = curve_fit(
            lambda x, tau, n, C: C * (1 - np.exp(-x / tau)) + n,
            a, b, p0=[tau0, n0, C0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
            method="trf"
        )
        predicted = popt[2] * (1 - np.exp(-a / popt[0])) + popt[1]
        error2 = np.sum((b - predicted) ** 2)
        AIC_values[1] = len(b) * np.log(error2 / len(b)) + 2 * 3
        models[1] = ("Exponential", popt)
    except Exception:
        pass

    # Model 2: Fine-tuned exponential: C*(1 - exp(-a^alpha/tau)) + n
    try:
        tau0 = r_grouped[np.argmax(v_grouped)]
        n0 = np.min(v)
        C0 = np.max(v) - n0
        popt, _ = curve_fit(
            lambda x, tau, n, C, alpha: C * (1 - np.exp(-x**alpha / tau)) + n,
            a, b, p0=[tau0, n0, C0, 1.0],
            bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]),
            method="trf"
        )
        predicted = popt[2] * (1 - np.exp(-a**popt[3] / popt[0])) + popt[1]
        error2 = np.sum((b - predicted) ** 2)
        AIC_values[2] = len(b) * np.log(error2 / len(b)) + 2 * 4
        models[2] = ("Finetuned exponential", popt)
    except Exception:
        pass

    # Select best model
    best_idx = int(np.argmin(AIC_values))
    best_model, best_params = models[best_idx], params[best_idx]

    result = {
        "Sill": np.nan, "Range": np.nan, "Nugget": np.nan,
        "Alpha": np.nan, "Total_variance": np.nan, "R2": np.nan,
        "Prop": np.nan, "AIC": AIC_values[best_idx],
        "Model": "Constant", "Vario_values": v_grouped.tolist()
    }

    if best_idx == 0 and models[0] is not None:
        _, popt = models[0]
        result["Sill"] = popt[0]
        result["Model"] = "Constant"
        predicted = np.full_like(b, popt[0])
        ss_res = np.sum((b - predicted) ** 2)
        ss_tot = np.sum((b - np.min(b)) ** 2)
        result["R2"] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    elif best_idx == 1 and models[1] is not None:
        _, popt = models[1]
        result["Sill"] = popt[2]
        result["Range"] = popt[0]
        result["Nugget"] = popt[1]
        result["Total_variance"] = popt[2] + popt[1]
        result["Prop"] = popt[2] / (popt[1] + popt[2]) if (popt[1] + popt[2]) > 0 else np.nan
        result["Model"] = "Exponential"
        predicted = popt[2] * (1 - np.exp(-a / popt[0])) + popt[1]
        ss_res = np.sum((b - predicted) ** 2)
        ss_tot = np.sum((b - np.mean(b)) ** 2)
        result["R2"] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    elif best_idx == 2 and models[2] is not None:
        _, popt = models[2]
        result["Sill"] = popt[2]
        result["Nugget"] = popt[1]
        result["Alpha"] = popt[3]
        result["Range"] = popt[0]
        result["Total_variance"] = popt[2] + popt[1]
        result["Prop"] = popt[2] / (popt[1] + popt[2]) if (popt[1] + popt[2]) > 0 else np.nan
        result["Model"] = "Finetuned exponential"
        predicted = popt[2] * (1 - np.exp(-a**popt[3] / popt[0])) + popt[1]
        ss_res = np.sum((b - predicted) ** 2)
        ss_tot = np.sum((b - np.mean(b)) ** 2)
        result["R2"] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return result


def compute_variogram(expression: np.ndarray, meta_x: np.ndarray, meta_y: np.ndarray,
                      automatic_pad: bool = True, n_pad: int = 20,
                      ncores: int = 1) -> dict:
    """Compute variogram for spatial gene expression data.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (cells x genes).
    meta_x, meta_y : np.ndarray
        Spatial coordinates.
    automatic_pad : bool
        Auto-compute padding from coordinates.
    n_pad : int
        Padding size (ignored if automatic_pad is True).
    ncores : int
        Number of cores (unused in Python, parallelizable externally).

    Returns
    -------
    dict
        Selected_genes and Parameters dataframe.
    """
    if automatic_pad:
        n_pad = pad_definitor(meta_x, meta_y)

    gene_names = list(range(expression.shape[1]))
    results = []
    for g in gene_names:
        res = process_gene(expression[:, g], n_pad, meta_x, meta_y)
        results.append(res)

    # Build parameters array
    param_keys = ["Sill", "Range", "Nugget", "Alpha", "Total_variance", "R2", "Prop", "AIC", "Model"]
    params = {k: [r[k] for r in results] for k in param_keys}
    params["gene_names"] = gene_names
    params["Vario_values"] = [r["Vario_values"] for r in results]

    # Range correction
    range_corrected = []
    for i, model in enumerate(params["Model"]):
        r = params["Range"][i]
        if model == "Exponential":
            range_corrected.append(r * np.log(2) if not np.isnan(r) else np.nan)
        elif model == "Finetuned exponential":
            a = params["Alpha"][i]
            range_corrected.append((r * np.log(2)) ** (1 / a) if not np.isnan(r) and not np.isnan(a) else np.nan)
        else:
            range_corrected.append(r)
    params["Range_corrected"] = range_corrected

    # Select spatially variable genes
    selected = []
    for i, g in enumerate(gene_names):
        r2 = params["R2"][i]
        prop = params["Prop"][i]
        tv = params["Total_variance"][i]
        if not np.isnan(r2) and not np.isnan(prop) and not np.isnan(tv):
            if r2 > 0.85 and prop > 0.3 and tv > 0.5:
                selected.append(g)

    return {"Selected_genes": selected, "Parameters": params}
