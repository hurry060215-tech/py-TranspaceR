"""Utility functions: color conversion, data curation, multi-sample handling."""

import numpy as np


def color_convertion(x: np.ndarray) -> list:
    """Map numeric values to white-yellow-orange-red hex colors.

    Parameters
    ----------
    x : np.ndarray
        Numeric vector.

    Returns
    -------
    list
        Hex color strings.
    """
    x = np.asarray(x, dtype=float)
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    if x_max == x_min:
        normed = np.full_like(x, 0.5)
    else:
        normed = (x - x_min) / (x_max - x_min)

    colors = []
    for v in normed:
        if np.isnan(v):
            colors.append("#FFFFFF")
        elif v < 0.25:
            r, g, b = 255, int(255 * v * 4), 0
        elif v < 0.5:
            r, g, b = 255, int(255 * (1 - (v - 0.25) * 4)), 0
        elif v < 0.75:
            r, g, b = int(255 * (1 - (v - 0.5) * 4)), 0, 0
        else:
            r, g, b = int(255 * (v - 0.75) * 4), 0, 0
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors


def string_to_colors(strings: list) -> dict:
    """Map unique strings to distinct colors.

    Parameters
    ----------
    strings : list
        List of string labels.

    Returns
    -------
    dict
        {string: color} mapping using HSV rainbow.
    """
    unique = sorted(set(strings))
    n = len(unique)
    colors = {}
    for i, s in enumerate(unique):
        hue = i / max(n, 1)
        # HSV to RGB
        h = hue * 6
        c = 1.0
        x = c * (1 - abs(h % 2 - 1))
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        colors[s] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return colors


def compute_radius(area=None, volume=None, platform: str = "default") -> np.ndarray:
    """Compute radius from area or volume.

    Parameters
    ----------
    area : np.ndarray, optional
        Cell areas.
    volume : np.ndarray, optional
        Cell volumes.
    platform : str
        Platform type.

    Returns
    -------
    np.ndarray
        Cell radii.
    """
    if area is not None:
        return np.sqrt(np.asarray(area) / np.pi)
    elif volume is not None:
        return (3 * np.asarray(volume) / (4 * np.pi)) ** (1 / 3)
    else:
        raise ValueError("Must provide area or volume")


def curate_data(expression, meta_data, lib_size_min: float = 0,
                lib_size_max: float = np.inf,
                radius_min: float = 0, radius_max: float = np.inf):
    """Filter cells by library size and radius.

    Parameters
    ----------
    expression : np.ndarray or sparse
        Expression matrix.
    meta_data : dict
        Must have 'lib_size' and 'radius' arrays.
    lib_size_min, lib_size_max : float
        Library size bounds.
    radius_min, radius_max : float
        Radius bounds.

    Returns
    -------
    tuple
        (filtered_expression, mask)
    """
    from scipy.sparse import issparse

    lib_size = meta_data.get("lib_size")
    radius = meta_data.get("radius")

    mask = np.ones(expression.shape[0], dtype=bool)
    if lib_size is not None:
        mask &= (lib_size >= lib_size_min) & (lib_size <= lib_size_max)
    if radius is not None:
        mask &= (radius >= radius_min) & (radius <= radius_max)

    if issparse(expression):
        return expression[mask, :], mask
    return expression[mask, :], mask


def fraction_multiple_samples(expression, meta_data, sample_col: str = "sample"):
    """Split expression and metadata by sample.

    Parameters
    ----------
    expression : np.ndarray or sparse
        Expression matrix.
    meta_data : dict or DataFrame-like
        Must contain sample_col.
    sample_col : str
        Column name for sample IDs.

    Returns
    -------
    dict
        {sample_id: (expression_subset, meta_subset)}
    """
    samples = meta_data[sample_col]
    unique_samples = np.unique(samples)
    result = {}
    for s in unique_samples:
        mask = samples == s
        if hasattr(expression, '__getitem__'):
            result[s] = (expression[mask], {k: v[mask] if hasattr(v, '__getitem__') else v
                                             for k, v in meta_data.items()})
    return result


def load_scimilarity_results(csv_path: str, threshold: float = 0.05) -> dict:
    """Load and threshold scimilarity cell type predictions.

    Parameters
    ----------
    csv_path : str
        Path to CSV file.
    threshold : float
        Minimum fraction to keep cell type (else "Other").

    Returns
    -------
    dict
        Cell type assignments.
    """
    import csv
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cell_types = [r.get("cell_type", r.get("Cell_type", "")) for r in rows]
    counts = {}
    for ct in cell_types:
        counts[ct] = counts.get(ct, 0) + 1
    total = len(cell_types)

    rare = {ct for ct, c in counts.items() if c / total < threshold}
    filtered = [ct if ct not in rare else "Other" for ct in cell_types]
    return {"cell_types": filtered}
