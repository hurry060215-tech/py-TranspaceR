"""Cell-size normalization."""

import numpy as np


def c_normalisation(y: np.ndarray, scaling_factor: np.ndarray) -> np.ndarray:
    """Normalize expression by cell size: y / (area + 1/tau).

    Parameters
    ----------
    y : np.ndarray
        Expression matrix (cells x genes).
    scaling_factor : np.ndarray
        Cell areas (length n_cells).

    Returns
    -------
    np.ndarray
        Normalized expression matrix.
    """
    cell_size = scaling_factor.astype(float)
    tau_parameter = np.mean(y / cell_size[:, np.newaxis], axis=0)
    scaling = cell_size[:, np.newaxis] + 1.0 / tau_parameter[np.newaxis, :]
    return y / scaling
