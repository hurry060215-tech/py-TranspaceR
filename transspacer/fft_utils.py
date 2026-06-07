"""FFT utility functions: fftshift, ifftshift, pad_definitor."""

import numpy as np


def fftshift(input_matrix: np.ndarray, dim: int = -1) -> np.ndarray:
    """Shift zero-frequency component to center of spectrum.

    Parameters
    ----------
    input_matrix : np.ndarray
        2D array to shift.
    dim : int
        -1 for both dimensions, 1 for rows, 2 for columns.

    Returns
    -------
    np.ndarray
        Shifted array.
    """
    rows, cols = input_matrix.shape

    def swap_up_down(m):
        rows_half = int(np.ceil(rows / 2))
        return np.vstack([m[rows_half:, :], m[:rows_half, :]])

    def swap_left_right(m):
        cols_half = int(np.ceil(cols / 2))
        return np.hstack([m[:, cols_half:], m[:, :cols_half]])

    if dim == -1:
        return swap_left_right(swap_up_down(input_matrix))
    elif dim == 1:
        return swap_up_down(input_matrix)
    elif dim == 2:
        return swap_left_right(input_matrix)
    else:
        raise ValueError("Invalid dimension parameter")


def ifftshift(input_matrix: np.ndarray, dim: int = -1) -> np.ndarray:
    """Inverse FFT shift.

    Parameters
    ----------
    input_matrix : np.ndarray
        2D array to shift.
    dim : int
        -1 for both dimensions, 1 for rows, 2 for columns.

    Returns
    -------
    np.ndarray
        Shifted array.
    """
    rows, cols = input_matrix.shape

    def swap_up_down(m):
        rows_half = int(np.floor(rows / 2))
        return np.vstack([m[rows_half:, :], m[:rows_half, :]])

    def swap_left_right(m):
        cols_half = int(np.floor(cols / 2))
        return np.hstack([m[:, cols_half:], m[:, :cols_half]])

    if dim == -1:
        return swap_up_down(swap_left_right(input_matrix))
    elif dim == 1:
        return swap_up_down(input_matrix)
    elif dim == 2:
        return swap_left_right(input_matrix)
    else:
        raise ValueError("Invalid dimension parameter")


def pad_definitor(meta_data_x: np.ndarray, meta_data_y: np.ndarray) -> int:
    """Compute automatic padding size from spatial coordinates.

    Parameters
    ----------
    meta_data_x : np.ndarray
        X coordinates of cell centroids.
    meta_data_y : np.ndarray
        Y coordinates of cell centroids.

    Returns
    -------
    int
        Padding size (minimum 1).
    """
    xrange = np.max(meta_data_x) - np.min(meta_data_x)
    yrange = np.max(meta_data_y) - np.min(meta_data_y)
    range_val = np.mean([xrange, yrange])
    n_pad = int(np.round(range_val / 200))
    if n_pad == 0:
        n_pad = 1
    return n_pad
