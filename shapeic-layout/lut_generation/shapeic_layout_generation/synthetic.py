from __future__ import annotations

import numpy as np


def primitive_matrices(
    primitive: str,
    port_count: int,
    length_m: float,
    finger_width_m: float,
    nf: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic passive matrices used for archive and integration tests."""
    c_scale = 8.0e-16 * nf * (finger_width_m / 1.0e-6) * (
        1.0 + length_m / 1.0e-6
    )
    primitive_scale = 0.75 + 0.05 * (sum(primitive.encode("utf-8")) % 6)
    # A one-port-per-net extrinsic model has no path for series metal R and
    # should not invent dielectric DC leakage between independent nets.
    conductance = np.zeros((port_count, port_count), dtype=np.float64)
    capacitance = _laplacian(port_count, c_scale * primitive_scale)
    return conductance, capacitance


def _laplacian(size: int, scale: float) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.float64)
    for row in range(size):
        for column in range(row + 1, size):
            weight = scale * (1.0 + ((row + 1) * (column + 3) % 7) / 10.0)
            matrix[row, row] += weight
            matrix[column, column] += weight
            matrix[row, column] -= weight
            matrix[column, row] -= weight
    return matrix
