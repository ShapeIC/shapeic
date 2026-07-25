from __future__ import annotations

import numpy as np


def reduce_first_order(
    conductance: np.ndarray,
    capacitance: np.ndarray,
    port_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Kron-reduce an RC network and retain the first derivative at s=0."""
    if conductance.shape != capacitance.shape or conductance.ndim != 2:
        raise ValueError("G and C must be equally-sized matrices")
    if conductance.shape[0] != conductance.shape[1]:
        raise ValueError("G and C must be square")
    if len(port_indices) < 2 or len(set(port_indices)) != len(port_indices):
        raise ValueError("at least two unique port indices are required")
    internal = [index for index in range(conductance.shape[0]) if index not in port_indices]
    p = np.asarray(port_indices, dtype=int)
    if not internal:
        return _clean(conductance[np.ix_(p, p)]), _clean(capacitance[np.ix_(p, p)])
    i = np.asarray(internal, dtype=int)
    gpp = conductance[np.ix_(p, p)]
    gpi = conductance[np.ix_(p, i)]
    gip = conductance[np.ix_(i, p)]
    gii = conductance[np.ix_(i, i)]
    cpp = capacitance[np.ix_(p, p)]
    cpi = capacitance[np.ix_(p, i)]
    cip = capacitance[np.ix_(i, p)]
    cii = capacitance[np.ix_(i, i)]

    if not np.any(conductance):
        try:
            a_cip = np.linalg.solve(cii, cip)
        except np.linalg.LinAlgError as error:
            raise ValueError("internal capacitance matrix is singular") from error
        reduced_c = cpp - cpi @ a_cip
        return _clean(gpp), _clean(reduced_c)

    try:
        a_gip = np.linalg.solve(gii, gip)
        a_cip = np.linalg.solve(gii, cip)
        a_cii_a_gip = np.linalg.solve(gii, cii @ a_gip)
    except np.linalg.LinAlgError as error:
        raise ValueError("internal DC conductance matrix is singular") from error
    reduced_g = gpp - gpi @ a_gip
    reduced_c = cpp - cpi @ a_gip - gpi @ a_cip + gpi @ a_cii_a_gip
    return _clean(reduced_g), _clean(reduced_c)


def _clean(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    scale = max(float(np.max(np.abs(matrix))), 1.0e-30)
    matrix[np.abs(matrix) < scale * 1.0e-12] = 0.0
    # A complete nodal model has no implicit ground; preserve exact KCL.
    diagonal = np.diag(matrix) - matrix.sum(axis=1)
    np.fill_diagonal(matrix, diagonal)
    _validate_passive(matrix)
    return matrix


def _validate_passive(matrix: np.ndarray) -> None:
    scale = max(float(np.max(np.abs(matrix))), 1.0e-30)
    tolerance = scale * 1.0e-7
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix, matrix.T, rtol=0.0, atol=tolerance
    ):
        raise ValueError("reduced matrix is not finite and symmetric")
    if np.any(np.diag(matrix) < -tolerance):
        raise ValueError("reduced matrix has a negative diagonal")
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    if np.any(off_diagonal > tolerance):
        raise ValueError("reduced matrix is not a passive nodal matrix")
