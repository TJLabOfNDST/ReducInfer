"""Physical vector layout helpers for pure-SQL factor GEMM."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def column_vectors(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return contiguous rows for columns of one FP32 right operand.

    TranSQL+ stores a GEMM right operand in transposed physical order. The
    one-time O(elements) copy spends one matrix-sized preprocessing buffer for
    sequential SQL dot products; the source layout is not persisted, so the
    numeric storage remains O(elements).
    TranSQL+ stores GEMM right operands transposed by column. This one-time
    O(elements) copy spends one matrix of preprocessing space to provide
    contiguous SQL dot-product access without duplicating persisted values.
    """

    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("factor matrix must be non-empty and two-dimensional")
    if matrix.dtype != np.float32:
        raise ValueError("factor matrix must use float32")
    if not np.isfinite(matrix).all():
        raise ValueError("factor matrix must contain only finite values")
    return np.ascontiguousarray(matrix.T)

