"""Offline block reduction that preserves ReducInfer's tSVD semantics."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import faiss
import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from .config import ReducInferConfig
from .types import FactorPair, TensorBlock, ValidationError


def transpose_linear_weight(weight: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert framework `(out, in)` weights into the paper's `(in, out)` layout."""

    if weight.ndim != 2:
        raise ValidationError("linear weight must be two-dimensional")
    return np.ascontiguousarray(weight.T, dtype=np.float32)


def split_tensor(
    values: NDArray[np.float32],
    *,
    model_id: str,
    tensor_name: str,
    block_h: int,
    block_w: int,
) -> list[TensorBlock]:
    """Split a two-dimensional tensor into padded fixed-shape blocks.

    Padding allows all cluster members to share a factor shape; `valid_shape`
    preserves the original boundary for SQL output trimming.
    """

    if values.ndim != 2:
        raise ValidationError("only two-dimensional tensors can be factorized")
    if block_h <= 0 or block_w <= 0:
        raise ValidationError("block dimensions must be positive")

    height, width = values.shape
    blocks: list[TensorBlock] = []
    for row_offset in range(0, height, block_h):
        for col_offset in range(0, width, block_w):
            valid_h = min(block_h, height - row_offset)
            valid_w = min(block_w, width - col_offset)
            padded = np.zeros((block_h, block_w), dtype=np.float32)
            padded[:valid_h, :valid_w] = values[
                row_offset : row_offset + valid_h,
                col_offset : col_offset + valid_w,
            ]
            blocks.append(
                TensorBlock(
                    block_id=f"{model_id}:{tensor_name}:{row_offset}:{col_offset}",
                    model_id=model_id,
                    tensor_name=tensor_name,
                    row_offset=row_offset,
                    col_offset=col_offset,
                    valid_shape=(valid_h, valid_w),
                    values=padded,
                )
            )
    return blocks


class ReducInferReducer:
    """Cluster normalized block features and factor each cluster exactly."""

    def __init__(self, config: ReducInferConfig) -> None:
        self._config = config
        generator = np.random.default_rng(config.random_seed)
        self._left_projection = generator.standard_normal(
            (config.brp_d1, config.block_h), dtype=np.float32
        )
        self._right_projection = generator.standard_normal(
            (config.brp_d2, config.block_w), dtype=np.float32
        )

    def reduce(self, blocks: Sequence[TensorBlock]) -> list[FactorPair]:
        """Jointly cluster blocks and return one D/C factorization per cluster."""

        if not blocks:
            return []
        self._validate_block_shapes(blocks)
        labels = self._cluster_labels(self._brp_features(blocks))
        grouped_indices: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels.tolist()):
            grouped_indices[label].append(index)

        return [
            self._factor_cluster(cluster_id, [blocks[index] for index in indices])
            for cluster_id, indices in sorted(grouped_indices.items())
        ]

    def _validate_block_shapes(self, blocks: Sequence[TensorBlock]) -> None:
        expected = (self._config.block_h, self._config.block_w)
        if any(block.values.shape != expected for block in blocks):
            raise ValidationError(f"all block values must have shape {expected}")

    def _brp_features(self, blocks: Sequence[TensorBlock]) -> NDArray[np.float32]:
        features: list[NDArray[np.float32]] = []
        for block in blocks:
            norm = float(np.linalg.norm(block.values, ord="fro"))
            normalized = block.values if norm == 0.0 else block.values / norm
            projected = self._left_projection @ normalized @ self._right_projection.T
            features.append(projected.reshape(-1))
        return np.ascontiguousarray(np.vstack(features), dtype=np.float32)

    def _cluster_labels(self, features: NDArray[np.float32]) -> NDArray[np.int64]:
        norms = np.linalg.norm(features, axis=1)
        nonzero_mask = norms > 0.0
        labels = np.zeros(features.shape[0], dtype=np.int64)
        nonzero_count = int(nonzero_mask.sum())
        if nonzero_count == 0:
            return labels

        normalized = np.ascontiguousarray(features[nonzero_mask] / norms[nonzero_mask, None])
        cluster_count = min(self._config.cluster_count, nonzero_count)
        kmeans = faiss.Kmeans(
            d=normalized.shape[1],
            k=cluster_count,
            niter=20,
            seed=self._config.random_seed,
            spherical=True,
            verbose=False,
        )
        kmeans.train(normalized)
        _, nonzero_labels = kmeans.index.search(normalized, 1)
        labels[nonzero_mask] = nonzero_labels.reshape(-1)

        # Keep all zero directions together but separate from valid cosine features
        # so they cannot distort nonzero feature centroids.
        if not bool(nonzero_mask.all()):
            labels[~nonzero_mask] = cluster_count
        return labels

    def _factor_cluster(self, cluster_id: int, blocks: Sequence[TensorBlock]) -> FactorPair:
        matrix = np.hstack([block.values for block in blocks]).astype(np.float32, copy=False)
        # Exact thin SVD uses additional workspace to preserve the paper's optimal
        # Frobenius-norm approximation instead of trading accuracy for speed.
        left, singular_values, right_t = linalg.svd(matrix, full_matrices=False)
        rank = min(self._config.rank, singular_values.size)
        shared_d = (left[:, :rank] * singular_values[:rank]).astype(np.float32)
        full_c = right_t[:rank, :].astype(np.float32)
        c_values = tuple(
            np.ascontiguousarray(full_c[:, index * self._config.block_w : (index + 1) * self._config.block_w])
            for index in range(len(blocks))
        )
        reconstructed = shared_d @ full_c
        tail_error = float(np.linalg.norm(singular_values[rank:]))
        frobenius_error = float(np.linalg.norm(matrix - reconstructed, ord="fro"))
        return FactorPair(
            factor_id=f"factor-{cluster_id}",
            shared_d=shared_d,
            c_blocks=tuple(blocks),
            c_values=c_values,
            tail_error=tail_error,
            frobenius_error=frobenius_error,
        )
