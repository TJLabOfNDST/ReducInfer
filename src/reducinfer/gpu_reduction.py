"""Bounded-memory CuPy primitives for the three-pass reduction."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .streaming_reduction import ClusterFactor


@dataclass(frozen=True, slots=True)
class GpuReductionConfig:
    device_id: int = 0
    batch_blocks: int = 1024

    def __post_init__(self) -> None:
        if self.device_id < 0 or self.batch_blocks <= 0:
            raise ValueError("device_id must be non-negative and batch_blocks positive")


def _batches(count: int, size: int) -> Iterator[slice]:
    for start in range(0, count, size):
        yield slice(start, min(start + size, count))


class CuPyReductionBackend:
    """Execute dense square-block kernels with a strict device-memory batch bound.

    Keeping only one block batch plus K Gram matrices gives O(batch*b^2 + K*b^2)
    device space. Smaller batches increase PCIe transfers, while larger batches
    trade device memory for throughput. The configured bound prevents all eight
    models from occupying device memory simultaneously.
    """

    def __init__(
        self,
        config: GpuReductionConfig,
        *,
        array_module: Any | None = None,
    ) -> None:
        if array_module is None:
            import cupy as cp

            cp.cuda.Device(config.device_id).use()
            array_module = cp
        self._config = config
        self._xp = array_module
        self._projection_key: tuple[int, int] | None = None
        self._left_projection_device: Any = None
        self._right_projection_device: Any = None
        self._factor_key: int | None = None
        self._factor_left_device: Any = None
        self._factor_inverse_device: Any = None

    @property
    def batch_blocks(self) -> int:
        return self._config.batch_blocks

    def _synchronize(self) -> None:
        cuda = getattr(self._xp, "cuda", None)
        if cuda is not None:
            cuda.get_current_stream().synchronize()

    def _to_device(self, values: NDArray[np.generic]) -> Any:
        return self._xp.asarray(values)

    def _to_host(self, values: Any) -> NDArray[np.generic]:
        asnumpy = getattr(self._xp, "asnumpy", np.asarray)
        result = np.asarray(asnumpy(values))
        return result

    def _kernel(self, operation: Any) -> Any:
        return operation()

    @staticmethod
    def _validate_blocks(blocks: NDArray[np.float32]) -> None:
        if blocks.ndim != 3 or blocks.shape[0] == 0:
            raise ValueError("blocks must have shape (count, rows, cols)")
        if not np.isfinite(blocks).all():
            raise ValueError("blocks must contain only finite values")

    def project_features(
        self,
        blocks: NDArray[np.float32],
        left_projection: NDArray[np.float32],
        right_projection: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        self._validate_blocks(blocks)
        if left_projection.shape[1] != blocks.shape[1] or right_projection.shape[1] != blocks.shape[2]:
            raise ValueError("projection dimensions do not match block dimensions")
        projection_key = (id(left_projection), id(right_projection))
        if projection_key != self._projection_key:
            # Projections are reused by every block batch. A few KiB of persistent
            # device memory removes O(batch count) PCIe copies.
            self._left_projection_device = self._to_device(
                left_projection.astype(np.float32, copy=False)
            )
            self._right_projection_device = self._to_device(
                right_projection.astype(np.float32, copy=False)
            )
            self._projection_key = projection_key
        left = self._left_projection_device
        right = self._right_projection_device
        output = np.zeros(
            (blocks.shape[0], left_projection.shape[0] * right_projection.shape[0]),
            dtype=np.float32,
        )
        nonzero = np.zeros(blocks.shape[0], dtype=np.bool_)
        for batch_slice in _batches(blocks.shape[0], self._config.batch_blocks):
            batch = blocks[batch_slice]
            device_blocks = self._to_device(batch)

            def calculate() -> Any:
                norms = self._xp.linalg.norm(device_blocks, axis=(1, 2))
                safe_norms = self._xp.where(norms > 0, norms, 1.0)
                projected = self._xp.matmul(
                    self._xp.matmul(left[None, :, :], device_blocks / safe_norms[:, None, None]),
                    right.T[None, :, :],
                ).reshape(len(batch), -1)
                feature_norms = self._xp.linalg.norm(projected, axis=1)
                safe_features = self._xp.where(feature_norms > 0, feature_norms, 1.0)
                return projected / safe_features[:, None], (norms > 0) & (feature_norms > 0)

            device_features, device_nonzero = self._kernel(calculate)
            output[batch_slice] = self._to_host(device_features).astype(np.float32, copy=False)
            nonzero[batch_slice] = self._to_host(device_nonzero).astype(np.bool_, copy=False)
        return output, nonzero

    def accumulate_grams(
        self,
        blocks: NDArray[np.float32],
        labels: NDArray[np.int32],
        cluster_count: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        self._validate_blocks(blocks)
        if labels.shape != (blocks.shape[0],) or cluster_count <= 0:
            raise ValueError("labels and cluster_count are incompatible with blocks")
        if np.any(labels < 0) or np.any(labels >= cluster_count):
            raise ValueError("cluster label is outside the configured range")
        device_grams = self._xp.zeros(
            (cluster_count, blocks.shape[1], blocks.shape[1]), dtype=self._xp.float64
        )
        counts = np.bincount(labels, minlength=cluster_count).astype(np.int64, copy=False)
        for batch_slice in _batches(blocks.shape[0], self._config.batch_blocks):
            batch = blocks[batch_slice]
            device_blocks = self._to_device(batch.astype(np.float64, copy=False))
            batch_labels = labels[batch_slice]
            device_labels = self._to_device(batch_labels)

            def calculate() -> None:
                # Indexed addition visits every block Gram once: O(batch*b^2)
                # aggregation after the unavoidable O(batch*b^3) GEMM, avoiding
                # repeated scans of the batch for each cluster.
                block_grams = self._xp.matmul(
                    device_blocks, self._xp.swapaxes(device_blocks, 1, 2)
                )
                self._xp.add.at(device_grams, device_labels, block_grams)

            self._kernel(calculate)
        grams = self._to_host(device_grams).astype(np.float64, copy=False)
        return grams, counts

    def accumulate_gram_batches(
        self,
        batches: Iterable[tuple[NDArray[np.float32], NDArray[np.int32]]],
        *,
        cluster_count: int,
        block_extent: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Accumulate streamed batches while transferring K Gram matrices only once."""

        if cluster_count <= 0 or block_extent <= 0:
            raise ValueError("cluster_count and block_extent must be positive")
        device_grams = self._xp.zeros(
            (cluster_count, block_extent, block_extent), dtype=self._xp.float64
        )
        counts = np.zeros(cluster_count, dtype=np.int64)
        for blocks, labels in batches:
            self._validate_blocks(blocks)
            if blocks.shape[1] != block_extent or labels.shape != (blocks.shape[0],):
                raise ValueError("streamed Gram batch shape mismatch")
            if np.any(labels < 0) or np.any(labels >= cluster_count):
                raise ValueError("cluster label is outside the configured range")
            device_blocks = self._to_device(blocks.astype(np.float64, copy=False))
            device_labels = self._to_device(labels)
            counts += np.bincount(labels, minlength=cluster_count)

            def calculate() -> None:
                block_grams = self._xp.matmul(
                    device_blocks, self._xp.swapaxes(device_blocks, 1, 2)
                )
                self._xp.add.at(device_grams, device_labels, block_grams)

            self._kernel(calculate)
        return self._to_host(device_grams).astype(np.float64, copy=False), counts

    def factorize_grams(
        self,
        grams: NDArray[np.float64],
        counts: NDArray[np.int64],
        *,
        rank: int,
    ) -> tuple[ClusterFactor, ...]:
        if grams.ndim != 3 or grams.shape[1] != grams.shape[2] or counts.shape != (grams.shape[0],):
            raise ValueError("Gram and count shapes are incompatible")
        if rank <= 0 or rank > grams.shape[1]:
            raise ValueError("rank must fit the Gram matrix extent")
        device_grams = self._to_device(grams)
        eigenvalues, eigenvectors = self._kernel(lambda: self._xp.linalg.eigh(device_grams))
        eigenvalues_host = self._to_host(eigenvalues).astype(np.float64, copy=False)
        eigenvectors_host = self._to_host(eigenvectors).astype(np.float64, copy=False)
        factors: list[ClusterFactor] = []
        for cluster_id in range(grams.shape[0]):
            order = np.argsort(eigenvalues_host[cluster_id])[::-1]
            values = np.maximum(eigenvalues_host[cluster_id][order], 0.0)
            vectors = eigenvectors_host[cluster_id][:, order]
            singular = np.sqrt(values)
            kept = singular[:rank]
            left = vectors[:, :rank]
            tolerance = np.finfo(np.float64).eps * grams.shape[1] * (float(kept[0]) if kept.size else 0.0)
            inverse = np.divide(1.0, kept, out=np.zeros_like(kept), where=kept > tolerance)
            factors.append(
                ClusterFactor(
                    cluster_id,
                    np.ascontiguousarray(left * kept, dtype=np.float32),
                    np.ascontiguousarray(left, dtype=np.float32),
                    np.ascontiguousarray(kept, dtype=np.float32),
                    np.ascontiguousarray(inverse, dtype=np.float32),
                    float(np.sqrt(np.sum(values[rank:], dtype=np.float64))),
                    int(counts[cluster_id]),
                )
            )
        return tuple(factors)

    def project_coefficients(
        self,
        blocks: NDArray[np.float32],
        labels: NDArray[np.int32],
        factors: Sequence[ClusterFactor],
    ) -> NDArray[np.float32]:
        self._validate_blocks(blocks)
        if labels.shape != (blocks.shape[0],) or not factors:
            raise ValueError("labels and factors are required for coefficient projection")
        rank = factors[0].shared_d.shape[1]
        if any(
            factor.left_vectors.shape != factors[0].left_vectors.shape
            or factor.inverse_singular_values.shape != (rank,)
            for factor in factors
        ):
            raise ValueError("all cluster factors must share one shape and rank")
        output = np.empty((blocks.shape[0], rank, blocks.shape[2]), dtype=np.float32)
        factor_key = id(factors)
        if factor_key != self._factor_key:
            # O(K*b*r) read-only factors remain resident through the coefficient
            # pass; this bounded allocation avoids uploading them once per tensor.
            self._factor_left_device = self._to_device(
                np.stack([factor.left_vectors for factor in factors]).astype(
                    np.float32, copy=False
                )
            )
            self._factor_inverse_device = self._to_device(
                np.stack([factor.inverse_singular_values for factor in factors]).astype(
                    np.float32, copy=False
                )
            )
            self._factor_key = factor_key
        device_left = self._factor_left_device
        device_inverse = self._factor_inverse_device
        for batch_slice in _batches(blocks.shape[0], self._config.batch_blocks):
            batch = blocks[batch_slice]
            batch_labels = labels[batch_slice]
            if np.any(batch_labels < 0) or np.any(batch_labels >= len(factors)):
                raise ValueError("cluster label is outside the factor range")
            device_blocks = self._to_device(batch)
            device_labels = self._to_device(batch_labels)

            def calculate() -> Any:
                selected_left = device_left[device_labels]
                projected = self._xp.matmul(
                    self._xp.swapaxes(selected_left, 1, 2), device_blocks
                )
                return projected * device_inverse[device_labels, :, None]

            output[batch_slice] = self._to_host(self._kernel(calculate)).astype(
                np.float32, copy=False
            )
        if not np.isfinite(output).all():
            raise ValueError("coefficient projection produced non-finite values")
        return output

    def release(self) -> None:
        """Release cached device blocks between ranks to keep peak RSS attributable."""

        cuda = getattr(self._xp, "cuda", None)
        if cuda is not None:
            self._synchronize()
            self._left_projection_device = None
            self._right_projection_device = None
            self._factor_left_device = None
            self._factor_inverse_device = None
            self._projection_key = None
            self._factor_key = None
            self._xp.get_default_memory_pool().free_all_blocks()
            self._xp.get_default_pinned_memory_pool().free_all_blocks()


def require_cuda_device(device_id: int) -> dict[str, object]:
    """Fail fast before any database reset when the selected CUDA device is unusable."""

    if device_id < 0:
        raise ValueError("device_id must be non-negative")
    try:
        import cupy as cp

        device = cp.cuda.Device(device_id)
        device.use()
        allocation = cp.empty(1, dtype=cp.float32)
        cp.cuda.get_current_stream().synchronize()
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        del allocation
        return {
            "device_id": device_id,
            "device_name": str(name),
            "compute_capability": f"{properties['major']}.{properties['minor']}",
            "total_memory_bytes": int(properties["totalGlobalMem"]),
        }
    except Exception as error:
        raise RuntimeError(f"CUDA device {device_id} is unavailable for CuPy preprocessing") from error
