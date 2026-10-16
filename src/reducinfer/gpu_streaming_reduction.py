"""Three-pass GPU preprocessing with import-ready continuous C artifacts."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .checkpoint import TensorBlockRecord
from .config import ReducInferConfig
from .gpu_reduction import CuPyReductionBackend
from .streaming_reduction import (
    BlockFactory,
    ReductionArtifactMetadata,
    StreamingReductionResult,
    ClusterFactor,
    automatic_cluster_count,
    cluster_features,
    save_reduction_result,
    load_reduction_result,
)

ProgressCallback = Callable[[str, int, int], None]

TOPOLOGY_DTYPE = np.dtype(
    [
        ("tensor_index", "<i4"),
        ("cluster_id", "<i4"),
        ("row_offset", "<i4"),
        ("col_offset", "<i4"),
        ("c_start", "<i8"),
        ("valid_rows", "<i2"),
        ("valid_cols", "<i2"),
    ]
)


@dataclass(frozen=True, slots=True)
class PackedTensor:
    tensor_index: int
    model_id: str
    tensor_name: str
    rows: int
    cols: int
    rank: int
    rank_stride: int
    c_element_offset: int
    c_element_count: int
    topology_start: int
    topology_count: int


@dataclass(frozen=True, slots=True)
class GpuStreamingReductionResult:
    reduction: StreamingReductionResult
    c_buffer_path: Path
    topology_path: Path
    tensor_manifest_path: Path
    tensors: tuple[PackedTensor, ...]


def _emit(progress: ProgressCallback | None, phase: str, completed: int, total: int) -> None:
    if progress is not None:
        progress(phase, completed, total)


def _block_batches(
    iterator: Iterator[TensorBlockRecord], batch_size: int
) -> Iterator[tuple[TensorBlockRecord, ...]]:
    batch: list[TensorBlockRecord] = []
    for block in iterator:
        batch.append(block)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


class GpuStreamingReducer:
    """Reduce checkpoints in three deterministic scans and persist import artifacts."""

    def __init__(
        self,
        config: ReducInferConfig,
        *,
        work_dir: Path,
        backend: CuPyReductionBackend,
        cluster_count_override: int | None = None,
    ) -> None:
        if cluster_count_override is not None and cluster_count_override <= 0:
            raise ValueError("cluster count override must be positive")
        self._config = config
        self._work_dir = work_dir
        self._backend = backend
        self._cluster_count_override = cluster_count_override
        generator = np.random.default_rng(config.random_seed)
        self._left_projection = generator.standard_normal(
            (config.brp_d1, config.block_h), dtype=np.float32
        )
        self._right_projection = generator.standard_normal(
            (config.brp_d2, config.block_w), dtype=np.float32
        )

    def _cluster_count(self, nonzero_count: int) -> int:
        """Resolve one fixed override without changing the automatic default.

        The override serves standalone models that require a fixed D shape. The
        default still derives the cluster count from nonzero blocks, preserving
        the established time complexity and numerical behavior.
        """

        if self._cluster_count_override is None:
            return automatic_cluster_count(nonzero_count)
        if self._cluster_count_override > nonzero_count:
            raise ValueError("cluster count override exceeds the nonzero block count")
        return self._cluster_count_override

    def reduce(
        self,
        block_factory: BlockFactory,
        *,
        metadata: ReductionArtifactMetadata,
        progress: ProgressCallback | None = None,
    ) -> GpuStreamingReductionResult:
        self._work_dir.mkdir(parents=True, exist_ok=True)
        feature_path = self._work_dir / "features.f32"
        ordinal_path = self._work_dir / "nonzero_ordinals.i64"
        block_count, nonzero_count, tensor_shapes = self._feature_pass(
            block_factory, feature_path, ordinal_path, progress
        )
        if block_count == 0:
            raise ValueError("checkpoint contains no reducible matrix blocks")
        cluster_count = self._cluster_count(nonzero_count)
        label_path = self._work_dir / "labels.i32"
        labels = np.memmap(label_path, mode="w+", dtype=np.int32, shape=(block_count,))
        labels[:] = -1
        _emit(progress, "clustering", 0, 1)
        if nonzero_count:
            features = np.memmap(
                feature_path,
                mode="r",
                dtype=np.float32,
                shape=(nonzero_count, self._config.brp_d1 * self._config.brp_d2),
            )
            ordinals = np.memmap(ordinal_path, mode="r", dtype=np.int64, shape=(nonzero_count,))
            labels[ordinals] = cluster_features(
                features, cluster_count, random_seed=self._config.random_seed
            )
        labels.flush()
        _emit(progress, "clustering", 1, 1)
        if not nonzero_count:
            raise ValueError("checkpoint contains only zero matrix blocks")
        grams, counts = self._gram_pass(block_factory, labels, cluster_count, block_count, progress)
        _emit(progress, "factorization", 0, cluster_count)
        factors = self._backend.factorize_grams(grams, counts, rank=self._config.rank)
        _emit(progress, "factorization", cluster_count, cluster_count)
        base = StreamingReductionResult(
            block_count,
            nonzero_count,
            cluster_count,
            feature_path,
            label_path,
            factors,
            tensor_shapes,
        )
        c_path, topology_path, manifest_path, tensors = self._packing_pass(
            block_factory, labels, factors, block_count, tensor_shapes, progress
        )
        save_reduction_result(base, self._work_dir, metadata=metadata)
        return GpuStreamingReductionResult(
            base,
            c_path,
            topology_path,
            manifest_path,
            tensors,
        )

    def _feature_pass(
        self,
        block_factory: BlockFactory,
        feature_path: Path,
        ordinal_path: Path,
        progress: ProgressCallback | None,
    ) -> tuple[int, int, tuple[tuple[str, str, int, int], ...]]:
        block_count = nonzero_count = 0
        shapes: dict[tuple[str, str], tuple[int, int]] = {}
        with feature_path.open("wb") as feature_stream, ordinal_path.open("wb") as ordinal_stream:
            for batch in _block_batches(block_factory(), self._backend.batch_blocks):
                for block in batch:
                    if block.ordinal != block_count:
                        raise ValueError("block ordinals must begin at zero and remain contiguous")
                    key = (block.model_id, block.tensor_name)
                    rows, cols = shapes.get(key, (0, 0))
                    shapes[key] = (
                        max(rows, block.row_offset + block.valid_shape[0]),
                        max(cols, block.col_offset + block.valid_shape[1]),
                    )
                    block_count += 1
                values = np.stack([block.values for block in batch]).astype(np.float32, copy=False)
                features, nonzero = self._backend.project_features(
                    values, self._left_projection, self._right_projection
                )
                features[nonzero].tofile(feature_stream)
                np.asarray([block.ordinal for block, keep in zip(batch, nonzero, strict=True) if keep], dtype=np.int64).tofile(ordinal_stream)
                nonzero_count += int(nonzero.sum())
                _emit(progress, "features", block_count, 0)
        tensor_shapes = tuple(
            (model_id, tensor_name, rows, cols)
            for (model_id, tensor_name), (rows, cols) in shapes.items()
        )
        _emit(progress, "features", block_count, block_count)
        return block_count, nonzero_count, tensor_shapes

    def _gram_pass(
        self,
        block_factory: BlockFactory,
        labels: NDArray[np.int32],
        cluster_count: int,
        block_count: int,
        progress: ProgressCallback | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        seen = 0

        def eligible_batches() -> Iterator[tuple[NDArray[np.float32], NDArray[np.int32]]]:
            nonlocal seen
            for batch in _block_batches(block_factory(), self._backend.batch_blocks):
                if any(block.ordinal != seen + index for index, block in enumerate(batch)):
                    raise ValueError("block ordinals changed during Gram pass")
                batch_labels = np.asarray([labels[block.ordinal] for block in batch], dtype=np.int32)
                values = np.stack([block.values for block in batch]).astype(np.float32, copy=False)
                keep = batch_labels >= 0
                seen += len(batch)
                _emit(progress, "grams", seen, block_count)
                if keep.any():
                    yield values[keep], batch_labels[keep]

        grams, counts = self._backend.accumulate_gram_batches(
            eligible_batches(),
            cluster_count=cluster_count,
            block_extent=self._config.block_h,
        )
        if seen != block_count:
            raise ValueError("block count changed during Gram pass")
        return grams, counts

    def _packing_pass(
        self,
        block_factory: BlockFactory,
        labels: NDArray[np.int32],
        factors: tuple[ClusterFactor, ...],
        block_count: int,
        tensor_shapes: tuple[tuple[str, str, int, int], ...],
        progress: ProgressCallback | None,
    ) -> tuple[Path, Path, Path, tuple[PackedTensor, ...]]:
        c_path = self._work_dir / "c_buffers.f32"
        topology_path = self._work_dir / "block_topology.bin"
        manifest_path = self._work_dir / "tensor_batches.jsonl"
        shape_by_key = {(m, n): (r, c) for m, n, r, c in tensor_shapes}
        tensors: list[PackedTensor] = []
        current_key: tuple[str, str] | None = None
        current_blocks: list[TensorBlockRecord] = []
        seen = 0
        c_elements = topology_records = 0

        with c_path.open("wb") as c_stream, topology_path.open("wb") as topology_stream, manifest_path.open("w", encoding="utf-8") as manifest_stream:
            def flush_tensor() -> None:
                nonlocal c_elements, topology_records
                if current_key is None:
                    return
                kept = [block for block in current_blocks if int(labels[block.ordinal]) >= 0]
                rows, cols = shape_by_key[current_key]
                if kept:
                    block_values = np.stack([block.values for block in kept]).astype(np.float32, copy=False)
                    block_labels = np.asarray([labels[block.ordinal] for block in kept], dtype=np.int32)
                    coefficients = self._backend.project_coefficients(block_values, block_labels, factors)
                    packed = np.ascontiguousarray(coefficients.transpose(1, 0, 2).reshape(self._config.rank, -1), dtype=np.float32)
                    packed.tofile(c_stream)
                    topology = np.empty(len(kept), dtype=TOPOLOGY_DTYPE)
                    for index, (block, cluster_id) in enumerate(zip(kept, block_labels, strict=True)):
                        topology[index] = (
                            len(tensors), int(cluster_id), block.row_offset, block.col_offset,
                            index * self._config.block_w, block.valid_shape[0], block.valid_shape[1],
                        )
                    topology.tofile(topology_stream)
                    stride = len(kept) * self._config.block_w
                else:
                    packed = np.empty((self._config.rank, 0), dtype=np.float32)
                    stride = 0
                tensor = PackedTensor(
                    len(tensors), current_key[0], current_key[1], rows, cols,
                    self._config.rank, stride, c_elements, int(packed.size),
                    topology_records, len(kept),
                )
                manifest_stream.write(json.dumps(asdict(tensor), ensure_ascii=False) + "\n")
                manifest_stream.flush()
                os.fsync(manifest_stream.fileno())
                tensors.append(tensor)
                c_elements += int(packed.size)
                topology_records += len(kept)

            for block in block_factory():
                if block.ordinal != seen:
                    raise ValueError("block ordinals changed during coefficient pass")
                key = (block.model_id, block.tensor_name)
                if current_key is not None and key != current_key:
                    flush_tensor()
                    current_blocks = []
                current_key = key
                current_blocks.append(block)
                seen += 1
                _emit(progress, "coefficients", seen, block_count)
            flush_tensor()
            c_stream.flush(); os.fsync(c_stream.fileno())
            topology_stream.flush(); os.fsync(topology_stream.fileno())
        if seen != block_count:
            raise ValueError("block count changed during coefficient pass")
        return c_path, topology_path, manifest_path, tuple(tensors)


def load_gpu_streaming_reduction(
    work_dir: Path,
    *,
    expected: ReductionArtifactMetadata | None = None,
) -> GpuStreamingReductionResult:
    """Load a completed import-ready reduction for an explicit resume."""

    base = load_reduction_result(work_dir / "reduction_manifest.json", expected=expected)
    tensors: list[PackedTensor] = []
    manifest_path = work_dir / "tensor_batches.jsonl"
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        tensors.append(PackedTensor(**json.loads(line)))
    result = GpuStreamingReductionResult(
        base,
        work_dir / "c_buffers.f32",
        work_dir / "block_topology.bin",
        manifest_path,
        tuple(tensors),
    )
    required = (result.c_buffer_path, result.topology_path, result.tensor_manifest_path)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("GPU reduction artifacts are incomplete")
    return result
