"""Three-pass global reduction with bounded raw-weight memory."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
import json
from math import floor, sqrt
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from .checkpoint import TensorBlockRecord
from .config import ReducInferConfig

BlockFactory = Callable[[], Iterator[TensorBlockRecord]]


def automatic_cluster_count(nonzero_block_count: int) -> int:
    """Return the approved nearest integer to `sqrt(N / 2)`."""

    if nonzero_block_count < 0:
        raise ValueError("nonzero block count must not be negative")
    if nonzero_block_count == 0:
        return 0
    return max(1, floor(sqrt(nonzero_block_count / 2.0) + 0.5))


def cluster_features(
    features: NDArray[np.float32], cluster_count: int, *, random_seed: int
) -> NDArray[np.int32]:
    """Run the approved deterministic spherical FAISS K-means contract."""

    if features.ndim != 2 or features.shape[0] == 0 or cluster_count <= 0:
        raise ValueError("non-empty two-dimensional features and clusters are required")
    if cluster_count > features.shape[0]:
        raise ValueError("cluster count cannot exceed feature count")
    if cluster_count == 1:
        return np.zeros(features.shape[0], dtype=np.int32)
    kmeans = faiss.Kmeans(
        d=features.shape[1],
        k=cluster_count,
        niter=20,
        seed=random_seed,
        spherical=True,
        verbose=False,
    )
    contiguous = np.ascontiguousarray(features, dtype=np.float32)
    kmeans.train(contiguous)
    _, labels = kmeans.index.search(contiguous, 1)
    return np.ascontiguousarray(labels.reshape(-1), dtype=np.int32)


@dataclass(frozen=True, slots=True)
class ClusterFactor:
    cluster_id: int
    shared_d: NDArray[np.float32]
    left_vectors: NDArray[np.float32]
    singular_values: NDArray[np.float32]
    inverse_singular_values: NDArray[np.float32]
    tail_error: float
    block_count: int


@dataclass(frozen=True, slots=True)
class CoefficientRecord:
    block: TensorBlockRecord
    cluster_id: int
    values: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class TensorCoefficientBatch:
    model_id: str
    tensor_name: str
    blocks: tuple[TensorBlockRecord, ...]
    cluster_ids: tuple[int, ...]
    slice_starts: tuple[int, ...]
    rank_stride: int
    values: NDArray[np.float32]


def group_tensor_coefficients(
    records: Iterator[CoefficientRecord],
) -> Iterator[TensorCoefficientBatch]:
    """Build one rank-major C buffer at a time from tensor-ordered records."""

    current_key: tuple[str, str] | None = None
    blocks: list[TensorBlockRecord] = []
    cluster_ids: list[int] = []
    values: list[NDArray[np.float32]] = []
    starts: list[int] = []
    stride = 0
    seen_keys: set[tuple[str, str]] = set()

    def build() -> TensorCoefficientBatch:
        if current_key is None or not values:
            raise RuntimeError("cannot build an empty tensor coefficient batch")
        return TensorCoefficientBatch(
            model_id=current_key[0],
            tensor_name=current_key[1],
            blocks=tuple(blocks),
            cluster_ids=tuple(cluster_ids),
            slice_starts=tuple(starts),
            rank_stride=stride,
            values=np.ascontiguousarray(np.concatenate(values, axis=1), dtype=np.float32),
        )

    for record in records:
        key = (record.block.model_id, record.block.tensor_name)
        if current_key is not None and key != current_key:
            seen_keys.add(current_key)
            yield build()
            blocks = []
            cluster_ids = []
            values = []
            starts = []
            stride = 0
        if key in seen_keys:
            raise ValueError("coefficient stream returned to an already completed tensor")
        if values and record.values.shape[0] != values[0].shape[0]:
            raise ValueError("one tensor cannot mix factor ranks")
        current_key = key
        blocks.append(record.block)
        cluster_ids.append(record.cluster_id)
        starts.append(stride)
        values.append(record.values)
        stride += record.values.shape[1]
    if current_key is not None:
        yield build()


def accumulate_cluster_grams(
    blocks: Iterator[TensorBlockRecord],
    labels: NDArray[np.int32],
    cluster_count: int,
    block_h: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Accumulate all `B @ B.T` matrices in one sequential scan.

    FP64 accumulation spends `O(K*b^2)` space to avoid an unbounded horizontal
    concatenation. It reduces rounding error across many block sums at the cost
    of doubling only the compact Gram arrays.
    """

    if cluster_count < 0 or block_h <= 0:
        raise ValueError("cluster_count must be non-negative and block_h positive")
    grams = np.zeros((cluster_count, block_h, block_h), dtype=np.float64)
    counts = np.zeros(cluster_count, dtype=np.int64)
    seen = 0
    for block in blocks:
        if block.ordinal != seen:
            raise ValueError("block ordinals must be contiguous and deterministic")
        if block.values.shape[0] != block_h:
            raise ValueError("block height differs from the reduction configuration")
        label = int(labels[block.ordinal])
        if label >= 0:
            if label >= cluster_count:
                raise ValueError("cluster label is outside the configured range")
            values = block.values.astype(np.float64, copy=False)
            grams[label] += values @ values.T
            counts[label] += 1
        seen += 1
    if seen != labels.size:
        raise ValueError("label count differs from the deterministic block scan")
    return grams, counts


def factor_cluster_grams(
    grams: NDArray[np.float64], counts: NDArray[np.int64], rank: int
) -> tuple[ClusterFactor, ...]:
    """Recover thin-SVD left factors from streamed cluster Gram matrices."""

    if grams.ndim != 3 or grams.shape[1] != grams.shape[2] or counts.shape != (grams.shape[0],):
        raise ValueError("Gram and count shapes are incompatible")
    if rank <= 0 or rank > grams.shape[1]:
        raise ValueError("rank must fit the Gram matrix extent")
    factors: list[ClusterFactor] = []
    for cluster_id, gram in enumerate(grams):
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        singular_values = np.sqrt(eigenvalues)
        kept = singular_values[:rank]
        left = eigenvectors[:, :rank]
        tolerance = np.finfo(np.float64).eps * gram.shape[0] * (float(kept[0]) if kept.size else 0.0)
        inverse = np.divide(1.0, kept, out=np.zeros_like(kept), where=kept > tolerance)
        factors.append(
            ClusterFactor(
                cluster_id=cluster_id,
                shared_d=np.ascontiguousarray(left * kept, dtype=np.float32),
                left_vectors=np.ascontiguousarray(left, dtype=np.float32),
                singular_values=np.ascontiguousarray(kept, dtype=np.float32),
                inverse_singular_values=np.ascontiguousarray(inverse, dtype=np.float32),
                tail_error=float(np.sqrt(np.sum(eigenvalues[rank:], dtype=np.float64))),
                block_count=int(counts[cluster_id]),
            )
        )
    return tuple(factors)


def project_coefficient(factor: ClusterFactor, block: NDArray[np.float32]) -> NDArray[np.float32]:
    """Project one block into its cluster basis without retaining neighboring blocks."""

    if block.ndim != 2 or block.shape[0] != factor.left_vectors.shape[0]:
        raise ValueError("block shape is incompatible with its cluster factor")
    projected = factor.left_vectors.T @ block
    return np.ascontiguousarray(factor.inverse_singular_values[:, None] * projected, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class StreamingReductionResult:
    block_count: int
    nonzero_block_count: int
    cluster_count: int
    feature_path: Path
    label_path: Path
    factors: tuple[ClusterFactor, ...]
    tensor_shapes: tuple[tuple[str, str, int, int], ...]

    def iter_coefficients(self, block_factory: BlockFactory) -> Iterator[CoefficientRecord]:
        labels = np.memmap(self.label_path, mode="r", dtype=np.int32, shape=(self.block_count,))
        seen = 0
        for block in block_factory():
            if block.ordinal != seen:
                raise ValueError("block ordinals changed between reduction passes")
            label = int(labels[block.ordinal])
            if label >= 0:
                yield CoefficientRecord(block, label, project_coefficient(self.factors[label], block.values))
            seen += 1
        if seen != self.block_count:
            raise ValueError("block count changed between reduction passes")


@dataclass(frozen=True, slots=True)
class ArtifactModel:
    model_id: str
    revision: str
    checkpoint_bytes: int

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision or self.checkpoint_bytes <= 0:
            raise ValueError("artifact model metadata is incomplete")


@dataclass(frozen=True, slots=True)
class ReductionArtifactMetadata:
    format_version: int
    block_h: int
    block_w: int
    rank: int
    models: tuple[ArtifactModel, ...]

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise ValueError("unsupported reduction artifact format version")
        if min(self.block_h, self.block_w, self.rank) <= 0:
            raise ValueError("artifact dimensions must be positive")
        if self.rank > min(self.block_h, self.block_w):
            raise ValueError("artifact rank exceeds its block extent")
        identifiers = tuple(model.model_id for model in self.models)
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("artifact models must be non-empty and unique")


def save_reduction_result(
    result: StreamingReductionResult,
    work_dir: Path,
    *,
    metadata: ReductionArtifactMetadata,
) -> Path:
    """Persist compact factors and topology metadata, never raw checkpoint blocks."""

    work_dir.mkdir(parents=True, exist_ok=True)
    factor_path = work_dir / "cluster_factors.npz"
    if result.factors:
        np.savez(
            factor_path,
            shared_d=np.stack([factor.shared_d for factor in result.factors]),
            left_vectors=np.stack([factor.left_vectors for factor in result.factors]),
            singular_values=np.stack([factor.singular_values for factor in result.factors]),
            inverse_singular_values=np.stack(
                [factor.inverse_singular_values for factor in result.factors]
            ),
            tail_errors=np.asarray([factor.tail_error for factor in result.factors], dtype=np.float64),
            block_counts=np.asarray([factor.block_count for factor in result.factors], dtype=np.int64),
        )
    else:
        np.savez(
            factor_path,
            shared_d=np.empty((0, 0, 0), dtype=np.float32),
            left_vectors=np.empty((0, 0, 0), dtype=np.float32),
            singular_values=np.empty((0, 0), dtype=np.float32),
            inverse_singular_values=np.empty((0, 0), dtype=np.float32),
            tail_errors=np.empty(0, dtype=np.float64),
            block_counts=np.empty(0, dtype=np.int64),
        )
    manifest_path = work_dir / "reduction_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "block_count": result.block_count,
                "nonzero_block_count": result.nonzero_block_count,
                "cluster_count": result.cluster_count,
                "feature_file": result.feature_path.name,
                "label_file": result.label_path.name,
                "factor_file": factor_path.name,
                "tensor_shapes": result.tensor_shapes,
                "metadata": {
                    **asdict(metadata),
                    "models": [asdict(model) for model in metadata.models],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def load_reduction_result(
    manifest_path: Path,
    *,
    expected: ReductionArtifactMetadata | None = None,
) -> StreamingReductionResult:
    """Load a resumable reduction result and validate its compact file set."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_row = manifest.get("metadata")
    if not isinstance(metadata_row, dict):
        raise ValueError("reduction artifact metadata is missing")
    try:
        metadata = ReductionArtifactMetadata(
            format_version=int(metadata_row["format_version"]),
            block_h=int(metadata_row["block_h"]),
            block_w=int(metadata_row["block_w"]),
            rank=int(metadata_row["rank"]),
            models=tuple(ArtifactModel(**model) for model in metadata_row["models"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("reduction artifact metadata is invalid") from error
    if expected is not None and metadata != expected:
        raise ValueError("reduction artifact metadata does not match the requested import")
    root = manifest_path.parent
    factor_path = root / str(manifest["factor_file"])
    if not factor_path.is_file():
        raise FileNotFoundError(f"reduction factor file is missing: {factor_path}")
    with np.load(factor_path, allow_pickle=False) as arrays:
        factors = tuple(
            ClusterFactor(
                cluster_id=cluster_id,
                shared_d=np.ascontiguousarray(arrays["shared_d"][cluster_id], dtype=np.float32),
                left_vectors=np.ascontiguousarray(
                    arrays["left_vectors"][cluster_id], dtype=np.float32
                ),
                singular_values=np.ascontiguousarray(
                    arrays["singular_values"][cluster_id], dtype=np.float32
                ),
                inverse_singular_values=np.ascontiguousarray(
                    arrays["inverse_singular_values"][cluster_id], dtype=np.float32
                ),
                tail_error=float(arrays["tail_errors"][cluster_id]),
                block_count=int(arrays["block_counts"][cluster_id]),
            )
            for cluster_id in range(int(manifest["cluster_count"]))
        )
    feature_path = root / str(manifest["feature_file"])
    label_path = root / str(manifest["label_file"])
    if not feature_path.is_file() or not label_path.is_file():
        raise FileNotFoundError("reduction feature or label file is missing")
    return StreamingReductionResult(
        block_count=int(manifest["block_count"]),
        nonzero_block_count=int(manifest["nonzero_block_count"]),
        cluster_count=int(manifest["cluster_count"]),
        feature_path=feature_path,
        label_path=label_path,
        factors=factors,
        tensor_shapes=tuple(
            (str(model_id), str(tensor_name), int(rows), int(cols))
            for model_id, tensor_name, rows, cols in manifest["tensor_shapes"]
        ),
    )


class StreamingReducer:
    """Run feature, Gram, and coefficient passes over a repeatable block source."""

    def __init__(self, config: ReducInferConfig, *, work_dir: Path) -> None:
        self._config = config
        self._work_dir = work_dir
        generator = np.random.default_rng(config.random_seed)
        self._left_projection = generator.standard_normal((config.brp_d1, config.block_h), dtype=np.float32)
        self._right_projection = generator.standard_normal((config.brp_d2, config.block_w), dtype=np.float32)

    def reduce(self, block_factory: BlockFactory) -> StreamingReductionResult:
        self._work_dir.mkdir(parents=True, exist_ok=True)
        feature_path = self._work_dir / "features.f32"
        ordinal_path = self._work_dir / "nonzero_ordinals.i64"
        block_count, nonzero_count, tensor_shapes = self._write_features(
            block_factory, feature_path, ordinal_path
        )
        if block_count == 0:
            raise ValueError("checkpoint contains no reducible matrix blocks")
        cluster_count = automatic_cluster_count(nonzero_count)
        label_path = self._work_dir / "labels.i32"
        labels = np.memmap(label_path, mode="w+", dtype=np.int32, shape=(block_count,))
        labels[:] = -1
        if nonzero_count:
            features = np.memmap(
                feature_path,
                mode="r",
                dtype=np.float32,
                shape=(nonzero_count, self._config.brp_d1 * self._config.brp_d2),
            )
            ordinals = np.memmap(ordinal_path, mode="r", dtype=np.int64, shape=(nonzero_count,))
            labels[ordinals] = self._cluster(features, cluster_count)
        labels.flush()
        grams, counts = accumulate_cluster_grams(
            block_factory(), labels, cluster_count, self._config.block_h
        )
        factors = factor_cluster_grams(grams, counts, self._config.rank) if cluster_count else ()
        return StreamingReductionResult(
            block_count=block_count,
            nonzero_block_count=nonzero_count,
            cluster_count=cluster_count,
            feature_path=feature_path,
            label_path=label_path,
            factors=factors,
            tensor_shapes=tensor_shapes,
        )

    def _write_features(
        self, block_factory: BlockFactory, feature_path: Path, ordinal_path: Path
    ) -> tuple[int, int, tuple[tuple[str, str, int, int], ...]]:
        block_count = 0
        nonzero_count = 0
        shapes: dict[tuple[str, str], tuple[int, int]] = {}
        with feature_path.open("wb") as feature_stream, ordinal_path.open("wb") as ordinal_stream:
            for block in block_factory():
                if block.ordinal != block_count:
                    raise ValueError("block ordinals must begin at zero and remain contiguous")
                if block.values.shape != (self._config.block_h, self._config.block_w):
                    raise ValueError("block shape differs from the reduction configuration")
                tensor_key = (block.model_id, block.tensor_name)
                previous_rows, previous_cols = shapes.get(tensor_key, (0, 0))
                shapes[tensor_key] = (
                    max(previous_rows, block.row_offset + block.valid_shape[0]),
                    max(previous_cols, block.col_offset + block.valid_shape[1]),
                )
                norm = float(np.linalg.norm(block.values, ord="fro"))
                if norm > 0.0:
                    normalized = block.values / norm
                    feature = (self._left_projection @ normalized @ self._right_projection.T).reshape(-1)
                    feature_norm = float(np.linalg.norm(feature))
                    if feature_norm > 0.0:
                        np.asarray(feature / feature_norm, dtype=np.float32).tofile(feature_stream)
                        np.asarray(block.ordinal, dtype=np.int64).tofile(ordinal_stream)
                        nonzero_count += 1
                block_count += 1
        return block_count, nonzero_count, tuple(
            (model_id, tensor_name, rows, cols)
            for (model_id, tensor_name), (rows, cols) in shapes.items()
        )

    def _cluster(self, features: NDArray[np.float32], cluster_count: int) -> NDArray[np.int32]:
        return cluster_features(features, cluster_count, random_seed=self._config.random_seed)
