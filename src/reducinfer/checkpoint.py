"""Deterministic, bounded-memory checkpoint tensor and block scanning."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .hf_models import ModelDefinition
from .model_adapter import adapt_matrix, adapt_vector


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    """Describe a source tensor without materializing its payload."""

    shape: tuple[int, ...]
    dtype: str
    is_floating: bool


def is_learned_parameter_name(name: str) -> bool:
    """Reject persistent framework buffers before their payload is loaded.

    Filtering by metadata and name first saves both I/O and conversion memory
    while excluding position indices, masks, and RoPE constants.
    """

    return not name.endswith(
        (
            ".attn.bias",
            ".attn.masked_bias",
            ".attention.bias",
            ".attention.masked_bias",
            ".rotary_emb.inv_freq",
            ".position_ids",
            ".token_type_ids",
        )
    )


class TensorSource(Protocol):
    """Expose one tensor at a time from a checkpoint-backed source."""

    def names(self) -> tuple[str, ...]: ...

    def metadata(self, name: str) -> TensorMetadata: ...

    def load(self, name: str) -> NDArray[np.generic]: ...

    def close(self) -> None: ...


class MappingTensorSource:
    """Adapt an existing tensor mapping to the scanner contract."""

    def __init__(self, tensors: Mapping[str, NDArray[np.generic]]) -> None:
        self._tensors = tensors

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tensors))

    def metadata(self, name: str) -> TensorMetadata:
        values = self._tensors[name]
        return TensorMetadata(
            shape=values.shape,
            dtype=str(values.dtype),
            is_floating=bool(np.issubdtype(values.dtype, np.floating)),
        )

    def load(self, name: str) -> NDArray[np.generic]:
        return self._tensors[name]

    def close(self) -> None:
        """Leave caller-owned mappings unchanged."""


class SafetensorSource:
    """Index local Safetensors shards and materialize only the requested tensor."""

    def __init__(self, checkpoint_dir: Path) -> None:
        from safetensors import safe_open

        if not checkpoint_dir.is_dir():
            raise ValueError(f"checkpoint directory does not exist: {checkpoint_dir}")
        tensor_paths: dict[str, Path] = {}
        tensor_metadata: dict[str, TensorMetadata] = {}
        dtype_names = {
            "BF16": "bfloat16",
            "F16": "float16",
            "F32": "float32",
            "F64": "float64",
            "I8": "int8",
            "I16": "int16",
            "I32": "int32",
            "I64": "int64",
            "U8": "uint8",
            "BOOL": "bool",
        }
        for shard_path in sorted(checkpoint_dir.glob("*.safetensors")):
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                for name in shard.keys():
                    if name in tensor_paths:
                        raise ValueError(f"duplicate tensor across Safetensors shards: {name}")
                    tensor_paths[name] = shard_path
                    tensor_slice = shard.get_slice(name)
                    source_dtype = str(tensor_slice.get_dtype())
                    tensor_metadata[name] = TensorMetadata(
                        shape=tuple(int(dimension) for dimension in tensor_slice.get_shape()),
                        dtype=dtype_names.get(source_dtype, source_dtype.lower()),
                        is_floating=source_dtype in {"BF16", "F16", "F32", "F64"},
                    )
        if not tensor_paths:
            raise ValueError(f"no Safetensors shards found in {checkpoint_dir}")
        self._tensor_paths = tensor_paths
        self._tensor_metadata = tensor_metadata

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tensor_paths))

    def metadata(self, name: str) -> TensorMetadata:
        try:
            return self._tensor_metadata[name]
        except KeyError as error:
            raise KeyError(f"tensor not found in Safetensors checkpoint: {name}") from error

    def load(self, name: str) -> NDArray[np.generic]:
        from safetensors import safe_open

        try:
            shard_path = self._tensor_paths[name]
        except KeyError as error:
            raise KeyError(f"tensor not found in Safetensors checkpoint: {name}") from error
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            tensor = shard.get_tensor(name)
            # PyTorch is used only at the file boundary because NumPy has no native
            # BF16 dtype. Converting only the current tensor avoids retaining the
            # entire checkpoint in memory.
            return np.ascontiguousarray(tensor.float().numpy(), dtype=np.float32)

    def close(self) -> None:
        """Safetensors opens and closes each shard inside ``load``."""


class LegacyTorchSource:
    """Memory-map one trusted Hugging Face PyTorch state dictionary safely."""

    def __init__(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise ValueError(f"legacy checkpoint does not exist: {checkpoint_path}")
        self._checkpoint_path = checkpoint_path
        self._state: Mapping[str, Any] | None = None

    def _ensure_state(self) -> Mapping[str, Any]:
        import torch

        if self._state is not None:
            return self._state
        try:
            state = torch.load(
                self._checkpoint_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except RuntimeError as error:
            if "mmap can only be used" not in str(error):
                raise
            # Pre-ZIP PyTorch files cannot be memory mapped. The compatibility
            # fallback retains only one trusted model at a time and still uses
            # weights_only; this spends bounded RAM to avoid rejecting official
            # legacy checkpoints.
            state = torch.load(
                self._checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        if not isinstance(state, Mapping) or any(not isinstance(name, str) for name in state):
            raise ValueError("legacy checkpoint must contain a string-keyed state dictionary")
        if any(not isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError("legacy checkpoint contains a non-tensor value")
        # One model's storages stay memory mapped; individual tensors are converted
        # only on demand to avoid copying all FP32 weights during deserialization.
        self._state = state
        return state

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._ensure_state()))

    def metadata(self, name: str) -> TensorMetadata:
        try:
            tensor = self._ensure_state()[name]
        except KeyError as error:
            raise KeyError(f"tensor not found in legacy checkpoint: {name}") from error
        dtype = str(tensor.dtype).removeprefix("torch.")
        return TensorMetadata(
            shape=tuple(int(dimension) for dimension in tensor.shape),
            dtype=dtype,
            is_floating=bool(tensor.is_floating_point()),
        )

    def load(self, name: str) -> NDArray[np.generic]:
        try:
            tensor = self._ensure_state()[name]
        except KeyError as error:
            raise KeyError(f"tensor not found in legacy checkpoint: {name}") from error
        return np.ascontiguousarray(tensor.float().numpy(), dtype=np.float32)

    def close(self) -> None:
        """Release the one legacy state before scanning the next model.

        Re-reading on the next pass trades sequential I/O for a strict one-model
        residency bound.
        """

        self._state = None


@dataclass(frozen=True, slots=True)
class TensorRecord:
    model_id: str
    tensor_name: str
    source_name: str
    source_dtype: str
    values: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class VectorRecord:
    model_id: str
    vector_name: str
    source_name: str
    source_dtype: str
    values: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class LearnedTensorRecord:
    """One adapted learned FP32 parameter used by dense and factor layouts."""

    model_id: str
    name: str
    source_name: str
    source_dtype: str
    kind: Literal["matrix", "vector"]
    values: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class TensorBlockRecord:
    ordinal: int
    model_id: str
    tensor_name: str
    row_offset: int
    col_offset: int
    valid_shape: tuple[int, int]
    values: NDArray[np.float32]


class CheckpointScanner:
    """Rescan model matrices in a stable order while retaining one tensor only."""

    def __init__(
        self,
        definition: ModelDefinition,
        source: TensorSource,
        *,
        model_config: Mapping[str, object] | None = None,
    ) -> None:
        self._definition = definition
        self._source = source
        self._model_config = dict(model_config or {})

    def _load_fp32(self, name: str) -> NDArray[np.float32]:
        return np.ascontiguousarray(self._source.load(name), dtype=np.float32)

    def _source_names(self) -> tuple[str, ...]:
        """Return the deterministic source order without retaining payloads."""

        return self._source.names()

    def _eligible_metadata(self, source_name: str, ndim: int) -> TensorMetadata | None:
        if not is_learned_parameter_name(source_name):
            return None
        metadata = self._source.metadata(source_name)
        if not metadata.is_floating or len(metadata.shape) != ndim:
            return None
        return metadata

    def iter_tensors(self) -> Iterator[TensorRecord]:
        for source_name in self._source_names():
            metadata = self._eligible_metadata(source_name, 2)
            if metadata is None:
                continue
            values = self._load_fp32(source_name)
            for adapted in adapt_matrix(self._definition, self._model_config, source_name, values):
                # Adaptation may split QKV but retains only the current source tensor
                # and its views or contiguous outputs.
                yield TensorRecord(
                    model_id=self._definition.repository,
                    tensor_name=adapted.name,
                    source_name=source_name,
                    source_dtype=metadata.dtype,
                    values=adapted.values,
                )

    def iter_vectors(self) -> Iterator[VectorRecord]:
        for source_name in self._source_names():
            metadata = self._eligible_metadata(source_name, 1)
            if metadata is None:
                continue
            values = self._load_fp32(source_name)
            for adapted in adapt_vector(self._definition, self._model_config, source_name, values):
                yield VectorRecord(
                    model_id=self._definition.repository,
                    vector_name=adapted.name,
                    source_name=source_name,
                    source_dtype=metadata.dtype,
                    values=adapted.values,
                )

    def iter_learned_tensors(self) -> Iterator[LearnedTensorRecord]:
        """Yield matrices and vectors in one stable scan with one payload resident."""

        for source_name in self._source_names():
            if not is_learned_parameter_name(source_name):
                continue
            metadata = self._source.metadata(source_name)
            if not metadata.is_floating or len(metadata.shape) not in (1, 2):
                continue
            values = self._load_fp32(source_name)
            if values.ndim == 2:
                for adapted in adapt_matrix(
                    self._definition, self._model_config, source_name, values
                ):
                    yield LearnedTensorRecord(
                        self._definition.repository,
                        adapted.name,
                        source_name,
                        metadata.dtype,
                        "matrix",
                        adapted.values,
                    )
            else:
                for adapted in adapt_vector(
                    self._definition, self._model_config, source_name, values
                ):
                    yield LearnedTensorRecord(
                        self._definition.repository,
                        adapted.name,
                        source_name,
                        metadata.dtype,
                        "vector",
                        adapted.values,
                    )

    def iter_blocks(self, *, block_h: int, block_w: int) -> Iterator[TensorBlockRecord]:
        if block_h <= 0 or block_w <= 0:
            raise ValueError("block dimensions must be positive")
        ordinal = 0
        for tensor in self.iter_tensors():
            rows, cols = tensor.values.shape
            for row_offset in range(0, rows, block_h):
                for col_offset in range(0, cols, block_w):
                    valid_h = min(block_h, rows - row_offset)
                    valid_w = min(block_w, cols - col_offset)
                    values = np.zeros((block_h, block_w), dtype=np.float32)
                    values[:valid_h, :valid_w] = tensor.values[
                        row_offset : row_offset + valid_h,
                        col_offset : col_offset + valid_w,
                    ]
                    yield TensorBlockRecord(
                        ordinal=ordinal,
                        model_id=tensor.model_id,
                        tensor_name=tensor.tensor_name,
                        row_offset=row_offset,
                        col_offset=col_offset,
                        valid_shape=(valid_h, valid_w),
                        values=values,
                    )
                    ordinal += 1
