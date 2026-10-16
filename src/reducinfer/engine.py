"""Typed library entry points for pure-SQL Transformer inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .storage import DatabaseStore
from .transformer_layer_sql import (
    TransformerStageResult,
    decode_transformer_layers,
    execute_transformer_layers,
    execute_transformer_tokens,
    prefill_transformer_layers,
)
from .transformer_sql import AttentionStageResult, execute_factorized_attention
from .types import TransformerLayerSpec


def _validated_hidden(hidden: NDArray[np.generic]) -> NDArray[np.float32]:
    if hidden.dtype != np.float32:
        raise ValueError("hidden input must use float32")
    if hidden.ndim != 3 or any(dimension <= 0 for dimension in hidden.shape):
        raise ValueError("hidden input must have non-empty shape (batch, sequence, width)")
    if not np.isfinite(hidden).all():
        raise ValueError("hidden input must contain only finite values")
    # SQL COPY benefits from sequential access. Copying only non-contiguous inputs
    # trades at most one input-sized buffer for predictable O(n) serialization.
    return np.ascontiguousarray(hidden)


def _validated_tokens(tokens: NDArray[np.generic]) -> NDArray[np.int64]:
    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("token input must use an integer dtype")
    if tokens.ndim != 2 or any(dimension <= 0 for dimension in tokens.shape):
        raise ValueError("token input must have non-empty shape (batch, sequence)")
    if np.any(tokens < 0):
        raise ValueError("token ids must be non-negative")
    return np.ascontiguousarray(tokens, dtype=np.int64)


def _validated_specs(
    specs: TransformerLayerSpec | Sequence[TransformerLayerSpec],
) -> tuple[TransformerLayerSpec, ...]:
    layer_specs = (specs,) if isinstance(specs, TransformerLayerSpec) else tuple(specs)
    if not layer_specs:
        raise ValueError("at least one Transformer layer is required")
    model_id = layer_specs[0].model_id
    if any(spec.model_id != model_id for spec in layer_specs):
        raise ValueError("all Transformer layers must belong to one model")
    return layer_specs


class InferenceEngine:
    """Execute one selected model without moving intermediate tensors to Python."""

    def __init__(self, store: DatabaseStore) -> None:
        self._store = store

    def infer_transformer_attention(
        self,
        spec: TransformerLayerSpec,
        hidden: NDArray[np.float32],
        *,
        causal: bool,
    ) -> AttentionStageResult:
        return execute_factorized_attention(
            self._store, spec, _validated_hidden(hidden), causal=causal
        )

    def infer_transformer(
        self,
        specs: TransformerLayerSpec | Sequence[TransformerLayerSpec],
        hidden: NDArray[np.float32],
    ) -> TransformerStageResult:
        return execute_transformer_layers(
            self._store, _validated_specs(specs), _validated_hidden(hidden)
        )

    def infer_transformer_tokens(
        self,
        specs: TransformerLayerSpec | Sequence[TransformerLayerSpec],
        tokens: NDArray[np.int64],
    ) -> TransformerStageResult:
        return execute_transformer_tokens(
            self._store, _validated_specs(specs), _validated_tokens(tokens)
        )

    def prefill_transformer(
        self,
        specs: TransformerLayerSpec | Sequence[TransformerLayerSpec],
        hidden: NDArray[np.float32],
        *,
        request_id: str,
    ) -> TransformerStageResult:
        values = _validated_hidden(hidden)
        if values.shape[0] != 1:
            raise ValueError("KV-cache prefill currently requires batch size 1")
        if not request_id:
            raise ValueError("request_id must not be empty")
        return prefill_transformer_layers(
            self._store, _validated_specs(specs), values, request_id=request_id
        )

    def decode_transformer(
        self,
        specs: TransformerLayerSpec | Sequence[TransformerLayerSpec],
        hidden: NDArray[np.float32],
        *,
        request_id: str,
        position: int,
    ) -> TransformerStageResult:
        values = _validated_hidden(hidden)
        if values.shape[:2] != (1, 1):
            raise ValueError("incremental decode requires hidden shape (1, 1, width)")
        if not request_id:
            raise ValueError("request_id must not be empty")
        if position < 0:
            raise ValueError("position must be non-negative")
        return decode_transformer_layers(
            self._store,
            _validated_specs(specs),
            values,
            request_id=request_id,
            position=position,
        )
