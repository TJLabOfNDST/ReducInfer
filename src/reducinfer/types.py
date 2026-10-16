"""Domain types shared by reduction, storage, and SQL execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class MatrixLayout(str, Enum):
    """Specify the multiplication convention persisted with each tensor."""

    RIGHT_MULTIPLY = "right_multiply"


class LayerTemplate(str, Enum):
    """Finite Transformer execution orders supported by the SQL compiler.

    Finite templates fix residual ordering at import time and avoid dynamic SQL
    composition during inference.
    """

    ENCODER_POST_NORM_MHA = "encoder_post_norm_mha"
    DECODER_PRE_NORM_MHA = "decoder_pre_norm_mha"
    DECODER_PARALLEL_MHA = "decoder_parallel_mha"
    DECODER_PRE_NORM_GQA = "decoder_pre_norm_gqa"


class PositionEncoding(str, Enum):
    """Supported position representations for a layer."""

    LEARNED = "learned"
    ROPE = "rope"


class NormKind(str, Enum):
    """Supported normalization operators."""

    LAYER = "layer_norm"
    RMS = "rms_norm"


class ActivationKind(str, Enum):
    """Activation identities kept explicit for checkpoint-compatible execution."""

    GELU_ERF = "gelu_erf"
    GELU_TANH = "gelu_tanh"
    RELU = "relu"
    SILU = "silu"
    SWIGLU = "swiglu"


class ValidationError(ValueError):
    """Raise one actionable domain error for invalid graph or tensor metadata."""


@dataclass(frozen=True, slots=True)
class TransformerLayerSpec:
    """Persisted, typed architecture settings for one Transformer layer.

    Head counts and position encoding are validated at import time so execution
    handles only known safe templates.
    """

    model_id: str
    layer_id: int
    template: LayerTemplate
    query_heads: int
    kv_heads: int
    head_dim: int
    position_encoding: PositionEncoding
    norm_kind: NormKind
    norm_epsilon: float
    activation: ActivationKind
    tensor_prefix: str
    rope_theta: float | None = None
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self) -> None:
        if not self.model_id or not self.tensor_prefix:
            raise ValidationError("model_id and tensor_prefix must not be empty")
        if self.layer_id < 0:
            raise ValidationError("layer_id must be non-negative")
        if self.query_heads <= 0 or self.kv_heads <= 0 or self.head_dim <= 0:
            raise ValidationError("head counts and head_dim must be positive")
        if self.query_heads % self.kv_heads:
            raise ValidationError("query_heads must be divisible by kv_heads")
        if self.norm_epsilon <= 0.0:
            raise ValidationError("norm_epsilon must be positive")
        if self.position_encoding is PositionEncoding.ROPE:
            if self.rope_theta is None or self.rope_theta <= 0.0:
                raise ValidationError("rope_theta must be positive when position encoding is RoPE")
            if self.head_dim % 2:
                raise ValidationError("RoPE requires an even head_dim")
        elif self.rope_theta is not None:
            raise ValidationError("learned positions cannot define rope_theta")

        expected_position, expected_norm = {
            LayerTemplate.ENCODER_POST_NORM_MHA: (PositionEncoding.LEARNED, NormKind.LAYER),
            LayerTemplate.DECODER_PRE_NORM_MHA: (PositionEncoding.LEARNED, NormKind.LAYER),
            LayerTemplate.DECODER_PARALLEL_MHA: (PositionEncoding.ROPE, NormKind.LAYER),
            LayerTemplate.DECODER_PRE_NORM_GQA: (PositionEncoding.ROPE, NormKind.RMS),
        }[self.template]
        if self.position_encoding is not expected_position:
            if expected_position is PositionEncoding.ROPE:
                raise ValidationError("template requires RoPE; learned positions are incompatible")
            raise ValidationError("template requires learned positions")
        if self.norm_kind is not expected_norm:
            raise ValidationError(f"template requires {expected_norm.value}")
        if self.template is LayerTemplate.DECODER_PRE_NORM_GQA and self.activation is not ActivationKind.SWIGLU:
            raise ValidationError("GQA template requires swiglu activation")

    @property
    def query_width(self) -> int:
        """Return Q/O feature width without duplicating stored dimensions."""

        return self.query_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        """Return K/V feature width for ordinary MHA or grouped-query attention."""

        return self.kv_heads * self.head_dim

    def kv_head_for_query(self, query_head: int) -> int:
        """Map one Q head to its shared KV head in O(1).

        Integer division avoids a head-map table; the validated divisibility
        invariant guarantees the GQA one-to-many relationship.
        """

        if not 0 <= query_head < self.query_heads:
            raise ValidationError("query_head is outside the configured range")
        return query_head // (self.query_heads // self.kv_heads)

    def tensor_name(self, component: str, parameter: str = "weight") -> str:
        """Return the sole runtime name for a declared layer parameter.

        Import canonicalizes framework differences, so runtime naming keeps SQL
        templates branch-free.
        """

        allowed = frozenset({"attn.q", "attn.k", "attn.v", "attn.o", "norm1", "norm2", "mlp_in", "mlp_out", "gate", "up", "down"})
        if component not in allowed or parameter not in {"weight", "bias"}:
            raise ValidationError("unknown canonical layer parameter")
        return f"{self.tensor_prefix}.{component}.{parameter}"


@dataclass(frozen=True, slots=True)
class TensorBlock:
    """A padded weight tile plus its original logical extent."""

    block_id: str
    model_id: str
    tensor_name: str
    row_offset: int
    col_offset: int
    valid_shape: tuple[int, int]
    values: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class FactorPair:
    """Store one shared D and every C block that references it."""

    factor_id: str
    shared_d: NDArray[np.float32]
    c_blocks: tuple[TensorBlock, ...]
    c_values: tuple[NDArray[np.float32], ...]
    tail_error: float
    frobenius_error: float


@dataclass(frozen=True, slots=True)
class TensorDescriptor:
    """Describe a right-multiplication tensor before persistence."""

    model_id: str
    name: str
    values: NDArray[np.float32]
    layout: MatrixLayout = MatrixLayout.RIGHT_MULTIPLY
