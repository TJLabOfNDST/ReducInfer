"""Translate Hugging Face config fields into validated runtime layer specs."""

from __future__ import annotations

from typing import Mapping

from .hf_models import ModelDefinition, ModelFamily
from .types import (
    ActivationKind,
    LayerTemplate,
    NormKind,
    PositionEncoding,
    TransformerLayerSpec,
)


def _positive_int(config: Mapping[str, object], *names: str) -> int:
    for name in names:
        value = config.get(name)
        if value is not None and int(value) > 0:
            return int(value)
    raise ValueError(f"model config requires one positive field from: {', '.join(names)}")


def _positive_float(config: Mapping[str, object], default: float, *names: str) -> float:
    for name in names:
        value = config.get(name)
        if value is not None and float(value) > 0.0:
            return float(value)
    return default


def build_layer_specs(
    definition: ModelDefinition, config: Mapping[str, object]
) -> tuple[TransformerLayerSpec, ...]:
    """Build one finite execution template per checkpoint layer."""

    family = definition.family
    if family is ModelFamily.GPT2:
        layer_count = _positive_int(config, "n_layer")
        hidden_size = _positive_int(config, "n_embd")
        query_heads = _positive_int(config, "n_head")
        kv_heads = query_heads
        template = LayerTemplate.DECODER_PRE_NORM_MHA
        position = PositionEncoding.LEARNED
        norm = NormKind.LAYER
        epsilon = _positive_float(config, 1e-5, "layer_norm_epsilon")
        activation = ActivationKind.GELU_TANH
        rope_theta = None
        attention_bias = True
        mlp_bias = True
    else:
        layer_count = _positive_int(config, "num_hidden_layers")
        hidden_size = _positive_int(config, "hidden_size")
        query_heads = _positive_int(config, "num_attention_heads")
        kv_heads = int(config.get("num_key_value_heads", query_heads))
        if family in {ModelFamily.BERT, ModelFamily.ROBERTA, ModelFamily.ELECTRA}:
            template = LayerTemplate.ENCODER_POST_NORM_MHA
            position = PositionEncoding.LEARNED
            norm = NormKind.LAYER
            epsilon = _positive_float(config, 1e-12, "layer_norm_eps")
            activation = ActivationKind.GELU_ERF
            rope_theta = None
            attention_bias = True
            mlp_bias = True
        elif family is ModelFamily.OPT:
            template = LayerTemplate.DECODER_PRE_NORM_MHA
            position = PositionEncoding.LEARNED
            norm = NormKind.LAYER
            epsilon = _positive_float(config, 1e-5, "layer_norm_eps")
            activation = ActivationKind.RELU
            rope_theta = None
            attention_bias = True
            mlp_bias = True
        elif family is ModelFamily.GPT_NEOX:
            if config.get("use_parallel_residual") is False:
                raise ValueError("GPT-NeoX checkpoint requires the parallel residual template")
            template = LayerTemplate.DECODER_PARALLEL_MHA
            position = PositionEncoding.ROPE
            norm = NormKind.LAYER
            epsilon = _positive_float(config, 1e-5, "layer_norm_eps")
            activation = ActivationKind.GELU_ERF
            rope_theta = _positive_float(config, 10_000.0, "rotary_emb_base")
            attention_bias = True
            mlp_bias = True
        elif family in {ModelFamily.QWEN2, ModelFamily.LLAMA}:
            template = LayerTemplate.DECODER_PRE_NORM_GQA
            position = PositionEncoding.ROPE
            norm = NormKind.RMS
            epsilon = _positive_float(config, 1e-6, "rms_norm_eps")
            activation = ActivationKind.SWIGLU
            rope_theta = _positive_float(config, 10_000.0, "rope_theta")
            attention_bias = family is ModelFamily.QWEN2
            mlp_bias = False
        else:
            raise ValueError(f"unsupported layer family: {family.value}")

    if hidden_size % query_heads:
        raise ValueError("hidden size must be divisible by attention heads")
    head_dim = int(config.get("head_dim", hidden_size // query_heads))
    return tuple(
        TransformerLayerSpec(
            model_id=definition.repository,
            layer_id=layer_id,
            template=template,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            position_encoding=position,
            norm_kind=norm,
            norm_epsilon=epsilon,
            activation=activation,
            tensor_prefix=f"layers.{layer_id}",
            rope_theta=rope_theta,
            attention_bias=attention_bias,
            mlp_bias=mlp_bias,
        )
        for layer_id in range(layer_count)
    )
