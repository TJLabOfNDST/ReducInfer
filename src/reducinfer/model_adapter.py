"""Map framework checkpoint matrices to the canonical SQL runtime contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .hf_models import ModelDefinition, ModelFamily


@dataclass(frozen=True, slots=True)
class AdaptedMatrix:
    name: str
    values: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class AdaptedVector:
    name: str
    values: NDArray[np.float32]


def _canonical_name(family: ModelFamily, source_name: str) -> str:
    embedding_names = {
        "bert.embeddings.word_embeddings.weight",
        "roberta.embeddings.word_embeddings.weight",
        "electra.embeddings.word_embeddings.weight",
        "transformer.wte.weight",
        "wte.weight",
        "model.decoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "gpt_neox.embed_in.weight",
        "model.embed_tokens.weight",
    }
    position_names = {
        "bert.embeddings.position_embeddings.weight",
        "roberta.embeddings.position_embeddings.weight",
        "electra.embeddings.position_embeddings.weight",
        "transformer.wpe.weight",
        "wpe.weight",
        "model.decoder.embed_positions.weight",
        "decoder.embed_positions.weight",
    }
    if source_name in embedding_names:
        return "token_embedding.weight"
    if source_name in position_names:
        return "position_embedding.weight"

    encoder = re.search(r"(?:bert|roberta|electra)\.encoder\.layer\.(\d+)\.(.+)", source_name)
    if encoder:
        layer, suffix = encoder.groups()
        component = {
            "attention.self.query.weight": "attn.q.weight",
            "attention.self.key.weight": "attn.k.weight",
            "attention.self.value.weight": "attn.v.weight",
            "attention.output.dense.weight": "attn.o.weight",
            "intermediate.dense.weight": "mlp_in.weight",
            "output.dense.weight": "mlp_out.weight",
        }.get(suffix)
        if component:
            return f"layers.{layer}.{component}"

    gpt2 = re.fullmatch(r"(?:transformer\.)?h\.(\d+)\.(.+)", source_name)
    if gpt2:
        layer, suffix = gpt2.groups()
        component = {
            "attn.c_proj.weight": "attn.o.weight",
            "mlp.c_fc.weight": "mlp_in.weight",
            "mlp.c_proj.weight": "mlp_out.weight",
        }.get(suffix)
        if component:
            return f"layers.{layer}.{component}"

    opt = re.fullmatch(r"(?:model\.)?decoder\.layers\.(\d+)\.(.+)", source_name)
    if opt:
        layer, suffix = opt.groups()
        component = {
            "self_attn.q_proj.weight": "attn.q.weight",
            "self_attn.k_proj.weight": "attn.k.weight",
            "self_attn.v_proj.weight": "attn.v.weight",
            "self_attn.out_proj.weight": "attn.o.weight",
            "fc1.weight": "mlp_in.weight",
            "fc2.weight": "mlp_out.weight",
        }.get(suffix)
        if component:
            return f"layers.{layer}.{component}"

    neox = re.search(r"gpt_neox\.layers\.(\d+)\.(.+)", source_name)
    if neox:
        layer, suffix = neox.groups()
        component = {
            "attention.dense.weight": "attn.o.weight",
            "mlp.dense_h_to_4h.weight": "mlp_in.weight",
            "mlp.dense_4h_to_h.weight": "mlp_out.weight",
        }.get(suffix)
        if component:
            return f"layers.{layer}.{component}"

    decoder = re.search(r"model\.layers\.(\d+)\.(.+)", source_name)
    if decoder:
        layer, suffix = decoder.groups()
        component = {
            "self_attn.q_proj.weight": "attn.q.weight",
            "self_attn.k_proj.weight": "attn.k.weight",
            "self_attn.v_proj.weight": "attn.v.weight",
            "self_attn.o_proj.weight": "attn.o.weight",
            "mlp.gate_proj.weight": "gate.weight",
            "mlp.up_proj.weight": "up.weight",
            "mlp.down_proj.weight": "down.weight",
        }.get(suffix)
        if component:
            return f"layers.{layer}.{component}"
    return source_name


def adapt_matrix(
    definition: ModelDefinition,
    model_config: Mapping[str, object],
    source_name: str,
    values: NDArray[np.generic],
) -> tuple[AdaptedMatrix, ...]:
    """Convert one source matrix without retaining another checkpoint tensor."""

    if values.ndim != 2 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("only floating-point matrices can be adapted")
    right_multiply = values.T if definition.needs_transpose(source_name) else values
    right_multiply = np.ascontiguousarray(right_multiply, dtype=np.float32)

    gpt2_qkv = re.fullmatch(r"(?:transformer\.)?h\.(\d+)\.attn\.c_attn\.weight", source_name)
    if gpt2_qkv:
        hidden_size = int(model_config.get("n_embd", right_multiply.shape[1] // 3))
        if right_multiply.shape[1] != 3 * hidden_size:
            raise ValueError("GPT-2 combined QKV width disagrees with n_embd")
        layer = gpt2_qkv.group(1)
        return tuple(
            AdaptedMatrix(
                f"layers.{layer}.attn.{component}.weight",
                np.ascontiguousarray(right_multiply[:, index * hidden_size : (index + 1) * hidden_size]),
            )
            for index, component in enumerate(("q", "k", "v"))
        )

    neox_qkv = re.fullmatch(r"gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.weight", source_name)
    if neox_qkv:
        hidden_size = int(model_config.get("hidden_size", 0))
        head_count = int(model_config.get("num_attention_heads", 0))
        if hidden_size <= 0 or head_count <= 0 or hidden_size % head_count:
            raise ValueError("GPT-NeoX config requires divisible hidden_size and num_attention_heads")
        if right_multiply.shape[1] != 3 * hidden_size:
            raise ValueError("GPT-NeoX combined QKV width disagrees with hidden_size")
        head_dim = hidden_size // head_count
        interleaved = right_multiply.reshape(right_multiply.shape[0], head_count, 3, head_dim)
        layer = neox_qkv.group(1)
        return tuple(
            AdaptedMatrix(
                f"layers.{layer}.attn.{component}.weight",
                np.ascontiguousarray(interleaved[:, :, index, :].reshape(right_multiply.shape[0], hidden_size)),
            )
            for index, component in enumerate(("q", "k", "v"))
        )

    return (AdaptedMatrix(_canonical_name(definition.family, source_name), right_multiply),)


def _canonical_vector_name(source_name: str) -> str:
    encoder = re.search(r"(?:bert|roberta|electra)\.encoder\.layer\.(\d+)\.(.+)", source_name)
    if encoder:
        layer, suffix = encoder.groups()
        component = {
            "attention.self.query.bias": "attn.q.bias",
            "attention.self.key.bias": "attn.k.bias",
            "attention.self.value.bias": "attn.v.bias",
            "attention.output.dense.bias": "attn.o.bias",
            "attention.output.LayerNorm.weight": "norm1.weight",
            "attention.output.LayerNorm.bias": "norm1.bias",
            "attention.output.LayerNorm.gamma": "norm1.weight",
            "attention.output.LayerNorm.beta": "norm1.bias",
            "intermediate.dense.bias": "mlp_in.bias",
            "output.dense.bias": "mlp_out.bias",
            "output.LayerNorm.weight": "norm2.weight",
            "output.LayerNorm.bias": "norm2.bias",
            "output.LayerNorm.gamma": "norm2.weight",
            "output.LayerNorm.beta": "norm2.bias",
        }.get(suffix)
        if component:
            return f"layers.{layer}.{component}"

    gpt2 = re.fullmatch(r"(?:transformer\.)?h\.(\d+)\.(.+)", source_name)
    if gpt2:
        layer, suffix = gpt2.groups()
        component = {
            "ln_1.weight": "norm1.weight",
            "ln_1.bias": "norm1.bias",
            "ln_2.weight": "norm2.weight",
            "ln_2.bias": "norm2.bias",
            "attn.c_proj.bias": "attn.o.bias",
            "mlp.c_fc.bias": "mlp_in.bias",
            "mlp.c_proj.bias": "mlp_out.bias",
        }.get(suffix)
        if component:
            return f"layers.{layer}.{component}"

    opt = re.fullmatch(r"(?:model\.)?decoder\.layers\.(\d+)\.(.+)", source_name)
    if opt:
        layer, suffix = opt.groups()
        component = {
            "self_attn_layer_norm.weight": "norm1.weight",
            "self_attn_layer_norm.bias": "norm1.bias",
            "final_layer_norm.weight": "norm2.weight",
            "final_layer_norm.bias": "norm2.bias",
            "self_attn.q_proj.bias": "attn.q.bias",
            "self_attn.k_proj.bias": "attn.k.bias",
            "self_attn.v_proj.bias": "attn.v.bias",
            "self_attn.out_proj.bias": "attn.o.bias",
            "fc1.bias": "mlp_in.bias",
            "fc2.bias": "mlp_out.bias",
        }.get(suffix)
        if component:
            return f"layers.{layer}.{component}"

    neox = re.search(r"gpt_neox\.layers\.(\d+)\.(.+)", source_name)
    if neox:
        layer, suffix = neox.groups()
        component = {
            "input_layernorm.weight": "norm1.weight",
            "input_layernorm.bias": "norm1.bias",
            "post_attention_layernorm.weight": "norm2.weight",
            "post_attention_layernorm.bias": "norm2.bias",
            "attention.dense.bias": "attn.o.bias",
            "mlp.dense_h_to_4h.bias": "mlp_in.bias",
            "mlp.dense_4h_to_h.bias": "mlp_out.bias",
        }.get(suffix)
        if component:
            return f"layers.{layer}.{component}"

    decoder = re.search(r"model\.layers\.(\d+)\.(.+)", source_name)
    if decoder:
        layer, suffix = decoder.groups()
        component = {
            "input_layernorm.weight": "norm1.weight",
            "post_attention_layernorm.weight": "norm2.weight",
            "self_attn.q_proj.bias": "attn.q.bias",
            "self_attn.k_proj.bias": "attn.k.bias",
            "self_attn.v_proj.bias": "attn.v.bias",
            "self_attn.o_proj.bias": "attn.o.bias",
        }.get(suffix)
        if component:
            return f"layers.{layer}.{component}"
    return source_name


def adapt_vector(
    definition: ModelDefinition,
    model_config: Mapping[str, object],
    source_name: str,
    values: NDArray[np.generic],
) -> tuple[AdaptedVector, ...]:
    """Canonicalize one dense vector, including architecture-specific QKV splits."""

    if values.ndim != 1 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("only floating-point vectors can be adapted")
    vector = np.ascontiguousarray(values, dtype=np.float32)

    gpt2_qkv = re.fullmatch(r"(?:transformer\.)?h\.(\d+)\.attn\.c_attn\.bias", source_name)
    if gpt2_qkv:
        hidden_size = int(model_config.get("n_embd", vector.size // 3))
        if vector.size != 3 * hidden_size:
            raise ValueError("GPT-2 combined QKV bias disagrees with n_embd")
        layer = gpt2_qkv.group(1)
        return tuple(
            AdaptedVector(
                f"layers.{layer}.attn.{component}.bias",
                np.ascontiguousarray(vector[index * hidden_size : (index + 1) * hidden_size]),
            )
            for index, component in enumerate(("q", "k", "v"))
        )

    neox_qkv = re.fullmatch(r"gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.bias", source_name)
    if neox_qkv:
        hidden_size = int(model_config.get("hidden_size", 0))
        head_count = int(model_config.get("num_attention_heads", 0))
        if hidden_size <= 0 or head_count <= 0 or hidden_size % head_count or vector.size != 3 * hidden_size:
            raise ValueError("GPT-NeoX combined QKV bias disagrees with model config")
        head_dim = hidden_size // head_count
        interleaved = vector.reshape(head_count, 3, head_dim)
        layer = neox_qkv.group(1)
        return tuple(
            AdaptedVector(
                f"layers.{layer}.attn.{component}.bias",
                np.ascontiguousarray(interleaved[:, index, :].reshape(hidden_size)),
            )
            for index, component in enumerate(("q", "k", "v"))
        )
    return (AdaptedVector(_canonical_vector_name(source_name), vector),)
