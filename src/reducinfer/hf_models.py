"""Immutable definitions for the approved real Hugging Face checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelFamily(str, Enum):
    """Identify layout rules without importing a framework model class."""

    BERT = "bert"
    ROBERTA = "roberta"
    ELECTRA = "electra"
    GPT2 = "gpt2"
    OPT = "opt"
    GPT_NEOX = "gpt_neox"
    QWEN2 = "qwen2"
    LLAMA = "llama"


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    """Describe one pinned model family and its matrix-layout convention."""

    repository: str
    architecture: str
    family: ModelFamily
    revision: str = "main"

    def needs_transpose(self, tensor_name: str) -> bool:
        """Return whether a checkpoint matrix must become `(input, output)`.

        Embedding tables are indexed by row and GPT-2 Conv1D already stores right-
        multiply matrices. Other PyTorch Linear weights are transposed to the
        canonical SQL right-multiplication layout.
        """

        lowered = tensor_name.lower()
        if lowered in {"wte.weight", "wpe.weight"}:
            return False
        if any(
            marker in lowered
            for marker in ("embedding", "embed_tokens", "embed_positions", ".embed_in.", ".wte.", ".wpe.")
        ):
            return False
        if self.family is ModelFamily.GPT2 and any(
            marker in lowered for marker in (".c_attn.weight", ".c_proj.weight", ".c_fc.weight")
        ):
            return False
        return True


REAL_MODEL_DEFINITIONS: tuple[ModelDefinition, ...] = (
    ModelDefinition("google-bert/bert-large-uncased", "encoder", ModelFamily.BERT),
    ModelDefinition("FacebookAI/roberta-large", "encoder", ModelFamily.ROBERTA),
    ModelDefinition("google/electra-large-discriminator", "encoder", ModelFamily.ELECTRA),
    ModelDefinition("openai-community/gpt2-medium", "decoder", ModelFamily.GPT2),
    ModelDefinition("facebook/opt-350m", "decoder", ModelFamily.OPT),
    ModelDefinition("EleutherAI/pythia-410m", "decoder", ModelFamily.GPT_NEOX),
    ModelDefinition("Qwen/Qwen2.5-0.5B", "decoder", ModelFamily.QWEN2),
    ModelDefinition("HuggingFaceTB/SmolLM2-360M", "decoder", ModelFamily.LLAMA),
)

_DEFINITIONS_BY_REPOSITORY = {definition.repository: definition for definition in REAL_MODEL_DEFINITIONS}


def model_definition(repository: str) -> ModelDefinition:
    """Resolve an approved repository and reject silent model substitutions."""

    try:
        return _DEFINITIONS_BY_REPOSITORY[repository]
    except KeyError as error:
        raise ValueError(f"unsupported Hugging Face repository: {repository}") from error
