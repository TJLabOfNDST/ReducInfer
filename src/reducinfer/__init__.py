"""ReducInfer pure-SQL inference system."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import ReducInferConfig
from .engine import InferenceEngine
from .hf_models import REAL_MODEL_DEFINITIONS, ModelDefinition
from .storage import DatabaseStore
from .types import TransformerLayerSpec

if TYPE_CHECKING:
    from .importer import DatabaseImporter

__all__ = (
    "DatabaseImporter",
    "DatabaseStore",
    "InferenceEngine",
    "ModelDefinition",
    "REAL_MODEL_DEFINITIONS",
    "ReducInferConfig",
    "TransformerLayerSpec",
)


def __getattr__(name: str) -> Any:
    """Load preprocessing-only APIs without taxing inference-only users.

    Lazy loading keeps database inference on base dependencies and imports
    FAISS/SciPy only when artifacts are imported.
    """

    if name == "DatabaseImporter":
        from .importer import DatabaseImporter

        return DatabaseImporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

