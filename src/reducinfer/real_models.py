"""Anonymous-first acquisition of the approved real model checkpoints."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from .checkpoint import (
    CheckpointScanner,
    LegacyTorchSource,
    SafetensorSource,
    TensorBlockRecord,
    TensorSource,
)
from .hf_models import ModelDefinition


def checkpoint_allow_patterns(repository_files: Sequence[str]) -> tuple[str, ...]:
    """Choose one safe weight representation without downloading duplicates."""

    if any(name.endswith(".safetensors") for name in repository_files):
        return ("config.json", "*.safetensors", "*.safetensors.index.json")
    if "pytorch_model.bin" in repository_files:
        return ("config.json", "pytorch_model.bin", "pytorch_model.bin.index.json")
    raise ValueError("repository has no supported Safetensors or weight-only PyTorch checkpoint")


@dataclass(frozen=True, slots=True)
class DownloadedCheckpoint:
    repository: str
    revision: str
    path: Path
    checkpoint_bytes: int
    format: str


def download_checkpoint(
    definition: ModelDefinition,
    destination_root: Path,
    *,
    token: str | None = None,
) -> DownloadedCheckpoint:
    """Download one public model while keeping credentials out of paths and reports."""

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)
    info = api.model_info(definition.repository, revision=definition.revision, files_metadata=True)
    repository_files = tuple(sibling.rfilename for sibling in info.siblings)
    allow_patterns = checkpoint_allow_patterns(repository_files)
    local_dir = destination_root / definition.repository.replace("/", "--")
    downloaded = Path(
        snapshot_download(
            repo_id=definition.repository,
            revision=info.sha,
            local_dir=local_dir,
            allow_patterns=list(allow_patterns),
            token=token,
        )
    )
    weight_files = tuple(downloaded.glob("*.safetensors")) or tuple(downloaded.glob("pytorch_model.bin"))
    if not weight_files:
        raise RuntimeError(f"downloaded checkpoint has no selected weights: {definition.repository}")
    checkpoint_format = "safetensors" if weight_files[0].suffix == ".safetensors" else "pytorch_bin"
    return DownloadedCheckpoint(
        repository=definition.repository,
        revision=str(info.sha),
        path=downloaded,
        checkpoint_bytes=sum(path.stat().st_size for path in weight_files),
        format=checkpoint_format,
    )


def checkpoint_source(checkpoint: DownloadedCheckpoint) -> TensorSource:
    """Open the representation selected during download."""

    if checkpoint.format == "safetensors":
        return SafetensorSource(checkpoint.path)
    if checkpoint.format == "pytorch_bin":
        return LegacyTorchSource(checkpoint.path / "pytorch_model.bin")
    raise ValueError(f"unknown checkpoint format: {checkpoint.format}")


@dataclass(frozen=True, slots=True)
class ModelCheckpoint:
    definition: ModelDefinition
    source: TensorSource
    config: dict[str, object]


class GlobalBlockSource:
    """Rescan heterogeneous checkpoints with one stable global block ordinal."""

    def __init__(
        self,
        checkpoints: Sequence[ModelCheckpoint],
        *,
        block_h: int,
        block_w: int,
    ) -> None:
        if not checkpoints or block_h <= 0 or block_w <= 0:
            raise ValueError("checkpoints and positive block dimensions are required")
        self._checkpoints = tuple(checkpoints)
        self._block_h = block_h
        self._block_w = block_w

    def __call__(self) -> Iterator[TensorBlockRecord]:
        ordinal = 0
        for checkpoint in self._checkpoints:
            scanner = CheckpointScanner(
                checkpoint.definition,
                checkpoint.source,
                model_config=checkpoint.config,
            )
            try:
                for block in scanner.iter_blocks(block_h=self._block_h, block_w=self._block_w):
                    # Replacing one integer is O(1); no block index table or resident
                    # multi-model block list is retained.
                    yield replace(block, ordinal=ordinal)
                    ordinal += 1
            finally:
                checkpoint.source.close()
