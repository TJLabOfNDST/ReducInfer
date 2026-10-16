"""Production command-line workflow for ReducInfer."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
import json
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import numpy as np

from .checkpoint import CheckpointScanner, VectorRecord
from .config import ReducInferConfig
from .engine import InferenceEngine
from .hf_layer_specs import build_layer_specs
from .hf_models import REAL_MODEL_DEFINITIONS, ModelDefinition, model_definition
from .real_models import (
    DownloadedCheckpoint,
    GlobalBlockSource,
    ModelCheckpoint,
    checkpoint_source,
    download_checkpoint,
)
from .storage import DatabaseStore

if TYPE_CHECKING:
    from .streaming_reduction import ReductionArtifactMetadata


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _add_models(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", dest="models", nargs="+", default=["all"])


def _add_reduction_layout(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--block-size", type=_positive_int, default=1024)
    parser.add_argument("--rank", type=_positive_int, default=256)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ReducInfer pure-SQL inference system")
    commands = parser.add_subparsers(dest="command", required=True)

    init_parser = commands.add_parser("init-db", help="initialize the ReducInfer schema")
    init_parser.add_argument("--allow-reset-database")

    download_parser = commands.add_parser("download", help="download approved checkpoints")
    _add_models(download_parser)
    download_parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))

    reduce_parser = commands.add_parser("reduce", help="create factor artifacts")
    _add_models(reduce_parser)
    _add_reduction_layout(reduce_parser)
    reduce_parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    reduce_parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    reduce_parser.add_argument("--device", type=_non_negative_int, default=0)
    reduce_parser.add_argument("--batch-blocks", type=_positive_int, default=256)

    import_parser = commands.add_parser("import-db", help="import factors into PostgreSQL")
    _add_models(import_parser)
    _add_reduction_layout(import_parser)
    import_parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    import_parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))

    infer_parser = commands.add_parser("infer", help="run an imported model")
    infer_parser.add_argument("--model", required=True)
    infer_parser.add_argument("--mode", choices=("hidden", "tokens", "prefill", "decode"), required=True)
    infer_parser.add_argument("--input", type=Path, required=True)
    infer_parser.add_argument("--output", type=Path, required=True)
    infer_parser.add_argument("--request-id")
    infer_parser.add_argument("--position", type=_non_negative_int)
    return parser


def _definitions(requested: Sequence[str] | None) -> tuple[ModelDefinition, ...]:
    names = list(requested or ["all"])
    if names == ["all"]:
        return REAL_MODEL_DEFINITIONS
    if "all" in names:
        raise ValueError("'all' cannot be combined with explicit repositories")
    definitions = tuple(model_definition(name) for name in names)
    if len(definitions) != len({definition.repository for definition in definitions}):
        raise ValueError("model selection contains duplicates")
    return definitions


def _required_dsn() -> str:
    dsn = os.getenv("REDUCINFER_DSN")
    if not dsn:
        raise ValueError("REDUCINFER_DSN is required")
    return dsn


def _redact(message: str) -> str:
    redacted = message
    for variable in ("REDUCINFER_DSN", "HF_TOKEN"):
        secret = os.getenv(variable)
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _metadata_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / ".reducinfer-checkpoint.json"


def _write_checkpoint_metadata(checkpoint: DownloadedCheckpoint) -> None:
    _metadata_path(checkpoint.path).write_text(
        json.dumps(
            {
                "repository": checkpoint.repository,
                "revision": checkpoint.revision,
                "checkpoint_bytes": checkpoint.checkpoint_bytes,
                "format": checkpoint.format,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _local_checkpoint(definition: ModelDefinition, root: Path) -> DownloadedCheckpoint:
    path = root.resolve() / definition.repository.replace("/", "--")
    safetensors = tuple(sorted(path.glob("*.safetensors")))
    legacy = path / "pytorch_model.bin"
    if safetensors:
        files = safetensors
        checkpoint_format = "safetensors"
    elif legacy.is_file():
        files = (legacy,)
        checkpoint_format = "pytorch_bin"
    else:
        raise FileNotFoundError(f"checkpoint not found: {definition.repository} at {path}")
    revision = definition.revision
    metadata_file = _metadata_path(path)
    if metadata_file.is_file():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata.get("repository") != definition.repository:
            raise ValueError(f"checkpoint metadata repository mismatch: {path}")
        revision = str(metadata["revision"])
    return DownloadedCheckpoint(
        repository=definition.repository,
        revision=revision,
        path=path,
        checkpoint_bytes=sum(file.stat().st_size for file in files),
        format=checkpoint_format,
    )


def _model_checkpoints(
    definitions: Sequence[ModelDefinition], root: Path
) -> tuple[ModelCheckpoint, ...]:
    checkpoints: list[ModelCheckpoint] = []
    try:
        for definition in definitions:
            downloaded = _local_checkpoint(definition, root)
            config_path = downloaded.path / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError(f"model config must be a JSON object: {definition.repository}")
            checkpoints.append(ModelCheckpoint(definition, checkpoint_source(downloaded), config))
    except BaseException:
        for checkpoint in checkpoints:
            checkpoint.source.close()
        raise
    return tuple(checkpoints)


def _artifact_metadata(
    definitions: Sequence[ModelDefinition],
    checkpoint_dir: Path,
    *,
    block_size: int,
    rank: int,
) -> ReductionArtifactMetadata:
    from .streaming_reduction import ArtifactModel, ReductionArtifactMetadata

    downloaded = tuple(_local_checkpoint(definition, checkpoint_dir) for definition in definitions)
    return ReductionArtifactMetadata(
        format_version=1,
        block_h=block_size,
        block_w=block_size,
        rank=rank,
        models=tuple(
            ArtifactModel(item.repository, item.revision, item.checkpoint_bytes)
            for item in downloaded
        ),
    )


def _iter_vectors(checkpoints: Sequence[ModelCheckpoint]) -> Iterator[VectorRecord]:
    for checkpoint in checkpoints:
        try:
            yield from CheckpointScanner(
                checkpoint.definition,
                checkpoint.source,
                model_config=checkpoint.config,
            ).iter_vectors()
        finally:
            checkpoint.source.close()


def _close_checkpoints(checkpoints: Sequence[ModelCheckpoint]) -> None:
    for checkpoint in checkpoints:
        checkpoint.source.close()


def _init_database(arguments: argparse.Namespace) -> int:
    with DatabaseStore(_required_dsn()) as store:
        if arguments.allow_reset_database:
            store.reset_project_schema(arguments.allow_reset_database)
        else:
            store.initialize()
    print("database schema is ready")
    return 0


def _download(arguments: argparse.Namespace) -> int:
    definitions = _definitions(arguments.models)
    root = arguments.checkpoint_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN") or None
    for index, definition in enumerate(definitions, start=1):
        checkpoint = download_checkpoint(definition, root, token=token)
        _write_checkpoint_metadata(checkpoint)
        print(
            f"download [{index}/{len(definitions)}] {definition.repository} "
            f"({checkpoint.checkpoint_bytes} bytes)"
        )
    return 0


def _config(block_size: int, rank: int) -> ReducInferConfig:
    return ReducInferConfig(
        block_h=block_size,
        block_w=block_size,
        rank=rank,
        brp_d1=8,
        brp_d2=8,
    )


def _reduce(arguments: argparse.Namespace) -> int:
    from .gpu_reduction import CuPyReductionBackend, GpuReductionConfig
    from .gpu_streaming_reduction import GpuStreamingReducer
    from .streaming_reduction import StreamingReducer, save_reduction_result

    definitions = _definitions(arguments.models)
    checkpoint_dir = arguments.checkpoint_dir.resolve()
    artifact_dir = arguments.artifact_dir.resolve()
    metadata = _artifact_metadata(
        definitions,
        checkpoint_dir,
        block_size=arguments.block_size,
        rank=arguments.rank,
    )
    checkpoints = _model_checkpoints(definitions, checkpoint_dir)
    blocks = GlobalBlockSource(
        checkpoints, block_h=arguments.block_size, block_w=arguments.block_size
    )
    config = _config(arguments.block_size, arguments.rank)
    try:
        print(f"reduction started: backend={arguments.backend}")
        if arguments.backend == "cpu":
            result = StreamingReducer(config, work_dir=artifact_dir).reduce(blocks)
            save_reduction_result(result, artifact_dir, metadata=metadata)
        else:
            backend = CuPyReductionBackend(
                GpuReductionConfig(
                    device_id=arguments.device,
                    batch_blocks=arguments.batch_blocks,
                )
            )
            try:
                reducer = GpuStreamingReducer(config, work_dir=artifact_dir, backend=backend)

                def progress(phase: str, completed: int, total: int) -> None:
                    suffix = str(total) if total else "?"
                    print(f"\r{phase}: {completed}/{suffix}", end="", flush=True)
                    if total and completed >= total:
                        print()

                reducer.reduce(blocks, metadata=metadata, progress=progress)
            finally:
                backend.release()
    finally:
        _close_checkpoints(checkpoints)
    print(f"reduction completed: {artifact_dir}")
    return 0


def _import_database(arguments: argparse.Namespace) -> int:
    from .gpu_streaming_reduction import load_gpu_streaming_reduction
    from .importer import DatabaseImporter
    from .streaming_reduction import load_reduction_result

    definitions = _definitions(arguments.models)
    checkpoint_dir = arguments.checkpoint_dir.resolve()
    artifact_dir = arguments.artifact_dir.resolve()
    expected = _artifact_metadata(
        definitions,
        checkpoint_dir,
        block_size=arguments.block_size,
        rank=arguments.rank,
    )
    checkpoints = _model_checkpoints(definitions, checkpoint_dir)
    models = {definition.repository: definition.architecture for definition in definitions}
    specs = tuple(
        spec
        for checkpoint in checkpoints
        for spec in build_layer_specs(checkpoint.definition, checkpoint.config)
    )
    try:
        with DatabaseStore(_required_dsn()) as store:
            store.initialize()
            importer = DatabaseImporter(store)
            if arguments.backend == "cpu":
                result = load_reduction_result(
                    artifact_dir / "reduction_manifest.json", expected=expected
                )
                blocks = GlobalBlockSource(
                    checkpoints,
                    block_h=arguments.block_size,
                    block_w=arguments.block_size,
                )
                importer.import_streaming(
                    models,
                    result,
                    blocks,
                    _iter_vectors(checkpoints),
                    specs,
                    expected,
                )
            else:
                result = load_gpu_streaming_reduction(artifact_dir, expected=expected)
                importer.import_packed(
                    models,
                    result,
                    _iter_vectors(checkpoints),
                    specs,
                    expected,
                )
    finally:
        _close_checkpoints(checkpoints)
    print(f"import completed: models={len(definitions)}")
    return 0


def _infer(arguments: argparse.Namespace) -> int:
    definition = model_definition(arguments.model)
    input_path = arguments.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input array not found: {input_path}")
    values = np.load(input_path, allow_pickle=False)
    if arguments.mode in {"prefill", "decode"} and not arguments.request_id:
        raise ValueError(f"{arguments.mode} mode requires --request-id")
    if arguments.mode == "decode" and arguments.position is None:
        raise ValueError("decode mode requires --position")
    if arguments.mode != "decode" and arguments.position is not None:
        raise ValueError("--position is valid only for decode mode")
    with DatabaseStore(_required_dsn()) as store:
        specs = store.layer_specs(definition.repository)
        if not specs:
            raise ValueError(f"model has no imported layer metadata: {definition.repository}")
        engine = InferenceEngine(store)
        if arguments.mode == "hidden":
            result = engine.infer_transformer(specs, values)
        elif arguments.mode == "tokens":
            result = engine.infer_transformer_tokens(specs, values)
        elif arguments.mode == "prefill":
            result = engine.prefill_transformer(
                specs, values, request_id=arguments.request_id
            )
        else:
            result = engine.decode_transformer(
                specs,
                values,
                request_id=arguments.request_id,
                position=arguments.position,
            )
    output_path = arguments.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result.values, allow_pickle=False)
    print(f"inference completed: {output_path}")
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(arguments)
    commands = {
        "init-db": _init_database,
        "download": _download,
        "reduce": _reduce,
        "import-db": _import_database,
        "infer": _infer,
    }
    try:
        return commands[parsed.command](parsed)
    except (Exception, KeyboardInterrupt) as error:
        print(
            f"error: {error.__class__.__name__}: {_redact(str(error))}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
