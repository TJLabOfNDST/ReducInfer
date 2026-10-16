"""Transactional import of CPU and CUDA reduction artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from psycopg import Cursor

from .checkpoint import VectorRecord
from .factor_layout import column_vectors
from .gpu_streaming_reduction import GpuStreamingReductionResult, PackedTensor, TOPOLOGY_DTYPE
from .pg_array import Float32Array
from .storage import DatabaseStore
from .streaming_reduction import (
    BlockFactory,
    ReductionArtifactMetadata,
    StreamingReductionResult,
    TensorCoefficientBatch,
    group_tensor_coefficients,
)
from .types import TransformerLayerSpec


class DatabaseImporter:
    """Import one validated factor bank without coupling inference to preprocessors."""

    def __init__(self, store: DatabaseStore) -> None:
        self._store = store

    @staticmethod
    def _validate_common(
        models: Mapping[str, str],
        result: StreamingReductionResult,
        expected: ReductionArtifactMetadata,
    ) -> None:
        requested_models = tuple(models)
        artifact_models = tuple(model.model_id for model in expected.models)
        if requested_models != artifact_models:
            raise ValueError("artifact metadata model order does not match the import request")
        tensor_models = {model_id for model_id, _, _, _ in result.tensor_shapes}
        if tensor_models != set(requested_models):
            raise ValueError("artifact tensor model set does not match its metadata")
        if result.cluster_count != len(result.factors) or result.cluster_count <= 0:
            raise ValueError("artifact cluster count is invalid")
        for cluster_id, factor in enumerate(result.factors):
            if factor.cluster_id != cluster_id:
                raise ValueError("artifact cluster identifiers are not contiguous")
            if factor.shared_d.shape != (expected.block_h, expected.rank):
                raise ValueError("artifact D shape does not match its metadata")
            if not np.isfinite(factor.shared_d).all():
                raise ValueError("artifact D contains non-finite values")

    @staticmethod
    def _validate_packed(result: GpuStreamingReductionResult) -> None:
        if not result.c_buffer_path.is_file() or not result.topology_path.is_file():
            raise FileNotFoundError("packed reduction C or topology artifact is missing")
        c_elements = result.c_buffer_path.stat().st_size // np.dtype(np.float32).itemsize
        topology_rows = result.topology_path.stat().st_size // TOPOLOGY_DTYPE.itemsize
        if result.c_buffer_path.stat().st_size % np.dtype(np.float32).itemsize:
            raise ValueError("packed C artifact has an invalid byte length")
        if result.topology_path.stat().st_size % TOPOLOGY_DTYPE.itemsize:
            raise ValueError("packed topology artifact has an invalid byte length")
        for tensor in result.tensors:
            if tensor.c_element_offset + tensor.c_element_count > c_elements:
                raise ValueError(f"packed C range exceeds its artifact: {tensor.tensor_name}")
            if tensor.topology_start + tensor.topology_count > topology_rows:
                raise ValueError(f"packed topology range exceeds its artifact: {tensor.tensor_name}")

    def import_streaming(
        self,
        models: Mapping[str, str],
        result: StreamingReductionResult,
        block_factory: BlockFactory,
        vectors: Iterable[VectorRecord],
        layer_specs: Sequence[TransformerLayerSpec],
        expected: ReductionArtifactMetadata,
    ) -> None:
        self._validate_common(models, result, expected)
        connection = self._store.connection
        context = "factor catalog"
        try:
            with connection.cursor() as cursor:
                factor_offset, model_keys, tensor_ids = self._prepare_catalog(
                    cursor, models, result.tensor_shapes, result.cluster_count
                )
                context = "shared D factors"
                self._copy_d(cursor, result, factor_offset)
                context = "streamed C factors"
                for batch in group_tensor_coefficients(result.iter_coefficients(block_factory)):
                    tensor_key = (batch.model_id, batch.tensor_name)
                    self._copy_tensor_coefficients(
                        cursor, tensor_ids[tensor_key], batch, factor_offset
                    )
                context = "dense vectors"
                self._copy_vectors(cursor, model_keys, vectors)
                context = "layer metadata"
                self._copy_layer_specs(cursor, model_keys, layer_specs)
                self._finish_import(cursor)
            connection.commit()
        except BaseException as error:
            connection.rollback()
            raise RuntimeError(f"database import failed while processing {context}") from error

    def import_packed(
        self,
        models: Mapping[str, str],
        result: GpuStreamingReductionResult,
        vectors: Iterable[VectorRecord],
        layer_specs: Sequence[TransformerLayerSpec],
        expected: ReductionArtifactMetadata,
    ) -> None:
        self._validate_common(models, result.reduction, expected)
        self._validate_packed(result)
        connection = self._store.connection
        context = "factor catalog"
        c_values = np.memmap(result.c_buffer_path, mode="r", dtype=np.float32)
        topology = np.memmap(result.topology_path, mode="r", dtype=TOPOLOGY_DTYPE)
        try:
            with connection.cursor() as cursor:
                factor_offset, model_keys, _ = self._prepare_catalog(
                    cursor,
                    models,
                    result.reduction.tensor_shapes,
                    result.reduction.cluster_count,
                    create_tensors=False,
                )
                context = "shared D factors"
                self._copy_d(cursor, result.reduction, factor_offset)
                tensor_ids: dict[int, int] = {}
                for tensor in result.tensors:
                    context = f"tensor {tensor.model_id}/{tensor.tensor_name}"
                    cursor.execute(
                        """
                        INSERT INTO tensors (model_key, tensor_name, rows, cols)
                        VALUES (%s, %s, %s, %s) RETURNING tensor_id
                        """,
                        (
                            model_keys[tensor.model_id],
                            tensor.tensor_name,
                            tensor.rows,
                            tensor.cols,
                        ),
                    )
                    tensor_ids[tensor.tensor_index] = int(cursor.fetchone()[0])
                    self._copy_packed_tensor(
                        cursor,
                        tensor_ids[tensor.tensor_index],
                        tensor,
                        c_values,
                        topology,
                        factor_offset,
                    )
                context = "dense vectors"
                self._copy_vectors(cursor, model_keys, vectors)
                context = "layer metadata"
                self._copy_layer_specs(cursor, model_keys, layer_specs)
                self._finish_import(cursor)
            connection.commit()
        except BaseException as error:
            connection.rollback()
            raise RuntimeError(f"database import failed while processing {context}") from error
        finally:
            del c_values
            del topology

    @staticmethod
    def _prepare_catalog(
        cursor: Cursor[tuple[object, ...]],
        models: Mapping[str, str],
        tensor_shapes: Sequence[tuple[str, str, int, int]],
        cluster_count: int,
        *,
        create_tensors: bool = True,
    ) -> tuple[int, dict[str, int], dict[tuple[str, str], int]]:
        # The exclusive lock makes max(d_id)+1 allocation deterministic under
        # concurrent imports. Brief serialization safely allocates contiguous IDs
        # without adding a metadata table.
        cursor.execute("LOCK TABLE shared_matrix_d_columns IN EXCLUSIVE MODE")
        cursor.execute("DELETE FROM models WHERE model_id = ANY(%s)", (list(models),))
        cursor.execute(
            """
            DELETE FROM shared_matrix_d_columns d
            WHERE NOT EXISTS (
                SELECT 1 FROM block_slice b WHERE b.factor_id = d.d_id
            )
            """
        )
        cursor.execute("SELECT COALESCE(MAX(d_id), 0) FROM shared_matrix_d_columns")
        factor_offset = int(cursor.fetchone()[0])
        if factor_offset + cluster_count > 2_147_483_647:
            raise OverflowError("factor identifier range exceeds PostgreSQL INTEGER")

        model_keys: dict[str, int] = {}
        for model_id, architecture in models.items():
            cursor.execute(
                """
                INSERT INTO models (model_id, architecture)
                VALUES (%s, %s) RETURNING model_key
                """,
                (model_id, architecture),
            )
            model_keys[model_id] = int(cursor.fetchone()[0])

        tensor_ids: dict[tuple[str, str], int] = {}
        if create_tensors:
            for model_id, tensor_name, rows, cols in tensor_shapes:
                cursor.execute(
                    """
                    INSERT INTO tensors (model_key, tensor_name, rows, cols)
                    VALUES (%s, %s, %s, %s) RETURNING tensor_id
                    """,
                    (model_keys[model_id], tensor_name, rows, cols),
                )
                tensor_ids[(model_id, tensor_name)] = int(cursor.fetchone()[0])
        return factor_offset, model_keys, tensor_ids

    @staticmethod
    def _copy_d(
        cursor: Cursor[tuple[object, ...]],
        result: StreamingReductionResult,
        factor_offset: int,
    ) -> None:
        # Binary COPY visits every stored D value exactly once and materializes only
        # one column vector, providing linear write complexity with one-column
        # auxiliary memory.
        with cursor.copy(
            "COPY shared_matrix_d_columns (d_id, rank_idx, d_values) "
            "FROM STDIN (FORMAT BINARY)"
        ) as copy:
            copy.set_types(("int4", "int2", "float4[]"))
            for factor in result.factors:
                for rank_idx, values in enumerate(column_vectors(factor.shared_d)):
                    copy.write_row(
                        (
                            factor_offset + factor.cluster_id + 1,
                            rank_idx,
                            Float32Array(values),
                        )
                    )

    @staticmethod
    def _copy_tensor_coefficients(
        cursor: Cursor[tuple[object, ...]],
        tensor_id: int,
        batch: TensorCoefficientBatch,
        factor_offset: int,
    ) -> None:
        rank = int(batch.values.shape[0])
        with cursor.copy(
            "COPY layer_c_vectors (tensor_id, factor_id, block_row_offset, "
            "output_col, rank, c_values) FROM STDIN (FORMAT BINARY)"
        ) as vector_copy:
            vector_copy.set_types(("int4", "int4", "int4", "int4", "int2", "float4[]"))
            for block, cluster_id, c_start in zip(
                batch.blocks, batch.cluster_ids, batch.slice_starts, strict=True
            ):
                block_c = np.ascontiguousarray(
                    batch.values[:, c_start : c_start + block.valid_shape[1]],
                    dtype=np.float32,
                )
                for local_col, vector in enumerate(column_vectors(block_c)):
                    vector_copy.write_row(
                        (
                            tensor_id,
                            factor_offset + cluster_id + 1,
                            block.row_offset,
                            block.col_offset + local_col,
                            rank,
                            Float32Array(vector),
                        )
                    )
        with cursor.copy(
            """
            COPY block_slice (
                tensor_id, factor_id, block_row_offset, block_col_offset,
                c_start, valid_rows, valid_cols
            ) FROM STDIN
            """
        ) as copy:
            for block, cluster_id, c_start in zip(
                batch.blocks, batch.cluster_ids, batch.slice_starts, strict=True
            ):
                copy.write_row(
                    (
                        tensor_id,
                        factor_offset + cluster_id + 1,
                        block.row_offset,
                        block.col_offset,
                        c_start,
                        block.valid_shape[0],
                        block.valid_shape[1],
                    )
                )

    @staticmethod
    def _copy_packed_tensor(
        cursor: Cursor[tuple[object, ...]],
        tensor_id: int,
        tensor: PackedTensor,
        c_values: np.memmap,
        topology: np.memmap,
        factor_offset: int,
    ) -> None:
        start = tensor.c_element_offset
        tensor_c = np.ascontiguousarray(
            c_values[start : start + tensor.c_element_count], dtype=np.float32
        )
        topology_start = tensor.topology_start
        tensor_topology = np.array(
            topology[topology_start : topology_start + tensor.topology_count],
            dtype=TOPOLOGY_DTYPE,
            copy=True,
        )
        if not tensor.topology_count:
            return
        c_matrix = tensor_c.reshape(tensor.rank, tensor.rank_stride)
        with cursor.copy(
            "COPY layer_c_vectors (tensor_id, factor_id, block_row_offset, "
            "output_col, rank, c_values) FROM STDIN (FORMAT BINARY)"
        ) as vector_copy:
            vector_copy.set_types(("int4", "int4", "int4", "int4", "int2", "float4[]"))
            for row in tensor_topology:
                valid_cols = int(row["valid_cols"])
                c_start = int(row["c_start"])
                block_c = np.ascontiguousarray(
                    c_matrix[:, c_start : c_start + valid_cols], dtype=np.float32
                )
                for local_col, vector in enumerate(column_vectors(block_c)):
                    vector_copy.write_row(
                        (
                            tensor_id,
                            factor_offset + int(row["cluster_id"]) + 1,
                            int(row["row_offset"]),
                            int(row["col_offset"]) + local_col,
                            tensor.rank,
                            Float32Array(vector),
                        )
                    )
        with cursor.copy(
            """
            COPY block_slice (
                tensor_id, factor_id, block_row_offset, block_col_offset,
                c_start, valid_rows, valid_cols
            ) FROM STDIN
            """
        ) as copy:
            for row in tensor_topology:
                copy.write_row(
                    (
                        tensor_id,
                        factor_offset + int(row["cluster_id"]) + 1,
                        int(row["row_offset"]),
                        int(row["col_offset"]),
                        int(row["c_start"]),
                        int(row["valid_rows"]),
                        int(row["valid_cols"]),
                    )
                )

    @staticmethod
    def _copy_vectors(
        cursor: Cursor[tuple[object, ...]],
        model_keys: Mapping[str, int],
        vectors: Iterable[VectorRecord],
    ) -> None:
        for vector in vectors:
            if vector.model_id not in model_keys:
                raise KeyError(f"vector model is outside the imported set: {vector.model_id}")
            values = np.asarray(vector.values)
            if values.ndim != 1 or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError(f"invalid dense vector: {vector.model_id}/{vector.vector_name}")
            cursor.execute(
                """
                INSERT INTO dense_vectors (model_key, vector_name, values)
                VALUES (%s, %s, %s)
                """,
                (
                    model_keys[vector.model_id],
                    vector.vector_name,
                    Float32Array(np.ascontiguousarray(values)),
                ),
            )

    @staticmethod
    def _copy_layer_specs(
        cursor: Cursor[tuple[object, ...]],
        model_keys: Mapping[str, int],
        specs: Sequence[TransformerLayerSpec],
    ) -> None:
        keys = [(spec.model_id, spec.layer_id) for spec in specs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate model/layer metadata")
        for spec in specs:
            if spec.model_id not in model_keys:
                raise KeyError(f"layer model is outside the imported set: {spec.model_id}")
            cursor.execute(
                """
                INSERT INTO model_layers (
                    model_key, layer_id, template, query_heads, kv_heads, head_dim,
                    position_encoding, norm_kind, norm_epsilon, activation,
                    tensor_prefix, rope_theta, attention_bias, mlp_bias
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    model_keys[spec.model_id],
                    spec.layer_id,
                    spec.template.value,
                    spec.query_heads,
                    spec.kv_heads,
                    spec.head_dim,
                    spec.position_encoding.value,
                    spec.norm_kind.value,
                    spec.norm_epsilon,
                    spec.activation.value,
                    spec.tensor_prefix,
                    spec.rope_theta,
                    spec.attention_bias,
                    spec.mlp_bias,
                ),
            )

    @staticmethod
    def _finish_import(cursor: Cursor[tuple[object, ...]]) -> None:
        for relation in (
            "models",
            "tensors",
            "shared_matrix_d_columns",
            "block_slice",
            "layer_c_vectors",
            "dense_vectors",
            "model_layers",
        ):
            cursor.execute(f"ANALYZE {relation}")
