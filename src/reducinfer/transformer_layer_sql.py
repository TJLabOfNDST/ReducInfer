"""Session-local pure-SQL Transformer layer composition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from psycopg import Cursor

from .activation_sql import (
    ProjectionExecutionTiming,
    ProjectionRequest,
    materialize_factorized_embedding,
    materialize_factorized_projection,
    materialize_factorized_projections,
    merge_projection_timings,
)
from .timing import PhaseTimings
from .storage import DatabaseStore
from .transformer_sql import materialize_factorized_attention
from .types import ActivationKind, LayerTemplate, NormKind, PositionEncoding, TransformerLayerSpec

TOKEN_EMBEDDING_TENSOR = "token_embedding.weight"
POSITION_EMBEDDING_TENSOR = "position_embedding.weight"


@dataclass(frozen=True, slots=True)
class TransformerStageResult:
    """Final vectors and mutually exclusive server execution phases."""

    values: NDArray[np.float32]
    input_stage_ms: float
    gemm_xd_ms: float
    gemm_c_ms: float
    kernel_select_ms: float
    compact_kernel_calls: int
    staged_kernel_calls: int
    non_gemm_ms: float
    kv_cache_read_ms: float
    kv_cache_write_ms: float
    final_transfer_ms: float
    end_to_end_ms: float

    def phase_timings(self) -> PhaseTimings:
        """Return non-overlapping operational phases without recomputation.

        Reusing execution-phase timings avoids rerunning SQL for observability.
        """

        return PhaseTimings(
            input_stage_ms=self.input_stage_ms,
            gemm_xd_ms=self.gemm_xd_ms,
            gemm_c_ms=self.gemm_c_ms,
            non_gemm_ms=self.non_gemm_ms,
            kv_cache_read_ms=self.kv_cache_read_ms,
            kv_cache_write_ms=self.kv_cache_write_ms,
            final_transfer_ms=self.final_transfer_ms,
            end_to_end_ms=self.end_to_end_ms,
        )


@dataclass(frozen=True, slots=True)
class TokenEmbeddingStage:
    output_table: str
    timing: ProjectionExecutionTiming
    non_gemm_ms: float


def _validate_table_names(*table_names: str) -> None:
    if any(not table_name.isidentifier() for table_name in table_names):
        raise ValueError("temporary Transformer table names must be identifiers")


def _materialize_norm(
    cursor: Cursor[tuple[object, ...]],
    spec: TransformerLayerSpec,
    component: str,
    *,
    source_table: str,
    output_table: str,
) -> float:
    """Normalize each token array while reading scale/bias directly in PostgreSQL."""

    _validate_table_names(source_table, output_table)
    gamma_name = spec.tensor_name(component)
    beta_name = spec.tensor_name(component, "bias")
    cursor.execute(
        "SELECT cardinality(v.values) FROM dense_vectors v JOIN models m USING (model_key) WHERE m.model_id = %s AND v.vector_name = %s",
        (spec.model_id, gamma_name),
    )
    gamma_row = cursor.fetchone()
    if gamma_row is None:
        raise ValueError(f"missing normalization scale: {spec.model_id}/{gamma_name}")

    started = perf_counter()
    if spec.norm_kind is NormKind.RMS:
        cursor.execute(
            f"""
            CREATE TEMP TABLE {output_table} ON COMMIT DROP AS
            WITH gamma AS (
                SELECT v.values FROM dense_vectors v JOIN models m USING (model_key)
                WHERE m.model_id = %s AND v.vector_name = %s
            ), expanded AS (
                SELECT s.batch_idx,s.token_idx,item.ordinality AS position,item.value
                FROM {source_table} s
                CROSS JOIN LATERAL unnest(s.values) WITH ORDINALITY AS item(value, ordinality)
            ), stats AS (
                SELECT batch_idx,token_idx,avg(value*value) AS mean_square
                FROM expanded GROUP BY batch_idx,token_idx
            )
            SELECT e.batch_idx,e.token_idx,
                   array_agg((e.value/sqrt(st.mean_square + %s) * g.values[e.position])::REAL
                             ORDER BY e.position)::REAL[] AS values
            FROM expanded e JOIN stats st USING(batch_idx,token_idx) CROSS JOIN gamma g
            GROUP BY e.batch_idx,e.token_idx
            """,
            (spec.model_id, gamma_name, spec.norm_epsilon),
        )
    else:
        cursor.execute(
            f"""
            CREATE TEMP TABLE {output_table} ON COMMIT DROP AS
            WITH gamma AS (
                SELECT v.values FROM dense_vectors v JOIN models m USING (model_key)
                WHERE m.model_id = %s AND v.vector_name = %s
            ), beta AS (
                SELECT v.values FROM dense_vectors v JOIN models m USING (model_key)
                WHERE m.model_id = %s AND v.vector_name = %s
            ), expanded AS (
                SELECT s.batch_idx,s.token_idx,item.ordinality AS position,item.value
                FROM {source_table} s
                CROSS JOIN LATERAL unnest(s.values) WITH ORDINALITY AS item(value, ordinality)
            ), stats AS (
                SELECT batch_idx,token_idx,avg(value) AS mean,
                       GREATEST(avg(value*value)-avg(value)*avg(value), 0.0) AS variance
                FROM expanded GROUP BY batch_idx,token_idx
            )
            SELECT e.batch_idx,e.token_idx,
                   array_agg((((e.value-st.mean)/sqrt(st.variance + %s))*g.values[e.position]
                              + COALESCE(b.values[e.position],0.0))::REAL ORDER BY e.position)::REAL[] AS values
            FROM expanded e JOIN stats st USING(batch_idx,token_idx)
            CROSS JOIN gamma g LEFT JOIN beta b ON TRUE
            GROUP BY e.batch_idx,e.token_idx
            """,
            (spec.model_id, gamma_name, spec.model_id, beta_name, spec.norm_epsilon),
        )
    return (perf_counter() - started) * 1_000.0


def _materialize_residual(
    cursor: Cursor[tuple[object, ...]], *, left_table: str, right_table: str, output_table: str
) -> float:
    """Add aligned token arrays without expanding them into a persistent row-per-scalar layout."""

    _validate_table_names(left_table, right_table, output_table)
    started = perf_counter()
    cursor.execute(
        f"""
        CREATE TEMP TABLE {output_table} ON COMMIT DROP AS
        SELECT l.batch_idx,l.token_idx,
               array_agg((l.values[position] + r.values[position])::REAL ORDER BY position)::REAL[] AS values
        FROM {left_table} l JOIN {right_table} r USING(batch_idx,token_idx)
        CROSS JOIN LATERAL generate_subscripts(l.values,1) AS position
        WHERE cardinality(l.values) = cardinality(r.values)
        GROUP BY l.batch_idx,l.token_idx
        """
    )
    return (perf_counter() - started) * 1_000.0


def _activation_expression(activation: ActivationKind, value: str) -> str:
    if activation is ActivationKind.RELU:
        return f"GREATEST({value},0.0)"
    if activation is ActivationKind.SILU:
        return f"CASE WHEN {value}>=40 THEN {value} WHEN {value}<=-40 THEN 0.0 ELSE {value}/(1.0+exp(-{value})) END"
    if activation is ActivationKind.GELU_TANH:
        return f"0.5*{value}*(1.0+tanh(sqrt(2.0/pi())*({value}+0.044715*power({value},3))))"
    if activation is ActivationKind.GELU_ERF:
        # Abramowitz-Stegun 7.1.26 keeps high precision in the PostgreSQL 15 core
        # function set without an extension; max absolute erf error is about 1.5e-7.
        z = f"({value}/sqrt(2.0))"
        t = f"(1.0/(1.0+0.3275911*abs({z})))"
        polynomial = f"(((((1.061405429*{t}-1.453152027)*{t}+1.421413741)*{t}-0.284496736)*{t}+0.254829592)*{t})"
        # PostgreSQL reports floating-point underflow instead of returning zero for
        # PostgreSQL raises on exp(-z^2) underflow. Beyond |z|=26 the omitted term
        # is below 3e-294 and cannot affect the FP32 result, so saturating it to the
        # mathematical limit is numerically equivalent and avoids the exception.
        decay = f"(CASE WHEN abs({z})>=26.0 THEN 0.0 ELSE exp(-{z}*{z}) END)"
        erf = f"(sign({z})*(1.0-{polynomial}*{decay}))"
        return f"0.5*{value}*(1.0+{erf})"
    raise ValueError(f"activation {activation.value} requires paired gate/up inputs")


def _materialize_activation(
    cursor: Cursor[tuple[object, ...]],
    activation: ActivationKind,
    *,
    source_table: str,
    output_table: str,
) -> float:
    _validate_table_names(source_table, output_table)
    expression = _activation_expression(activation, "item.value")
    started = perf_counter()
    cursor.execute(
        f"""
        CREATE TEMP TABLE {output_table} ON COMMIT DROP AS
        SELECT s.batch_idx,s.token_idx,
               array_agg(({expression})::REAL ORDER BY item.ordinality)::REAL[] AS values
        FROM {source_table} s
        CROSS JOIN LATERAL unnest(s.values) WITH ORDINALITY AS item(value, ordinality)
        GROUP BY s.batch_idx,s.token_idx
        """
    )
    return (perf_counter() - started) * 1_000.0


def _materialize_swiglu(
    cursor: Cursor[tuple[object, ...]], *, projection_table: str, output_table: str
) -> float:
    _validate_table_names(projection_table, output_table)
    silu = _activation_expression(ActivationKind.SILU, "g.values[position]")
    started = perf_counter()
    cursor.execute(
        f"""
        CREATE TEMP TABLE {output_table} ON COMMIT DROP AS
        SELECT g.batch_idx,g.token_idx,
               array_agg((({silu})*u.values[position])::REAL ORDER BY position)::REAL[] AS values
        FROM {projection_table} g
        JOIN {projection_table} u USING(batch_idx,token_idx)
        CROSS JOIN LATERAL generate_subscripts(g.values,1) AS position
        WHERE g.projection_id=0 AND u.projection_id=1
          AND cardinality(g.values)=cardinality(u.values)
        GROUP BY g.batch_idx,g.token_idx
        """
    )
    return (perf_counter() - started) * 1_000.0


def _materialize_mlp(
    cursor: Cursor[tuple[object, ...]],
    spec: TransformerLayerSpec,
    *,
    source_table: str,
    output_table: str,
    table_prefix: str,
) -> tuple[ProjectionExecutionTiming, float]:
    _validate_table_names(source_table, output_table, table_prefix)
    non_gemm_ms = 0.0
    if spec.activation is ActivationKind.SWIGLU:
        gate_up_table = f"{table_prefix}_gate_up"
        input_timing = materialize_factorized_projections(
            cursor,
            spec.model_id,
            (
                ProjectionRequest(0, spec.tensor_name("gate")),
                ProjectionRequest(1, spec.tensor_name("up")),
            ),
            source_table=source_table,
            xd_table=f"{gate_up_table}_xd",
            output_table=gate_up_table,
        )
        activated_table = f"{table_prefix}_activated"
        non_gemm_ms += _materialize_swiglu(
            cursor, projection_table=gate_up_table, output_table=activated_table
        )
        output_component = "down"
    else:
        intermediate_table = f"{table_prefix}_intermediate"
        input_timing = materialize_factorized_projection(
            cursor, spec.model_id, spec.tensor_name("mlp_in"), source_table=source_table,
            xd_table=f"{intermediate_table}_xd", output_table=intermediate_table,
        )
        activated_table = f"{table_prefix}_activated"
        non_gemm_ms += _materialize_activation(
            cursor, spec.activation, source_table=intermediate_table, output_table=activated_table
        )
        output_component = "mlp_out"
    output_timing = materialize_factorized_projection(
        cursor, spec.model_id, spec.tensor_name(output_component), source_table=activated_table,
        xd_table=f"{table_prefix}_output_xd", output_table=output_table,
    )
    return merge_projection_timings((input_timing, output_timing)), non_gemm_ms


def materialize_transformer_layer(
    cursor: Cursor[tuple[object, ...]],
    spec: TransformerLayerSpec,
    *,
    source_table: str,
    output_table: str,
    table_prefix: str,
    cache_scope: str | None = None,
    read_cache: bool = False,
    write_cache: bool = False,
) -> tuple[ProjectionExecutionTiming, float, float, float]:
    """Compose one of the four declared residual templates entirely in PostgreSQL."""

    _validate_table_names(source_table, output_table, table_prefix)
    projection_timings: list[ProjectionExecutionTiming] = []
    non_gemm_total = cache_read_total = cache_write_total = 0.0

    def add_attention(
        timings: tuple[ProjectionExecutionTiming, float, float, float],
    ) -> None:
        nonlocal non_gemm_total, cache_read_total, cache_write_total
        projection_timing, non_gemm_ms, cache_read_ms, cache_write_ms = timings
        projection_timings.append(projection_timing)
        non_gemm_total += non_gemm_ms
        cache_read_total += cache_read_ms
        cache_write_total += cache_write_ms

    def add_mlp(timings: tuple[ProjectionExecutionTiming, float]) -> None:
        nonlocal non_gemm_total
        projection_timing, non_gemm_ms = timings
        projection_timings.append(projection_timing)
        non_gemm_total += non_gemm_ms

    def finish() -> tuple[ProjectionExecutionTiming, float, float, float]:
        return (
            merge_projection_timings(projection_timings),
            non_gemm_total,
            cache_read_total,
            cache_write_total,
        )

    causal = spec.template is not LayerTemplate.ENCODER_POST_NORM_MHA
    if spec.template is LayerTemplate.ENCODER_POST_NORM_MHA:
        attention_table = f"{table_prefix}_attention"
        add_attention(materialize_factorized_attention(
            cursor, spec, source_table=source_table, output_table=attention_table,
            causal=False, table_prefix=f"{table_prefix}_attn",
            cache_scope=cache_scope, read_cache=read_cache, write_cache=write_cache,
        ))
        residual1 = f"{table_prefix}_residual1"
        non_gemm_total += _materialize_residual(
            cursor, left_table=source_table, right_table=attention_table, output_table=residual1
        )
        after_attention = f"{table_prefix}_after_attention"
        non_gemm_total += _materialize_norm(
            cursor, spec, "norm1", source_table=residual1, output_table=after_attention
        )
        mlp_table = f"{table_prefix}_mlp"
        add_mlp(_materialize_mlp(
            cursor, spec, source_table=after_attention, output_table=mlp_table,
            table_prefix=f"{table_prefix}_ffn",
        ))
        residual2 = f"{table_prefix}_residual2"
        non_gemm_total += _materialize_residual(
            cursor, left_table=after_attention, right_table=mlp_table, output_table=residual2
        )
        non_gemm_total += _materialize_norm(
            cursor, spec, "norm2", source_table=residual2, output_table=output_table
        )
        return finish()

    normalized = f"{table_prefix}_normalized"
    non_gemm_total += _materialize_norm(cursor, spec, "norm1", source_table=source_table, output_table=normalized)
    attention_table = f"{table_prefix}_attention"
    add_attention(materialize_factorized_attention(
        cursor, spec, source_table=normalized, output_table=attention_table,
        causal=causal, table_prefix=f"{table_prefix}_attn",
        cache_scope=cache_scope, read_cache=read_cache, write_cache=write_cache,
    ))
    if spec.template is LayerTemplate.DECODER_PARALLEL_MHA:
        mlp_table = f"{table_prefix}_mlp"
        add_mlp(_materialize_mlp(
            cursor, spec, source_table=normalized, output_table=mlp_table,
            table_prefix=f"{table_prefix}_ffn",
        ))
        partial = f"{table_prefix}_partial"
        non_gemm_total += _materialize_residual(
            cursor, left_table=source_table, right_table=attention_table, output_table=partial
        )
        non_gemm_total += _materialize_residual(
            cursor, left_table=partial, right_table=mlp_table, output_table=output_table
        )
        return finish()

    after_attention = f"{table_prefix}_after_attention"
    non_gemm_total += _materialize_residual(
        cursor, left_table=source_table, right_table=attention_table, output_table=after_attention
    )
    normalized2 = f"{table_prefix}_normalized2"
    non_gemm_total += _materialize_norm(
        cursor, spec, "norm2", source_table=after_attention, output_table=normalized2
    )
    mlp_table = f"{table_prefix}_mlp"
    add_mlp(_materialize_mlp(
        cursor, spec, source_table=normalized2, output_table=mlp_table,
        table_prefix=f"{table_prefix}_ffn",
    ))
    non_gemm_total += _materialize_residual(
        cursor, left_table=after_attention, right_table=mlp_table, output_table=output_table
    )
    return finish()


def _validated_layer_specs(specs: Sequence[TransformerLayerSpec]) -> tuple[TransformerLayerSpec, ...]:
    layer_specs = tuple(specs)
    if not layer_specs:
        raise ValueError("at least one Transformer layer is required")
    if len({spec.model_id for spec in layer_specs}) != 1:
        raise ValueError("all Transformer layers in one pipeline must belong to the same model")
    layer_ids = [spec.layer_id for spec in layer_specs]
    if len(layer_ids) != len(set(layer_ids)):
        raise ValueError("Transformer layer ids must be unique within one pipeline")
    return layer_specs


def _materialize_layer_sequence(
    cursor: Cursor[tuple[object, ...]],
    specs: Sequence[TransformerLayerSpec],
    *,
    source_table: str,
    cache_scope: str | None = None,
    read_cache: bool = False,
    write_cache: bool = False,
) -> tuple[str, ProjectionExecutionTiming, float, float, float]:
    """Connect layer tables without exposing any intermediate vector to Python."""

    current_table = source_table
    projection_total: ProjectionExecutionTiming | None = None
    non_gemm_total = cache_read_total = cache_write_total = 0.0
    for index, spec in enumerate(specs):
        next_table = f"reducinfer_layer_output_{index}"
        projection_timing, non_gemm_ms, cache_read_ms, cache_write_ms = materialize_transformer_layer(
            cursor,
            spec,
            source_table=current_table,
            output_table=next_table,
            table_prefix=f"reducinfer_layer_{index}_{spec.layer_id}",
            cache_scope=cache_scope,
            read_cache=read_cache,
            write_cache=write_cache,
        )
        projection_total = (
            projection_timing
            if projection_total is None
            else merge_projection_timings((projection_total, projection_timing))
        )
        non_gemm_total += non_gemm_ms
        cache_read_total += cache_read_ms
        cache_write_total += cache_write_ms
        current_table = next_table
    if projection_total is None:
        raise ValueError("at least one Transformer layer is required")
    return current_table, projection_total, non_gemm_total, cache_read_total, cache_write_total


def materialize_transformer_layers(
    cursor: Cursor[tuple[object, ...]],
    specs: Sequence[TransformerLayerSpec],
    *,
    source_table: str,
    cache_scope: str | None = None,
    read_cache: bool = False,
    write_cache: bool = False,
) -> tuple[str, ProjectionExecutionTiming, float, float, float]:
    """Expose the validated server-side layer chain without a client transfer."""

    return _materialize_layer_sequence(
        cursor,
        _validated_layer_specs(specs),
        source_table=source_table,
        cache_scope=cache_scope,
        read_cache=read_cache,
        write_cache=write_cache,
    )


def materialize_token_embeddings(
    cursor: Cursor[tuple[object, ...]],
    specs: Sequence[TransformerLayerSpec],
    *,
    token_table: str,
    table_prefix: str,
) -> TokenEmbeddingStage:
    """Materialize canonical token/position embeddings for an existing SQL token table."""

    _validate_table_names(token_table, table_prefix)
    layer_specs = _validated_layer_specs(specs)
    position_kinds = {spec.position_encoding for spec in layer_specs}
    if len(position_kinds) != 1:
        raise ValueError("all layers in one model must use the same position encoding")
    model_id = layer_specs[0].model_id
    token_embedding = f"{table_prefix}_token"
    xd_total, c_total = materialize_factorized_embedding(
        cursor,
        model_id,
        TOKEN_EMBEDDING_TENSOR,
        source_table=token_table,
        index_column="token_id",
        xd_table=f"{token_embedding}_xd",
        output_table=token_embedding,
    )
    output_table = token_embedding
    non_gemm_ms = 0.0
    if next(iter(position_kinds)) is PositionEncoding.LEARNED:
        position_embedding = f"{table_prefix}_position"
        xd_ms, c_ms = materialize_factorized_embedding(
            cursor,
            model_id,
            POSITION_EMBEDDING_TENSOR,
            source_table=token_table,
            index_column="token_idx",
            xd_table=f"{position_embedding}_xd",
            output_table=position_embedding,
        )
        xd_total += xd_ms
        c_total += c_ms
        output_table = f"{table_prefix}_embedded"
        non_gemm_ms = _materialize_residual(
            cursor,
            left_table=token_embedding,
            right_table=position_embedding,
            output_table=output_table,
        )
    return TokenEmbeddingStage(
        output_table,
        ProjectionExecutionTiming(
            xd_ms=xd_total,
            c_ms=c_total,
            kernel_select_ms=0.0,
            compact_kernel_calls=0,
            staged_kernel_calls=0,
        ),
        non_gemm_ms,
    )


def execute_transformer_layers(
    store: DatabaseStore,
    specs: Sequence[TransformerLayerSpec],
    hidden: NDArray[np.float32],
    *,
    cache_scope: str | None = None,
    read_cache: bool = False,
    write_cache: bool = False,
    clear_cache: bool = False,
    position_offset: int = 0,
) -> TransformerStageResult:
    """Run ordered layers with one client upload and one terminal download.

    Layers exchange only session-local temporary tables, so space is bounded by
    the current transaction's layer output. Avoiding Python-held intermediates
    removes O(layer_count) client round trips at the cost of transaction-local
    database pages.
    """

    if hidden.ndim != 3 or 0 in hidden.shape:
        raise ValueError("hidden must have non-empty shape (batch, tokens, dimension)")
    if position_offset < 0:
        raise ValueError("position_offset must be non-negative")
    if (read_cache or write_cache or clear_cache) and (cache_scope is None or hidden.shape[0] != 1):
        raise ValueError("cached Transformer execution requires cache_scope and batch size one")
    layer_specs = _validated_layer_specs(specs)

    end_to_end_started = perf_counter()
    with store.connection.cursor() as cursor:
        cache_clear_ms = 0.0
        if clear_cache:
            cache_started = perf_counter()
            cursor.execute("DELETE FROM kv_cache WHERE request_id = %s", (cache_scope,))
            cache_clear_ms = (perf_counter() - cache_started) * 1_000.0
        cursor.execute(
            "CREATE TEMP TABLE reducinfer_layer_input "
            "(batch_idx INT, token_idx INT, values REAL[]) ON COMMIT DROP"
        )
        input_started = perf_counter()
        with cursor.copy("COPY reducinfer_layer_input (batch_idx, token_idx, values) FROM STDIN") as copy:
            for batch_idx, batch in enumerate(hidden):
                for token_idx, vector in enumerate(batch):
                    copy.write_row((batch_idx, position_offset + token_idx, vector.tolist()))
        input_stage_ms = (perf_counter() - input_started) * 1_000.0
        current_table, projection_timing, non_gemm_total, cache_read_total, cache_write_total = (
            materialize_transformer_layers(
                cursor,
                layer_specs,
                source_table="reducinfer_layer_input",
                cache_scope=cache_scope,
                read_cache=read_cache,
                write_cache=write_cache,
            )
        )
        cache_write_total += cache_clear_ms
        transfer_started = perf_counter()
        cursor.execute(f"SELECT values FROM {current_table} ORDER BY batch_idx,token_idx")
        rows = cursor.fetchall()
        final_transfer_ms = (perf_counter() - transfer_started) * 1_000.0
    store.connection.commit()
    end_to_end_ms = (perf_counter() - end_to_end_started) * 1_000.0
    expected_rows = hidden.shape[0] * hidden.shape[1]
    if len(rows) != expected_rows:
        raise ValueError(f"layer output is incomplete for {layer_specs[0].model_id}")
    values = np.asarray([row[0] for row in rows], dtype=np.float32).reshape(*hidden.shape[:2], -1)
    return TransformerStageResult(
        values=values,
        input_stage_ms=input_stage_ms,
        gemm_xd_ms=projection_timing.xd_ms,
        gemm_c_ms=projection_timing.c_ms,
        kernel_select_ms=projection_timing.kernel_select_ms,
        compact_kernel_calls=projection_timing.compact_kernel_calls,
        staged_kernel_calls=projection_timing.staged_kernel_calls,
        non_gemm_ms=non_gemm_total,
        kv_cache_read_ms=cache_read_total,
        kv_cache_write_ms=cache_write_total,
        final_transfer_ms=final_transfer_ms,
        end_to_end_ms=end_to_end_ms,
    )


def execute_transformer_tokens(
    store: DatabaseStore,
    specs: Sequence[TransformerLayerSpec],
    tokens: NDArray[np.int64],
) -> TransformerStageResult:
    """Look up canonical embeddings and execute all layers in one database transaction."""

    if tokens.ndim != 2 or 0 in tokens.shape or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("tokens must be a non-empty two-dimensional integer array")
    layer_specs = _validated_layer_specs(specs)
    model_id = layer_specs[0].model_id
    end_to_end_started = perf_counter()
    with store.connection.cursor() as cursor:
        cursor.execute(
            "CREATE TEMP TABLE reducinfer_tokens "
            "(batch_idx INT,token_idx INT,token_id INT) ON COMMIT DROP"
        )
        input_started = perf_counter()
        with cursor.copy("COPY reducinfer_tokens (batch_idx,token_idx,token_id) FROM STDIN") as copy:
            for batch_idx, batch in enumerate(tokens):
                for token_idx, token_id in enumerate(batch):
                    copy.write_row((batch_idx, token_idx, int(token_id)))
        input_stage_ms = (perf_counter() - input_started) * 1_000.0

        embedding = materialize_token_embeddings(
            cursor,
            layer_specs,
            token_table="reducinfer_tokens",
            table_prefix="reducinfer_embedding",
        )
        current_table, layer_timing, layer_non_gemm, cache_read, cache_write = (
            materialize_transformer_layers(
                cursor, layer_specs, source_table=embedding.output_table
            )
        )
        projection_timing = merge_projection_timings((embedding.timing, layer_timing))
        non_gemm_total = embedding.non_gemm_ms + layer_non_gemm
        transfer_started = perf_counter()
        cursor.execute(f"SELECT values FROM {current_table} ORDER BY batch_idx,token_idx")
        rows = cursor.fetchall()
        final_transfer_ms = (perf_counter() - transfer_started) * 1_000.0
    store.connection.commit()
    end_to_end_ms = (perf_counter() - end_to_end_started) * 1_000.0
    expected_rows = tokens.shape[0] * tokens.shape[1]
    if len(rows) != expected_rows:
        raise ValueError(f"token pipeline output is incomplete for {model_id}")
    values = np.asarray([row[0] for row in rows], dtype=np.float32).reshape(*tokens.shape, -1)
    return TransformerStageResult(
        values=values,
        input_stage_ms=input_stage_ms,
        gemm_xd_ms=projection_timing.xd_ms,
        gemm_c_ms=projection_timing.c_ms,
        kernel_select_ms=projection_timing.kernel_select_ms,
        compact_kernel_calls=projection_timing.compact_kernel_calls,
        staged_kernel_calls=projection_timing.staged_kernel_calls,
        non_gemm_ms=non_gemm_total,
        kv_cache_read_ms=cache_read,
        kv_cache_write_ms=cache_write,
        final_transfer_ms=final_transfer_ms,
        end_to_end_ms=end_to_end_ms,
    )


def execute_transformer_layer(
    store: DatabaseStore, spec: TransformerLayerSpec, hidden: NDArray[np.float32]
) -> TransformerStageResult:
    """Run a single layer through the same database-resident pipeline."""

    return execute_transformer_layers(store, (spec,), hidden)


def prefill_transformer_layers(
    store: DatabaseStore,
    specs: Sequence[TransformerLayerSpec],
    hidden: NDArray[np.float32],
    *,
    request_id: str,
) -> TransformerStageResult:
    """Run a causal prompt and atomically replace this request's per-layer KV rows."""

    if not request_id:
        raise ValueError("request_id must not be empty")
    layer_specs = tuple(specs)
    if any(spec.template is LayerTemplate.ENCODER_POST_NORM_MHA for spec in layer_specs):
        raise ValueError("KV cache is only valid for decoder templates")
    cache_scope = f"{layer_specs[0].model_id}:{request_id}" if layer_specs else request_id
    return execute_transformer_layers(
        store,
        layer_specs,
        hidden,
        cache_scope=cache_scope,
        write_cache=True,
        clear_cache=True,
    )


def decode_transformer_layers(
    store: DatabaseStore,
    specs: Sequence[TransformerLayerSpec],
    hidden: NDArray[np.float32],
    *,
    request_id: str,
    position: int,
) -> TransformerStageResult:
    """Decode one new token using only cache rows in the same model/request scope."""

    if hidden.ndim != 3 or hidden.shape[:2] != (1, 1):
        raise ValueError("decode hidden input must have shape (1, 1, dimension)")
    if not request_id:
        raise ValueError("request_id must not be empty")
    layer_specs = tuple(specs)
    if any(spec.template is LayerTemplate.ENCODER_POST_NORM_MHA for spec in layer_specs):
        raise ValueError("KV cache is only valid for decoder templates")
    cache_scope = f"{layer_specs[0].model_id}:{request_id}" if layer_specs else request_id
    return execute_transformer_layers(
        store,
        layer_specs,
        hidden,
        cache_scope=cache_scope,
        read_cache=True,
        write_cache=True,
        position_offset=position,
    )
