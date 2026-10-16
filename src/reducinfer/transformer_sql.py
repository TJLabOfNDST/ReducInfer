"""Pure-SQL multi-head attention over session-local factorized projections."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from psycopg import Cursor

from .activation_sql import (
    ProjectionExecutionTiming,
    ProjectionRequest,
    materialize_factorized_projection,
    materialize_factorized_projections,
    merge_projection_timings,
)
from .storage import DatabaseStore
from .types import TransformerLayerSpec


@dataclass(frozen=True, slots=True)
class AttentionStageResult:
    values: NDArray[np.float32]
    gemm_xd_ms: float
    gemm_c_ms: float
    non_gemm_ms: float
    kernel_select_ms: float
    compact_kernel_calls: int
    staged_kernel_calls: int


def _checked_identifiers(*names: str) -> None:
    """Reject identifiers that could alter a generated temporary-table statement.

    Temporary names are generated internally but remain boundary-validated so a
    future caller cannot pass data as a SQL identifier.
    """

    if any(not name.isidentifier() for name in names):
        raise ValueError("temporary attention table names must be identifiers")


def materialize_factorized_attention(
    cursor: Cursor[tuple[object, ...]],
    spec: TransformerLayerSpec,
    *,
    source_table: str,
    output_table: str,
    causal: bool,
    table_prefix: str,
    cache_scope: str | None = None,
    read_cache: bool = False,
    write_cache: bool = False,
) -> tuple[ProjectionExecutionTiming, float, float, float]:
    """Materialize attention from one session table into another.

    Q, K, V, context, and O stay in the PostgreSQL session. Array rows avoid the
    tuple-header overhead of element-wise relations at the cost of ``unnest`` at
    operator boundaries, the locality-optimal choice under pure SQL without
    extensions.
    """

    qkv_table = f"{table_prefix}_qkv"
    qkv_xd_table = f"{table_prefix}_qkv_xd"
    context_table = f"{table_prefix}_context"
    cache_table = f"{table_prefix}_cache"
    _checked_identifiers(
        source_table,
        output_table,
        table_prefix,
        qkv_table,
        qkv_xd_table,
        context_table,
        cache_table,
    )
    if (read_cache or write_cache) and cache_scope is None:
        raise ValueError("cache_scope is required when reading or writing KV cache")

    # Q/K/V share one source expansion and catalog pass while each C buffer is
    # still scanned exactly once, without copying or merging persisted weights.
    qkv_timing = materialize_factorized_projections(
        cursor,
        spec.model_id,
        tuple(
            ProjectionRequest(projection_id, spec.tensor_name(f"attn.{component}"))
            for projection_id, component in enumerate(("q", "k", "v"))
        ),
        source_table=source_table,
        xd_table=qkv_xd_table,
        output_table=qkv_table,
    )

    cache_read_ms = 0.0
    if read_cache:
        cache_started = perf_counter()
        cursor.execute(
            f"""
            CREATE TEMP TABLE {cache_table} ON COMMIT DROP AS
            SELECT head_id,position_idx,key_values,value_values
            FROM kv_cache WHERE request_id=%s AND layer_id=%s
            """,
            (cache_scope, spec.layer_id),
        )
        cache_read_ms = (perf_counter() - cache_started) * 1_000.0

    non_gemm_started = perf_counter()
    uses_rope = spec.rope_theta is not None
    rotation = (
        "q.value"
        if not uses_rope
        else "CASE WHEN mod(q.col_idx, %s) %% 2 = 0 "
        "THEN q.value*cos(q.token_idx*power(%s::double precision, -2*floor(mod(q.col_idx, %s)/2)::double precision/%s))"
        "-pair.value*sin(q.token_idx*power(%s::double precision, -2*floor(mod(q.col_idx, %s)/2)::double precision/%s)) "
        "ELSE q.value*cos(q.token_idx*power(%s::double precision, -2*floor(mod(q.col_idx, %s)/2)::double precision/%s))"
        "+pair.value*sin(q.token_idx*power(%s::double precision, -2*floor(mod(q.col_idx, %s)/2)::double precision/%s)) END"
    )
    key_rotation = rotation.replace("q.", "k.")
    query_pair_join = (
        ""
        if not uses_rope
        else "JOIN q_raw pair ON pair.batch_idx=q.batch_idx AND pair.token_idx=q.token_idx "
        "AND pair.col_idx=q.col_idx+CASE WHEN mod(q.col_idx, %s) %% 2=0 THEN 1 ELSE -1 END"
    )
    key_pair_join = query_pair_join.replace("q_", "k_").replace("q.", "k.")
    rotation_parameters: list[object] = []
    if uses_rope:
        # Parameter order mirrors four frequency terms and avoids a persisted angle cache.
        rotation_parameters = [spec.head_dim, *([spec.rope_theta, spec.head_dim, spec.head_dim] * 4)]
    mask = "AND k.token_idx <= q.token_idx" if causal else ""
    cache_ctes = ""
    key_union = "SELECT * FROM k_current"
    value_union = "SELECT * FROM v_current"
    if read_cache:
        cache_ctes = f"""
        , cached_k AS (
            SELECT 0 AS batch_idx,c.position_idx AS token_idx,
                   c.head_id*{spec.head_dim} + item.ordinality-1 AS col_idx,item.value
            FROM {cache_table} c
            CROSS JOIN LATERAL unnest(c.key_values) WITH ORDINALITY AS item(value, ordinality)
        ), cached_v AS (
            SELECT 0 AS batch_idx,c.position_idx AS token_idx,c.head_id,
                   item.ordinality-1 AS feature,item.value
            FROM {cache_table} c
            CROSS JOIN LATERAL unnest(c.value_values) WITH ORDINALITY AS item(value, ordinality)
        )
        """
        key_union += " UNION ALL SELECT * FROM cached_k"
        value_union += " UNION ALL SELECT * FROM cached_v"
    cursor.execute(
        f"""
        CREATE TEMP TABLE {context_table} ON COMMIT DROP AS
        WITH q_raw AS (
            SELECT batch_idx, token_idx, item.ordinality-1 AS col_idx, item.value
            FROM {qkv_table} CROSS JOIN LATERAL unnest(values) WITH ORDINALITY AS item(value, ordinality)
            WHERE projection_id=0
        ), k_current AS (
            SELECT batch_idx, token_idx, item.ordinality-1 AS col_idx, item.value
            FROM {qkv_table} CROSS JOIN LATERAL unnest(values) WITH ORDINALITY AS item(value, ordinality)
            WHERE projection_id=1
        ), v_current AS (
            SELECT batch_idx, token_idx, floor((item.ordinality-1)/%s)::INT AS head_id,
                   mod(item.ordinality-1,%s)::INT AS feature, item.value
            FROM {qkv_table} CROSS JOIN LATERAL unnest(values) WITH ORDINALITY AS item(value, ordinality)
            WHERE projection_id=2
        )
        {cache_ctes}
        , k_raw AS (
            {key_union}
        ), v_raw AS (
            {value_union}
        ), q AS (
            SELECT q.batch_idx,q.token_idx,floor(q.col_idx/%s)::INT AS head_id,
                   mod(q.col_idx,%s)::INT AS feature,{rotation} AS value
            FROM q_raw q {query_pair_join}
        ), k AS (
            SELECT k.batch_idx,k.token_idx,floor(k.col_idx/%s)::INT AS head_id,
                   mod(k.col_idx,%s)::INT AS feature,{key_rotation} AS value
            FROM k_raw k {key_pair_join}
        ), scores AS (
            SELECT q.batch_idx,q.token_idx AS query_idx,k.token_idx AS key_idx,q.head_id,
                   SUM(q.value*k.value)/sqrt(%s::REAL) AS value
            FROM q JOIN k ON k.batch_idx=q.batch_idx AND k.head_id=floor(q.head_id/%s)
                         AND k.feature=q.feature {mask}
            GROUP BY q.batch_idx,q.token_idx,k.token_idx,q.head_id
        ), maxima AS (
            SELECT batch_idx,query_idx,head_id,max(value) AS value
            FROM scores GROUP BY batch_idx,query_idx,head_id
        ), probs AS (
            SELECT s.*,exp(s.value-m.value)/sum(exp(s.value-m.value))
                OVER (PARTITION BY s.batch_idx,s.query_idx,s.head_id) AS probability
            FROM scores s JOIN maxima m USING(batch_idx,query_idx,head_id)
        ), raw AS (
            SELECT p.batch_idx,p.query_idx AS token_idx,p.head_id*%s+v.feature AS col_idx,
                   SUM(p.probability*v.value)::REAL AS value
            FROM probs p JOIN v_raw v ON v.batch_idx=p.batch_idx AND v.token_idx=p.key_idx
                                      AND v.head_id=floor(p.head_id/%s)
            GROUP BY p.batch_idx,p.query_idx,p.head_id,v.feature
        )
        SELECT a.batch_idx,a.token_idx,
               array_agg(COALESCE(r.value,0.0)::REAL ORDER BY positions.col_idx)::REAL[] AS values
        FROM {source_table} a
        CROSS JOIN LATERAL generate_series(0,%s-1) AS positions(col_idx)
        LEFT JOIN raw r ON r.batch_idx=a.batch_idx AND r.token_idx=a.token_idx AND r.col_idx=positions.col_idx
        GROUP BY a.batch_idx,a.token_idx
        """,
        [
            spec.head_dim, spec.head_dim,
            spec.head_dim, spec.head_dim, *rotation_parameters, *([spec.head_dim] if uses_rope else []),
            spec.head_dim, spec.head_dim, *rotation_parameters, *([spec.head_dim] if uses_rope else []),
            spec.head_dim, spec.query_heads // spec.kv_heads,
            spec.head_dim, spec.query_heads // spec.kv_heads, spec.query_width,
        ],
    )
    non_gemm_ms = (perf_counter() - non_gemm_started) * 1_000.0
    cache_write_ms = 0.0
    if write_cache:
        cache_started = perf_counter()
        cursor.execute(
            f"""
            INSERT INTO kv_cache (request_id,layer_id,head_id,position_idx,key_values,value_values)
            SELECT %s,%s,head.head_id,k.token_idx,
                   k.values[(head.head_id*%s+1):((head.head_id+1)*%s)],
                   v.values[(head.head_id*%s+1):((head.head_id+1)*%s)]
            FROM {qkv_table} k JOIN {qkv_table} v USING(batch_idx,token_idx)
            CROSS JOIN LATERAL generate_series(0,%s-1) AS head(head_id)
            WHERE k.projection_id=1 AND v.projection_id=2 AND k.batch_idx=0
            ON CONFLICT (request_id,layer_id,head_id,position_idx) DO UPDATE
            SET key_values=EXCLUDED.key_values,value_values=EXCLUDED.value_values
            """,
            (
                cache_scope, spec.layer_id,
                spec.head_dim, spec.head_dim, spec.head_dim, spec.head_dim, spec.kv_heads,
            ),
        )
        cache_write_ms = (perf_counter() - cache_started) * 1_000.0
    output_timing = materialize_factorized_projection(
        cursor,
        spec.model_id,
        spec.tensor_name("attn.o"),
        source_table=context_table,
        xd_table=f"{table_prefix}_o_xd",
        output_table=output_table,
    )
    return (
        merge_projection_timings((qkv_timing, output_timing)),
        non_gemm_ms,
        cache_read_ms,
        cache_write_ms,
    )


def execute_factorized_attention(
    store: DatabaseStore, spec: TransformerLayerSpec, hidden: NDArray[np.float32], *, causal: bool
) -> AttentionStageResult:
    """Stage only the public input/output while all attention intermediates stay server-side."""

    if hidden.ndim != 3 or 0 in hidden.shape:
        raise ValueError("hidden must have non-empty shape (batch, tokens, dimension)")
    connection = store.connection
    with connection.cursor() as cursor:
        cursor.execute(
            "CREATE TEMP TABLE reducinfer_attention_input "
            "(batch_idx INT, token_idx INT, values REAL[]) ON COMMIT DROP"
        )
        with cursor.copy("COPY reducinfer_attention_input (batch_idx, token_idx, values) FROM STDIN") as copy:
            for batch_idx, batch in enumerate(hidden):
                for token_idx, vector in enumerate(batch):
                    copy.write_row((batch_idx, token_idx, vector.tolist()))
        projection_timing, non_gemm_ms, _, _ = materialize_factorized_attention(
            cursor,
            spec,
            source_table="reducinfer_attention_input",
            output_table="reducinfer_attention_output",
            causal=causal,
            table_prefix="reducinfer_attention",
        )
        cursor.execute("SELECT values FROM reducinfer_attention_output ORDER BY batch_idx, token_idx")
        rows = cursor.fetchall()
    connection.commit()
    if len(rows) != hidden.shape[0] * hidden.shape[1]:
        raise ValueError(f"missing or incompatible attention weights for {spec.model_id}/{spec.tensor_prefix}")
    values = np.asarray([row[0] for row in rows], dtype=np.float32).reshape(*hidden.shape[:2], -1)
    return AttentionStageResult(
        values=values,
        gemm_xd_ms=projection_timing.xd_ms,
        gemm_c_ms=projection_timing.c_ms,
        non_gemm_ms=non_gemm_ms,
        kernel_select_ms=projection_timing.kernel_select_ms,
        compact_kernel_calls=projection_timing.compact_kernel_calls,
        staged_kernel_calls=projection_timing.staged_kernel_calls,
    )
