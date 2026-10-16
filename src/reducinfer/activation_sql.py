"""Array-vector activation staging and separately timed factorized GEMM phases."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from psycopg import Cursor

from .storage import DatabaseStore


# The selector preserves a stable operational timing contract while ROW2COL uses
# one physical kernel for every size.
_FP32_BYTES = 4
_COMPACT_MAXIMUM_C_BYTES = 512 * 1024
_COMPACT_TOTAL_C_BYTES = 6 * 1024 * 1024
_COMPACT_WEIGHTED_C_BYTES = 2 * 1024 * 1024


class ProjectionKernel(str, Enum):
    """Safe single-pass C execution strategies."""

    COMPACT = "compact_single_pass"
    STAGED = "analyzed_staged"


@dataclass(frozen=True, slots=True)
class ProjectionWorkloadStats:
    """Scalar-only metadata used to choose a projection kernel."""

    source_rows: int
    tensor_count: int
    block_count: int
    maximum_c_elements: int
    total_c_elements: int

    def __post_init__(self) -> None:
        counts = (
            self.source_rows,
            self.tensor_count,
            self.block_count,
            self.maximum_c_elements,
            self.total_c_elements,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("projection workload counts must be positive")
        if self.maximum_c_elements > self.total_c_elements:
            raise ValueError("maximum C elements cannot exceed total C elements")


@dataclass(frozen=True, slots=True)
class ProjectionKernelDecision:
    """One deterministic decision and overlapping time."""

    kernel: ProjectionKernel
    select_ms: float
    stats: ProjectionWorkloadStats


@dataclass(frozen=True, slots=True)
class ProjectionExecutionTiming:
    """Non-overlapping GEMM phases plus selector diagnostics."""

    xd_ms: float
    c_ms: float
    kernel_select_ms: float
    compact_kernel_calls: int
    staged_kernel_calls: int


def merge_projection_timings(
    timings: Sequence[ProjectionExecutionTiming],
) -> ProjectionExecutionTiming:
    """Merge timings in O(n) time and O(1) auxiliary space."""

    if not timings:
        raise ValueError("at least one projection timing is required")
    xd_ms = c_ms = kernel_select_ms = 0.0
    compact_kernel_calls = staged_kernel_calls = 0
    for timing in timings:
        xd_ms += timing.xd_ms
        c_ms += timing.c_ms
        kernel_select_ms += timing.kernel_select_ms
        compact_kernel_calls += timing.compact_kernel_calls
        staged_kernel_calls += timing.staged_kernel_calls
    return ProjectionExecutionTiming(
        xd_ms=xd_ms,
        c_ms=c_ms,
        kernel_select_ms=kernel_select_ms,
        compact_kernel_calls=compact_kernel_calls,
        staged_kernel_calls=staged_kernel_calls,
    )


def choose_projection_kernel(stats: ProjectionWorkloadStats) -> ProjectionKernel:
    """Choose in O(1) from scalar metadata without reading C."""

    maximum_bytes = stats.maximum_c_elements * _FP32_BYTES
    total_bytes = stats.total_c_elements * _FP32_BYTES
    weighted_bytes = stats.source_rows * maximum_bytes
    if (
        maximum_bytes <= _COMPACT_MAXIMUM_C_BYTES
        and total_bytes <= _COMPACT_TOTAL_C_BYTES
        and weighted_bytes <= _COMPACT_WEIGHTED_C_BYTES
    ):
        return ProjectionKernel.COMPACT
    return ProjectionKernel.STAGED


@dataclass(frozen=True, slots=True)
class ProjectionStageResult:
    """Terminal vectors and factor GEMM timings."""

    values: NDArray[np.float32]
    xd_ms: float
    c_ms: float
    kernel_select_ms: float
    compact_kernel_calls: int
    staged_kernel_calls: int


@dataclass(frozen=True, slots=True)
class ProjectionRequest:
    """Identify one tensor inside a grouped projection."""

    projection_id: int
    tensor_name: str
    bias_name: str | None = None


@dataclass(frozen=True, slots=True)
class SqlStatement:
    """A trusted SQL fragment and its bound values."""

    query: str
    parameters: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class ProjectionStageNames:
    """The only materialized ROW2COL boundaries."""

    xd: str
    output: str


@dataclass(frozen=True, slots=True)
class ProjectionStages:
    """Shared pure SELECT stages for execution and plan inspection."""

    names: ProjectionStageNames
    xd: SqlStatement
    output: SqlStatement


def _validate_projection_identifier(identifier: str) -> None:
    """Reject names that cannot be interpolated as unquoted PostgreSQL identifiers."""

    if not identifier.isidentifier() or len(identifier.encode("utf-8")) > 63:
        raise ValueError("temporary projection table names must be valid PostgreSQL identifiers")


def _validated_projection_requests(
    requests: Sequence[ProjectionRequest],
) -> tuple[ProjectionRequest, ...]:
    """Validate a grouped request once before any SQL side effect."""

    grouped_requests = tuple(requests)
    if not grouped_requests:
        raise ValueError("at least one projection request is required")
    projection_ids = [request.projection_id for request in grouped_requests]
    if len(projection_ids) != len(set(projection_ids)):
        raise ValueError("projection ids must be unique")
    if any(projection_id < 0 or projection_id > 32_767 for projection_id in projection_ids):
        raise ValueError("projection ids must fit PostgreSQL SMALLINT")
    if any(not request.tensor_name for request in grouped_requests):
        raise ValueError("projection tensor names must not be empty")
    return grouped_requests


def _projection_request_values(
    requests: Sequence[ProjectionRequest],
) -> tuple[str, tuple[object, ...]]:
    """Build one typed VALUES list shared by metadata and execution SQL."""

    request_rows_sql = ",".join(
        "(%s::SMALLINT,%s::TEXT,%s::TEXT)" for _ in requests
    )
    request_parameters: list[object] = []
    for request in requests:
        request_parameters.extend(
            (
                request.projection_id,
                request.tensor_name,
                request.bias_name
                if request.bias_name is not None
                else request.tensor_name.removesuffix("weight") + "bias",
            )
        )
    return request_rows_sql, tuple(request_parameters)


def select_projection_kernel(
    cursor: Cursor[tuple[object, ...]],
    model_id: str,
    requests: Sequence[ProjectionRequest],
    *,
    source_table: str,
) -> ProjectionKernelDecision:
    """Choose a kernel from one scalar-only metadata query.

    C values are intentionally absent, so selection costs O(P+B+S) catalog/source
    rows and O(1) client space instead of detoasting payloads. Avoiding C reads
    prevents selection itself from expanding large objects.
    """

    _validate_projection_identifier(source_table)
    grouped_requests = _validated_projection_requests(requests)
    request_rows_sql, request_values = _projection_request_values(grouped_requests)
    started = perf_counter()
    cursor.execute(
        f"""
        WITH requested(projection_id,tensor_name,bias_name) AS (
            VALUES {request_rows_sql}
        ), requested_tensors AS MATERIALIZED (
            SELECT r.projection_id,t.tensor_id,t.rows,
                   SUM(cardinality(c.c_values))::BIGINT AS c_elements
            FROM requested r
            JOIN models m ON m.model_id=%s
            JOIN tensors t
              ON t.model_key=m.model_key AND t.tensor_name=r.tensor_name
            JOIN layer_c_vectors c USING(tensor_id)
            GROUP BY r.projection_id,t.tensor_id,t.rows
        ), tensor_stats AS (
            SELECT count(*) AS tensor_count,
                   max(c_elements) AS maximum_c_elements,
                   sum(c_elements) AS total_c_elements,
                   min(rows) AS minimum_input_width,
                   max(rows) AS maximum_input_width
            FROM requested_tensors
        ), topology AS (
            SELECT count(*) AS block_count
            FROM requested_tensors t
            JOIN block_slice b USING(tensor_id)
        ), source AS (
            SELECT count(*) AS source_rows,
                   count(*) FILTER (
                       WHERE cardinality(values)<>(SELECT minimum_input_width FROM tensor_stats)
                   ) AS invalid_source_rows
            FROM {source_table}
        )
        SELECT source_rows,tensor_count,block_count,
               maximum_c_elements,total_c_elements,
               minimum_input_width,maximum_input_width,invalid_source_rows
        FROM source CROSS JOIN tensor_stats CROSS JOIN topology
        """,
        (*request_values, model_id),
    )
    row = cursor.fetchone()
    select_ms = (perf_counter() - started) * 1_000.0
    if row is None:
        raise ValueError("projection metadata query returned no row")
    (
        source_rows,
        tensor_count,
        block_count,
        maximum_c_elements,
        total_c_elements,
        minimum_input_width,
        maximum_input_width,
        invalid_source_rows,
    ) = row
    if int(source_rows) == 0:
        raise ValueError("projection source must not be empty")
    if int(tensor_count) != len(grouped_requests):
        raise ValueError(f"missing projection tensor metadata for {model_id}")
    if int(block_count) == 0:
        raise ValueError(f"missing projection block topology for {model_id}")
    if maximum_c_elements is None or total_c_elements is None:
        raise ValueError(f"missing projection C metadata for {model_id}")
    if minimum_input_width != maximum_input_width or int(invalid_source_rows):
        raise ValueError("projection source width does not match requested tensors")
    stats = ProjectionWorkloadStats(
        source_rows=int(source_rows),
        tensor_count=int(tensor_count),
        block_count=int(block_count),
        maximum_c_elements=int(maximum_c_elements),
        total_c_elements=int(total_c_elements),
    )
    return ProjectionKernelDecision(choose_projection_kernel(stats), select_ms, stats)


def build_projection_stages(
    model_id: str,
    requests: Sequence[ProjectionRequest],
    *,
    source_table: str,
    xd_table: str,
    output_table: str,
) -> ProjectionStages:
    """Build two deterministic TranSQL+ factor-vector stages.

    Both right operands are pre-transposed. Correlated array dots keep the
    optimal arithmetic count while reducing outer relational rows from scalar
    factors to result vectors.

    PostgreSQL core has neither TranSQL+'s POSITIONAL JOIN nor a numeric-array
    dot operator, so explicit keys plus correlated ``unnest(a,b)``/``SUM`` are
    the pure-SQL equivalent. This preserves the result and arithmetic count but
    cannot reproduce the SIMD execution behavior of a columnar engine.
    """

    grouped_requests = _validated_projection_requests(requests)
    names = ProjectionStageNames(xd=xd_table, output=output_table)
    for identifier in (source_table, names.xd, names.output):
        _validate_projection_identifier(identifier)
    request_rows_sql, request_values = _projection_request_values(grouped_requests)

    xd = SqlStatement(
        query=f"""
        WITH requested(projection_id,tensor_name,bias_name) AS (
            VALUES {request_rows_sql}
        ), target AS MATERIALIZED (
            SELECT DISTINCT r.projection_id,b.factor_id AS d_id,
                   b.block_row_offset,b.valid_rows
            FROM requested r
            JOIN models m ON m.model_id=%s
            JOIN tensors t ON t.model_key=m.model_key AND t.tensor_name=r.tensor_name
            JOIN block_slice b USING(tensor_id)
        ), rank_dots AS (
            SELECT target.projection_id,source.batch_idx,source.token_idx,
                   target.d_id,target.block_row_offset,d.rank_idx,
                   dot.value
            FROM target
            CROSS JOIN {source_table} source
            JOIN shared_matrix_d_columns d ON d.d_id=target.d_id
            CROSS JOIN LATERAL (
                SELECT SUM(pair.input_value*pair.d_value)::REAL AS value
                FROM unnest(
                    source.values[
                        target.block_row_offset+1:
                        target.block_row_offset+target.valid_rows
                    ],
                    d.d_values[1:target.valid_rows]
                ) AS pair(input_value,d_value)
            ) dot
        )
        SELECT projection_id,batch_idx,token_idx,d_id,block_row_offset,
               array_agg(value ORDER BY rank_idx)::REAL[] AS values
        FROM rank_dots
        GROUP BY projection_id,batch_idx,token_idx,d_id,block_row_offset
        """,
        parameters=(*request_values, model_id),
    )
    output = SqlStatement(
        query=f"""
        WITH requested(projection_id,tensor_name,bias_name) AS (
            VALUES {request_rows_sql}
        ), target AS MATERIALIZED (
            SELECT r.projection_id,t.tensor_id,t.cols
            FROM requested r
            JOIN models m ON m.model_id=%s
            JOIN tensors t ON t.model_key=m.model_key AND t.tensor_name=r.tensor_name
        ), raw AS (
            SELECT x.projection_id,x.batch_idx,x.token_idx,c.output_col,
                   SUM(dot.value)::REAL AS value
            FROM target
            JOIN layer_c_vectors c USING(tensor_id)
            JOIN {names.xd} x
              ON x.projection_id=target.projection_id
             AND x.d_id=c.factor_id
             AND x.block_row_offset=c.block_row_offset
            CROSS JOIN LATERAL (
                SELECT SUM(pair.x_value*pair.c_value)::REAL AS value
                FROM unnest(x.values,c.c_values) AS pair(x_value,c_value)
            ) dot
            WHERE cardinality(x.values)=c.rank
            GROUP BY x.projection_id,x.batch_idx,x.token_idx,c.output_col
        ), bias AS (
            SELECT r.projection_id,v.values
            FROM requested r
            JOIN models m ON m.model_id=%s
            JOIN dense_vectors v
              ON v.model_key=m.model_key AND v.vector_name=r.bias_name
        )
        SELECT s.projection_id,a.batch_idx,a.token_idx,
               array_agg(
                   (COALESCE(raw.value,0.0)+COALESCE(b.values[column_idx+1],0.0))::REAL
                   ORDER BY column_idx
               )::REAL[] AS values
        FROM target s
        CROSS JOIN {source_table} a
        CROSS JOIN LATERAL generate_series(0,s.cols-1) AS column_idx
        LEFT JOIN raw
          ON raw.projection_id=s.projection_id
         AND raw.batch_idx=a.batch_idx
         AND raw.token_idx=a.token_idx
         AND raw.output_col=column_idx
        LEFT JOIN bias b ON b.projection_id=s.projection_id
        GROUP BY s.projection_id,a.batch_idx,a.token_idx
        """,
        parameters=(*request_values, model_id, model_id),
    )
    return ProjectionStages(
        names=names,
        xd=xd,
        output=output,
    )


def _create_projection_stage(
    cursor: Cursor[tuple[object, ...]],
    table_name: str,
    statement: SqlStatement,
) -> None:
    """Materialize one validated stage."""

    cursor.execute(
        f"CREATE TEMP TABLE {table_name} ON COMMIT DROP AS {statement.query}",
        statement.parameters,
    )


def _analyze_projection_stage(
    cursor: Cursor[tuple[object, ...]], table_name: str
) -> None:
    """Expose exact temporary cardinality to PostgreSQL."""

    cursor.execute(f"ANALYZE {table_name}")


def _drop_projection_stages(
    cursor: Cursor[tuple[object, ...]], *table_names: str
) -> None:
    """Release helpers after their final consumer."""

    cursor.execute("DROP TABLE " + ",".join(table_names))


def _execute_projection_kernel(
    cursor: Cursor[tuple[object, ...]],
    stages: ProjectionStages,
    kernel: ProjectionKernel,
) -> None:
    """Execute the single ROW2COL C stage after XD exists.

    Both selector outcomes map to the same vector-dot statement because ROW2COL
    bounds C access without a second materialization path.
    """

    if kernel not in (ProjectionKernel.COMPACT, ProjectionKernel.STAGED):
        raise ValueError(f"unsupported projection kernel: {kernel!r}")
    _create_projection_stage(cursor, stages.names.output, stages.output)


def materialize_factorized_embedding(
    cursor: Cursor[tuple[object, ...]],
    model_id: str,
    tensor_name: str,
    *,
    source_table: str,
    index_column: str,
    xd_table: str,
    output_table: str,
) -> tuple[float, float]:
    """Look up factorized embedding rows without constructing one-hot inputs.

    Row indices directly access the target elements in pre-transposed D columns
    before multiplying by C column vectors. The operation costs
    O(token_count * rank * output_width), avoids an
    O(token_count * vocabulary_size) one-hot GEMM, and trades indexed topology
    lookups for asymptotically optimal row access.
    """

    identifiers = (source_table, index_column, xd_table, output_table)
    if any(not identifier.isidentifier() for identifier in identifiers):
        raise ValueError("embedding table and column names must be identifiers")
    cursor.execute(
        f"""
        SELECT min(s.{index_column}),max(s.{index_column}),count(*),t.rows
        FROM {source_table} s
        CROSS JOIN tensors t
        JOIN models m USING(model_key)
        WHERE m.model_id=%s AND t.tensor_name=%s
        GROUP BY t.rows
        """,
        (model_id, tensor_name),
    )
    bounds = cursor.fetchone()
    if bounds is None:
        raise ValueError(f"missing embedding tensor: {model_id}/{tensor_name}")
    minimum, maximum, count, row_count = bounds
    if int(count) == 0:
        raise ValueError("embedding source must not be empty")
    minimum = int(minimum)
    maximum = int(maximum)
    row_count = int(row_count)
    if minimum < 0 or maximum >= row_count:
        raise ValueError(
            f"{tensor_name} index outside [0, {row_count}): min={minimum}, max={maximum}"
        )

    xd_started = perf_counter()
    cursor.execute(
        f"""
        CREATE TEMP TABLE {xd_table} ON COMMIT DROP AS
        WITH target AS (
            SELECT b.factor_id AS d_id,b.block_row_offset,b.valid_rows
            FROM block_slice b JOIN tensors t USING(tensor_id) JOIN models m USING(model_key)
            WHERE m.model_id=%s AND t.tensor_name=%s
        ), active AS (
            SELECT DISTINCT d_id,block_row_offset,valid_rows FROM target
        )
        SELECT s.batch_idx,s.token_idx,a.d_id,a.block_row_offset,
               array_agg(
                   d.d_values[s.{index_column}-a.block_row_offset+1]
                   ORDER BY d.rank_idx
               )::REAL[] AS values
        FROM {source_table} s
        JOIN active a ON s.{index_column}>=a.block_row_offset
                     AND s.{index_column}<a.block_row_offset+a.valid_rows
        JOIN shared_matrix_d_columns d ON d.d_id=a.d_id
        GROUP BY s.batch_idx,s.token_idx,a.d_id,a.block_row_offset
        """,
        (model_id, tensor_name),
    )
    xd_ms = (perf_counter() - xd_started) * 1_000.0

    c_started = perf_counter()
    cursor.execute(
        f"""
        CREATE TEMP TABLE {output_table} ON COMMIT DROP AS
        WITH target AS (
            SELECT b.tensor_id
            FROM block_slice b JOIN tensors t USING(tensor_id) JOIN models m USING(model_key)
            WHERE m.model_id=%s AND t.tensor_name=%s
            GROUP BY b.tensor_id
        ), raw AS (
            SELECT x.batch_idx,x.token_idx,c.output_col AS col_idx,
                   SUM(dot.value)::REAL AS value
            FROM {xd_table} x
            CROSS JOIN target t
            JOIN layer_c_vectors c
              ON c.tensor_id=t.tensor_id
             AND c.factor_id=x.d_id
             AND c.block_row_offset=x.block_row_offset
            CROSS JOIN LATERAL (
                SELECT SUM(pair.x_value*pair.c_value)::REAL AS value
                FROM unnest(x.values,c.c_values) AS pair(x_value,c_value)
            ) dot
            WHERE cardinality(x.values)=c.rank
            GROUP BY x.batch_idx,x.token_idx,c.output_col
        ), shape AS (
            SELECT t.cols FROM tensors t JOIN models m USING(model_key)
            WHERE m.model_id=%s AND t.tensor_name=%s
        )
        SELECT s.batch_idx,s.token_idx,
               array_agg(COALESCE(raw.value,0.0)::REAL ORDER BY col_idx)::REAL[] AS values
        FROM {source_table} s
        CROSS JOIN LATERAL generate_series(0,(SELECT cols-1 FROM shape)) AS col_idx
        LEFT JOIN raw USING(batch_idx,token_idx,col_idx)
        GROUP BY s.batch_idx,s.token_idx
        """,
        (model_id, tensor_name, model_id, tensor_name),
    )
    return xd_ms, (perf_counter() - c_started) * 1_000.0


def materialize_factorized_projections(
    cursor: Cursor[tuple[object, ...]],
    model_id: str,
    requests: Sequence[ProjectionRequest],
    *,
    source_table: str,
    xd_table: str,
    output_table: str,
) -> ProjectionExecutionTiming:
    """Materialize compatible projections through the ROW2COL kernel.

    The historical selector call now validates scalar metadata and remains timed
    inside C exactly once. XD is the physical boundary, and selector time is
    counted in the C stage exactly once.
    """

    stages = build_projection_stages(
        model_id,
        requests,
        source_table=source_table,
        xd_table=xd_table,
        output_table=output_table,
    )

    xd_started = perf_counter()
    _create_projection_stage(cursor, stages.names.xd, stages.xd)
    _analyze_projection_stage(cursor, stages.names.xd)
    xd_ms = (perf_counter() - xd_started) * 1_000.0

    c_started = perf_counter()
    decision = select_projection_kernel(
        cursor,
        model_id,
        requests,
        source_table=source_table,
    )
    _execute_projection_kernel(cursor, stages, ProjectionKernel.COMPACT)
    _drop_projection_stages(cursor, stages.names.xd)
    c_ms = (perf_counter() - c_started) * 1_000.0
    return ProjectionExecutionTiming(
        xd_ms=xd_ms,
        c_ms=c_ms,
        kernel_select_ms=decision.select_ms,
        compact_kernel_calls=1,
        staged_kernel_calls=0,
    )


def materialize_factorized_projection(
    cursor: Cursor[tuple[object, ...]],
    model_id: str,
    tensor_name: str,
    *,
    source_table: str,
    xd_table: str,
    output_table: str,
) -> ProjectionExecutionTiming:
    """Use the shared grouped kernel for one projection."""

    return materialize_factorized_projections(
        cursor,
        model_id,
        (ProjectionRequest(0, tensor_name),),
        source_table=source_table,
        xd_table=xd_table,
        output_table=output_table,
    )


def execute_factorized_projection(
    store: DatabaseStore,
    model_id: str,
    tensor_name: str,
    activations: NDArray[np.float32],
) -> ProjectionStageResult:
    """Evaluate one `X(D·C)` projection while retaining XD inside PostgreSQL.

    XD remains in a session-local table instead of returning to Python. Two SQL
    statements provide non-overlapping timing for the two GEMM stages.
    """

    if activations.ndim != 3 or 0 in activations.shape:
        raise ValueError("activations must have non-empty shape (batch, tokens, dimension)")
    connection = store.connection
    with connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS reducinfer_activation")
        cursor.execute("DROP TABLE IF EXISTS reducinfer_xd")
        cursor.execute("DROP TABLE IF EXISTS reducinfer_projection")
        cursor.execute(
            "CREATE TEMP TABLE reducinfer_activation (batch_idx INT, token_idx INT, values REAL[]) ON COMMIT DROP"
        )
        with cursor.copy("COPY reducinfer_activation (batch_idx, token_idx, values) FROM STDIN") as copy:
            for batch_idx, batch in enumerate(activations):
                for token_idx, vector in enumerate(batch):
                    copy.write_row((batch_idx, token_idx, vector.tolist()))
        timing = materialize_factorized_projection(
            cursor,
            model_id,
            tensor_name,
            source_table="reducinfer_activation",
            xd_table="reducinfer_xd",
            output_table="reducinfer_projection",
        )
        cursor.execute(
            "SELECT batch_idx,token_idx,values FROM reducinfer_projection "
            "WHERE projection_id=0 ORDER BY batch_idx,token_idx"
        )
        rows = cursor.fetchall()
    connection.commit()
    if len(rows) != activations.shape[0] * activations.shape[1]:
        raise ValueError(f"unknown or incompatible tensor: {model_id}/{tensor_name}")
    values = np.asarray([row[2] for row in rows], dtype=np.float32)
    return ProjectionStageResult(
        values=values.reshape(activations.shape[0], activations.shape[1], -1),
        xd_ms=timing.xd_ms,
        c_ms=timing.c_ms,
        kernel_select_ms=timing.kernel_select_ms,
        compact_kernel_calls=timing.compact_kernel_calls,
        staged_kernel_calls=timing.staged_kernel_calls,
    )
