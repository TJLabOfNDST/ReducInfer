"""PostgreSQL connection ownership and runtime catalog reads."""

from __future__ import annotations

from pathlib import Path
import sysconfig

import numpy as np
from numpy.typing import NDArray
import psycopg

from .pg_array import Float32Array, Float32ArrayBinaryDumper
from .types import (
    ActivationKind,
    LayerTemplate,
    NormKind,
    PositionEncoding,
    TransformerLayerSpec,
)


def _schema_path() -> Path:
    """Resolve source and installed layouts without embedding a machine path.

    The source layout supports development, while the installed layout keeps the
    wheel self-contained. Checking only these two fixed locations avoids an
    unbounded filesystem search and accidental schema substitution.
    """

    candidates = (
        Path(__file__).resolve().parents[2] / "sql" / "schema.sql",
        Path(sysconfig.get_path("data")) / "share" / "reducinfer" / "schema.sql",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("packaged ReducInfer schema.sql was not found")


class DatabaseStore:
    """Own one PostgreSQL connection used by import and pure-SQL execution."""

    def __init__(self, dsn: str | None) -> None:
        if not dsn:
            raise ValueError("REDUCINFER_DSN is required")
        self._connection: psycopg.Connection[tuple[object, ...]] = psycopg.connect(dsn)
        self._connection.adapters.register_dumper(Float32Array, Float32ArrayBinaryDumper)
        with self._connection.cursor() as cursor:
            cursor.execute("SET search_path TO reducinfer, public")
        self._connection.commit()

    def initialize(self) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(_schema_path().read_text(encoding="utf-8"))
        self._connection.commit()

    def reset_project_schema(self, expected_database: str) -> None:
        """Drop only this project schema after an exact database-name check."""

        if not expected_database:
            raise ValueError("expected database name is required")
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT current_database()")
            actual_database = str(cursor.fetchone()[0])
            if actual_database != expected_database:
                raise ValueError(
                    "database reset confirmation mismatch: "
                    f"expected {expected_database}, got {actual_database}"
                )
            cursor.execute("DROP SCHEMA IF EXISTS reducinfer CASCADE")
        self._connection.commit()
        self.initialize()

    def layer_specs(self, model_id: str) -> tuple[TransformerLayerSpec, ...]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT layer_id, template, query_heads, kv_heads, head_dim,
                       position_encoding, norm_kind, norm_epsilon, activation,
                       tensor_prefix, rope_theta, attention_bias, mlp_bias
                FROM model_layers l JOIN models m USING (model_key)
                WHERE m.model_id = %s ORDER BY layer_id
                """,
                (model_id,),
            )
            rows = cursor.fetchall()
        return tuple(
            TransformerLayerSpec(
                model_id=model_id,
                layer_id=int(row[0]),
                template=LayerTemplate(str(row[1])),
                query_heads=int(row[2]),
                kv_heads=int(row[3]),
                head_dim=int(row[4]),
                position_encoding=PositionEncoding(str(row[5])),
                norm_kind=NormKind(str(row[6])),
                norm_epsilon=float(row[7]),
                activation=ActivationKind(str(row[8])),
                tensor_prefix=str(row[9]),
                rope_theta=None if row[10] is None else float(row[10]),
                attention_bias=bool(row[11]),
                mlp_bias=bool(row[12]),
            )
            for row in rows
        )

    def load_vector(self, model_id: str, vector_name: str) -> NDArray[np.float32]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT v.values FROM dense_vectors v
                JOIN models m USING (model_key)
                WHERE m.model_id = %s AND v.vector_name = %s
                """,
                (model_id, vector_name),
            )
            row = cursor.fetchone()
        if row is None:
            raise KeyError(f"vector not found: {model_id}/{vector_name}")
        return np.asarray(row[0], dtype=np.float32)

    @property
    def connection(self) -> psycopg.Connection[tuple[object, ...]]:
        return self._connection

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "DatabaseStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
