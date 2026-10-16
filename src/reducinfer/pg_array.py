"""Bounded-memory PostgreSQL binary adaptation for contiguous FP32 arrays."""

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np
from numpy.typing import NDArray
from psycopg.adapt import Dumper
from psycopg.pq import Format


_FLOAT4_OID = 700
_FLOAT4_ARRAY_OID = 1021
_MAX_POSTGRES_VALUE_BYTES = (1 << 30) - 1


@dataclass(frozen=True, slots=True)
class Float32Array:
    """Mark one flat C-contiguous array for binary ``REAL[]`` adaptation."""

    values: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.values.ndim != 1 or self.values.dtype != np.float32:
            raise ValueError("Float32Array requires a one-dimensional float32 array")
        if not self.values.flags.c_contiguous:
            raise ValueError("Float32Array requires C-contiguous storage")


class Float32ArrayBinaryDumper(Dumper):
    """Encode PostgreSQL's one-dimensional binary array format in O(n).

    A structured NumPy buffer interleaves each network-order element length and
    value without O(n) Python objects. It trades roughly twice the numeric payload
    size for linear encoding and avoids the much larger temporary allocation of
    ``tolist()``.
    """

    format = Format.BINARY
    oid = _FLOAT4_ARRAY_OID

    def dump(self, obj: Float32Array) -> bytes:
        element_count = int(obj.values.size)
        payload_bytes = element_count * 8
        if 20 + payload_bytes > _MAX_POSTGRES_VALUE_BYTES:
            raise ValueError("float4[] binary value exceeds PostgreSQL's 1 GiB datum limit")
        payload = np.empty(
            element_count,
            dtype=np.dtype([("length", ">i4"), ("value", ">f4")]),
        )
        payload["length"] = 4
        payload["value"] = obj.values
        header = struct.pack(
            "!iiiii",
            1,
            0,
            _FLOAT4_OID,
            element_count,
            1,
        )
        return header + payload.tobytes()
