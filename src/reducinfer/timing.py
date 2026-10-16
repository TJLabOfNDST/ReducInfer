"""Runtime timing values exposed by SQL execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PhaseTimings:
    """Store mutually exclusive server-side execution phases.

    Mutually exclusive fields avoid double counting and can be summed without
    exposing the SQL implementation phases to callers.
    """

    input_stage_ms: float
    gemm_xd_ms: float
    gemm_c_ms: float
    non_gemm_ms: float
    kv_cache_read_ms: float
    kv_cache_write_ms: float
    final_transfer_ms: float
    end_to_end_ms: float

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in asdict(self).values()):
            raise ValueError("phase timings must be non-negative")
