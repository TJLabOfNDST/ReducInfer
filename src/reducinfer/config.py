"""Typed configuration for ReducInfer preprocessing and storage."""

from __future__ import annotations

from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True, slots=True)
class ReducInferConfig:
    """Keep numerical dimensions immutable so every persisted factor is compatible."""

    block_h: int = 64
    block_w: int = 64
    rank: int = 16
    brp_d1: int = 8
    brp_d2: int = 8
    cluster_count: int = 16
    random_seed: int = 42
    database_dsn: str | None = None

    def __post_init__(self) -> None:
        dimensions = {
            "block_h": self.block_h,
            "block_w": self.block_w,
            "rank": self.rank,
            "brp_d1": self.brp_d1,
            "brp_d2": self.brp_d2,
            "cluster_count": self.cluster_count,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rank > min(self.block_h, self.block_w):
            raise ValueError("rank must not exceed the smallest block extent")

    @classmethod
    def from_environment(cls) -> "ReducInferConfig":
        """Read only the connection string from the environment, never source code."""

        return cls(database_dsn=getenv("REDUCINFER_DSN"))

