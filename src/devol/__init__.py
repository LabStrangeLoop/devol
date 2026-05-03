"""Devol: Diffusion Evolution Algorithm."""

from importlib.metadata import PackageNotFoundError, version

from devol.algorithm import DiffusionEvolution
from devol.config import (
    DiffusionConfig,
    DistanceConfig,
    DistanceType,
    FitnessConfig,
    FitnessMapping,
    ScheduleConfig,
    ScheduleType,
)

try:
    __version__ = version("devol")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "DiffusionEvolution",
    "DiffusionConfig",
    "DistanceConfig",
    "DistanceType",
    "FitnessConfig",
    "FitnessMapping",
    "ScheduleConfig",
    "ScheduleType",
]
