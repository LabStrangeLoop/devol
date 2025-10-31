"""Devol: Diffusion Evolution Algorithm."""

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
from devol.exceptions import (
    ConfigurationError,
    DevolError,
    EvolutionError,
    FitnessComputationError,
)

__version__ = "0.1.0"

__all__ = [
    "DiffusionEvolution",
    "DiffusionConfig",
    "DistanceConfig",
    "DistanceType",
    "FitnessConfig",
    "FitnessMapping",
    "ScheduleConfig",
    "ScheduleType",
    "DevolError",
    "ConfigurationError",
    "EvolutionError",
    "FitnessComputationError",
]
