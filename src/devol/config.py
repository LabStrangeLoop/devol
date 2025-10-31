"""Configuration and settings models for Diffusion Evolution."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from devol.exceptions import ConfigurationError


class ScheduleType(str, Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    DDPM = "ddpm"


class FitnessMapping(str, Enum):
    DIRECT = "direct"
    EXPONENTIAL = "exponential"
    RANK = "rank"


class DistanceType(str, Enum):
    EUCLIDEAN = "euclidean"
    LATENT = "latent"
    COSINE = "cosine"


class ScheduleConfig(BaseModel):
    """Configuration for the alpha schedule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: ScheduleType = ScheduleType.COSINE
    epsilon: float = Field(default=1e-4, gt=0, lt=1)


class FitnessConfig(BaseModel):
    """Configuration for mapping raw fitness values to weights."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mapping: FitnessMapping = FitnessMapping.DIRECT
    temperature: float = Field(default=1.0, gt=0)
    normalize: bool = True
    shift_negative: bool = True


class DistanceConfig(BaseModel):
    """Configuration for the distance metric used during evolution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: DistanceType = DistanceType.EUCLIDEAN
    latent_dim: int = Field(default=2, ge=1)


class DiffusionConfig(BaseModel):
    """Top-level configuration for the Diffusion Evolution algorithm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    population_size: int = Field(default=512, ge=1)
    num_steps: int = Field(default=50, ge=1)
    param_dim: int = Field(ge=1)
    sigma_m: float = Field(default=1.0, ge=0, le=1)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    fitness: FitnessConfig = Field(default_factory=FitnessConfig)
    distance: DistanceConfig = Field(default_factory=DistanceConfig)
    seed: int | None = None

    @field_validator("sigma_m")
    @classmethod
    def validate_sigma_m(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("sigma_m must be between 0 and 1")
        return v


class DiffusionSettings(BaseSettings):
    """Environment-driven overrides for :class:`DiffusionConfig` values.

    Values are loaded from environment variables prefixed with ``DEVOL_`` and support
    nested fields using ``__`` delimiters (for example ``DEVOL_FITNESS__MAPPING``).
    """

    model_config = SettingsConfigDict(
        env_prefix="DEVOL_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    population_size: int | None = None
    num_steps: int | None = None
    param_dim: int | None = None
    sigma_m: float | None = None
    schedule: ScheduleConfig | dict[str, Any] | None = None
    fitness: FitnessConfig | dict[str, Any] | None = None
    distance: DistanceConfig | dict[str, Any] | None = None
    seed: int | None = None

    def to_config(self, base: DiffusionConfig | None = None, **overrides: Any) -> DiffusionConfig:
        """Merge loaded settings with an optional base config and explicit overrides."""
        merged: dict[str, Any] = {}
        if base is not None:
            merged.update(base.model_dump())

        env_values = self._normalize_nested_models()
        merged.update(env_values)
        merged.update({key: value for key, value in overrides.items() if value is not None})

        if "param_dim" not in merged or merged["param_dim"] is None:
            msg = "param_dim must be provided via configuration, environment, or overrides."
            raise ConfigurationError(msg)

        return DiffusionConfig(**merged)

    def _normalize_nested_models(self) -> dict[str, Any]:
        """Convert loaded values into a dictionary suitable for DiffusionConfig."""
        values = self.model_dump(exclude_none=True)

        normalized: dict[str, Any] = {}
        for key, value in values.items():
            if isinstance(value, BaseModel):
                normalized[key] = value.model_dump()
            elif key in {"schedule", "fitness", "distance"} and isinstance(value, dict):
                normalized[key] = value
            else:
                normalized[key] = value
        return normalized
