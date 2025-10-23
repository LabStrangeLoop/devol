"""Configuration models for Diffusion Evolution."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


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


class ScheduleConfig(BaseModel, frozen=True):
    type: ScheduleType = ScheduleType.COSINE
    epsilon: float = Field(default=1e-4, gt=0, lt=1)


class FitnessConfig(BaseModel, frozen=True):
    mapping: FitnessMapping = FitnessMapping.DIRECT
    temperature: float = Field(default=1.0, gt=0)
    normalize: bool = True
    shift_negative: bool = True


class DistanceConfig(BaseModel, frozen=True):
    type: DistanceType = DistanceType.EUCLIDEAN
    latent_dim: int = Field(default=2, ge=1)


class DiffusionConfig(BaseModel, frozen=True):
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
