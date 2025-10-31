"""Tests for configuration models and settings."""

import pytest

from devol.config import DiffusionConfig, DiffusionSettings, ScheduleType
from devol.exceptions import ConfigurationError


def test_diffusion_config_sigma_bounds() -> None:
    """sigma_m must be within [0, 1]."""
    with pytest.raises(ValueError):
        DiffusionConfig(param_dim=4, sigma_m=1.5)


def test_diffusion_settings_to_config_merges_overrides() -> None:
    """Environment settings merge with overrides to produce a full config."""
    settings = DiffusionSettings(
        population_size=64,
        schedule={"type": "linear", "epsilon": 1e-5},
    )

    config = settings.to_config(param_dim=3, num_steps=10, seed=123)
    assert config.param_dim == 3
    assert config.population_size == 64
    assert config.schedule.type is ScheduleType.LINEAR
    assert config.schedule.epsilon == pytest.approx(1e-5)
    assert config.num_steps == 10
    assert config.seed == 123


def test_diffusion_settings_requires_param_dim() -> None:
    """The derived configuration must include param_dim."""
    settings = DiffusionSettings(population_size=16)

    with pytest.raises(ConfigurationError):
        settings.to_config()
