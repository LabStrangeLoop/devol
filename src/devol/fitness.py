"""Fitness-to-probability mapping strategies."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from devol.config import FitnessMapping, NormalType


class FitnessMapper(Protocol):
    def __call__(self, fitness: NDArray) -> NDArray:
        """Map fitness values to probability weights."""
        ...


class FitnessNormalizer(Protocol):
    def __call__(self, fitness: NDArray) -> NDArray:
        """Normalize fitness values to be within acceptable range or gaussian."""
        ...


class Identity:
    """Identity fitness mapping function."""

    def __init__(self, l2_factor=0.0):
        self.l2_factor = l2_factor

    def l2(self, x):
        return np.linalg.norm(x, axis=-1) ** 2

    def forward(self, x):
        return x

    def __call__(self, fitness: NDArray) -> NDArray:
        return self.forward(fitness) * np.exp(-1.0 * self.l2(fitness) * self.l2_factor)


class DirectMapper:
    def __call__(self, fitness: NDArray) -> NDArray:
        return fitness


class Energy(Identity):
    """Fitness mapping function that treats the fitness as energy.

    Args:
        temperature: float, the temperature of the system.

    Returns:
        p: torch.Tensor, the probability of the fitness. Compute by exp(-x / temperature).
    """

    def __init__(self, temperature=1.0, l2_factor=0.0):
        super().__init__(l2_factor=l2_factor)
        self.temperature = temperature

    def forward(self, x):
        power = -x / self.temperature
        power = power - power.max() + 5  # avoid overflow
        p = np.exp(power)
        return p


class ExponentialMapper:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def __call__(self, fitness: NDArray) -> NDArray:
        scaled = fitness / self.temperature
        exp_fitness = np.exp(scaled - np.max(scaled))
        return exp_fitness / np.sum(exp_fitness)


class RankMapper:
    def __call__(self, fitness: NDArray) -> NDArray:
        ranks = np.argsort(np.argsort(fitness)) + 1
        return ranks / len(ranks)


def create_fitness_mapper(
    mapping_type: FitnessMapping,
    temperature: float = 1.0,
) -> FitnessMapper:
    mapper: FitnessMapper
    if mapping_type == FitnessMapping.EXPONENTIAL:
        mapper = ExponentialMapper(temperature)
    elif mapping_type == FitnessMapping.RANK:
        mapper = RankMapper()
    elif mapping_type == FitnessMapping.DIRECT:
        mapper = DirectMapper()
    elif mapping_type == FitnessMapping.ENERGY:
        return Energy(temperature=temperature)
    elif mapping_type == FitnessMapping.IDENTITY:
        return Identity()
    else:
        raise ValueError(f"Unknown fitness mapping: {mapping_type}")

    return mapper


class MaxScaleNormalizer:
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon

    def __call__(self, fitness: NDArray) -> NDArray:
        max_abs = np.max(np.abs(fitness))
        if max_abs < self.epsilon:
            return np.zeros_like(fitness)
        return fitness / max_abs


class MinMaxNormalizer:
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon

    def __call__(self, fitness: NDArray) -> NDArray:
        min_val = np.min(fitness)
        max_val = np.max(fitness)
        span = max_val - min_val
        if span < self.epsilon:
            return np.zeros_like(fitness)
        return (fitness - min_val) / span


class ZScoreNormalizer:
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon

    def __call__(self, fitness: NDArray) -> NDArray:
        mean = np.mean(fitness)
        std = np.std(fitness)
        if std < self.epsilon:
            return np.zeros_like(fitness)
        return (fitness - mean) / std


class SumToOneNormalizer:
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon

    def __call__(self, fitness: NDArray) -> NDArray:
        total = np.sum(np.abs(fitness))
        if total < self.epsilon:
            return np.zeros_like(fitness)
        return fitness / total


class IdentityNormalizer:
    def __call__(self, fitness: NDArray) -> NDArray:
        return fitness


def create_fitness_normalizer(normalize_type: NormalType = NormalType.MAX_SCALE) -> FitnessNormalizer:
    if normalize_type == NormalType.MAX_SCALE:
        return MaxScaleNormalizer()
    if normalize_type == NormalType.MIN_MAX:
        return MinMaxNormalizer()
    if normalize_type == NormalType.Z_SCORE:
        return ZScoreNormalizer()
    if normalize_type == NormalType.SUM_TO_ONE:
        return SumToOneNormalizer()
    if normalize_type == NormalType.IDENTITY:
        return IdentityNormalizer()

    raise ValueError(f"Unknown normalizer type: {normalize_type}")
