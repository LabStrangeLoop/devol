"""Fitness-to-probability mapping strategies."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class FitnessMapper(Protocol):
    def __call__(self, fitness: NDArray) -> NDArray:
        """Map fitness values to probability weights."""
        ...


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
    mapping_type: str,
    temperature: float = 1.0,
) -> FitnessMapper:
    mapper: FitnessMapper
    if mapping_type == "exponential":
        mapper = ExponentialMapper(temperature)
    elif mapping_type == "rank":
        mapper = RankMapper()
    else:
        raise ValueError(f"Unknown fitness mapping: {mapping_type}")

    return mapper
