"""Fitness-to-probability mapping strategies."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class FitnessMapper(Protocol):
    def __call__(self, fitness: NDArray) -> NDArray:
        """Map fitness values to probability weights."""
        ...


class DirectMapper:
    def __call__(self, fitness: NDArray) -> NDArray:
        return fitness


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


def preprocess_fitness(
    fitness: NDArray, shift_negative: bool = True, normalize: bool = True
) -> NDArray:
    result = fitness.copy()

    if shift_negative:
        min_val = np.min(result)
        if min_val < 0:
            result = result - min_val

    if normalize:
        max_val = np.max(result)
        if max_val > 0:
            result = result / max_val

    return result


def create_fitness_mapper(
    mapping_type: str, temperature: float = 1.0, normalize: bool = True, shift_negative: bool = True
) -> tuple[FitnessMapper, bool, bool]:
    mapper: FitnessMapper
    if mapping_type == "direct":
        mapper = DirectMapper()
    elif mapping_type == "exponential":
        mapper = ExponentialMapper(temperature)
        normalize = False
    elif mapping_type == "rank":
        mapper = RankMapper()
        normalize = False
    else:
        raise ValueError(f"Unknown fitness mapping: {mapping_type}")

    return mapper, normalize, shift_negative
