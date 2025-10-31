"""Fitness-to-probability mapping strategies."""

from typing import Protocol

import numpy as np

from devol.config import FitnessMapping
from devol.types import FloatArray


class FitnessMapper(Protocol):
    def __call__(self, fitness: FloatArray) -> FloatArray:
        """Map fitness values to probability weights."""
        ...


class DirectMapper:
    def __call__(self, fitness: FloatArray) -> FloatArray:
        return fitness


class ExponentialMapper:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def __call__(self, fitness: FloatArray) -> FloatArray:
        scaled = fitness / self.temperature
        exp_fitness = np.exp(scaled - np.max(scaled))
        return exp_fitness / np.sum(exp_fitness)


class RankMapper:
    def __call__(self, fitness: FloatArray) -> FloatArray:
        ranks = np.argsort(np.argsort(fitness)) + 1
        return ranks / len(ranks)


def preprocess_fitness(
    fitness: FloatArray, shift_negative: bool = True, normalize: bool = True
) -> FloatArray:
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
    mapping_type: FitnessMapping | str,
    temperature: float = 1.0,
    normalize: bool = True,
    shift_negative: bool = True,
) -> tuple[FitnessMapper, bool, bool]:
    mapper: FitnessMapper

    if isinstance(mapping_type, FitnessMapping):
        mapping_key = mapping_type.value
    else:
        mapping_key = mapping_type

    if mapping_key == "direct":
        mapper = DirectMapper()
    elif mapping_key == "exponential":
        mapper = ExponentialMapper(temperature)
        normalize = False
    elif mapping_key == "rank":
        mapper = RankMapper()
        normalize = False
    else:
        raise ValueError(f"Unknown fitness mapping: {mapping_type}")

    return mapper, normalize, shift_negative
