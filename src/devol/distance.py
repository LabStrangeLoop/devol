"""Distance computation strategies."""

from typing import Protocol

import numpy as np

from devol.config import DistanceType
from devol.types import FloatArray



class DistanceComputer(Protocol):
    def compute_distances(self, x_i: FloatArray, population: FloatArray) -> FloatArray:
        """Compute distances from x_i to all individuals in population."""
        ...


class EuclideanDistance:
    def compute_distances(self, x_i: FloatArray, population: FloatArray) -> FloatArray:
        diff = population - x_i
        return np.sum(diff * diff, axis=1)


class LatentDistance:
    def __init__(self, param_dim: int, latent_dim: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.projection = rng.normal(0, 1 / np.sqrt(param_dim), (latent_dim, param_dim))

    def compute_distances(self, x_i: FloatArray, population: FloatArray) -> FloatArray:
        z_i = self.projection @ x_i
        z_pop = self.projection @ population.T
        diff = z_pop.T - z_i
        return np.sum(diff * diff, axis=1)


class CosineDistance:
    def compute_distances(self, x_i: FloatArray, population: FloatArray) -> FloatArray:
        norm_i = np.linalg.norm(x_i)
        norms_pop = np.linalg.norm(population, axis=1)

        if norm_i == 0 or np.any(norms_pop == 0):
            return np.ones(len(population))

        similarity = population @ x_i / (norms_pop * norm_i)
        similarity = np.clip(similarity, -1, 1)
        return 1 - similarity


def create_distance_computer(
    distance_type: DistanceType | str,
    param_dim: int,
    latent_dim: int = 2,
    seed: int | None = None,
) -> DistanceComputer:
    if isinstance(distance_type, DistanceType):
        distance_key = distance_type.value
    else:
        distance_key = distance_type

    if distance_key == "euclidean":
        return EuclideanDistance()
    if distance_key == "latent":
        return LatentDistance(param_dim, latent_dim, seed)
    if distance_key == "cosine":
        return CosineDistance()
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")
