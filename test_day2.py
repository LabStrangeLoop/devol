"""Test Day 2 implementations."""

import numpy as np

from devol.distance import create_distance_computer
from devol.fitness import create_fitness_mapper, preprocess_fitness


def test_fitness_pipeline():
    fitness = np.array([-1.0, 0.5, 2.0, 1.5])
    preprocessed = preprocess_fitness(fitness, shift_negative=True, normalize=True)
    print(f"✓ Preprocessed fitness: {preprocessed}")

    mapper, _, _ = create_fitness_mapper("exponential", temperature=1.0)
    weights = mapper(preprocessed)
    print(f"✓ Exponential weights (sum={weights.sum():.4f}): {weights}")

    rank_mapper, _, _ = create_fitness_mapper("rank")
    ranks = rank_mapper(preprocessed)
    print(f"✓ Rank weights: {ranks}")


def test_distance_pipeline():
    population = np.random.randn(10, 5)
    x_i = population[0]

    euclidean = create_distance_computer("euclidean", param_dim=5)
    distances = euclidean.compute_distances(x_i, population)
    print(f"✓ Euclidean distances (first 3): {distances[:3]}")

    latent = create_distance_computer("latent", param_dim=5, latent_dim=2, seed=42)
    distances_latent = latent.compute_distances(x_i, population)
    print(f"✓ Latent distances (first 3): {distances_latent[:3]}")

    cosine = create_distance_computer("cosine", param_dim=5)
    distances_cosine = cosine.compute_distances(x_i, population)
    print(f"✓ Cosine distances (first 3): {distances_cosine[:3]}")


if __name__ == "__main__":
    test_fitness_pipeline()
    print()
    test_distance_pipeline()
    print("\n✓ Day 2 complete!")
