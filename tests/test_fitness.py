"""Tests for fitness mapping functions."""

import numpy as np
import pytest

from devol.fitness import DirectMapper, ExponentialMapper, RankMapper, preprocess_fitness


def test_direct_mapper():
    mapper = DirectMapper()
    fitness = np.array([0.1, 0.5, 0.9])
    result = mapper(fitness)
    np.testing.assert_array_equal(result, fitness)


def test_exponential_mapper():
    mapper = ExponentialMapper(temperature=1.0)
    fitness = np.array([1.0, 2.0, 3.0])
    result = mapper(fitness)
    assert np.isclose(np.sum(result), 1.0)
    assert np.all(result > 0)
    assert result[2] > result[1] > result[0]


def test_rank_mapper():
    mapper = RankMapper()
    fitness = np.array([0.1, 0.9, 0.5])
    result = mapper(fitness)
    expected = np.array([1 / 3, 3 / 3, 2 / 3])
    np.testing.assert_array_almost_equal(result, expected)


def test_preprocess_fitness_shift():
    fitness = np.array([-1.0, 0.0, 1.0])
    result = preprocess_fitness(fitness, shift_negative=True, normalize=False)
    expected = np.array([0.0, 1.0, 2.0])
    np.testing.assert_array_equal(result, expected)


def test_preprocess_fitness_normalize():
    fitness = np.array([0.0, 5.0, 10.0])
    result = preprocess_fitness(fitness, shift_negative=False, normalize=True)
    expected = np.array([0.0, 0.5, 1.0])
    np.testing.assert_array_equal(result, expected)
