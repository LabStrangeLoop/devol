"""Tests for fitness mapping functions."""

import numpy as np

from devol.fitness import ExponentialMapper, RankMapper


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
