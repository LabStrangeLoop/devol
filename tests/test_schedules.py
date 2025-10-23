"""Tests for schedule implementations."""

import numpy as np
import pytest

from devol.schedules import CosineSchedule, DDPMSchedule, LinearSchedule, create_alpha_schedule


def test_linear_schedule():
    schedule = LinearSchedule()
    assert schedule(0, 10) == 1.0
    assert schedule(5, 10) == 0.5
    assert schedule(10, 10) == 0.0


def test_cosine_schedule():
    schedule = CosineSchedule()
    assert np.isclose(schedule(0, 10), 1.0)
    assert np.isclose(schedule(10, 10), 0.0)
    assert 0 < schedule(5, 10) < 1


def test_ddpm_schedule():
    schedule = DDPMSchedule(epsilon=1e-4)
    assert np.isclose(schedule(0, 10), 1.0 - 1e-4, atol=1e-3)
    assert schedule(10, 10) < 0.1


def test_create_alpha_schedule():
    alpha = create_alpha_schedule("cosine", 50, 1e-4)
    assert len(alpha) == 51
    assert np.isclose(alpha[0], 1.0)
    assert alpha[-1] < 0.1
    assert np.all(np.diff(alpha) <= 0)
