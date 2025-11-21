import os

import pytest

from tests.ci.n_peaks import FitnessMapping, NormalType, run_multi_peak


@pytest.mark.parametrize("num_peaks", [2, 3, 5, 7, 11])
def test_multi_peak_convergence(num_peaks: int) -> None:
    os.environ.setdefault("DEVOL_DEBUG_FITNESS", "0")
    run_multi_peak(
        num_peaks=num_peaks,
        population_size=512,
        num_steps=50,
        convergence_radius=0.1,
        peak_seed=123,
        fitness_mapping=FitnessMapping.DIRECT,
        normalization=NormalType.MIN_MAX,
    )
