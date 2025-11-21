"""N-peak optimization smoke test.

Run with: `python tests/ci/n_peaks.py 3 --plot prime_peaks_3.png`
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

try:  # Matplotlib is optional, only needed when plotting.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

from devol import DiffusionConfig, DiffusionEvolution
from devol.config import FitnessConfig, FitnessMapping, NormalType


def create_peak_positions(
    num_peaks: int,
    *,
    bounds: tuple[float, float] = (-1.0, 1.0),
    seed: int | None = 123,
) -> NDArray:
    """Place peaks on a lattice inside the provided bounds.

    Peaks are spread evenly across the square area, using interior lattice points
    (endpoints are omitted so peaks stay visible on plots). When fewer peaks than
    lattice slots are needed, the order is shuffled deterministically with the
    provided seed before truncation.
    """
    if num_peaks < 1:
        raise ValueError("num_peaks must be > 0")

    low, high = bounds

    grid_size = math.ceil(math.sqrt(num_peaks))
    lattice_coords = np.linspace(low, high, grid_size + 2)[1:-1]
    grid_x, grid_y = np.meshgrid(lattice_coords, lattice_coords)
    lattice_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    rng = np.random.default_rng(seed)
    rng.shuffle(lattice_points)

    return lattice_points[:num_peaks]


def make_multi_peak_function(peaks: NDArray, width: float = 0.02):
    """Return a callable fitness function for the provided peak coordinates."""

    def _fitness(x: NDArray) -> float:
        diffs = x - peaks
        dist_sq = np.sum(diffs * diffs, axis=1)
        contributions = np.exp(-dist_sq / width)
        return float(np.mean(contributions))

    return _fitness


def verify_convergence(population: NDArray, peaks: NDArray, tolerance: float) -> list[bool]:
    """Check that each peak has at least one individual within the tolerance."""
    flags: list[bool] = []
    for peak in peaks:
        dists = np.linalg.norm(population - peak, axis=1)
        flags.append(np.min(dists) <= tolerance)
    return flags


def render_population(
    population: NDArray,
    peaks: NDArray,
    fitness_fn,
    out_path: Path | None,
    bounds: tuple[float, float] = (-1.2, 1.2),
) -> None:
    """Save a contour plot of the multi-peak landscape if matplotlib is ready."""
    if out_path is None:
        return

    if plt is None:
        print("Matplotlib not available; skipping visualization.")
        return

    x = np.linspace(bounds[0], bounds[1], 120)
    y = np.linspace(bounds[0], bounds[1], 120)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fitness_fn(np.array([X[i, j], Y[i, j]]))

    plt.figure(figsize=(8, 7))
    plt.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.6)
    plt.colorbar(label="Fitness")
    plt.scatter(population[:, 0], population[:, 1], c="red", s=10, alpha=0.4, label="Population")
    plt.scatter(peaks[:, 0], peaks[:, 1], c="white", s=80, edgecolors="black", label="Peaks")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Multi-peak population snapshot ({len(peaks)} peaks)")
    plt.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {out_path}")


def run_multi_peak(
    num_peaks: int,
    population_size: int = 512,
    num_steps: int = 50,
    convergence_radius: float = 0.25,
    plot_path: Path | None = None,
    peak_seed: int | None = 123,
    fitness_mapping: FitnessMapping = FitnessMapping.DIRECT,
    normalization: NormalType = NormalType.SUM_TO_ONE,
) -> None:
    """Run diffusion evolution and assert convergence for each target peak."""
    peaks = create_peak_positions(num_peaks, seed=peak_seed)
    fitness_fn = make_multi_peak_function(peaks)

    config = DiffusionConfig(
        population_size=population_size,
        num_steps=num_steps,
        param_dim=2,
        sigma_m=1.0,
        seed=42,
        fitness=FitnessConfig(mapping=fitness_mapping, normalize=normalization),
    )

    algo = DiffusionEvolution(config, fitness_fn)
    final_population = algo.run()

    flags = verify_convergence(final_population, peaks, tolerance=convergence_radius)

    for idx, success in enumerate(flags, start=1):
        status = "✅" if success else "❌"
        peak_coords = peaks[idx - 1]
        print(f"{status} Peak {idx}: ({peak_coords[0]:+.3f}, {peak_coords[1]:+.3f})")

    render_population(final_population, peaks, fitness_fn, plot_path)

    if not all(flags):
        missing = [str(i + 1) for i, ok in enumerate(flags) if not ok]
        raise RuntimeError(f"Failed to converge on peaks: {', '.join(missing)}")

    print(f"Successfully converged on all {num_peaks} peaks.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prime the diffusion evolution algorithm on N peaks.")
    parser.add_argument("num_peaks", type=int, help="Number of target peaks (>=1)")
    parser.add_argument("-o", "--plot", type=Path, default=None, help="Optional path to save a contour visualization.")
    parser.add_argument("-p", "--population", type=int, default=512, help="Population size (default: 512)")
    parser.add_argument("-s", "--steps", type=int, default=50, help="Number of denoising steps (default: 50)")
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=0.25,
        help="Distance threshold to count a peak as converged (default: 0.25)",
    )
    parser.add_argument(
        "-k",
        "--peak-seed",
        type=int,
        default=123,
        help="Seed used to shuffle lattice peak positions (default: 123)",
    )
    parser.add_argument(
        "-m",
        "--mapping",
        type=FitnessMapping,
        choices=list(FitnessMapping),
        default=FitnessMapping.DIRECT,
        help="Fitness mapping strategy (default: direct)",
    )
    parser.add_argument(
        "-n",
        "--normalize",
        type=NormalType,
        choices=list(NormalType),
        default=NormalType.SUM_TO_ONE,
        help="Fitness normalization strategy (default: sum_to_one)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run_multi_peak(
            num_peaks=args.num_peaks,
            population_size=args.population,
            num_steps=args.steps,
            convergence_radius=args.radius,
            plot_path=args.plot,
            peak_seed=args.peak_seed,
            fitness_mapping=args.mapping,
            normalization=args.normalize,
        )
    except RuntimeError as exc:  # Ensure CI failure on missed peaks.
        print(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
