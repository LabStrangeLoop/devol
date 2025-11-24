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


def average_min_peak_distance(population: NDArray, peaks: NDArray) -> float:
    """Average distance from each individual to its closest peak."""
    distances = np.linalg.norm(population[:, None, :] - peaks[None, :, :], axis=2)
    nearest = np.min(distances, axis=1)
    return float(np.mean(nearest))


def nearest_peak_stats(population: NDArray, peaks: NDArray) -> tuple[NDArray, NDArray]:
    """Return nearest distances and peak indices for each individual."""
    distances = np.linalg.norm(population[:, None, :] - peaks[None, :, :], axis=2)
    nearest_idx = np.argmin(distances, axis=1)
    nearest_dist = distances[np.arange(len(population)), nearest_idx]
    return nearest_dist, nearest_idx


def chi_square_p_value(statistic: float, dof: int) -> float:
    """Right-tail p-value for chi-square using normal approximation (mean=k, var=2k)."""
    if dof <= 0:
        return 1.0
    mean = dof
    std = math.sqrt(2 * dof)
    z = (statistic - mean) / std
    return 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def estimate_expected_peak_probs(peaks: NDArray, *, sample_size: int = 50_000, seed: int | None = 123) -> NDArray:
    """Estimate nearest-peak probabilities for an initial N(0, I) population via Monte Carlo."""
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal((sample_size, peaks.shape[1]))
    distances = np.linalg.norm(samples[:, None, :] - peaks[None, :, :], axis=2)
    nearest_idx = np.argmin(distances, axis=1)
    counts = np.bincount(nearest_idx, minlength=len(peaks))
    probs = counts / np.sum(counts)

    # Avoid zero probability due to sampling noise; renormalize after floor.
    probs = np.maximum(probs, 1e-12)
    probs /= np.sum(probs)
    return probs


def estimate_expected_peak_probs_conditioned(
    peaks: NDArray,
    threshold: float,
    *,
    sample_size: int = 50_000,
    seed: int | None = 123,
) -> tuple[NDArray, int]:
    """Estimate nearest-peak probabilities conditioned on being within the threshold of a peak."""
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal((sample_size, peaks.shape[1]))
    distances = np.linalg.norm(samples[:, None, :] - peaks[None, :, :], axis=2)
    nearest_idx = np.argmin(distances, axis=1)
    nearest_dist = distances[np.arange(len(samples)), nearest_idx]
    mask = nearest_dist <= threshold
    kept_idx = nearest_idx[mask]
    kept_total = len(kept_idx)

    if kept_total == 0:
        raise RuntimeError(
            "Fairness baseline failed: zero Monte Carlo samples fell within the assignment threshold. "
            "Increase --fair-samples or loosen the threshold."
        )

    counts = np.bincount(kept_idx, minlength=len(peaks))
    probs = counts / kept_total
    probs = np.maximum(probs, 1e-12)
    probs /= np.sum(probs)
    return probs, kept_total


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
    min_improvement_ratio: float = 2.0,
    temperature: float = 0.25,
    fairness_alpha: float = 0.05,
    min_expected_per_peak: int = 5,
    fairness_sample_size: int = 50_000,
) -> None:
    """Run diffusion evolution and assert convergence for each target peak."""
    peaks = create_peak_positions(num_peaks, seed=peak_seed)
    fitness_fn = make_multi_peak_function(peaks)

    config = DiffusionConfig(
        population_size=population_size,
        num_steps=num_steps,
        param_dim=2,
        sigma_m=1.0,
        seed=333,
        fitness=FitnessConfig(
            mapping=fitness_mapping,
            normalize=normalization,
            temperature=temperature,
        ),
    )

    algo = DiffusionEvolution(config, fitness_fn)
    initial_population = algo.initialize_population()
    initial_avg_distance = average_min_peak_distance(initial_population, peaks)

    final_population = algo.run(initial_population)
    final_distances, nearest_peaks = nearest_peak_stats(final_population, peaks)
    final_avg_distance = float(np.mean(final_distances))
    final_std_distance = float(np.std(final_distances))
    improvement_ratio = math.inf if final_avg_distance == 0 else initial_avg_distance / final_avg_distance

    flags = verify_convergence(final_population, peaks, tolerance=convergence_radius)

    for idx, success in enumerate(flags, start=1):
        status = "✅" if success else "❌"
        peak_coords = peaks[idx - 1]
        print(f"{status} Peak {idx}: ({peak_coords[0]:+.3f}, {peak_coords[1]:+.3f})")

    print(
        "Average nearest-peak distance: "
        f"start {initial_avg_distance:.3f} -> end {final_avg_distance:.3f} (std {final_std_distance:.3f})"
    )
    print(f"Improvement ratio (start/end): {improvement_ratio:.2f}x")

    render_population(final_population, peaks, fitness_fn, plot_path)

    missing_peaks = [str(i + 1) for i, ok in enumerate(flags) if not ok]
    if missing_peaks:
        print(f"Note: peaks lacking neighbors within radius {convergence_radius}: {', '.join(missing_peaks)}")

    if improvement_ratio < min_improvement_ratio:
        raise RuntimeError(
            f"Insufficient improvement: {improvement_ratio:.2f}x (<{min_improvement_ratio:.2f}x target) "
            f"[start {initial_avg_distance:.3f}, end {final_avg_distance:.3f}]"
        )

    assignment_threshold = final_avg_distance + final_std_distance
    assigned_mask = final_distances <= assignment_threshold
    assigned_indices = nearest_peaks[assigned_mask]
    assigned_counts = np.bincount(assigned_indices, minlength=num_peaks)
    assigned_total = int(np.sum(assigned_counts))

    expected_probs, baseline_total = estimate_expected_peak_probs_conditioned(
        peaks,
        assignment_threshold,
        sample_size=fairness_sample_size,
        seed=peak_seed,
    )
    expected_counts = expected_probs * assigned_total

    if np.any(expected_counts < min_expected_per_peak):
        smallest = float(np.min(expected_counts))
        raise RuntimeError(
            "Population too small for fair-spread check with biased expectations: "
            f"smallest expected count {smallest:.2f} (<{min_expected_per_peak}). "
            "Increase population size or lower --min-expected."
        )

    chi_square_stat = float(np.sum((assigned_counts - expected_counts) ** 2 / expected_counts))
    chi_dof = num_peaks - 1
    chi_p_value = chi_square_p_value(chi_square_stat, chi_dof)

    print(
        f"Fairness check: threshold {assignment_threshold:.3f}, assigned {assigned_total} individuals; "
        f"expected probs (conditioned) {np.round(expected_probs, 4).tolist()} from {baseline_total} baseline samples; "
        f"counts per peak {assigned_counts.tolist()}, chi2={chi_square_stat:.3f}, dof={chi_dof}, p={chi_p_value:.3f}"
    )

    if chi_p_value < fairness_alpha:
        raise RuntimeError(
            f"Unbalanced allocation across peaks (p={chi_p_value:.3f} < {fairness_alpha:.3f}); "
            "population collapsed unevenly relative to expected bias."
        )

    print(
        f"Achieved {improvement_ratio:.2f}x improvement on average nearest-peak distance "
        f"(target: {min_improvement_ratio:.2f}x) and passed fairness check (p={chi_p_value:.3f})."
    )


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
        "-t",
        "--temperature",
        type=float,
        default=0.25,
        help="Temperature to use in fitness mapping (default: 0.25)",
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
    parser.add_argument(
        "-i",
        "--improvement",
        type=float,
        default=2.0,
        help="Required improvement ratio of average nearest-peak distance (start/end). Default: 2.0x",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for chi-square fairness test (default: 0.05).",
    )
    parser.add_argument(
        "--min-expected",
        type=int,
        default=5,
        help="Minimum expected count per peak to run chi-square test (default: 5).",
    )
    parser.add_argument(
        "--fair-samples",
        type=int,
        default=50_000,
        help="Sample size for Monte Carlo estimation of biased peak probabilities (default: 50k).",
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
            min_improvement_ratio=args.improvement,
            temperature=args.temperature,
            fairness_alpha=args.alpha,
            min_expected_per_peak=args.min_expected,
            fairness_sample_size=args.fair_samples,
        )
    except RuntimeError as exc:  # Ensure CI failure on missed peaks.
        print(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
