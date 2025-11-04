"""Comprehensive benchmark comparing schedule types across parameter ranges."""

import numpy as np

from benchmark import GridSearchRunner, create_visualizations, display_results


def rasputin(x: np.ndarray) -> float:
    """Rasputin function (aka Rastrigin): highly multimodal test function.

    f(x) = -[10n + Σ(x_i² - 10cos(2πx_i))]

    Characteristics:
    - Many local minima arranged in a regular lattice
    - Global optimum at origin with value 0
    - Tests exploration vs exploitation balance
    - Standard domain: [-5.12, 5.12]^n
    """
    n = len(x)
    return -(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def main():
    """Run comprehensive grid search benchmark."""
    print("=" * 80)
    print("DIFFUSION EVOLUTION: SCHEDULE BENCHMARK")
    print("=" * 80)
    print("\nObjective: Maximize Rasputin function (Rastrigin with many local minima)")
    print("Testing: linear, cosine, and ddpm schedules\n")

    # Grid parameters (coarse search)
    runner = GridSearchRunner(
        fitness_fn=rasputin,
        schedule_types=["linear", "cosine", "ddpm"],
        population_sizes=[64, 128, 256],
        num_steps_list=[64, 128, 256, 512],
        param_dims=[8, 16, 32, 64],
        sigma_m_values=[0.2, 0.5, 0.8, 1.0],
        seeds=[666, 1337],
    )

    # Run experiments
    results = runner.run(verbose=True)

    # Display results in terminal
    display_results(results)

    # Create visualizations
    create_visualizations(results, output_dir="benchmark_results")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
