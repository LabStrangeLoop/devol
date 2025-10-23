"""Compare different alpha schedules and their impact."""

import numpy as np

from devol import DiffusionConfig, DiffusionEvolution
from devol.schedules import create_alpha_schedule


def sphere_function(x: np.ndarray) -> float:
    return -np.sum(x**2)


def visualize_schedules():
    """Show how different schedules look."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to see visualization: uv pip install matplotlib")
        return

    steps = 50
    linear = create_alpha_schedule("linear", steps, 1e-4)
    cosine = create_alpha_schedule("cosine", steps, 1e-4)
    ddpm = create_alpha_schedule("ddpm", steps, 1e-4)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(linear, label="Linear", linewidth=2)
    plt.plot(cosine, label="Cosine (Accelerated)", linewidth=2)
    plt.plot(ddpm, label="DDPM", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("α_t")
    plt.title("Alpha Schedules Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    variance_linear = 1 - linear
    variance_cosine = 1 - cosine
    variance_ddpm = 1 - ddpm
    plt.plot(variance_linear, label="Linear", linewidth=2)
    plt.plot(variance_cosine, label="Cosine (Accelerated)", linewidth=2)
    plt.plot(variance_ddpm, label="DDPM", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Variance (1 - α_t)")
    plt.title("Noise Variance Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("schedule_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved schedule_comparison.png")
    plt.show()


def benchmark_schedules():
    """Compare performance of different schedules."""
    print("Benchmarking schedules on 10D sphere function...\n")

    configs = {
        "Linear": DiffusionConfig(
            population_size=256,
            num_steps=100,
            param_dim=10,
            sigma_m=1.0,
            schedule={"type": "linear"},
            seed=42,
        ),
        "Cosine (Accelerated)": DiffusionConfig(
            population_size=256,
            num_steps=100,
            param_dim=10,
            sigma_m=1.0,
            schedule={"type": "cosine"},
            seed=42,
        ),
        "DDPM": DiffusionConfig(
            population_size=256,
            num_steps=100,
            param_dim=10,
            sigma_m=1.0,
            schedule={"type": "ddpm"},
            seed=42,
        ),
    }

    results = {}

    for name, config in configs.items():
        print(f"Testing {name} schedule...")
        algo = DiffusionEvolution(config, sphere_function)
        final_pop = algo.run()
        best_individual, best_fitness = algo.get_best_individual()

        fitness_values = np.array([sphere_function(ind) for ind in final_pop])
        results[name] = {
            "best": best_fitness,
            "mean": np.mean(fitness_values),
            "std": np.std(fitness_values),
            "distance_from_origin": np.linalg.norm(best_individual),
        }

        print(f"  Best fitness: {best_fitness:.6f}")
        print(f"  Mean fitness: {results[name]['mean']:.6f}")
        print(f"  Distance from origin: {results[name]['distance_from_origin']:.6f}\n")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best_schedule = min(results.items(), key=lambda x: x[1]["distance_from_origin"])
    print(f"Best schedule: {best_schedule[0]}")
    print(f"  Achieved distance: {best_schedule[1]['distance_from_origin']:.6f}")
    print("\nNote: With only 25 steps, Cosine schedule typically performs best")
    print("      due to better exploration-exploitation balance.")


if __name__ == "__main__":
    visualize_schedules()
    print("\n")
    benchmark_schedules()
