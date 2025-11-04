"""Grid search runner with multiprocessing support."""

import time
from itertools import product
from multiprocessing import Pool, cpu_count
from typing import Any

import numpy as np
from tqdm import tqdm

from benchmark.metrics import (
    BenchmarkMetrics,
    FitnessFunction,
    calculate_population_diversity,
    evaluate_population_fitness,
)
from devol import DiffusionConfig, DiffusionEvolution


def run_single_experiment(params: dict[str, Any]) -> BenchmarkMetrics:
    """Run a single experiment with given parameters.

    Args:
        params: Dictionary with experiment parameters

    Returns:
        BenchmarkMetrics object with results
    """
    schedule_type = params["schedule_type"]
    population_size = params["population_size"]
    num_steps = params["num_steps"]
    param_dim = params["param_dim"]
    sigma_m = params["sigma_m"]
    seed = params["seed"]
    fitness_fn = params["fitness_fn"]

    config = DiffusionConfig(
        population_size=population_size,
        num_steps=num_steps,
        param_dim=param_dim,
        sigma_m=sigma_m,
        schedule={"type": schedule_type},
        seed=seed,
    )

    start_time = time.time()
    algo = DiffusionEvolution(config, fitness_fn)
    final_pop = algo.run()
    runtime = time.time() - start_time

    best_individual, best_fitness = algo.get_best_individual()
    fitness_values = evaluate_population_fitness(final_pop, fitness_fn)

    final_diversity = calculate_population_diversity(fitness_values)

    return BenchmarkMetrics(
        schedule_type=schedule_type,
        population_size=population_size,
        num_steps=num_steps,
        param_dim=param_dim,
        sigma_m=sigma_m,
        seed=seed,
        best_fitness=float(best_fitness),
        mean_fitness=float(np.mean(fitness_values)),
        std_fitness=float(np.std(fitness_values)),
        distance_from_origin=float(np.linalg.norm(best_individual)),
        final_diversity=final_diversity,
        runtime_seconds=runtime,
    )


class GridSearchRunner:
    """Run grid search experiments with multiprocessing."""

    def __init__(
        self,
        fitness_fn: FitnessFunction,
        schedule_types: list[str],
        population_sizes: list[int],
        num_steps_list: list[int],
        param_dims: list[int],
        sigma_m_values: list[float],
        seeds: list[int],
        n_workers: int | None = None,
    ):
        """Initialize grid search runner.

        Args:
            fitness_fn: Fitness function to optimize
            schedule_types: List of schedule types to test
            population_sizes: List of population sizes
            num_steps_list: List of num_steps values
            param_dims: List of parameter dimensions
            sigma_m_values: List of sigma_m values
            seeds: List of random seeds
            n_workers: Number of parallel workers (default: CPU count)
        """
        self.fitness_fn = fitness_fn
        self.schedule_types = schedule_types
        self.population_sizes = population_sizes
        self.num_steps_list = num_steps_list
        self.param_dims = param_dims
        self.sigma_m_values = sigma_m_values
        self.seeds = seeds
        self.n_workers = n_workers or cpu_count()

    def generate_experiments(self) -> list[dict[str, Any]]:
        """Generate all experiment configurations."""
        configs = []
        for schedule, pop_size, steps, dim, sigma, seed in product(
            self.schedule_types,
            self.population_sizes,
            self.num_steps_list,
            self.param_dims,
            self.sigma_m_values,
            self.seeds,
        ):
            configs.append({
                "schedule_type": schedule,
                "population_size": pop_size,
                "num_steps": steps,
                "param_dim": dim,
                "sigma_m": sigma,
                "seed": seed,
                "fitness_fn": self.fitness_fn,
            })
        return configs

    def run(self, verbose: bool = True) -> list[BenchmarkMetrics]:
        """Run all experiments in parallel.

        Args:
            verbose: Print progress information

        Returns:
            List of BenchmarkMetrics for all experiments
        """
        experiments = self.generate_experiments()
        total = len(experiments)

        if verbose:
            print(f"Running {total} experiments with {self.n_workers} workers...\n")

        with Pool(self.n_workers) as pool:
            if verbose:
                results = list(
                    tqdm(
                        pool.imap(run_single_experiment, experiments),
                        total=total,
                        desc="Experiments",
                        unit="exp",
                    )
                )
            else:
                results = pool.map(run_single_experiment, experiments)

        if verbose:
            print(f"\nCompleted {total} experiments!")

        return results
