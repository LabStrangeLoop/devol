"""Grid search runner with multiprocessing support."""

import time
from dataclasses import dataclass
from itertools import product
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from benchmark.metrics import (
    BenchmarkMetrics,
    FitnessFunction,
    calculate_population_diversity,
    evaluate_population_fitness,
)
from devol import DiffusionConfig, DiffusionEvolution
from devol.config import ScheduleConfig, ScheduleType


@dataclass
class ExperimentConfig:
    """Configuration for a single benchmark experiment."""

    config: DiffusionConfig
    fitness_fn: FitnessFunction


def run_single_experiment(experiment: ExperimentConfig) -> BenchmarkMetrics:
    """Run a single experiment with given configuration.

    Args:
        experiment: ExperimentConfig with DiffusionConfig and fitness function

    Returns:
        BenchmarkMetrics object with results
    """
    config = experiment.config
    fitness_fn = experiment.fitness_fn

    start_time = time.time()
    algo = DiffusionEvolution(config, fitness_fn)
    final_pop = algo.run(initial_population=None)
    runtime = time.time() - start_time

    best_individual, best_fitness = algo.get_best_individual()
    fitness_values = evaluate_population_fitness(final_pop, fitness_fn)
    final_diversity = calculate_population_diversity(fitness_values)

    return BenchmarkMetrics(
        schedule_type=config.schedule.type.value,
        population_size=config.population_size,
        num_steps=config.num_steps,
        param_dim=config.param_dim,
        sigma_m=config.sigma_m,
        seed=config.seed,
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
        schedule_types: list[ScheduleType | str],
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
        self.schedule_types = [ScheduleType(s) if isinstance(s, str) else s for s in schedule_types]
        self.population_sizes = population_sizes
        self.num_steps_list = num_steps_list
        self.param_dims = param_dims
        self.sigma_m_values = sigma_m_values
        self.seeds = seeds
        self.n_workers = n_workers or cpu_count()

    def generate_experiments(self) -> list[ExperimentConfig]:
        """Generate all experiment configurations."""
        experiments = []
        for schedule_type, pop_size, steps, dim, sigma, seed in product(
            self.schedule_types,
            self.population_sizes,
            self.num_steps_list,
            self.param_dims,
            self.sigma_m_values,
            self.seeds,
        ):
            config = DiffusionConfig(
                population_size=pop_size,
                num_steps=steps,
                param_dim=dim,
                sigma_m=sigma,
                schedule=ScheduleConfig(type=schedule_type),
                seed=seed,
            )
            experiments.append(ExperimentConfig(config=config, fitness_fn=self.fitness_fn))
        return experiments

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
