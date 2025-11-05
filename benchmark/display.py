"""Rich terminal display for benchmark results."""

import numpy as np
from rich.console import Console
from rich.table import Table

from benchmark.metrics import BenchmarkMetrics


def display_results(results: list[BenchmarkMetrics]) -> None:
    """Display benchmark results in rich tables.

    Args:
        results: List of benchmark metrics
    """
    console = Console()

    # Overall summary by schedule type
    console.print("\n[bold cyan]Overall Performance by Schedule[/bold cyan]\n")
    display_schedule_summary(results, console)

    # Best configurations per schedule
    console.print("\n[bold cyan]Best Configurations per Schedule[/bold cyan]\n")
    display_best_configs(results, console)

    # Parameter sensitivity analysis
    console.print("\n[bold cyan]Parameter Sensitivity[/bold cyan]\n")
    display_parameter_sensitivity(results, console)


def display_schedule_summary(
    results: list[BenchmarkMetrics], console: Console
) -> None:
    """Display summary statistics for each schedule type."""
    table = Table(title="Schedule Performance Summary")

    table.add_column("Schedule", style="cyan", no_wrap=True)
    table.add_column("Avg Distance", justify="right", style="green")
    table.add_column("Best Distance", justify="right", style="green")
    table.add_column("Avg Runtime (s)", justify="right", style="magenta")
    table.add_column("Runs", justify="right")

    schedules = {}
    for result in results:
        sched = result.schedule_type
        if sched not in schedules:
            schedules[sched] = []
        schedules[sched].append(result)

    for sched, sched_results in sorted(schedules.items()):
        distances = [r.distance_from_origin for r in sched_results]
        runtimes = [r.runtime_seconds for r in sched_results]

        table.add_row(
            sched,
            f"{np.mean(distances):.4f} ± {np.std(distances):.4f}",
            f"{np.min(distances):.4f}",
            f"{np.mean(runtimes):.2f}",
            str(len(sched_results)),
        )

    console.print(table)


def display_best_configs(results: list[BenchmarkMetrics], console: Console) -> None:
    """Display best configuration for each schedule type."""
    table = Table(title="Best Configurations (by distance)")

    table.add_column("Schedule", style="cyan")
    table.add_column("Pop Size", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Param Dim", justify="right")
    table.add_column("Sigma M", justify="right")
    table.add_column("Distance", justify="right", style="green")

    schedules = {}
    for result in results:
        sched = result.schedule_type
        if sched not in schedules or result.distance_from_origin < schedules[sched].distance_from_origin:
            schedules[sched] = result

    for sched, best in sorted(schedules.items()):
        table.add_row(
            sched,
            str(best.population_size),
            str(best.num_steps),
            str(best.param_dim),
            f"{best.sigma_m:.2f}",
            f"{best.distance_from_origin:.4f}",
        )

    console.print(table)


def display_parameter_sensitivity(
    results: list[BenchmarkMetrics], console: Console
) -> None:
    """Display how performance varies with each parameter."""
    params = ["population_size", "num_steps", "param_dim", "sigma_m"]

    for param_name in params:
        table = Table(title=f"Sensitivity to {param_name}")
        table.add_column(param_name.replace("_", " ").title(), justify="right")
        table.add_column("Avg Distance", justify="right", style="green")
        table.add_column("Std Distance", justify="right", style="yellow")
        table.add_column("Count", justify="right")

        param_groups = {}
        for result in results:
            param_val = getattr(result, param_name)
            if param_val not in param_groups:
                param_groups[param_val] = []
            param_groups[param_val].append(result.distance_from_origin)

        for val in sorted(param_groups.keys()):
            distances = param_groups[val]
            table.add_row(
                str(val),
                f"{np.mean(distances):.4f}",
                f"{np.std(distances):.4f}",
                str(len(distances)),
            )

        console.print(table)
        console.print()
