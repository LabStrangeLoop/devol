"""Visualization functions for benchmark results."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from benchmark.metrics import BenchmarkMetrics


def create_visualizations(results: list[BenchmarkMetrics], output_dir: str = ".") -> None:
    """Create all visualization plots.

    Args:
        results: List of benchmark metrics
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    create_heatmaps(results, output_dir)
    create_line_plots(results, output_dir)
    create_boxplots(results, output_dir)
    print(f"\nVisualizations saved to {output_dir}/")


def create_heatmaps(results: list[BenchmarkMetrics], output_dir: str) -> None:
    """Create heatmaps for parameter interactions.

    Args:
        results: List of benchmark metrics
        output_dir: Directory to save plots
    """
    schedules = sorted(set(r.schedule_type for r in results))
    fig, axes = plt.subplots(1, len(schedules), figsize=(6 * len(schedules), 5))
    if len(schedules) == 1:
        axes = [axes]

    # First pass: collect all heatmap data and find global min/max
    all_heatmaps = []
    global_min = float("inf")
    global_max = float("-inf")

    for sched in schedules:
        sched_results = [r for r in results if r.schedule_type == sched]

        pop_sizes = sorted(set(r.population_size for r in sched_results))
        step_counts = sorted(set(r.num_steps for r in sched_results))

        heatmap_data = np.zeros((len(step_counts), len(pop_sizes)))
        counts = np.zeros((len(step_counts), len(pop_sizes)))

        for result in sched_results:
            i = step_counts.index(result.num_steps)
            j = pop_sizes.index(result.population_size)
            heatmap_data[i, j] += result.distance_from_origin
            counts[i, j] += 1

        heatmap_data = np.divide(heatmap_data, counts, where=counts > 0)

        all_heatmaps.append((heatmap_data, pop_sizes, step_counts))
        valid_data = heatmap_data[counts > 0]
        if len(valid_data) > 0:
            global_min = min(global_min, valid_data.min())
            global_max = max(global_max, valid_data.max())

    # Second pass: plot with consistent scale
    for ax, sched, (heatmap_data, pop_sizes, step_counts) in zip(axes, schedules, all_heatmaps):
        im = ax.imshow(heatmap_data, cmap="RdYlGn_r", aspect="auto", vmin=global_min, vmax=global_max)
        ax.set_title(f"{sched} Schedule")
        ax.set_xlabel("Population Size")
        ax.set_ylabel("Num Steps")
        ax.set_xticks(range(len(pop_sizes)))
        ax.set_xticklabels(pop_sizes, rotation=45)
        ax.set_yticks(range(len(step_counts)))
        ax.set_yticklabels(step_counts)
        plt.colorbar(im, ax=ax, label="Avg Distance from Origin")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_pop_vs_steps.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_line_plots(results: list[BenchmarkMetrics], output_dir: str) -> None:
    """Create line plots showing parameter scaling.

    Args:
        results: List of benchmark metrics
        output_dir: Directory to save plots
    """
    params = [
        ("population_size", "Population Size"),
        ("num_steps", "Number of Steps"),
        ("param_dim", "Parameter Dimension"),
        ("sigma_m", "Sigma M"),
    ]

    for param_name, param_label in params:
        fig, ax = plt.subplots(figsize=(10, 6))
        schedules = sorted(set(r.schedule_type for r in results))

        for sched in schedules:
            sched_results = [r for r in results if r.schedule_type == sched]

            param_vals = sorted(set(getattr(r, param_name) for r in sched_results))
            means = []
            stds = []

            for val in param_vals:
                distances = [r.distance_from_origin for r in sched_results if getattr(r, param_name) == val]
                means.append(np.mean(distances))
                stds.append(np.std(distances))

            ax.errorbar(
                param_vals,
                means,
                yerr=stds,
                marker="o",
                label=sched,
                capsize=5,
                linewidth=2,
            )

        ax.set_xlabel(param_label, fontsize=12)
        ax.set_ylabel("Distance from Origin", fontsize=12)
        ax.set_title(f"Performance vs {param_label}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if param_name in ["population_size", "num_steps", "param_dim"]:
            ax.set_xscale("log")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/lineplot_{param_name}.png", dpi=150, bbox_inches="tight")
        plt.close()


def create_boxplots(results: list[BenchmarkMetrics], output_dir: str) -> None:
    """Create boxplots comparing schedule robustness.

    Args:
        results: List of benchmark metrics
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    schedules = sorted(set(r.schedule_type for r in results))
    data = [[r.distance_from_origin for r in results if r.schedule_type == sched] for sched in schedules]

    bp = ax.boxplot(data, labels=schedules, patch_artist=True)

    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")

    ax.set_ylabel("Distance from Origin", fontsize=12)
    ax.set_title("Schedule Robustness Across All Configurations", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/boxplot_schedules.png", dpi=150, bbox_inches="tight")
    plt.close()
