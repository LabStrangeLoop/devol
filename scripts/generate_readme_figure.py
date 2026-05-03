"""Generate the Rastrigin denoising trajectory figures used in the README.

Produces:
  docs/images/denoising-trajectory.png  – 4-panel static snapshot
  docs/images/denoising-trajectory.gif  – animated version across every step

The script is deterministic (seeded) and depends only on devol + matplotlib.
Rerun with: `uv run scripts/generate_readme_figure.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from devol import DiffusionConfig, DiffusionEvolution
from devol.config import FitnessConfig, FitnessMapping, NormalType
from devol.distance import FloatArray

# --- Configuration knobs ---------------------------------------------------

SEED = 42
POPULATION_SIZE = 1024
NUM_STEPS = 120
PARAM_DIM = 2
SIGMA_M = 0.5

# How far to stretch the initial N(0,1) noise. Pushes the starting population
# close to the plot edges so the "noise → clusters" collapse is visually strong.
INIT_SCALE = 4.6

# Exponential fitness mapping with a moderate temperature keeps enough selection
# pressure to find peaks without collapsing the whole population to the global
# max, so the final population visibly spreads across several Rastrigin peaks.
FITNESS_CONFIG = FitnessConfig(
    mapping=FitnessMapping.EXPONENTIAL,
    temperature=2.0,
    normalize=NormalType.IDENTITY,
)

BOUNDS = (-5.12, 5.12)  # standard Rastrigin search region
GRID_RESOLUTION = 200

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"
STATIC_PATH = OUTPUT_DIR / "denoising-trajectory.png"
GIF_PATH = OUTPUT_DIR / "denoising-trajectory.gif"


def rastrigin(x: FloatArray) -> float:
    """Rastrigin in 2D, converted to a maximization problem.

    Global maximum at the origin; many regular local maxima surround it.
    """
    a = 10.0
    n = x.shape[0]
    return float(-(a * n + np.sum(x**2 - a * np.cos(2 * np.pi * x))))


class RecordingEvolution(DiffusionEvolution):
    """DiffusionEvolution that stores a copy of the population after every step."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.trajectory: list[NDArray[np.float64]] = []

    def step(self, timestamp: int, population: NDArray[np.float64]) -> NDArray[np.float64]:
        new_population = super().step(timestamp, population)
        self.trajectory.append(new_population.copy())
        return new_population


def build_landscape_grid() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate Rastrigin over a regular grid for contour plotting."""
    axis = np.linspace(BOUNDS[0], BOUNDS[1], GRID_RESOLUTION)
    xx, yy = np.meshgrid(axis, axis)
    stacked = np.stack([xx.ravel(), yy.ravel()], axis=1)
    zz = np.array([rastrigin(point) for point in stacked]).reshape(xx.shape)
    return xx, yy, zz


def run_evolution() -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
    """Run the seeded evolution and return the trajectory (initial + every step)."""
    config = DiffusionConfig(
        population_size=POPULATION_SIZE,
        num_steps=NUM_STEPS,
        param_dim=PARAM_DIM,
        sigma_m=SIGMA_M,
        seed=SEED,
        fitness=FITNESS_CONFIG,
    )
    algo = RecordingEvolution(config, rastrigin)

    # Scale initial noise to cover the landscape. devol's default init is N(0,1); we
    # rescale once so the starting cloud fills the Rastrigin bounds for a stronger
    # "noise → structure" visual.
    initial_population = algo.initialize_population() * INIT_SCALE

    algo.run(initial_population)
    trajectory = [initial_population.copy(), *algo.trajectory]
    return trajectory, initial_population


def draw_landscape(ax: Axes, xx: NDArray[np.float64], yy: NDArray[np.float64], zz: NDArray[np.float64]) -> None:
    ax.contourf(xx, yy, zz, levels=30, cmap="Greys_r", alpha=0.55)
    ax.set_xlim(BOUNDS)
    ax.set_ylim(BOUNDS)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


SCATTER_KW: dict[str, Any] = dict(s=18, c="#FF3366", edgecolors="white", linewidths=0.6, alpha=0.95)


def make_static_figure(
    trajectory: list[NDArray[np.float64]],
    xx: NDArray[np.float64],
    yy: NDArray[np.float64],
    zz: NDArray[np.float64],
) -> Figure:
    """Four-panel snapshot showing noise → convergence.

    The interesting part of the denoising happens early (by t ~= T/4 the
    population has collapsed onto the basin), so panels are front-loaded
    rather than evenly spaced.
    """
    num_frames = len(trajectory)
    last = num_frames - 1
    panel_indices = [0, max(1, num_frames // 6), max(1, num_frames // 3), last]
    panel_titles = [
        f"t = {panel_indices[0]} (pure noise)",
        f"t = {panel_indices[1]}",
        f"t = {panel_indices[2]}",
        f"t = {panel_indices[3]} (converged)",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, idx, title in zip(axes, panel_indices, panel_titles):
        draw_landscape(ax, xx, yy, zz)
        population = trajectory[idx]
        ax.scatter(population[:, 0], population[:, 1], **SCATTER_KW)
        ax.set_title(title, fontsize=12, pad=8)

    fig.suptitle(
        "Diffusion Evolution on Rastrigin (2D): noise → convergence",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def make_gif(
    trajectory: list[NDArray[np.float64]],
    xx: NDArray[np.float64],
    yy: NDArray[np.float64],
    zz: NDArray[np.float64],
    out_path: Path,
) -> None:
    """Animate every recorded step."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    draw_landscape(ax, xx, yy, zz)
    scatter = ax.scatter([], [], **SCATTER_KW)
    title = ax.set_title("t = 0", fontsize=12, pad=8)

    def update(frame: int) -> list[Artist]:
        population = trajectory[frame]
        scatter.set_offsets(population)
        title.set_text(f"t = {frame}")
        return [scatter, title]

    anim = FuncAnimation(fig, update, frames=len(trajectory), interval=120, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=12))
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running evolution (seed={SEED}, steps={NUM_STEPS}, population={POPULATION_SIZE})...")
    trajectory, _ = run_evolution()
    print(f"Captured {len(trajectory)} frames.")

    print("Building landscape grid...")
    xx, yy, zz = build_landscape_grid()

    print(f"Writing static figure to {STATIC_PATH}")
    fig = make_static_figure(trajectory, xx, yy, zz)
    fig.savefig(STATIC_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Writing animated figure to {GIF_PATH}")
    make_gif(trajectory, xx, yy, zz, GIF_PATH)

    print("Done.")


if __name__ == "__main__":
    main()
