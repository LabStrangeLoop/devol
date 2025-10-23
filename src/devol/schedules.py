"""Alpha and sigma schedule implementations."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Schedule(Protocol):
    def __call__(self, t: int, total_steps: int) -> float:
        """Compute schedule value at timestep t."""
        ...


class LinearSchedule:
    def __call__(self, t: int, total_steps: int) -> float:
        return 1.0 - t / total_steps


class CosineSchedule:
    def __call__(self, t: int, total_steps: int) -> float:
        return 0.5 * np.cos(np.pi * t / total_steps) + 0.5


class DDPMSchedule:
    def __init__(self, epsilon: float = 1e-4):
        self.epsilon = epsilon

    def __call__(self, t: int, total_steps: int) -> float:
        if total_steps == 0:
            return 1.0 - self.epsilon
        beta_0 = -np.log(self.epsilon) / total_steps
        gamma = beta_0
        return float(np.exp(-beta_0 * t - gamma * t * t / total_steps))


def create_alpha_schedule(schedule_type: str, total_steps: int, epsilon: float) -> NDArray:
    schedule: Schedule
    if schedule_type == "linear":
        schedule = LinearSchedule()
    elif schedule_type == "cosine":
        schedule = CosineSchedule()
    elif schedule_type == "ddpm":
        schedule = DDPMSchedule(epsilon)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    return np.array([schedule(t, total_steps) for t in range(total_steps + 1)])


def create_sigma_schedule(alpha: NDArray, sigma_m: float) -> NDArray:
    sigma = np.zeros(len(alpha))
    for t in range(1, len(alpha)):
        if alpha[t] < alpha[t - 1]:
            ratio = (1 - alpha[t - 1]) / (1 - alpha[t])
            alpha_ratio = 1 - alpha[t] / alpha[t - 1]
            sigma[t] = sigma_m * np.sqrt(ratio * alpha_ratio)
    return sigma
