"""Quick test to verify setup."""

from devol import DiffusionConfig, DistanceType, FitnessMapping, ScheduleType
from devol.schedules import create_alpha_schedule, create_sigma_schedule


def test_config():
    config = DiffusionConfig(
        param_dim=10,
        population_size=128,
        num_steps=25,
        distance={"type": DistanceType.LATENT, "latent_dim": 5},
    )
    print(f"✓ Config created: {config.population_size} individuals, {config.num_steps} steps")


def test_schedules():
    alpha = create_alpha_schedule("cosine", 50, 1e-4)
    sigma = create_sigma_schedule(alpha, 1.0)
    print(f"✓ Schedules created: alpha range [{alpha.min():.4f}, {alpha.max():.4f}]")
    print(f"  sigma range [{sigma.min():.4f}, {sigma.max():.4f}]")


if __name__ == "__main__":
    test_config()
    test_schedules()
    print("\n✓ Day 1 setup complete!")
