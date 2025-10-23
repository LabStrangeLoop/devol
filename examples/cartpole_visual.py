"""Cart-pole reinforcement learning example with visual demonstrations."""

import numpy as np
from numpy.typing import NDArray

from devol import DiffusionConfig, DiffusionEvolution


def create_neural_network(params: NDArray, input_size: int, hidden_size: int, output_size: int):
    """Create neural network from parameter vector."""
    idx = 0
    w1_size = input_size * hidden_size
    w1 = params[idx : idx + w1_size].reshape(input_size, hidden_size)
    idx += w1_size

    b1 = params[idx : idx + hidden_size]
    idx += hidden_size

    w2_size = hidden_size * output_size
    w2 = params[idx : idx + w2_size].reshape(hidden_size, output_size)
    idx += w2_size

    b2 = params[idx : idx + output_size]

    def forward(x: NDArray) -> NDArray:
        h = np.maximum(0, x @ w1 + b1)
        return h @ w2 + b2

    return forward


def evaluate_cartpole(params: NDArray, num_episodes: int = 3, render: bool = False) -> float:
    """Evaluate neural network controller on CartPole."""
    try:
        import gymnasium as gym
    except ImportError:
        raise ImportError("Install gymnasium: uv pip install gymnasium")

    input_size, hidden_size, output_size = 4, 2, 2
    network = create_neural_network(params, input_size, hidden_size, output_size)

    total_reward = 0.0
    render_mode = "human" if render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)

    for episode in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0.0

        for step in range(500):
            action_values = network(observation)
            action = int(np.argmax(action_values))
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        total_reward += episode_reward
        if render:
            print(f"  Episode {episode + 1}: {episode_reward:.0f} steps")

    env.close()
    return total_reward / num_episodes


def demonstrate_controller(params: NDArray, num_demos: int = 3) -> None:
    """Visually demonstrate a controller."""
    print(f"\nDemonstrating controller for {num_demos} episodes...")
    print("(Close the window to continue)")
    evaluate_cartpole(params, num_episodes=num_demos, render=True)


def demonstrate_evolution_progress(algo: DiffusionEvolution, demo_interval: int = 10) -> None:
    """Demonstrate evolution progress at intervals."""
    try:
        import gymnasium as gym
    except ImportError:
        return

    class DemonstratingEvolution(DiffusionEvolution):
        def __init__(self, *args, demo_interval: int = 10, **kwargs):
            super().__init__(*args, **kwargs)
            self.demo_interval = demo_interval
            self.step_count = 0

        def step(self, t: int, population: NDArray) -> NDArray:
            result = super().step(t, population)
            self.step_count += 1

            if self.step_count % self.demo_interval == 0:
                fitness = self.evaluate_fitness(population)
                best_idx = np.argmax(fitness)
                best_individual = population[best_idx]
                best_fitness = fitness[best_idx]

                print(f"\nGeneration {self.step_count}/{self.config.num_steps}")
                print(f"Best fitness so far: {best_fitness:.2f}")
                print("Demonstrating current best controller...")

                demonstrate_controller(best_individual, num_demos=1)

            return result

    return DemonstratingEvolution


def compare_random_vs_evolved(evolved_params: NDArray) -> None:
    """Compare random controller vs evolved controller side by side."""
    print("\n" + "=" * 60)
    print("COMPARISON: Random Controller vs Evolved Controller")
    print("=" * 60)

    input_size, hidden_size, output_size = 4, 2, 2
    param_dim = input_size * hidden_size + hidden_size + hidden_size * output_size + output_size
    random_params = np.random.randn(param_dim)

    print("\n1. Random Controller Performance:")
    random_fitness = evaluate_cartpole(random_params, num_episodes=5, render=False)
    print(f"   Average reward: {random_fitness:.2f}")

    print("\n2. Evolved Controller Performance:")
    evolved_fitness = evaluate_cartpole(evolved_params, num_episodes=5, render=False)
    print(f"   Average reward: {evolved_fitness:.2f}")

    print(f"\n   Improvement: {evolved_fitness - random_fitness:.2f} steps")
    print(f"   Improvement: {((evolved_fitness / max(random_fitness, 1)) - 1) * 100:.1f}%")

    print("\nDemonstrating Random Controller:")
    demonstrate_controller(random_params, num_demos=2)

    print("\nDemonstrating Evolved Controller:")
    demonstrate_controller(evolved_params, num_demos=2)


def run_cartpole(show_progress: bool = False, compare: bool = True) -> None:
    """Run CartPole evolution with optional visualizations."""
    input_size, hidden_size, output_size = 4, 2, 2
    param_dim = input_size * hidden_size + hidden_size + hidden_size * output_size + output_size

    print(f"CartPole neural network: {param_dim} parameters")
    print(f"Architecture: {input_size} -> {hidden_size} -> {output_size}")

    config = DiffusionConfig(
        population_size=128,
        num_steps=30,
        param_dim=param_dim,
        sigma_m=1.0,
        distance={"type": "cosine"},
        seed=42,
    )

    print(f"\nRunning evolution:")
    print(f"  Population size: {config.population_size}")
    print(f"  Evolution steps: {config.num_steps}")
    #    print(f"  Distance metric: latent space (dim={config.distance.latent_dim})")

    if show_progress:
        print("\nLive demonstrations will be shown every 10 generations")
        EvolutionClass = demonstrate_evolution_progress(None, demo_interval=10)
        algo = EvolutionClass(
            config,
            lambda x: evaluate_cartpole(x, num_episodes=3, render=False),
            demo_interval=10,
        )
    else:
        print("\nRunning evolution (no live demos)...")
        algo = DiffusionEvolution(config, lambda x: evaluate_cartpole(x, num_episodes=3, render=False))

    final_population = algo.run()
    best_individual, best_fitness = algo.get_best_individual()

    print(f"\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Best fitness achieved: {best_fitness:.2f}")

    print("\nEvaluating best controller over 10 test episodes...")
    test_reward = sum(evaluate_cartpole(best_individual, num_episodes=1) for _ in range(10)) / 10
    print(f"Average test reward: {test_reward:.2f}")

    if test_reward >= 475:
        print("✓ Task SOLVED! (reward >= 475)")
    else:
        print(f"✗ Task not solved yet (need >= 475)")

    print("\nDemonstrating best evolved controller:")
    demonstrate_controller(best_individual, num_demos=3)

    if compare:
        compare_random_vs_evolved(best_individual)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CartPole evolution with visualization")
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show live demos during evolution (every 10 generations)",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip comparison with random controller",
    )
    args = parser.parse_args()

    run_cartpole(show_progress=args.show_progress, compare=not args.no_compare)
