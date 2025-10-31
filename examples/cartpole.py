"""Cart-pole reinforcement learning example using gymnasium."""

import numpy as np

from devol import DiffusionConfig, DiffusionEvolution
from devol.types import FloatArray


def create_neural_network(params: FloatArray, input_size: int, hidden_size: int, output_size: int):
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

    def forward(x: FloatArray) -> FloatArray:
        h = np.maximum(0, x @ w1 + b1)
        return h @ w2 + b2

    return forward


def evaluate_cartpole(params: FloatArray, num_episodes: int = 3) -> float:
    """Evaluate neural network controller on CartPole."""
    try:
        import gymnasium as gym
    except ImportError:
        raise ImportError("Install gymnasium: uv pip install gymnasium")

    input_size, hidden_size, output_size = 4, 8, 2
    network = create_neural_network(params, input_size, hidden_size, output_size)

    total_reward = 0.0
    env = gym.make("CartPole-v1")

    for _ in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0.0

        for _ in range(500):
            action_values = network(observation)
            action = int(np.argmax(action_values))
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        total_reward += episode_reward

    env.close()
    return total_reward / num_episodes


def run_cartpole() -> None:
    input_size, hidden_size, output_size = 4, 8, 2
    param_dim = input_size * hidden_size + hidden_size + hidden_size * output_size + output_size

    print(f"CartPole neural network: {param_dim} parameters")

    config = DiffusionConfig(
        population_size=128,
        num_steps=25,
        param_dim=param_dim,
        sigma_m=1.0,
        distance={"type": "latent", "latent_dim": 5},
        seed=42,
    )

    print(f"Running evolution with {config.population_size} individuals...")

    algo = DiffusionEvolution(config, evaluate_cartpole)
    final_population = algo.run()

    best_individual, best_fitness = algo.get_best_individual()

    print(f"\nBest fitness: {best_fitness:.2f}")
    print("Testing best controller over 10 episodes...")

    test_reward = sum(evaluate_cartpole(best_individual, num_episodes=1) for _ in range(10)) / 10
    print(f"Average test reward: {test_reward:.2f}")

    if test_reward >= 475:
        print("✓ Task solved! (reward >= 475)")
    else:
        print(f"Task not yet solved. Try more steps or larger population.")


if __name__ == "__main__":
    run_cartpole()
