"""Main training script for MNIST evolution."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic_yaml import parse_yaml_file_as

from devol.algorithm import DiffusionEvolution
from examples.mnist.config import MNISTConfig
from examples.mnist.fitness import MNISTFitnessEvaluator
from examples.mnist.lenet5 import count_parameters, create_lenet5, create_lenet_mini
from examples.mnist.serialization import create_random_individual, serialize_model


class MNISTEvolution(DiffusionEvolution):
    """Diffusion Evolution adapted for MNIST with early stopping."""

    def __init__(self, config: MNISTConfig, evaluator: MNISTFitnessEvaluator):
        self.mnist_config = config
        self.evaluator = evaluator

        def fitness_fn(params: NDArray) -> float:
            images, labels = evaluator.sample_stratified_batch(config.mini_batch_size)
            return evaluator.evaluate_single(params, images, labels)

        super().__init__(config.evolution, fitness_fn)

        self.best_val_acc = 0.0
        self.best_params: NDArray | None = None
        self.patience_counter = 0
        self.generation = 0
        self.fitness_history: list[dict[str, float]] = []

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def evaluate_fitness(self, population: NDArray) -> NDArray:
        """Override to use batched GPU evaluation."""
        return self.evaluator.evaluate_population(population)

    def run(self) -> NDArray:
        """Run evolution with early stopping."""
        population = self.initialize_population()

        for t in range(self.config.num_steps, 0, -1):
            self.generation = self.config.num_steps - t + 1

            if self.mnist_config.verbose:
                print(f"\nGeneration {self.generation}/{self.config.num_steps}")

            population = self.step(t, population)

            fitness = self.evaluate_fitness(population)
            best_idx = np.argmax(fitness)
            current_best_fitness = fitness[best_idx]

            stats = {
                "generation": self.generation,
                "mean_fitness": float(np.mean(fitness)),
                "max_fitness": float(np.max(fitness)),
                "min_fitness": float(np.min(fitness)),
                "std_fitness": float(np.std(fitness)),
            }
            self.fitness_history.append(stats)

            if self.mnist_config.verbose:
                print(
                    f"  Fitness: mean={stats['mean_fitness']:.2f}%, "
                    f"max={stats['max_fitness']:.2f}%, "
                    f"min={stats['min_fitness']:.2f}%"
                )

            if self.generation % self.mnist_config.validation_check_interval == 0:
                full_val_acc = self.evaluator.evaluate_full_validation(population[best_idx])

                if self.mnist_config.verbose:
                    print(f"  Full validation accuracy: {full_val_acc:.2f}%")

                if full_val_acc > self.best_val_acc:
                    self.best_val_acc = full_val_acc
                    self.best_params = population[best_idx].copy()
                    self.patience_counter = 0

                    if self.mnist_config.save_checkpoints:
                        self._save_checkpoint(population[best_idx], full_val_acc)

                    if self.mnist_config.verbose:
                        print(f"  ✓ New best validation accuracy: {full_val_acc:.2f}%")
                else:
                    self.patience_counter += 1
                    if self.mnist_config.verbose:
                        print(f"  No improvement ({self.patience_counter}/{self.mnist_config.early_stopping_patience})")

                if full_val_acc >= self.mnist_config.target_accuracy:
                    print(f"\n✓ Target accuracy {self.mnist_config.target_accuracy}% reached!")
                    break

                if self.patience_counter >= self.mnist_config.early_stopping_patience:
                    print(f"\n✗ Early stopping triggered (patience exhausted)")
                    break

        self.population = population
        return population

    def _save_checkpoint(self, params: NDArray, val_accuracy: float) -> None:
        """Save checkpoint with model parameters and metadata."""
        checkpoint_path = Path(self.mnist_config.checkpoint_dir)
        generation_str = f"gen_{self.generation:04d}"

        model_path = checkpoint_path / f"{generation_str}_model.pt"
        torch.save(
            {"params": params, "state_dict": self._params_to_state_dict(params)},
            model_path,
        )

        model_name = "LeNet-Mini" if self.mnist_config.model_type == "mini" else "LeNet5"
        total_params = 7674 if self.mnist_config.model_type == "mini" else 61706

        metadata = {
            "generation": self.generation,
            "validation_accuracy": float(val_accuracy),
            "best_fitness": float(self.fitness_history[-1]["max_fitness"]),
            "param_dim": int(self.config.param_dim),
            "population_size": int(self.config.population_size),
            "latent_dim": int(self.config.distance.latent_dim),
            "model_type": self.mnist_config.model_type,
            "architecture": {
                "type": model_name,
                "input_channels": 1,
                "num_classes": 10,
                "total_parameters": total_params,
            },
        }

        metadata_path = checkpoint_path / f"{generation_str}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if self.mnist_config.verbose:
            print(f"  Checkpoint saved: {model_path.name}")

    def _params_to_state_dict(self, params: NDArray) -> dict[str, torch.Tensor]:
        """Convert parameter array to PyTorch state dict."""
        from examples.mnist.serialization import deserialize_model

        model = deserialize_model(params, self.mnist_config.device, self.mnist_config.model_type)
        return model.state_dict()

    def save_final_results(self) -> None:
        """Save final results and fitness history."""
        checkpoint_path = Path(self.mnist_config.checkpoint_dir)

        if self.best_params is not None:
            test_acc, _ = self.evaluator.evaluate_test_set(self.best_params)
            print(f"\nFinal test accuracy: {test_acc:.2f}%")

            final_model_path = checkpoint_path / "best_model.pt"
            torch.save(
                {"params": self.best_params, "state_dict": self._params_to_state_dict(self.best_params)},
                final_model_path,
            )

            model_name = "LeNet-Mini" if self.mnist_config.model_type == "mini" else "LeNet5"
            total_params = 7674 if self.mnist_config.model_type == "mini" else 61706

            final_metadata = {
                "best_generation": self.generation,
                "best_validation_accuracy": float(self.best_val_acc),
                "test_accuracy": float(test_acc),
                "total_generations": self.generation,
                "param_dim": int(self.config.param_dim),
                "population_size": int(self.config.population_size),
                "latent_dim": int(self.config.distance.latent_dim),
                "model_type": self.mnist_config.model_type,
                "architecture": {
                    "type": model_name,
                    "input_channels": 1,
                    "num_classes": 10,
                    "total_parameters": total_params,
                },
            }

            with open(checkpoint_path / "best_metadata.json", "w") as f:
                json.dump(final_metadata, f, indent=2)

        history_path = checkpoint_path / "fitness_history.json"
        with open(history_path, "w") as f:
            json.dump(self.fitness_history, f, indent=2)

        print(f"\nResults saved to {checkpoint_path}")


def main() -> None:
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description="Train LeNet on MNIST with Diffusion Evolution")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model-type", type=str, choices=["lenet5", "mini"], default="lenet5")
    parser.add_argument("--population-size", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--latent-dim", type=int, default=50)
    parser.add_argument("--sigma-m", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.config:
        config = parse_yaml_file_as(MNISTConfig, Path(args.config))
    else:
        from examples.mnist.config import create_default_config

        param_dim = 7674 if args.model_type == "mini" else 61706
        latent_dim = args.latent_dim if args.latent_dim != 50 else (30 if args.model_type == "mini" else 50)

        config = create_default_config(
            population_size=args.population_size,
            num_steps=args.num_steps,
            param_dim=param_dim,
            latent_dim=latent_dim,
            sigma_m=args.sigma_m,
            model_type=args.model_type,
            seed=args.seed,
        )

    model_name = "LeNet-Mini" if config.model_type == "mini" else "LeNet5"
    print("=" * 70)
    print(f"MNIST EVOLUTION - {model_name} with Diffusion Evolution")
    print("=" * 70)
    print(f"Model: {model_name} ({config.evolution.param_dim} parameters)")
    print(f"Population size: {config.evolution.population_size}")
    print(f"Evolution steps: {config.evolution.num_steps}")
    print(f"Latent dimension: {config.evolution.distance.latent_dim}")
    print(f"Sigma_m: {config.evolution.sigma_m}")
    print(f"Target accuracy: {config.target_accuracy}%")
    print(f"Device: {config.device}")
    print("=" * 70)

    evaluator = MNISTFitnessEvaluator(
        mini_batch_size=config.mini_batch_size,
        eval_batch_size=config.eval_batch_size,
        device=config.device,
        model_type=config.model_type,
        seed=config.evolution.seed or 42,
    )

    algo = MNISTEvolution(config, evaluator)
    algo.run()
    algo.save_final_results()


if __name__ == "__main__":
    main()
