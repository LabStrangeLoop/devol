"""GPU-batched fitness evaluation for MNIST."""

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from examples.mnist.serialization import deserialize_model


class MNISTFitnessEvaluator:
    """Evaluates LeNet models on MNIST with GPU batching."""

    def __init__(
        self,
        mini_batch_size: int = 512,
        eval_batch_size: int = 32,
        device: str = "cuda",
        model_type: str = "lenet5",
        seed: int = 42,
    ):
        self.mini_batch_size = mini_batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.model_type = model_type
        self.rng = np.random.default_rng(seed)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
        )
        self.test_dataset = val_dataset

        print(f"Model type: {model_type.upper()}")
        print(
            f"Dataset sizes: train={len(self.train_dataset)}, "
            f"val={len(self.val_dataset)}, test={len(self.test_dataset)}"
        )

    def sample_stratified_batch(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a stratified mini-batch from validation set."""
        indices = self.rng.choice(len(self.val_dataset), size=size, replace=False)
        subset = Subset(self.val_dataset, indices)
        loader = DataLoader(subset, batch_size=size, shuffle=False)
        images, labels = next(iter(loader))
        return images.to(self.device), labels.to(self.device)

    def evaluate_single(self, params: NDArray, images: torch.Tensor, labels: torch.Tensor) -> float:
        """Evaluate single individual on given batch."""
        model = deserialize_model(params, self.device, self.model_type)
        model.eval()

        with torch.no_grad():
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == labels).float().mean().item()

        return accuracy * 100.0

    def evaluate_population(self, population: NDArray) -> NDArray:
        """Evaluate entire population with batched GPU computation."""
        num_individuals = len(population)
        accuracies = np.zeros(num_individuals)

        for i in range(0, num_individuals, self.eval_batch_size):
            batch_end = min(i + self.eval_batch_size, num_individuals)
            batch_size = batch_end - i

            batch_accuracies = []
            for j in range(batch_size):
                images, labels = self.sample_stratified_batch(self.mini_batch_size)
                acc = self.evaluate_single(population[i + j], images, labels)
                batch_accuracies.append(acc)

            accuracies[i:batch_end] = batch_accuracies

        return accuracies

    def evaluate_full_validation(self, params: NDArray) -> float:
        """Evaluate individual on full validation set."""
        model = deserialize_model(params, self.device, self.model_type)
        model.eval()

        loader = DataLoader(self.val_dataset, batch_size=256, shuffle=False)
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return 100.0 * correct / total

    def evaluate_test_set(self, params: NDArray) -> tuple[float, NDArray]:
        """Evaluate on test set and return accuracy + predictions."""
        model = deserialize_model(params, self.device, self.model_type)
        model.eval()

        loader = DataLoader(self.test_dataset, batch_size=256, shuffle=False)
        correct = 0
        total = 0
        all_predictions = []

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                all_predictions.append(predictions.cpu().numpy())

        accuracy = 100.0 * correct / total
        predictions = np.concatenate(all_predictions)
        return accuracy, predictions
