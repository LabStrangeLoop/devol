"""Parameter serialization and deserialization for LeNet models."""

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from examples.mnist.lenet5 import LeNet5, LeNetMini, create_lenet5, create_lenet_mini


def serialize_model(model: nn.Module) -> NDArray:
    """Flatten all model parameters into 1D numpy array."""
    params = []
    for param in model.parameters():
        params.append(param.detach().cpu().numpy().flatten())
    return np.concatenate(params)


def deserialize_model(params: NDArray, device: str = "cuda", model_type: str = "lenet5") -> nn.Module:
    """Reconstruct LeNet model from 1D parameter array."""
    if model_type == "mini":
        model = LeNetMini().to(device)
    else:
        model = LeNet5().to(device)

    idx = 0
    for param in model.parameters():
        param_size = param.numel()
        param_shape = param.shape

        param_values = params[idx : idx + param_size]
        param.data = torch.from_numpy(param_values.reshape(param_shape)).float().to(device)

        idx += param_size

    return model


def create_random_individual(param_dim: int, device: str = "cuda", model_type: str = "lenet5") -> NDArray:
    """Create a randomly initialized individual."""
    if model_type == "mini":
        model = create_lenet_mini(device)
    else:
        model = create_lenet5(device)
    return serialize_model(model)


def deserialize_population(population: NDArray, device: str = "cuda", model_type: str = "lenet5") -> list[nn.Module]:
    """Deserialize entire population into list of models."""
    return [deserialize_model(params, device, model_type) for params in population]
