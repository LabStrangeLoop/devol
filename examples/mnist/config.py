"""Configuration for MNIST evolution experiments."""

from pydantic import BaseModel, Field, field_validator

from devol.config import DiffusionConfig


class MNISTConfig(BaseModel, frozen=True):
    """Configuration for MNIST evolution experiment."""

    evolution: DiffusionConfig
    mini_batch_size: int = Field(default=512, ge=1)
    eval_batch_size: int = Field(default=32, ge=1)
    early_stopping_patience: int = Field(default=10, ge=1)
    validation_check_interval: int = Field(default=5, ge=1)
    target_accuracy: float = Field(default=99.0, ge=0, le=100)
    device: str = Field(default="cuda")
    save_checkpoints: bool = Field(default=True)
    checkpoint_dir: str = Field(default="./mnist_checkpoints")
    verbose: bool = Field(default=True)
    model_type: str = Field(default="lenet5")

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        if v not in ["lenet5", "mini"]:
            raise ValueError("model_type must be 'lenet5' or 'mini'")
        return v


def create_default_config(
    population_size: int = 256,
    num_steps: int = 50,
    param_dim: int = 61706,
    latent_dim: int = 50,
    sigma_m: float = 1.0,
    model_type: str = "lenet5",
    seed: int = 42,
) -> MNISTConfig:
    """Create default configuration for MNIST evolution."""
    evolution_config = DiffusionConfig(
        population_size=population_size,
        num_steps=num_steps,
        param_dim=param_dim,
        sigma_m=sigma_m,
        schedule={"type": "cosine", "epsilon": 1e-4},
        fitness={"mapping": "exponential"},
        distance={"type": "latent", "latent_dim": latent_dim},
        seed=seed,
    )

    target_accuracy = 98.0 if model_type == "mini" else 99.0
    checkpoint_dir = f"./mnist_checkpoints_{model_type}"

    return MNISTConfig(
        evolution=evolution_config,
        mini_batch_size=512,
        eval_batch_size=32,
        early_stopping_patience=10,
        validation_check_interval=5,
        target_accuracy=target_accuracy,
        device="cuda",
        save_checkpoints=True,
        checkpoint_dir=checkpoint_dir,
        verbose=True,
        model_type=model_type,
    )
