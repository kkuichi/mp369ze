"""
Module for imputer pipeline configuration classes.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import FieldValidationInfo
from src.pipeline.config.shared_config import BaseConfig
from src.pipeline.constants import ORIGINAL_MEASUREMENT_COLS


class ImputerDataConfig(BaseConfig):
    """
    Config for imputer data.

    Attributes:
        - T_max (int):
        - corrupt_rate (float):
        From BaseConfig inherits:
        - device
        - seed
        - experiment_name
        - test_size
        - event_col
        - group_col
    """

    T_max: int = 50
    corrupt_rate: float = 0.2

    # pylint: disable=invalid-name
    @field_validator("T_max")
    @classmethod
    def validate_T_max(cls, v: int) -> int:
        """Validate T_max"""
        if v <= 0:
            raise ValueError("T_max must be positive")
        return v

    @field_validator("corrupt_rate")
    @classmethod
    def validate_corrupt_rate(cls, v: float) -> float:
        """Validate corrupt_rate"""
        if not 0.0 < v < 1.0:
            raise ValueError("corrupt_rate must be in (0, 1)")
        return v


# Trainer config
class ImputerTrainerConfig(BaseConfig):
    """
    Class for imputer trainer configuration.

    Attributes:
        - features (list[str]): Features - test measurements.
        - epochs (int): Number of training epochs.
        - patience (int): Number of epochs to wait for improvement before stopping.
        - batch_size (int): Batch size.
        - lr (float): Learning rate.
        - weight_decay (float): Weight decay.
        - n_trials (int): Number of trials for hyperparameter search.
        - batch_size_choices (list[int]): list of batch sizes to choose from.
        - hidden_size_choices (list[int]): list of hidden sizes to choose from.
        - dropout_min (float): Minimum dropout rate.
        - dropout_max (float): Maximum dropout rate.
        - lr_min (float): Minimum learning rate.
        - lr_max (float): Maximum learning rate.
        - weight_decay_min (float): Minimum weight decay.
        - weight_decay_max (float): Maximum weight decay.
        From BaseConfig inherits:
        - device
        - seed
        - experiment_name
        - test_size
    """

    # more data related, but... why not
    features: list[str] = ORIGINAL_MEASUREMENT_COLS
    # training loop
    epochs: int = 30
    patience: int = 5
    batch_size: int = 32
    shuffle: bool = True
    lr: float = 1e-3
    weight_decay: float = 0.05
    # Optuna
    n_trials: int = 5
    # Optuna search space
    batch_size_choices: list[int] = [32, 64]
    hidden_size_choices: list[int] = [32, 64]
    dropout_min: float = 0.0
    dropout_max: float = 0.3
    lr_min: float = 1e-4
    lr_max: float = 1e-2
    weight_decay_min: float = 1e-4
    weight_decay_max: float = 1e-2

    base: BaseConfig = Field(default_factory=BaseConfig)

    @field_validator("epochs")
    @classmethod
    def validate_epochs(cls, v: int) -> int:
        """Validate epochs."""
        if v < 0:
            raise ValueError("epochs must be positive")
        return v

    @field_validator("patience")
    @classmethod
    def validate_patience(cls, v: int, info: FieldValidationInfo) -> int:
        """Validate patience."""
        epochs = info.data.get("epochs", 10)
        if v >= epochs:
            raise ValueError("patience must be less than epochs")
        if v < 0:
            raise ValueError("patience must be positive")
        return v

    @field_validator("n_trials")
    @classmethod
    def validate_n_trials(cls, v: int) -> int:
        """Validate n_trials."""
        if v < 0:
            raise ValueError("n_trials must be positive")
        return v

    @field_validator("batch_size_choices")
    @classmethod
    def validate_batch_size_choices(cls, v: list[int]) -> list[int]:
        """Validate batch_size_choices."""
        if not all(x > 0 for x in v):
            raise ValueError("batch_size_choices must be positive")
        return v

    @field_validator("hidden_size_choices")
    @classmethod
    def validate_hidden_size_choices(cls, v: list[int]) -> list[int]:
        """Validate hidden_size_choices."""
        if not all(x > 0 for x in v):
            raise ValueError("hidden_size_choices must be positive")
        return v

    @field_validator(
        "dropout_min",
        "dropout_max",
        "lr_min",
        "lr_max",
        "weight_decay_min",
        "weight_decay_max",
        mode="after",
    )
    @classmethod
    def validate_bounds(cls, v: float, info: FieldValidationInfo) -> float:
        """Validate bounds."""
        if v < 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v


# Model config
class ImputerModelConfig(BaseModel):
    """
    Class for imputer model configuration.

    Attributes:
        - hidden_size (int): Size of the hidden layers. Must be between 1 and 1024.
          Determines the capacity of the model.
        - n_layers (int): Number of layers in the model. Must be at least 1.
          Determines the depth of the model.
        - dropout (float): Dropout rate. Must be 0.0 if n_layers is 1.
          Helps prevent overfitting by randomly setting a fraction of input units
          to 0 during training.
    """

    hidden_size: int = 64
    n_layers: int = 1
    dropout: float = 0.0

    @field_validator("hidden_size")
    @classmethod
    def validate_hidden_size(cls, v, min_v: int = 1, max_v: int = 1024) -> int:
        """Validate hidden_size."""
        if v < min_v or v > max_v:
            raise ValueError(f"hidden_size must be {min_v}-{max_v}")
        return v

    @field_validator("n_layers")
    @classmethod
    def validate_n_layers(cls, v, min_v: int = 1) -> int:
        """Validate n_layers."""
        if v < min_v:
            raise ValueError(f"n_layers must be >= {min_v}")
        return v

    @field_validator("dropout")
    @classmethod
    def validate_dropout(cls, v: float, info: FieldValidationInfo) -> float:
        """Validate dropout."""
        n_layers = info.data.get("n_layers", 1)
        if n_layers == 1 and v != 0.0:
            raise ValueError("dropout must be 0.0 if n_layers == 1")
        return v


# Pipeline config
class ImputerConfig(BaseModel):
    """
    Class for the overall imputer pipeline configuration.

    Attributes:
        - base (BaseConfig): Base configuration.
        - data (ImputerDataConfig): Data configuration.
        - model (ImputerModelConfig): Model configuration.
        - train (ImputerTrainerConfig): Trainer configuration.
    """

    base: BaseConfig = Field(default_factory=BaseConfig)
    data: ImputerDataConfig = Field(default_factory=ImputerDataConfig)
    model: ImputerModelConfig = Field(default_factory=ImputerModelConfig)
    train: ImputerTrainerConfig = Field(default_factory=ImputerTrainerConfig)
