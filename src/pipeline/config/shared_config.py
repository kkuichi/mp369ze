"""
Module for shared pipeline configuration components.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

from typing import Literal

from pydantic import BaseModel, field_validator


class BaseConfig(BaseModel):
    """
    Base configuration shared across pipelines.

    This configuration controls global runtime behavior such as
    device selection and random seed initialization.

    Attributes:
        - device (Literal["auto", "cpu", "cuda"]):
            Device to use for computation.
            - "auto": automatically select available device.
            - "cpu": force CPU execution.
            - "cuda": force GPU execution.
        - seed (int): Random seed used to initialize all stochastic components
            for reproducibility.
        - experiment_name (str): Set name for the experiment.
        - test_size (float): Set the size of dataframe to be used as a test set.
    """

    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = 42
    experiment_name: str = "experiment_name"
    test_size: float = 0.2

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        """Validate device."""
        if value not in ["auto", "cpu", "cuda"]:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        return value

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, value: int) -> int:
        """Validate seed."""
        if value < 0:
            raise ValueError("seed must be a non-negative integer")
        return value

    @field_validator("experiment_name")
    @classmethod
    def validate_experiment_name(cls, value: str) -> str:
        """Validate experiment_name"""
        if value is None:
            raise ValueError("experiment_name cann't be None")
        return value

    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, value: float) -> float:
        """Validate test size."""
        if not 0.0 < value < 1.0:
            raise ValueError("test_size must be in the interval (0, 1)")
        return value
