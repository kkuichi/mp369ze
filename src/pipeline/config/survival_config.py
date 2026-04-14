"""
Module for survival pipeline configuration classes.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core.core_schema import FieldValidationInfo
from src.pipeline.config.shared_config import BaseConfig


class SurvivalTrainConfig(BaseConfig):
    """
    Class for survival training.

    Attributes:
        - n_splits (int): Number of times to repeat cross-validation.
        - shuffle (bool): Whether to shuffle the data before splitting.
        - n_trials (int): Number of trials for hyperparameter optimization.
        - direction (str): Direction for optimization. In survival analysis, we
            typically want to maximize the C-index, so "maximize"
            is the common choice.
    """

    n_splits: int = 3
    shuffle: bool = True
    n_trials: int = 30
    direction: str = "maximize"

    @field_validator("n_splits")
    @classmethod
    def validate_n_splits(cls, value: int) -> int:
        """Validate number of splits."""
        if value < 1:
            raise ValueError("n_splits must be >= 1")
        return value

    @field_validator("n_trials")
    @classmethod
    def validate_n_trials(cls, n_trials: int, min_v: int = 1) -> int:
        """
        Validate number of OPTUNA trails"""
        if n_trials < min_v:
            raise ValueError(f"Number of OPTUNA trails must be greater than {min_v}")
        return n_trials


class SurvivalModelConfig(BaseModel):
    """
    Class for survival model configuration.

    Attributes:
        - model_name (Literal["xgb_cox", "xgb_aft", "rsf"]):
            Selection of the survival model architecture.
        - verbosity (int): Controls the amount of logging output during training.

    XGBoost parameters (Common for Cox and AFT):
        - num_boost_round (int): Maximum number of boosting iterations (trees).
            Controls model capacity.
        - early_stopping_rounds (int): Patience for early stopping to prevent
            overfitting when validation loss plateaus.
        - eta_min/max (float): Range for the learning rate.
            Smaller values make learning more robust but slower.
        - max_depth_min/max (int): Range for tree depth. Deeper trees capture
            complex interactions but risk overfitting.
        - subsample_min/max (float): Fraction of patients sampled per tree.
            Prevents overfitting to specific samples.
        - colsample_bytree_min/max (float): Fraction of features sampled per tree.
            Useful for high-dimensional clinical data.
        - lambda_min/max (float): L2 regularization term. Penalizes large weights
            to handle correlated lab features.
        - alpha_min/max (float): L1 regularization term. Encourages sparsity,
            acting as an implicit feature selector.
        - gamma_min/max (float): Minimum loss reduction required for a split.
            Higher values lead to more conservative models.
        - min_child_weight_min/max (int): Minimum sum of instance weights in a leaf.
            Ensures statistical reliability of nodes.
    XGBoost AFT specific:
        - aft_loss_distribution (Literal["normal", "logistic", "extreme"]):
            The assumed probability distribution of survival times.

    Random Survival Forest (RSF) parameters:
        - n_estimators_min/max (int): Number of trees in the forest.
            More trees improve stability and accuracy.
        - min_samples_split_min/max (int): Minimum samples required to
            split a node. Prevents trees from learning noise.
        - min_samples_leaf_min/max (int): Minimum samples required in a leaf.
            Smoothes survival estimates.
        - max_features_choices (list): List of strategies (e.g., "sqrt", "log2")
            to select features for each split.
    """

    model_name: Literal["xgb_cox", "xgb_aft", "rsf"] = "xgb_aft"
    verbosity: int = 0
    # XGB_AFT specific
    aft_loss_distribution: Literal["normal", "logistic", "extreme"] = "normal"
    # XGB params
    num_boost_round: int = 600
    early_stopping_rounds: int = 250
    # Optuna searchspace
    eta_min: float = 0.005
    eta_max: float = 0.3
    max_depth_min: int = 2
    max_depth_max: int = 6
    subsample_min: float = 0.5
    subsample_max: float = 1.0
    colsample_bytree_min: float = 0.5
    colsample_bytree_max: float = 1.0
    lambda_min: float = 0.01
    lambda_max: float = 10.0
    alpha_min: float = 0.01
    alpha_max: float = 10.0
    gamma_min: float = 0.0
    gamma_max: float = 5.0
    min_child_weight_min: int = 1
    min_child_weight_max: int = 10

    # RSF params - Optuna searchspace
    n_estimators_min: int = 100
    n_estimators_max: int = 1000
    min_samples_split_min: int = 2
    min_samples_split_max: int = 20
    min_samples_leaf_min: int = 2
    min_samples_leaf_max: int = 20
    max_features_min: Literal["auto", "sqrt", "log2"] = "sqrt"
    max_features_choices: list[str | float | int] = ["sqrt", "log2"]

    base: BaseConfig = Field(default_factory=BaseConfig)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, name: str) -> str:
        """Validate model name."""
        valid_names = ["xgb_cox", "xgb_aft", "rsf"]
        if name not in valid_names:
            raise ValueError(f"Model name must be one of {valid_names}")
        return name

    @field_validator(
        "n_estimators_min",
        "n_estimators_max",
        "max_depth_min",
        "max_depth_max",
        "min_samples_leaf_min",
        "min_samples_leaf_max",
        mode="after",
    )
    @classmethod
    def validate_positive_ints(cls, v: int, info: FieldValidationInfo) -> int:
        """Validate positive integers."""
        if v < 1:
            raise ValueError(f"{info.field_name} must be >= 1")
        return v

    @field_validator("min_samples_split_min", "min_samples_split_max", mode="after")
    @classmethod
    def validate_min_samples_split(cls, v: int, info: FieldValidationInfo) -> int:
        """Validate min_samples_split."""
        if v < 2:
            raise ValueError(f"{info.field_name} must be >= 2")
        return v

    @field_validator(
        "subsample_min",
        "subsample_max",
        "colsample_bytree_min",
        "colsample_bytree_max",
        mode="after",
    )
    @classmethod
    def validate_probabilities(cls, v: float, info: FieldValidationInfo) -> float:
        """Validate probabilities (0, 1]."""
        if not 0 < v <= 1.0:
            raise ValueError(f"{info.field_name} must be in (0, 1]")
        return v

    @field_validator(
        "eta_min",
        "eta_max",
        "lambda_min",
        "lambda_max",
        "alpha_min",
        "alpha_max",
        mode="after",
    )
    @classmethod
    def validate_positive_floats(cls, v: float, info: FieldValidationInfo) -> float:
        """Validate positive floats."""
        if v <= 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return v

    @field_validator("max_features_choices", mode="after")
    @classmethod
    def validate_max_features_choices(cls, v: list) -> list:
        """Validate max_features choices."""
        valid_strings = ["auto", "sqrt", "log2"]
        for item in v:
            if isinstance(item, str):
                if item not in valid_strings:
                    raise ValueError(
                        f"Invalid string choice in max_features_choices: {item}."
                        f"Must be one of {valid_strings}"
                    )
            elif isinstance(item, float):
                if not 0 < item <= 1.0:
                    raise ValueError(
                        f"Float choice in max_features_choices must be in (0, 1]: {item}"
                    )
            elif isinstance(item, int):
                if item < 1:
                    raise ValueError(f"Int choice in max_features_choices must be >= 1: {item}")
        return v

    @field_validator("num_boost_round")
    @classmethod
    def validate_num_boost_round(cls, rounds: int, min_v: int = 10, max_v: int = 1_000_000) -> int:
        """Validate number of boosting rounds."""
        if (rounds < min_v) or (rounds > max_v):
            raise ValueError(
                f"Number of boosting rounds is not within valid range: {min_v}-{max_v}"
            )
        return rounds

    @field_validator("early_stopping_rounds")
    @classmethod
    def validate_early_stopping_rounds(
        cls,
        rounds: int,
        info: FieldValidationInfo,
        min_v: int = 1,
        max_v: int = 100_000,
    ) -> int:
        """Validate number of early stopping rounds."""
        num_boost_round = info.data.get("num_boost_round")

        if not isinstance(num_boost_round, int):
            raise ValueError("num_boost_round must be a integer.")

        if rounds > num_boost_round:
            raise ValueError(
                "Number of early stopping rounds is greater than numbrt of boost rounds"
            )
        if (rounds < min_v) or (rounds > max_v):
            raise ValueError(
                f"Number of early stopping rounds is not within valid range: {min_v}-{max_v}"
            )
        return rounds

    @model_validator(mode="after")
    def validate_ranges(self):
        """Validate that min values are less than max values."""
        ranges = [
            # XGB_COX
            ("eta_min", "eta_max"),
            ("max_depth_min", "max_depth_max"),
            ("subsample_min", "subsample_max"),
            ("colsample_bytree_min", "colsample_bytree_max"),
            ("lambda_min", "lambda_max"),
            ("alpha_min", "alpha_max"),
            # RSF
            ("n_estimators_min", "n_estimators_max"),
            ("min_samples_split_min", "min_samples_split_max"),
            ("min_samples_leaf_min", "min_samples_leaf_max"),
        ]

        for low, high in ranges:
            val_low = getattr(self, low)
            val_high = getattr(self, high)
            if val_low >= val_high:
                raise ValueError(f"{low} ({val_low}) must be < {high} ({val_high})")
        return self


class SurvivalConfig(BaseModel):
    """
    Class for survival pipeline configuration.

    Attributes:
        - base (BaseConfig): Base configuration
        - data (SurvivalDataConfig): Data configuration
        - train (SurvivalTrainConfig): Training configuration
        - model (SurvivalModelConfig): Model configuration
    """

    base: BaseConfig = Field(default_factory=BaseConfig)
    train: SurvivalTrainConfig = Field(default_factory=SurvivalTrainConfig)
    model: SurvivalModelConfig = Field(default_factory=SurvivalModelConfig)
