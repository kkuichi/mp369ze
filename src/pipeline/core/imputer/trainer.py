"""
Module for training the GRU-D imputer.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

# pylint: disable=import-error
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals

from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from src.pipeline.config.imputer_config import (
    ImputerDataConfig,
    ImputerModelConfig,
    ImputerTrainerConfig,
)
from src.pipeline.constants import GROUP_COL, TIME_COL
from src.pipeline.core.imputer.datasets import LongitudinalDataset
from src.pipeline.core.imputer.models import GRUDImputer
from src.pipeline.core.imputer.preprocessing import DataPreprocessor
from torch import nn
from torch.utils.data import DataLoader


class ImputerTrainer:
    """
    Advanced trainer for GRU-D imputer with optimized mapping and Optuna integration.
    """

    def __init__(
        self,
        data_config: ImputerDataConfig,
        trainer_config: ImputerTrainerConfig,
        model_config: ImputerModelConfig,
        model: nn.Module | None = None,
        preprocessor: DataPreprocessor | None = None,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            data_config (ImputerDataConfig): Imputer data configuration.
            trainer_config (ImputerTrainerConfig): Imputer trainer configuration.
            model_config (ImputerModelConfig): Imputer model configuration.
            model (nn.Module): Pre-initialized model. Defaults to None.
            preprocessor (DataPreprocessor): Pre-initialized preprocessor. Defaults to None.
        """
        self.data_config = data_config
        self.trainer_config = trainer_config
        self.model_config = model_config

        device_setting = trainer_config.base.device
        if device_setting == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device_setting

        self.device = torch.device(resolved_device)

        self.model = model.to(self.device) if model else None
        self.preprocessor = preprocessor or DataPreprocessor(
            config=self.data_config, features=self.trainer_config.features
        )
        self.loss_fn = nn.L1Loss(reduction="sum")

    def _create_optimizer(
        self, model: nn.Module, lr: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        """
        Create a new optimizer instance for a specific model.

        Args:
            model (nn.Module): The model instance.
            lr (float): Learning rate.
            weight_decay (float): Weight decay (L2).

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(
        self, loader: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Train the model for one epoch.

        Args:
            loader (DataLoader): Training data loader.
            model (nn.Module): Model to train.
            optimizer (torch.optim.Optimizer): Optimizer to use.

        Returns:
            float: Average loss per imputed element.
        """
        model.train()
        total_loss: float = 0.0
        total_count: int = 0

        for Xb, Mb, Db, target_mask, Xtrue in loader:
            Xb, Mb, Db = Xb.to(self.device), Mb.to(self.device), Db.to(self.device)
            target_mask, Xtrue = target_mask.to(self.device), Xtrue.to(self.device)

            preds = model(Xb, Mb, Db)
            if target_mask.sum() == 0:
                continue

            # Calculate loss only on artificially masked values (Self-Supervision)
            loss = self.loss_fn(preds[target_mask], Xtrue[target_mask])
            scaled_loss = loss / target_mask.sum().clamp(min=1)

            optimizer.zero_grad()
            scaled_loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_count += target_mask.sum().item()

        return total_loss / max(1, total_count)

    def _prepare_data(self, df: pd.DataFrame) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[int],
        LongitudinalDataset,
    ]:
        """
        Preprocess DataFrame and prepare self-supervised training artifacts.

        Args:
            df (pd.DataFrame): Input long-format or pivoted DataFrame.

        Returns:
            tuple: Contains (X_raw, X_norm, M, D, corrupt_mask, patients, dataset).
        """
        # fit_transform handles normalization and conversion to 3D tensors (N, T, D)
        X_raw, X_norm, M, D, _, patients = self.preprocessor.fit_transform(df)

        # Create artificial missingness (corruption) for the model to learn to fill
        corrupt_mask = (np.random.rand(*M.shape) < self.data_config.corrupt_rate) & (M == 1)

        # M_train hides values that are present in X_norm but marked in corrupt_mask
        M_train = M.copy()
        M_train[corrupt_mask] = 0

        dataset = LongitudinalDataset(
            X=X_norm, M=M_train, D=D, X_true_norm=X_norm, target_mask=corrupt_mask
        )

        return X_raw, X_norm, M, D, corrupt_mask, patients, dataset

    def _optuna_objective(
        self,
        trial: optuna.Trial,
        X_norm: np.ndarray,
        M: np.ndarray,
        D: np.ndarray,
        corrupt_mask: np.ndarray,
        dataset: LongitudinalDataset,
    ) -> float:
        """
        Optuna objective for hyperparameter tuning.

        Args:
            trial (optuna.Trial): Optuna trial.
            X_norm (np.ndarray): Normalized input data
            M (np.ndarray): Observation mask.
            D (np.ndarray): Delta-times.
            corrupt_mask (np.ndarray): Artificially introduced missingness mask.
            dataset (LongitudinalDataset): Dataset for training.

        Returns:
            float: Validation MAE loss.
        """
        # Suggest hyperparameters
        hp = {
            "batch_size": trial.suggest_categorical(
                "batch_size", self.trainer_config.batch_size_choices
            ),
            "hidden_size": trial.suggest_categorical(
                "hidden_size", self.trainer_config.hidden_size_choices
            ),
            "dropout": trial.suggest_float(
                "dropout",
                self.trainer_config.dropout_min,
                self.trainer_config.dropout_max,
            ),
            "lr": trial.suggest_float(
                "lr", self.trainer_config.lr_min, self.trainer_config.lr_max, log=True
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay",
                self.trainer_config.weight_decay_min,
                self.trainer_config.weight_decay_max,
                log=True,
            ),
        }

        trial_model_config = self.model_config.model_copy(
            update={"hidden_size": hp["hidden_size"], "dropout": hp["dropout"]}
        )

        # Initialize temporary model and optimizer
        trial_model = GRUDImputer(config=trial_model_config, input_size=X_norm.shape[2]).to(
            self.device
        )

        optimizer = self._create_optimizer(trial_model, hp["lr"], hp["weight_decay"])
        loader = DataLoader(
            dataset, batch_size=hp["batch_size"], shuffle=self.trainer_config.shuffle
        )

        # Training loop for HPO
        for _ in range(self.trainer_config.epochs):
            self.train_epoch(loader, trial_model, optimizer)

        # Validation on hidden (corrupted) values
        trial_model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
            Mt = torch.tensor(M, dtype=torch.float32, device=self.device)
            Dt = torch.tensor(D, dtype=torch.float32, device=self.device)
            mask = torch.tensor(corrupt_mask, device=self.device)

            preds = trial_model(Xt, Mt, Dt)
            val_loss = torch.abs(preds[mask] - Xt[mask]).mean().item()

        return val_loss

    def fit(self, df_train: pd.DataFrame, study_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Full pipeline: Optuna HPO + final training + imputation.

        Args:
            df_train (pd.DataFrame): Training data.
            study_name (str): Unique name for the study.

        Returns:
            tuple: (Imputed DataFrame, Meta-information).
        """
        X_raw, X_norm, M, D, corrupt_mask, patients, dataset = self._prepare_data(df_train)

        # Hyperparameter Optimization
        study = optuna.create_study(
            direction="minimize", study_name=study_name, load_if_exists=True
        )
        study.optimize(
            lambda t: self._optuna_objective(t, X_norm, M, D, corrupt_mask, dataset),
            n_trials=self.trainer_config.n_trials,
        )

        bp = study.best_params

        # Final Training with best params
        final_model_config = self.model_config.model_copy(
            update={"hidden_size": bp["hidden_size"], "dropout": bp["dropout"]}
        )
        self.model = GRUDImputer(config=final_model_config, input_size=X_norm.shape[2]).to(
            self.device
        )

        optimizer = self._create_optimizer(self.model, bp["lr"], bp["weight_decay"])
        loader = DataLoader(dataset, batch_size=bp["batch_size"], shuffle=True)

        for _ in range(self.trainer_config.epochs):
            self.train_epoch(loader, self.model, optimizer)

        # Final Imputation of training data
        self.model.eval()
        with torch.no_grad():
            Xt, Mt, Dt = [
                torch.tensor(t, dtype=torch.float32, device=self.device) for t in [X_norm, M, D]
            ]
            X_filled = self.preprocessor.denormalize_positive(self.model(Xt, Mt, Dt).cpu().numpy())

        X_final = X_raw.copy()
        X_final[M == 0] = X_filled[M == 0]

        # Efficiently map back to DF
        df_imputed = self._map_back(df_train, X_final, patients)

        metadata = {
            "best_params": bp,
            "train_loss": self.train_epoch(loader, self.model, optimizer),
            "best_val_loss": study.best_value,
            "seed": self.trainer_config.seed,
        }
        return df_imputed, metadata

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute new data using the fitted model.

        Args:
            df (pd.DataFrame): Data to impute.

        Returns:
            pd.DataFrame: Imputed data.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling transform.")

        X_raw, X_norm, M, D, _, patients = self.preprocessor.transform(df)

        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
            Mt = torch.tensor(M, dtype=torch.float32, device=self.device)
            Dt = torch.tensor(D, dtype=torch.float32, device=self.device)
            X_filled_norm = self.model(Xt, Mt, Dt).cpu().numpy()

        X_filled = self.preprocessor.denormalize_positive(X_filled_norm)
        X_final = X_raw.copy()
        X_final[M == 0] = X_filled[M == 0]

        return self._map_back(df, X_final, patients)

    def _map_back(
        self, df_original: pd.DataFrame, X_filled: np.ndarray, patients: list[int]
    ) -> pd.DataFrame:
        """
        Efficiently map 3D imputed tensors back to a long-format DataFrame.

        Args:
            df_original (pd.DataFrame): Original DataFrame to use as a template.
            X_filled (np.ndarray): Imputed 3D tensor (N, T, D).
            patients (list[int]): List of patient IDs corresponding to N.

        Returns:
            pd.DataFrame: Completed DataFrame.
        """
        df = df_original.copy().sort_values([GROUP_COL, TIME_COL])
        patient_groups = df.groupby(GROUP_COL).groups

        for i, pid in enumerate(patients):
            if pid in patient_groups:
                idx = patient_groups[pid]
                actual_len = len(idx)
                df.loc[idx, self.trainer_config.features] = X_filled[i, :actual_len, :]

        return df
