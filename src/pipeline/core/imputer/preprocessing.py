"""
Module for preprocessing longitudinal patient data for GRU-D imputer.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

import numpy as np
import pandas as pd
from src.pipeline.config.imputer_config import ImputerDataConfig
from src.pipeline.constants import (  # pylint: disable=import-error
    ADMISSION_COL,
    GROUP_COL,
    TIME_COL,
)


# pylint: disable=invalid-name
class DataPreprocessor:
    """
    Prepare longitudinal patient data for imputation models.
    Converts long-format data into dense tensors suitable for GRU-D models.

    Attributes:
        features (list[str]): List of feature column names to process.
        T_max (int): Maximum number of time steps to consider.
        feat_means (np.ndarray): Feature-wise means for normalization.
        feat_stds (np.ndarray): Feature-wise standard deviations for normalization.
    """

    def __init__(self, config: ImputerDataConfig, features: list[str]) -> None:
        """
        Initialize the data preprocessor.

        Args:
            config (ImputerDataConfig): Configuration for data preprocessing.
            features (list[str]): Feature column names.
        """
        self.config = config
        self.features = features
        self.feat_means: np.ndarray | None = None
        self.feat_stds: np.ndarray | None = None

    # pylint: disable=too-many-locals
    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
        """
        Convert a long-format dataframe into padded tensors.

        Args:
            df (pd.DataFrame): Input long-format dataframe.

        Returns:
            tuple containing:
                - X_raw: Raw values with NaNs
                - X_norm: Normalized values with NaNs replaced by zeros
                - M: Observation mask
                - D: Delta-time tensor
                - Tgrid: Time grid tensor
                - patient_ids: list of patient IDs
        """
        df = df.copy()
        df[TIME_COL] = (
            pd.to_datetime(df[TIME_COL]) - pd.to_datetime(df[ADMISSION_COL])
        ).dt.days.astype(int)

        patient_ids = sorted(df[GROUP_COL].unique())
        n_patients = len(patient_ids)
        n_features = len(self.features)

        X = np.full((n_patients, self.config.T_max, n_features), np.nan, dtype=float)
        M = np.zeros_like(X, dtype=float)
        Tgrid = np.zeros((n_patients, self.config.T_max), dtype=float)

        for i, pid in enumerate(patient_ids):
            sub = df[df[GROUP_COL] == pid].sort_values(TIME_COL).reset_index(drop=True)
            length = min(len(sub), self.config.T_max)
            for t in range(length):
                Tgrid[i, t] = sub.loc[t, TIME_COL]
                for f_idx, feature in enumerate(self.features):
                    value = sub.loc[t, feature]
                    if pd.notna(value):
                        X[i, t, f_idx] = float(value)
                        M[i, t, f_idx] = 1.0

        D = np.zeros_like(X)
        for i in range(n_patients):
            for f in range(n_features):
                last_time = None
                for t in range(self.config.T_max):
                    if Tgrid[i, t] == 0 and t > 0:
                        continue
                    if M[i, t, f] == 1:
                        D[i, t, f] = 0.0 if last_time is None else Tgrid[i, t] - last_time
                        last_time = Tgrid[i, t]
                    else:
                        D[i, t, f] = 0.0 if last_time is None else Tgrid[i, t] - last_time

        self.feat_means = np.nanmean(X, axis=(0, 1))
        self.feat_stds = np.nanstd(X, axis=(0, 1))
        self.feat_stds[self.feat_stds == 0] = 1.0

        X_norm = (X - self.feat_means) / self.feat_stds
        X_norm = np.nan_to_num(X_norm, nan=0.0)

        return X, X_norm, M, D, Tgrid, patient_ids

    def denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Convert normalized predictions back to original scale.

        Args:
            X_norm (np.ndarray): Normalized predictions.

        Returns:
            Denormalized predictions.
        """
        if self.feat_means is None or self.feat_stds is None:
            raise RuntimeError("Preprocessor must be fitted before denormalization.")

        return X_norm * self.feat_stds + self.feat_means

    def denormalize_positive(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Convert normalized predictions back to original scale and ensure non-negative values.

        This method applies a Softplus activation after denormalization to ensure
        all imputed values are strictly positive. The Softplus function smoothly
        approximates ReLU and is differentiable, allowing the model to learn
        to produce values that remain positive after denormalization.

        Args:
            X_norm (np.ndarray): Normalized predictions.

        Returns:
            Denormalized predictions with non-negative constraint applied.
        """
        if self.feat_means is None or self.feat_stds is None:
            raise RuntimeError("Preprocessor must be fitted before denormalization.")

        # Denormalize first
        X_raw = X_norm * self.feat_stds + self.feat_means

        # Apply smooth non-negative constraint using Softplus
        # Softplus(x) = log(1 + exp(x))
        # Clipping prevents numerical overflow for large values
        eps = 1e-6
        return np.log(1 + np.exp(np.clip(X_raw, -20, 20))) + eps

    def transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
        """
        Transform new data using already fitted preprocessing parameters.
        """
        if self.feat_means is None or self.feat_stds is None:
            raise RuntimeError("DataPreprocessor must be fitted before transform().")

        df = df.copy()
        df[TIME_COL] = (
            pd.to_datetime(df[TIME_COL]) - pd.to_datetime(df[ADMISSION_COL])
        ).dt.days.astype(int)

        patient_ids = sorted(df[GROUP_COL].unique())
        n_patients = len(patient_ids)
        n_features = len(self.features)

        X = np.full((n_patients, self.config.T_max, n_features), np.nan, dtype=float)
        M = np.zeros_like(X, dtype=float)
        Tgrid = np.zeros((n_patients, self.config.T_max), dtype=float)

        for i, pid in enumerate(patient_ids):
            sub = df[df[GROUP_COL] == pid].sort_values(TIME_COL).reset_index(drop=True)
            length = min(len(sub), self.config.T_max)
            for t in range(length):
                Tgrid[i, t] = sub.loc[t, TIME_COL]
                for f_idx, feature in enumerate(self.features):
                    value = sub.loc[t, feature]
                    if pd.notna(value):
                        X[i, t, f_idx] = float(value)
                        M[i, t, f_idx] = 1.0

        D = np.zeros_like(X)
        for i in range(n_patients):
            for f in range(n_features):
                last_time = None
                for t in range(self.config.T_max):
                    if Tgrid[i, t] == 0 and t > 0:
                        continue
                    if M[i, t, f] == 1:
                        D[i, t, f] = 0.0 if last_time is None else Tgrid[i, t] - last_time
                        last_time = Tgrid[i, t]
                    else:
                        D[i, t, f] = 0.0 if last_time is None else Tgrid[i, t] - last_time

        X_norm = (X - self.feat_means) / self.feat_stds
        X_norm = np.nan_to_num(X_norm, nan=0.0)

        return X, X_norm, M, D, Tgrid, patient_ids
