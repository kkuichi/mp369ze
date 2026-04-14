"""
Module for PyTorch Dataset handling longitudinal data.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

from typing import TypeAlias

import numpy as np
import torch
from torch.utils.data import Dataset

ReturnType: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class LongitudinalDataset(Dataset[ReturnType]):
    """
    PyTorch Dataset for longitudinal tensors.

    Args:
        X (np.ndarray): Normalized input tensor of shape (N, T, F).
        M (np.ndarray): Observation mask tensor of shape (N, T, F).
        D (np.ndarray): Delta-time tensor of shape (N, T, F).
        X_true_norm (np.ndarray): Ground-truth normalized tensor of shape (N, T, F).
        target_mask (np.ndarray | None): Boolean mask indicating where loss
            should be computed. Defaults to missing values (M == 0).
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    # pylint: disable=invalid-name
    def __init__(
        self,
        X: np.ndarray,
        M: np.ndarray,
        D: np.ndarray,
        X_true_norm: np.ndarray,
        target_mask: np.ndarray | None = None,
    ) -> None:
        if not X.shape == M.shape == D.shape == X_true_norm.shape:
            raise ValueError("All input arrays must have the same shape")

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.M = torch.as_tensor(M, dtype=torch.float32)
        self.D = torch.as_tensor(D, dtype=torch.float32)
        self.X_true_norm = torch.as_tensor(X_true_norm, dtype=torch.float32)

        if target_mask is not None:
            if target_mask.shape != X.shape:
                raise ValueError("target_mask must have the same shape as X")
            self.target_mask = torch.as_tensor(target_mask, dtype=torch.bool)
        else:
            self.target_mask = self.M == 0

    def __len__(self) -> int:
        """
        Return the number of patients in the dataset.

        Returns:
            Number of samples (patients).
        """
        return self.X.shape[0]

    def __getitem__(self, index: int) -> ReturnType:
        """
        Retrieve a single patient sample.

        Args:
            index (int): Index of the patient.

        Returns:
            ReturnType, tuple containing:
                - X: Input tensor
                - M: Mask tensor
                - D: Delta-time tensor
                - target_mask: Mask for loss computation
                - X_true_norm: Ground-truth normalized tensor
        """
        return (
            self.X[index],
            self.M[index],
            self.D[index],
            self.target_mask[index],
            self.X_true_norm[index],
        )
