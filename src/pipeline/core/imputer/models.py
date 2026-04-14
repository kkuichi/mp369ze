"""
Module with GRU-D neural network for longitudinal data imputation.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

from abc import abstractmethod

import torch
from src.pipeline.config.imputer_config import ImputerModelConfig
from torch import nn


class BaseImputer(nn.Module):
    """
    Abstract base class for imputers with shared input preprocessing.
    """

    def __init__(self, input_size: int) -> None:
        """
        Args:
            input_size (int): Number of input features.
        """
        super().__init__()
        self.input_size = input_size

    def prepare_input(self, x: torch.Tensor, m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        Prepare model input by concatenating values, masks, and log-scaled deltas.

        Args:
            x (torch.Tensor): Input values, shape (B, T, F).
            m (torch.Tensor): Observation mask, shape (B, T, F).
            d (torch.Tensor): Delta-times, shape (B, T, F).

        Returns:
            Concatenated tensor of shape (B, T, 3F).
        """
        d_scaled = torch.log1p(d)
        return torch.cat([x, m, d_scaled], dim=-1)

    @abstractmethod
    def forward(self, x: torch.Tensor, m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the imputer.

        Args:
            x (torch.Tensor): Input values, shape (B, T, F).
            m (torch.Tensor): Observation mask, shape (B, T, F).
            d (torch.Tensor): Delta-times, shape (B, T, F).

        Returns:
            Imputed values, shape (B, T, F).
        """


class GRUDImputer(BaseImputer):
    """
    GRU-D style imputer for longitudinal time series.
    """

    def __init__(self, config: ImputerModelConfig, input_size: int) -> None:
        """
        Initialize GRU-D imputer.

        Args:
            config (ImputerConfig): Model architecture configuration.
            input_size (int): Number of input features.
        """
        super().__init__(input_size)
        self.config = config

        self.rnn = nn.GRU(
            input_size=input_size * 3,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.n_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.n_layers > 1 else 0.0,
        )
        self.out = nn.Linear(self.config.hidden_size, input_size)

    def forward(self, x: torch.Tensor, m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GRU-D model.

        Args:
            x (torch.Tensor): Input values, shape (B, T, F).
            m (torch.Tensor): Observation mask, shape (B, T, F).
            d (torch.Tensor): Delta-times, shape (B, T, F).

        Returns:
            Imputed values, shape (B, T, F).
        """
        inp = self.prepare_input(x, m, d)
        hidden, _ = self.rnn(inp)
        return self.out(hidden)
