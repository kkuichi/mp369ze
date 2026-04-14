"""
Module for training survival models.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.pipeline.config.survival_config import SurvivalConfig
from src.pipeline.constants import DURATION_COL, EVENT_COL, GROUP_COL
from src.pipeline.core.survival.data import Data
from src.pipeline.core.survival.models import RandomSurvivalForest, XGBModel


# pylint: disable=too-few-public-methods
class Trainer:
    """
    Trainer class for training survival models.

    Attributes:
        config (SurvivalConfig): Survival configuration.
        data (Data): Data object.

    Methods:
        - train: Trains survival model on imputed pivot train.
        - _assert_no_group_leakage: Raises an error if there is overlap
            between training and validation groups.
        - _objective: Objective function for Optuna optimization.
        - _get_model_class: Returns the model class based on configuration.
    """

    def __init__(self, config: SurvivalConfig, data: Data) -> None:
        self.config = config
        self.data = data

    def _get_model_class(self) -> type:
        """Returns the model class based on configuration."""
        model_name = self.config.model.model_name

        if model_name in ["xgb_cox", "xgb_aft"]:
            return XGBModel
        return RandomSurvivalForest

    def _assert_no_group_leakage(self, train_groups: pd.Series, val_groups: pd.Series) -> None:
        """
        Raise an error if there is overlap between training and validation groups.

        Args:
            train_groups (pd.Series): Training groups.
            val_groups (pd.Series): Validation groups.

        Raises:
            AssertionError: If there is overlap between training and validation groups.

        Returns:
            None
        """
        overlap = set(train_groups.unique()) & set(val_groups.unique())
        if overlap:
            raise AssertionError(f"Group leakage detected in CV: {len(overlap)} patients")

    # pylint: disable=too-many-locals
    def _objective(
        self, trial: optuna.Trial, x: pd.DataFrame, y: pd.DataFrame, groups: pd.Series
    ) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial (optuna.Trial): Optuna trial object.
            x (pd.DataFrame): Features.
            y (pd.DataFrame): Labels.
            groups (pd.Series): Groups.

        Returns:
            float: C-index.
        """

        ModelClass = self._get_model_class()
        params = ModelClass.suggest_params(trial, self.config.model) # type: ignore

        cv = StratifiedGroupKFold(
            n_splits=self.config.train.n_splits,
            shuffle=self.config.train.shuffle,
            random_state=self.config.base.seed,
        )

        scores: list[float] = []

        for train_idx, val_idx in cv.split(x, y[EVENT_COL], groups=groups):
            x_tr, x_val = x.iloc[train_idx], x.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            g_tr, g_val = groups.iloc[train_idx], groups.iloc[val_idx]

            self._assert_no_group_leakage(g_tr, g_val)

            model = ModelClass(config=self.config.model, params=params)
            model.fit(x_tr, y_tr, x_val, y_val)

            score = model.score(x_val, y_val)
            scores.append(score)

        return float(np.mean(scores))

    def train(self, imputed_pivot_train_df: pd.DataFrame, study_name: str | None = None) -> dict:
        """
        Trains survival model on imputed pivot train.

        Args:
            imputed_pivot_train_df (pd.DataFrame): Imputed pivot train dataframe.
            study_name (str | None): Study name.

        Returns:
            dict: Dictionary with survival model, study, best params and best val c-index.
        """
        logging.info("Training survival model on imputed pivot train")

        assert imputed_pivot_train_df[GROUP_COL].isna().sum() == 0
        assert imputed_pivot_train_df[EVENT_COL].isin([0, 1]).all()
        assert (imputed_pivot_train_df[DURATION_COL] >= 0).all()

        x, y, groups = self.data.split_xy(imputed_pivot_train_df)

        for col in [GROUP_COL, DURATION_COL, EVENT_COL]:
            assert col not in x.columns, f"{col} leaked into features"

        study = optuna.create_study(
            direction=self.config.train.direction,
            study_name=study_name,
            load_if_exists=True,
        )

        study.optimize(
            lambda trial: self._objective(trial, x, y, groups),
            n_trials=self.config.train.n_trials,
        )

        best_params = study.best_params

        ModelClass = self._get_model_class()
        final_model = ModelClass(config=self.config.model, params=best_params)

        if self.config.model.model_name == "xgb_cox":

            # Split training data for final model validation (to enable early stopping)
            cv = StratifiedGroupKFold(
                n_splits=self.config.train.n_splits,
                shuffle=self.config.train.shuffle,
                random_state=self.config.base.seed,
            )

            train_idx, val_idx = next(cv.split(x, y[EVENT_COL], groups=groups))
            x_train_final = x.iloc[train_idx]
            y_train_final = y.iloc[train_idx]
            x_val_final = x.iloc[val_idx]
            y_val_final = y.iloc[val_idx]

            # Verify no patient leakage
            groups_train = groups.iloc[train_idx]
            groups_val = groups.iloc[val_idx]
            self._assert_no_group_leakage(groups_train, groups_val)

            # Train with early stopping on validation set
            final_model.fit(x_train_final, y_train_final, x_val_final, y_val_final)

        else:
            final_model.fit(x, y)

        # Calculate train C-index on the full training set
        train_c_index = final_model.score(x, y)

        return {
            "model": final_model,
            "study": study,
            "best_params": best_params,
            "train_c_index": train_c_index,
            "best_val_c_index": study.best_value,
        }
