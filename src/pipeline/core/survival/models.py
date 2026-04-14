"""
Module implementing survival analysis models.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest as RSF
from sksurv.util import Surv
from src.pipeline.config.survival_config import SurvivalModelConfig
from src.pipeline.constants import DURATION_COL, EVENT_COL


class XGBModel:
    """
    XGBoost Cox proportional hazards model.

    Attributes:
        config (SurvivalModelConfig): Model configuration.
        params (dict): Model hyperparameters.
        model (xgb.Booster | None): Trained XGBoost model.

    Methods:
        - fit: Fit the model to training data.
        - predict_risk: Predict risk scores for given features.
        - score: Calculate concordance index on given data.
        - suggest_params: Suggest hyperparameters for tuning.
    """

    def __init__(self, config: SurvivalModelConfig, params: dict | None = None) -> None:
        self.config = config
        self.params = params or {}
        self.model: xgb.Booster | None = None

    def _make_dmatrix(self, x: pd.DataFrame, y: pd.DataFrame) -> xgb.DMatrix:
        """
        Make DMatrix for XGBoost training.

        Args:
            x (pd.DataFrame): Features.
            y (pd.DataFrame): Labels.

        Returns:
            xgb.DMatrix: DMatrix for XGBoost.
        """
        durations = y[DURATION_COL].astype(float)
        events = y[EVENT_COL].astype(int)

        dmat = xgb.DMatrix(x, label=durations)

        dmat.set_float_info("label_lower_bound", durations)
        dmat.set_float_info("label_upper_bound", np.where(events == 1, durations, np.inf))
        return dmat

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
    ) -> None:
        """
        Fit the XGBoost model.

        Args:
            x_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Training labels.
            x_val (pd.DataFrame)| None: Validation features.
            y_val (pd.DataFrame)| None: Validation labels.

        Returns:
            None
        """

        dtrain = self._make_dmatrix(x_train, y_train)
        evals = [(dtrain, "train")]

        if x_val is not None and y_val is not None:
            dval = self._make_dmatrix(x_val, y_val)
            evals.append((dval, "val"))

        if self.config.model_name == "xgb_cox":
            params = {"objective": "survival:cox", "eval_metric": "cox-nloglik"}

        else:
            params: dict[str, int | float | str] = {
                "objective": "survival:aft",
                "eval_metric": "aft-nloglik",
                "aft_loss_distribution": self.config.aft_loss_distribution,
            }

        params.update(dict(
                verbosity=self.config.verbosity,
                seed=self.config.base.seed,
                **self.params,
        ))

        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.config.num_boost_round,
            evals=evals,
            early_stopping_rounds=(self.config.early_stopping_rounds if len(evals) > 1 else None),
            verbose_eval=False,
        )

    def predict_risk(self, x: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores.

        Args:
            x (pd.DataFrame): Features.

        Returns:
            np.ndarray: Predicted risk scores.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return self.model.predict(xgb.DMatrix(x))

    def score(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Calculate the concordance index.

        Args:
            x (pd.DataFrame): Features.
            y (pd.DataFrame): Labels.

        Returns:
            float: Concordance index.
        """
        preds = self.predict_risk(x)
        return concordance_index(y[DURATION_COL], preds, y[EVENT_COL])

    @staticmethod
    def suggest_params(trial: optuna.Trial, config: SurvivalModelConfig) -> dict[str, float | int]:
        """
        Suggest parameters for hyperparameter tuning.

        Args:
            trial (optuna.Trial): Trial object from Optuna.
            config (SurvivalModelConfig):

        Returns:
            dict[str, float | int]: Suggested parameters.
        """

        params = {
            "eta": trial.suggest_float("eta", config.eta_min, config.eta_max, log=True),
            "max_depth": trial.suggest_int(
                "max_depth", config.max_depth_min, config.max_depth_max, log=False
            ),
            "subsample": trial.suggest_float(
                "subsample", config.subsample_min, config.subsample_max, log=False
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                config.colsample_bytree_min,
                config.colsample_bytree_max,
                log=False,
            ),
            "lambda": trial.suggest_float("lambda", config.lambda_min, config.lambda_max, log=True),
            "alpha": trial.suggest_float("alpha", config.alpha_min, config.alpha_max, log=True),
            "gamma": trial.suggest_float("gamma", config.gamma_min, config.gamma_max, log=False),
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                config.min_child_weight_min,
                config.min_child_weight_max,
                log=False,
            ),
        }

        return params


class RandomSurvivalForest:
    """
    Class for Random Survival Forest model implementation.
    """

    def __init__(self, config: SurvivalModelConfig, params: dict | None = None) -> None:
        self.config = config
        self.params = params or {}
        self.model: RSF | None = None

    # pylint: disable=unused-argument
    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
    ) -> None:
        """
        Fit the Random Survival Forest model.

        Args:
            x_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Training labels.
        """
        y_structured = Surv.from_dataframe(EVENT_COL, DURATION_COL, y_train)

        params = {
            "n_estimators": 100,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": self.config.base.seed,
            "verbose": 0,
            **self.params,
        }

        self.model = RSF(**params)
        self.model.fit(x_train, y_structured)

    def predict_risk(self, x: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores.

        Args:
            x (pd.DataFrame): Features.

        Returns:
            np.ndarray: Predicted risk scorres.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return self.model.predict(x)

    def score(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Calculate the concordance index.

        Args:
            x (pd.DataFrame): Features.
            y (pd.DataFrame): Labels.

        Returns:
            float: Concordance index (C-index).
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        y_structured = Surv.from_dataframe(EVENT_COL, DURATION_COL, y)
        return self.model.score(x, y_structured)

    @staticmethod
    def suggest_params(
        trial: optuna.Trial, config: SurvivalModelConfig
    ) -> dict[str, int | float | str | None]:
        """
        Suggest hyperparameters for RSF using an Optuna trial.

        Args:
            trial (optuna.Trial): Optuna trial object.
            config (SurvivalModelConfig): Model-specific configuration.

        Returns:
            dict[str, int | float | str | None]: Suggested hyperparameters.
        """
        params: dict[str, int | float | str | None] = {
            "n_estimators": trial.suggest_int(
                "n_estimators", config.n_estimators_min, config.n_estimators_max
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                config.min_samples_split_min,
                config.min_samples_split_max,
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                config.min_samples_leaf_min,
                config.min_samples_leaf_max,
            ),
        }

        if hasattr(config, "max_features_choices") and config.max_features_choices:
            params["max_features"] = trial.suggest_categorical(
                "max_features", config.max_features_choices
            )

        return params
