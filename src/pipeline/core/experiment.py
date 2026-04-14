"""
Module for executing survival analysis experiments
across arbitrary waves.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

import logging
from typing import TypedDict, Union

import numpy as np
import pandas as pd
from src.pipeline.config.imputer_config import (
    ImputerDataConfig,
    ImputerModelConfig,
    ImputerTrainerConfig,
)
from src.pipeline.config.survival_config import SurvivalConfig
from src.pipeline.constants import DURATION_COL, EVENT_COL, GROUP_COL
from src.pipeline.core.imputer.trainer import ImputerTrainer
from src.pipeline.core.survival.data import Data
from src.pipeline.core.survival.models import RandomSurvivalForest, XGBModel
from src.pipeline.core.survival.trainer import Trainer
from src.pipeline.utilities.data_loader import ImputerCacheLoader, PivotedCache
from src.pipeline.utilities.experiment_logger import ExperimentLogger

SurvivalModel = Union[XGBModel, RandomSurvivalForest]

WaveDFDict = dict[str, pd.DataFrame]
TrainTestDict = dict[str, tuple[pd.DataFrame, pd.DataFrame]]
EvalDict = dict[str, dict[str, float]]


class WaveResult(TypedDict):
    test_c_index: float
    risk_scores: np.ndarray

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
class ExperimentRunner:
    """
    This class serves as high-level experiment orchestrator.
    Responsibilities:
        - wave-wise preprocessing
        - train/test splitting
        - GRU-D imputation (fit on train, transform test)
        - survival model training

    Attributes:
        data (Data): Data object for data loading and preprocessing.
        imputer_config (ImputerConfig): Configuration for the imputer.
        survival_config (SurvivalConfig): Configuration for the survival model.

    Methods:
        _prepare_single_wave: Prepares single wave, preprocessing steps:
            - prepare original dataframe (calls data.prepare_origial_data)
            - prepare prepared dataframe (calls data.prepare_prepared_data)
            - merge original and prepared dataframes (calls data.merge_original_and_prepared)
            - parse merged dataframe (calls data.parse_merged)
            - pivot parsed dataframe (calls data.pivot_parsed_with_static)
        prepare_waves: Prepares all waves - applies all preprocessing steps by calling
            self._prepare_single_wave() method.
        split_waves: Split each wave into train/test sets.
        impute_waves: Fit GRU-D imputer on train set and impute both train and test sets.
        train_survival: Train survival model on imputed pivot train.
        evaluate: Evaluate survival model on imputed pivot test.
    """

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments

    def __init__(
        self,
        data: Data,
        imputer_data_config: ImputerDataConfig,
        imputer_trainer_config: ImputerTrainerConfig,
        imputer_model_config: ImputerModelConfig,
        survival_config: SurvivalConfig,
        logger: ExperimentLogger,
    ) -> None:
        self.data = data
        self.imp_data_config = imputer_data_config
        self.imp_tr_config = imputer_trainer_config
        self.imp_mdl_config = imputer_model_config
        self.survival_config = survival_config
        self.logger = logger
        self.pivoted_cache = PivotedCache()
        self.imputer_cache_loader = ImputerCacheLoader()

    def _prepare_single_wave(
        self, original_df: pd.DataFrame, prepared_df: pd.DataFrame, wave_name: str
    ) -> pd.DataFrame:
        """
        Prepares single wave, preprocessing steps:
            - prepare original dataframe (calls data.prepare_origial_data)
            - prepare prepared dataframe (calls data.prepare_prepared_data)
            - merge original and prepared dataframes (calls data.merge_original_and_prepared)
            - parse merged dataframe (calls data.parse_merged)
            - pivot parsed dataframe (calls data.pivot_parsed_with_static)

        Args:
            original_df (pd.DataFrame): Original dataframe.
            prepared_df (pd.DataFrame): Prepared dataframe.
            wave_name (str): Name of the COVID-19 wave (working with 1st and 4th wave)

        Returns:
            pd.DataFrame: Pivoted dataframe of one COVID-19 wave. This dataframe
                will contain missing values in tests, that have been taken
                for each paient. These miisng values will be imputed further on
                with GRU-D imputer.
        """
        prepared_original_df = self.data.prepare_original_data(
            original_df=original_df, wave_name=wave_name
        )

        prepared_prepared_df = self.data.prepare_prepared_data(
            original_df=original_df, prepared_df=prepared_df, wave_name=wave_name
        )

        merged_df = self.data.merge_original_and_prepared(
            original_df=prepared_original_df, prepared_df=prepared_prepared_df
        )

        parsed_df = self.data.parse_merged(merged_df=merged_df)

        pivoted_df = self.data.pivot_parsed_with_static(parsed_df=parsed_df)

        return pivoted_df

    def prepare_waves(
        self,
        original_df_dict: WaveDFDict,
        prepared_df_dict: WaveDFDict,
        wave_names: list[str],
    ) -> WaveDFDict:
        """
        Prepares all waves - applies all preprocessing steps by calling
        self._prepare_single_wave() method.

        Args:
            original_df_dict (WaveDFDict): Original dataframe.
            prepared_df_dict (WaveDFDict): Prepared dataframe.
            wave_names: (list[str]): Names of all COVID-19 waves chosen
                to work with.

        Returns:
            WaveDFDict: Dictionary with wave names and preprocessed
                (pivoted with missings) dataframes.
        """
        pivoted_by_wave: WaveDFDict = {}

        for wave_name in wave_names:
            pivoted_by_wave[wave_name] = self._prepare_single_wave(
                original_df=original_df_dict[wave_name],
                prepared_df=prepared_df_dict[wave_name],
                wave_name=wave_name,
            )

        return pivoted_by_wave

    def split_waves(self, pivoted_by_wave_dict: WaveDFDict) -> TrainTestDict:
        """
        Split each wave into train/test sets.

        Args:
            pivoted_by_wave_dict (WaveDFDict): Dictionary with
                pivoted wave dataframes.

        Returns:
            TrainTestDict: Dict of wave name and its
                train and test sets in Tuple.
        """
        splits: TrainTestDict = {}

        for wave, df in pivoted_by_wave_dict.items():
            train_df, test_df = self.data.split_pivoted_to_train_test(pivoted_df=df)
            splits[wave] = (train_df, test_df)
        return splits

    def impute_waves(self, splits: TrainTestDict) -> tuple[TrainTestDict, dict[str, dict]]:
        """
        Fit GRU-D imputer on train set and impute both train and test sets.

        Args:
            splits (TrainTestDict): Dict of wave name and its
                train and test sets in Tuple.

        Returns:
            tuple: Dict of wave name and its imputed train and test
                sets, metadata.
        """
        # Initialize ImputerTrainer
        imputer = ImputerTrainer(
            data_config=self.imp_data_config,
            trainer_config=self.imp_tr_config,
            model_config=self.imp_mdl_config,
        )

        # Impute per wave
        imputed: TrainTestDict = {}
        metadata_by_wave: dict[str, dict] = {}

        for wave, (train_df, test_df) in splits.items():
            logging.info("Imputing wave: %s", wave)
            # Get optuna storage url
            # FIT: Performs HPO with Optuna, trains final model on train_df
            imputed_train, metadata = imputer.fit(train_df, study_name=f"{wave}_imputer")
            # TRANSFORM: Impute test_df using fitted model
            # Preprocessor state (means/stds) from fit is reused
            imputed_test = imputer.transform(test_df)

            # Store results
            imputed[wave] = (imputed_train, imputed_test)
            metadata_by_wave[wave] = metadata

        return imputed, metadata_by_wave

    def train_survival_model(self, imputed_splits: TrainTestDict) -> tuple[dict[str, dict], dict[str, dict]]:
        """
        Train suvival model per wave.

        Args:
            imputed_splits (TrainTestDict): Dict of wave name and its
                imputed train and test sets in Tuple.

        Returns:
            tuple[dict[str, dict], dict[str, dict]]: Tuple with results of Trainer.train mathod in
                dict and metadata of each wave in dict.
        """
        trainer = Trainer(self.survival_config, self.data)
        results: dict[str, dict] = {}
        metadata_by_wave: dict[str, dict] = {}

        for wave, (train_df, _) in imputed_splits.items():
            logging.info("Training survival model for wave: %s", wave)
            results[wave] = trainer.train(train_df, study_name=f"{wave}_survival")
            metadata_by_wave[wave] = {
                "best_params": results[wave]["best_params"],
                "train_c_index": results[wave]["train_c_index"],
                "best_val_c_index": results[wave]["best_val_c_index"],
            }

        return results, metadata_by_wave

    def _evaluate_single_wave(
        self, model: SurvivalModel, imputed_test_df: pd.DataFrame
    ) -> WaveResult:
        """
        Evaluate trained survival model on a single wave.

        Args:
            model (SurvivalModel): Trained survival model.
            imputed_test_df (pd.DataFrame): Imputed test set.

        Returns:
            WaveResult: Result of the evaluation.
        """

        logging.info("Running survival model inference.")

        x_test, y_test, _ = self.data.split_xy(imputed_test_df)

        assert len(x_test) == len(y_test)
        assert x_test.index.equals(y_test.index)

        # Ensure there is no leakage
        for col in [EVENT_COL, DURATION_COL, GROUP_COL]:
            assert col not in x_test.columns, f"{col} leaked into X at inference"

        risk_scores = model.predict_risk(x_test).tolist()
        test_c_index = model.score(x_test, y_test)

        return {"test_c_index": test_c_index, "risk_scores": risk_scores}

    def evaluate_waves(
        self, models_by_wave: dict[str, SurvivalModel], imputed_splits: TrainTestDict
    ) -> dict[str, WaveResult]:
        """
        Evaluate survival models on test sets for all waves.

        Args:
           models_by_wave (dict[str, SurvivalModel]): Dictionary of trained models per wave.
           imputed_splits (TrainTestDict): Dictionary containing train/test splits.

        Returns:
            dict[str, WaveResult]: Dictionary containing evaluation metrics for each wave.
        """
        results: dict[str, WaveResult] = {}

        for wave, model in models_by_wave.items():
            logging.info("Evaluating survival model for wave: %s", wave)

            _, test_df = imputed_splits[wave]

            metrics = self._evaluate_single_wave(model=model, imputed_test_df=test_df)

            results[wave] = metrics

        return results

    def run(self, run_id: int | None = None) -> dict[str, dict]:
        """
        Run full end-to-end experiment with multi-level caching.
        Steps:
            1. Check for imputed data (Level 1 cache).
            2. Check for pivoted data (Level 2 cache) if L1 missing.
            3. Generate data from scratch if no cache found.
            4. Impute using GRU-D and save to cache.
            5. Train Survival model (XGB/RSF) and evaluate.

        Args:
            run_id (int | None): The run identifier.
        """
        logging.info("=== EXPERIMENT RUN STARTED (Run ID: %s) ===", run_id)

        original_df_dict = self.data.load_original_data_dict()
        prepared_df_dict = self.data.load_prepared_data_dict()

        if original_df_dict.keys() != prepared_df_dict.keys():
            raise ValueError("Wave names mismatch between original and prepared data.")

        wave_names = list(original_df_dict.keys())
        logging.info("Waves identified for processing: %s", wave_names)

        imputed_splits_by_wave: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        imputer_metadata: dict[str, dict] = {}

        for wave in wave_names:
            logging.info("--- Processing Wave: %s ---", wave)

            cache = self.imputer_cache_loader.load_cache(wave)
            if cache is not None:
                cached_splits, cached_meta = cache
                # cached_splits, cached_meta = self.imputer_cache_loader.load_cache(wave)
                if cached_splits is not None:
                    logging.info("Level 1 Cache Hit: Using imputed data for wave %s", wave)
                    imputed_splits_by_wave[wave] = cached_splits
                    imputer_metadata[wave] = cached_meta
                    continue
            else:
                logging.info("Level 1 Cache Miss: No imputed data for wave %s", wave)

            pivoted_df = self.pivoted_cache.load_cache(wave)
            if pivoted_df is None:
                logging.info("Generating pivoted data for wave %s", wave)
                pivoted_df = self._prepare_single_wave(
                    original_df=original_df_dict[wave],
                    prepared_df=prepared_df_dict[wave],
                    wave_name=wave,
                )
                self.pivoted_cache.save_cache(wave, pivoted_df)
            else:
                logging.info("Using pivoted data for wave %s", wave)

            train_df, test_df = self.data.split_pivoted_to_train_test(pivoted_df=pivoted_df)

            current_split = {wave: (train_df, test_df)}
            new_imputed_splits, new_imputer_metadata = self.impute_waves(current_split)

            self.imputer_cache_loader.save_cache(
                wave, new_imputed_splits[wave], new_imputer_metadata[wave]
            )

            imputed_splits_by_wave[wave] = new_imputed_splits[wave]
            imputer_metadata[wave] = new_imputer_metadata[wave]

        training_results, survival_metadata = self.train_survival_model(imputed_splits_by_wave)

        models_by_wave = {wave: result["model"] for wave, result in training_results.items()}

        evaluation_results = self.evaluate_waves(
            models_by_wave=models_by_wave, imputed_splits=imputed_splits_by_wave
        )

        for wave in wave_names:
            wave_results = {
                "wave": wave,
                "imputer": imputer_metadata.get(wave, {}),
                "survival": {
                    "model_name": self.survival_config.model.model_name,
                    **survival_metadata.get(wave, {}),
                },
                "evaluation": evaluation_results.get(wave, {}),
            }

            self.logger.save_results(wave_results, wave, run_id=run_id)

            if wave in models_by_wave:
                self.logger.save_model(
                    models_by_wave[wave],
                    wave,
                    run_id=run_id,
                    model_name=self.survival_config.model.model_name,
                )

        return {
            "imputer": imputer_metadata,
            "survival": survival_metadata,
            "evaluation": evaluation_results,
        }

    def run_multiple(self, n_runs: int = 10) -> None:
        """
        Run experiment multiple times and aggregate results.

        Args:
            n_runs (int): Number of runs.
        """
        all_results = []
        for i in range(n_runs):
            logging.info("Starting Run %d/%d", i + 1, n_runs)
            run_result = self.run(run_id=i + 1)
            all_results.append(run_result)

        # Aggregation logic
        aggregated_data = {}
        wave_names = all_results[0]["evaluation"].keys()

        for wave in wave_names:
            train_c_indices = [res["survival"][wave]["train_c_index"] for res in all_results]
            val_c_indices = [res["survival"][wave]["best_val_c_index"] for res in all_results]
            test_c_indices = [res["evaluation"][wave]["test_c_index"] for res in all_results]

            aggregated_data[wave] = {
                "mean_train_c_index": float(np.mean(train_c_indices)),
                "mean_val_c_index": float(np.mean(val_c_indices)),
                "mean_test_c_index": float(np.mean(test_c_indices)),
                "std_train_c_index": float(np.std(train_c_indices)),
                "std_val_c_index": float(np.std(val_c_indices)),
                "std_test_c_index": float(np.std(test_c_indices)),
                "min_train_c_index": float(np.min(train_c_indices)),
                "min_val_c_index": float(np.min(val_c_indices)),
                "min_test_c_index": float(np.min(test_c_indices)),
                "max_train_c_index": float(np.max(train_c_indices)),
                "max_val_c_index": float(np.max(val_c_indices)),
                "max_test_c_index": float(np.max(test_c_indices)),
                "train - test c_index gap (mean)": float(
                    np.mean(train_c_indices) - np.mean(test_c_indices)
                ),
                "all_train_c_indices": train_c_indices,
                "all_val_c_indices": val_c_indices,
                "all_test_c_indices": test_c_indices,
            }

        self.logger.save_aggregated_results(aggregated_data)
