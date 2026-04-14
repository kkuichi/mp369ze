"""
Module for logging experiment results, configurations, and artifacts in a structured manner.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from src.pipeline.config.shared_config import BaseConfig
from src.pipeline.constants import OUTPUT_ROOT
from src.pipeline.core.survival.models import RandomSurvivalForest, XGBModel


class ExperimentLogger:
    """
    Handles logging of experiment results, configurations, and artifacts.
    """

    def __init__(self, config: BaseConfig, base_output_dir: Path = OUTPUT_ROOT) -> None:
        self.config = config
        self.base_output_dir = base_output_dir
        self.timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
        self.experiment_dir = (
            self.base_output_dir / f"{self.timestamp}_{self.config.experiment_name}"
        )

        self._create_experiment_dir()

    def _create_experiment_dir(self) -> None:
        """Creates the main experiment directory."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created experiment directory: {self.experiment_dir}")

    def save_config(self, config: dict[str, Any]) -> None:
        """
        Saves the experiment configuration to a YAML file.

        Args:
            config (dict[str, Any]): The configuration to save.

        Returns:
            None
        """
        config_path = self.experiment_dir / "config.yaml"
        with open(config_path, "w", encoding="UTF-8") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Saved config to: {config_path}")

    def get_wave_dir(
        self, wave_name: str, run_id: int | None = None, model_name: str | None = None
    ) -> Path:
        """
        Creates and returns the directory for a specific wave.
        If run_id is provided, creates a subdirectory for that run.

        Args:
            wave_name (str): The name of the wave.
            run_id (int | None): The run identifier.
            model_name (str | None): The name of the model (e.g., 'xgb_aft').

        Returns:
            Path: The path to the wave directory.
        """
        if run_id is not None:
            dir_name = f"{wave_name}_run_{run_id}"
            if model_name:
                dir_name += f"_model_{model_name}"
            wave_dir = self.experiment_dir / dir_name
        else:
            wave_dir = self.experiment_dir / wave_name

        wave_dir.mkdir(parents=True, exist_ok=True)
        return wave_dir

    def save_results(
        self, results: dict[str, Any], wave_name: str, run_id: int | None = None
    ) -> None:
        """
        Saves results to a JSON file in the wave directory.

        Args:
            results (dict[str, Any]): The results to save.
            wave_name (str): The name of the wave.
            run_id (int | None): The run identifier.

        Returns:
            None
        """
        # Extract model name from results if available, to use in directory name
        model_name = results.get("survival", {}).get("model_name", "unknown")

        wave_dir = self.get_wave_dir(wave_name, run_id, model_name)
        results_path = wave_dir / "results.json"

        # Convert non-serializable types if necessary
        # For now, assuming results are JSON-serializable
        with open(results_path, "w", encoding="UTF-8") as f:
            json.dump(results, f, indent=4)
        print(f"Saved results to: {results_path}")

    def save_model(
        self,
        model: XGBModel | RandomSurvivalForest,
        wave_name: str,
        run_id: int | None = None,
        model_name: str = "unknown",
    ) -> None:
        """
        Save the trained model to a pickle file.

        Args:
            model: The trained model instance.
            wave_name (str): The name of the wave.
            run_id (int | None): The run identifier.
            model_name (str): The name of the model.
        """
        wave_dir = self.get_wave_dir(wave_name, run_id, model_name)
        model_path = wave_dir / "model.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved model to: {model_path}")

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def save_vizualization(
        self,
        fig,
        wave_name: str,
        filename: str,
        run_id: int | None = None,
        model_name: str | None = None,
    ) -> None:
        """
        Save a visualization figure to the wave directory.

        Args:
            fig: The figure object to save.
            wave_name (str): The name of the wave.
            filename (str): The filename for the saved figure.
            run_id (int | None): The run identifier.
            model_name (str | None): The name of the model.
        """
        wave_dir = self.get_wave_dir(wave_name, run_id, model_name)
        viz_path = wave_dir / filename

        fig.savefig(viz_path)
        print(f"Saved visualization to: {viz_path}")

    def save_aggregated_results(self, aggregated_results: dict[str, Any]) -> None:
        """
        Saves aggregated results to a JSON file in the final_aggregated_summary directory.

        Args:
            aggregated_results (dict[str, Any]): The aggregated results to save.
        """
        summary_dir = self.experiment_dir / "final_aggregated_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        results_path = summary_dir / "results.json"

        with open(results_path, "w", encoding="UTF-8") as f:
            json.dump(aggregated_results, f, indent=4)
        print(f"Saved aggregated results to: {results_path}")
