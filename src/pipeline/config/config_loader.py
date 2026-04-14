"""
Module for loading the pipeline config.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

from pathlib import Path

import yaml
from src.pipeline.config.imputer_config import ImputerConfig
from src.pipeline.config.survival_config import SurvivalConfig


def _load_config(path: str | Path) -> dict:
    """
    Load and validate pipeline configuration from YAML.

    Args:
        path (str | Path): Path to YAML config file.

    Returns:
        Config file
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError("Config file is empty")

    return raw_config


def load_survival_config(path: str | Path) -> SurvivalConfig:
    """
    Load survival configuration from YAML.

    Args:
        path (str | Path): Path to YAML config file.

    Returns:
        SurvivalConfig: Validated pipeline configuration.
    """
    raw_config = _load_config(path)
    survival_args = raw_config["survival"]

    # Inject base config if present
    if "base" in raw_config:
        survival_args["base"] = raw_config["base"]

    return SurvivalConfig(**survival_args)


def load_imputer_config(path: str | Path) -> ImputerConfig:
    """
    Load imputer configuration from YAML.

    Args:
        path (str | Path): Path to YAML config file.

    Returns:
        ImputerConfig: Validated pipeline configuration.
    """
    raw_config = _load_config(path)
    imputer_args = raw_config["imputer"]

    # Inject base config if present
    if "base" in raw_config:
        imputer_args["base"] = raw_config["base"]

    return ImputerConfig(**imputer_args)
