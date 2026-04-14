"""
Module for loading data in the pipeline.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from src.pipeline.constants import (
    IMPUTED_CACHE_DIR,
    ORIGINAL_WAVE_1,
    ORIGINAL_WAVE_2,
    ORIGINAL_WAVE_3,
    ORIGINAL_WAVE_4,
    PIVOTED_CACHE_DIR,
    PREPARED_WAVE_1,
    PREPARED_WAVE_2,
    PREPARED_WAVE_3,
    PREPARED_WAVE_4,
)


class DataLoader:
    """Class for data loading"""

    def __init__(self):
        """Initialize data paths"""
        self.original_paths: dict[str, Path] = {
            "wave_1": ORIGINAL_WAVE_1,
            "wave_2": ORIGINAL_WAVE_2,
            "wave_3": ORIGINAL_WAVE_3,
            "wave_4": ORIGINAL_WAVE_4,
        }

        self.prepared_paths: dict[str, Path] = {
            "wave_1": PREPARED_WAVE_1,
            "wave_2": PREPARED_WAVE_2,
            "wave_3": PREPARED_WAVE_3,
            "wave_4": PREPARED_WAVE_4,
        }

    def load_wave(
        self,
        paths: dict[str, Path],
    ) -> dict[str, pd.DataFrame]:
        """
        Helper method for loading wave data.

        Args:
            paths (dict[str, Path]): Dictionary with wave name and it's path.

        Returns:
            dict[str, pd.DataFrame]: Dictionary with wave name and loaded DataFrame.
        """
        data: dict[str, pd.DataFrame] = {}

        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist.")

            logging.info("Loading %s from %s", name, path)
            data[name] = pd.read_excel(path)

        return data

    def load_original_waves(self) -> dict[str, pd.DataFrame]:
        """
        Load original wave data from excel file

        Returns:
            dict[str, pd.DataFrame]: Mapping wave name to dataframe
        """
        logging.info("Loading original wave dataset")
        logging.info("Keys: %s", self.original_paths.keys())

        return self.load_wave(paths=self.original_paths)

    def load_prepared_waves(self) -> dict[str, pd.DataFrame]:
        """
        Load prepared wave data from excel file

        Returns:
            dict[str, pd.DataFrame]: Mapping wave name to dataframe
        """
        logging.info("Loading prepared wave dataset")

        return self.load_wave(paths=self.prepared_paths)


class PivotedCache:
    """Class for loading pivoted data cache (with missing values)"""

    def __init__(self, cache_dir: Path = PIVOTED_CACHE_DIR) -> None:
        self.cache_dir = cache_dir

    def _get_cache_path(self, wave_name: str) -> Path:
        """Get path for cached pivoted data."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{wave_name}_pivoted.parquet"

    def load_cache(self, wave_name: str) -> pd.DataFrame | None:
        """Load pivoted data from cache."""
        path = self._get_cache_path(wave_name)
        if not path.exists():
            return None

        logging.info("Loading cached pivoted data for %s from %s", wave_name, path)
        return pd.read_parquet(path)

    def save_cache(self, wave_name: str, df: pd.DataFrame) -> None:
        """Save pivoted data to cache."""
        path = self._get_cache_path(wave_name)
        logging.info("Saving pivoted data for %s to %s", wave_name, path)
        df.to_parquet(path)


class ImputerCacheLoader:
    """Class for loading cache data"""

    def __init__(self, cache_dir: Path = IMPUTED_CACHE_DIR) -> None:
        self.cache_dir = cache_dir

    def _get_cache_path(self, wave_name: str) -> Path:
        """Get path for cached imputed data."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{wave_name}_imputed.parquet"

    def load_cache(self, wave_name: str) -> tuple[tuple[pd.DataFrame, pd.DataFrame], dict] | None:
        """Load imputed data from cache."""
        path = self._get_cache_path(wave_name)
        if not path.exists():
            return None

        logging.info("Loading cached imputed data for %s from %s", wave_name, path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["splits"], data["metadata"]

    def save_cache(
        self, wave_name: str, splits: tuple[pd.DataFrame, pd.DataFrame], metadata: dict
    ) -> None:
        """Save imputed data to cache."""
        path = self._get_cache_path(wave_name)
        logging.info("Saving imputed data for %s to %s", wave_name, path)

        data = {"splits": splits, "metadata": metadata}
        with open(path, "wb") as f:
            pickle.dump(data, f)
