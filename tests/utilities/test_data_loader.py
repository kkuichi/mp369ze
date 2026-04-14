"""
Module for testing utilities/data_loader.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.pipeline.utilities.data_loader import DataLoader


class TestDataLoader:
    """Class for utilities/data_loader tests."""

    def test_load_wave(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        loader = DataLoader()

        fake_file = tmp_path / "wave.xlsx"
        fake_file.touch()

        with patch("pandas.read_excel", return_value=sample_df):
            result = loader.load_wave({"wave_1": fake_file})

        assert "wave_1" in result
        assert result["wave_1"].equals(sample_df)
