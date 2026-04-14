"""
Module for data operations in the pipeline.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

# pylint: disable=too-many-lines

import logging
from collections.abc import Callable

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.pipeline.config.survival_config import SurvivalConfig
from src.pipeline.constants import (
    ADMISSION_COL,
    DISCHARGE_COL,
    DURATION_COL,
    EVENT_COL,
    GENDER_COL,
    GROUP_COL,
    ORIGINAL_DROP_COLS,
    ORIGINAL_MEASUREMENT_COLS,
    ORIGINAL_RENAME_MAP,
    PARSING_STATIC_COLS,
    PIVOT_STATIC_COLS,
    PREPARED_COLS_TO_USE,
    PREPARED_RENAME_MAP,
)
from src.pipeline.utilities.data_exploration import DataExploration
from src.pipeline.utilities.data_loader import DataLoader
from src.pipeline.utilities.data_preparation import DataPreparation


# pylint: disable=too-many-public-methods
class Data:
    """Class representing data flow in the pipeline."""

    def __init__(
        self,
        config: SurvivalConfig,
        data_loader: DataLoader,
        data_preparation: DataPreparation,
        data_exploration: DataExploration | None = None,
    ) -> None:
        self.config = config
        self.data_loader = data_loader
        self.data_preparation = data_preparation
        self.data_exploration = data_exploration or DataExploration()
        self.test_log_df = pd.DataFrame(
            columns=pd.Index([
                "test",
                "wave",
                "preprocessing_step",
                "n_measurements_before",
                "n_patients_before",
                "avg_per_patient_before",
                "n_measurements_after",
                "n_patients_after",
                "avg_per_patient_after",
                "removed",
                "removed_pct",
            ])
        )
        self.feature_cols = ORIGINAL_MEASUREMENT_COLS

    def _create_patient_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates ID for each patient

        Args:
            df(pd.DataFrame): Any dataframe

        Returns:
            pd.DataFrame: Dataftame with one column - patient ID
        """
        logging.info("Creating patient ID")
        df = df.copy()

        if "Poradie" in df.columns:
            logging.info("Using 'Poradie' as ID")
            df = df.rename(columns={"Poradie": "ID"})
            df["ID"] = df["ID"].astype(str)
            assert df["ID"].is_unique, "ID (Poradie) is not unique per patient"
            return df

        if "ID" in df.columns:
            logging.info("ID already exists in dataframe - skipping ID creation")
            return df

        logging.warning("No ID or Poradie column found - creating integer ID")
        df["ID"] = pd.RangeIndex(start=1, stop=len(df) + 1, step=1).astype(str)

        assert df["ID"].is_unique, "ID is not unique per patient"

        return df

    def load_original_data_dict(self) -> dict[str, pd.DataFrame]:
        """
        Load original data in dictionary and create patient IDs.

        Args:
            None.

        Returns:
            dict[str, pd.DataFrame]: Original dataframe in dict with key.
        """
        logging.info("Loading original data")

        data = self.data_loader.load_original_waves()

        for name, df in data.items():
            data[name] = self._create_patient_id(df)

        return data

    def _add_wave(self, df: pd.DataFrame, wave_name: str) -> pd.DataFrame:
        """
        Add Wave column to dataframe based on dictionary key

        Args:
            df (pd.DataFrame): DataFrame to which Wave column will be added
            wave_name (str): dictionary key, e.g., 'wave_1' or 'wave_4'

        Returns:
            pd.DataFrame: DataFrame with added Wave column
        """
        df = df.copy()
        try:
            wave_number = int(wave_name.rsplit("_", maxsplit=1)[-1])
        except (ValueError, IndexError):
            logging.warning("Cannot parse wave number from key '%s', defaulting to 0", wave_name)
            wave_number = 0

        df["Wave"] = wave_number
        return df

    def drop_original_cols(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop unnecessary original columns

        Args:
            original_df(pd.DataFrame): Original dataframe

        Returns:
            pd.DataFrame: Dataframe with dropped columns
        """
        logging.info("Dropping unnecessary columns from original dataset")
        return original_df.drop(columns=ORIGINAL_DROP_COLS, axis=1)

    def rename_original_cols(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename chosen original columns

        Args:
            original_df(pd.DataFrame): Original dataframe

        Returns:
            pd.DataFrame: Dataframe with renamed columns
        """
        logging.info("Renaming chosen attributes in original dataset")

        if not ORIGINAL_RENAME_MAP:
            logging.warning("No rename map defined!!!")

        return original_df.rename(mapper=ORIGINAL_RENAME_MAP, axis=1)

    def convert_original_gender_to_numbers(
        self, original_df: pd.DataFrame, gender_col: str = GENDER_COL
    ) -> pd.DataFrame:
        """
        Convert gender values into numeric data type

        Args:
            original_df(pd.DataFrame): Original dataframe
            gender_col(str): Gender column name

        Returns:
            pd.DataFrame: Dataframe with converted gender column
        """
        if gender_col in original_df.columns:
            original_df[gender_col] = original_df[gender_col].replace({"Muž": 0, "Žena": 1})
        return original_df

    def convert_original_admission_and_discharge_to_datetime(
        self,
        original_df: pd.DataFrame,
        admission_col: str = ADMISSION_COL,
        discharge_col: str = DISCHARGE_COL,
    ) -> pd.DataFrame:
        """
        Convert admission column into datetime data type

        Args:
            original_df(pd.DataFrame): Original dataframe
            admission_col(str): Admission column
            discharge_col(str): Discharge column

        Returns:
            pd.DataFrame: Dataframe with converted
            admission and discharge columns
        """
        logging.info("Converting %s and %s to datetime", admission_col, discharge_col)

        if (admission_col in original_df.columns) and (discharge_col in original_df.columns):
            original_df[admission_col] = pd.to_datetime(
                original_df[admission_col], format="%d.%m.%Y", dayfirst=True
            )
            original_df[discharge_col] = pd.to_datetime(
                original_df[discharge_col], format="%d.%m.%Y", dayfirst=True
            )

        return original_df

    def calculate_duration(
        self,
        original_df: pd.DataFrame,
        admission_col: str = ADMISSION_COL,
        discharge_col: str = DISCHARGE_COL,
    ) -> pd.DataFrame:
        """
        Calculate duration of patient hosital stay:
        duration = discharge_col - admission_col

        Args:
            original_df(pd.DataFrame): Original dataframe
            admission_col(str): Admission column
            discharge_col(str): Discharge column

        Returns:
            pd.Dataframe original datarame with duration
        """
        logging.info("Calculating duration")

        original_df["duration"] = (original_df[discharge_col] - original_df[admission_col]).dt.days

        # Filter out invalid durations
        n_invalid = (original_df["duration"] <= 0).sum()
        if n_invalid > 0:
            logging.warning("Removing %d patients with duration <= 0", n_invalid)
            original_df = original_df[original_df["duration"] > 0].reset_index(drop=True)

        assert DURATION_COL not in ORIGINAL_MEASUREMENT_COLS
        assert (original_df["duration"] > 0).all()

        return original_df

    def _count_measurements_in_cells(self, df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
        """
        Count actual measurements inside cells (split by ';') for given columns.

        Delegates to DataExploration.count_measurements_per_test and returns
        a simple {column: count} dict for backward compatibility.

        Args:
            df(pd.DataFrame): Dataframe
            columns(list[str]): list of columns to count measurements in

        Returns:
            dict[str, int]: counts of measurements per column
        """
        summary = self.data_exploration.count_measurements_per_test(df, columns)
        return dict(zip(summary["test"], summary["n_measurements"]))

    def _log_test_changes(
        self,
        before: dict[str, int],
        removed_values: dict[str, list],
        preprocessing_step: str,
        wave: int,
    ) -> None:
        """
        Log changes in test measurements after preprocessing (legacy helper).

        Used by steps that return explicit removed_values dicts.
        For steps wrapped with _snapshot_step, use _append_snapshot_rows instead.

        Args:
            before(dict[str, int]): counts of measurements before preprocessing
            removed_values(dict[str, list]): removed values per test
            preprocessing_step(str): name of preprocessing step
            wave(int): wave number

        Returns:
            None
        """
        rows = []
        for test, before_count in before.items():
            removed_list = removed_values.get(test, [])
            removed_count = len(removed_list)
            after_count = before_count - removed_count

            if removed_count <= 0:
                continue

            removed_pct = removed_count / (before_count if before_count > 0 else 1) * 100

            rows.append(
                {
                    "test": test,
                    "wave": wave,
                    "preprocessing_step": preprocessing_step,
                    "n_measurements_before": before_count,
                    "n_patients_before": None,
                    "avg_per_patient_before": None,
                    "n_measurements_after": after_count,
                    "n_patients_after": None,
                    "avg_per_patient_after": None,
                    "removed": removed_count,
                    "removed_pct": round(removed_pct, 2),
                }
            )

        if rows:
            self.test_log_df = pd.concat([self.test_log_df, pd.DataFrame(rows)], ignore_index=True)

    def _append_snapshot_rows(
        self,
        before_df: pd.DataFrame,
        after_df: pd.DataFrame,
        preprocessing_step: str,
        wave: int,
    ) -> None:
        """
        Append before/after measurement statistics to test_log_df.

        Compares two DataFrames produced by count_measurements_per_test
        and appends one row per test where measurements were removed.

        Args:
            before_df(pd.DataFrame): snapshot before the preprocessing step
            after_df(pd.DataFrame): snapshot after the preprocessing step
            preprocessing_step(str): name of the preprocessing step
            wave(int): wave number

        Returns:
            None
        """
        merged = before_df.merge(after_df, on="test", suffixes=("_before", "_after"))
        rows = []

        for _, row in merged.iterrows():
            removed = row["n_measurements_before"] - row["n_measurements_after"]
            if removed <= 0:
                continue

            removed_pct = (
                removed / row["n_measurements_before"] * 100
                if row["n_measurements_before"] > 0
                else 0.0
            )

            rows.append(
                {
                    "test": row["test"],
                    "wave": wave,
                    "preprocessing_step": preprocessing_step,
                    "n_measurements_before": row["n_measurements_before"],
                    "n_patients_before": row["n_patients_with_test_before"],
                    "avg_per_patient_before": row["avg_per_patient_before"],
                    "n_measurements_after": row["n_measurements_after"],
                    "n_patients_after": row["n_patients_with_test_after"],
                    "avg_per_patient_after": row["avg_per_patient_after"],
                    "removed": removed,
                    "removed_pct": round(removed_pct, 2),
                }
            )

        if rows:
            self.test_log_df = pd.concat([self.test_log_df, pd.DataFrame(rows)], ignore_index=True)

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def _snapshot_step(
        self,
        df: pd.DataFrame,
        step_fn: Callable[..., pd.DataFrame],
        step_name: str,
        test_cols: list[str],
        wave: int,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Execute a preprocessing step and record before/after measurement statistics.

        Calls count_measurements_per_test before and after the step, then
        appends the diff to test_log_df via _append_snapshot_rows.

        Args:
            df(pd.DataFrame): Input dataframe
            step_fn(Callable): Preprocessing function to call with df as first arg
            step_name(str): Label for this step in test_log_df
            test_cols(list[str]): Test columns to count measurements in
            wave(int): Wave number
            **kwargs: Additional keyword arguments forwarded to step_fn

        Returns:
            pd.DataFrame: Result of step_fn
        """
        before_df = self.data_exploration.count_measurements_per_test(df, test_cols)
        result_df = step_fn(df, **kwargs)
        after_df = self.data_exploration.count_measurements_per_test(result_df, test_cols)
        self._append_snapshot_rows(before_df, after_df, step_name, wave)
        return result_df

    # pylint: disable=dangerous-default-value
    def clear_original_tests(
        self,
        original_df: pd.DataFrame,
        measurement_cols: list[str] = ORIGINAL_MEASUREMENT_COLS,
    ) -> pd.DataFrame:
        """
        Remove invalid values from tests and log removed measurements.

        Args:
            original_df(pd.DataFrame): original dataframe
            measurement_cols(list[str]): list of measurement columns

        Returns:
            pd.DataFrame: Dataframe with cleared invalid test values
        """
        logging.info("Identifying invalid measurement values in tests")

        # Clean cells and get removed values
        cleaned_df, _ = self.data_preparation.clean_invalid_rows_in_cells(
            df=original_df, columns=measurement_cols
        )

        # Double-check
        invalid_df_after = self.data_preparation.identify_invalid_test_values(
            df=cleaned_df, columns=measurement_cols
        )
        logging.info(
            "Number of invalid measurement values after clearing: %s",
            len(invalid_df_after),
        )

        if len(invalid_df_after) != 0:
            logging.warning("--- SOME INVALID MEASUREMENT VALUES HAS NOT BEEN REMOVED!!! ---")

        return cleaned_df

    def sort_original_test_measurements(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort test measurements chronologically

        Args:
            original_df(pd.DataFrame): original dataframe

        Returns:
            pd.DataFrame: Dataframe with sorted test measurements
        """
        logging.info("Sorting test measurements chronologically")

        return self.data_preparation.sort_datetime_values(df=original_df)

    def normalize_original_test_measurements(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize test measurement timestamps to days since admission

        Args:
            original_df(pd.DataFrame): original dataframe

        Returns:
            pd.DataFrame: Dataframe with normalized test measurement timestamps to days
        """
        logging.info("Normalizing test measurement values to days")

        return self.data_preparation.normalize_test_datetimes_to_days(
            df=original_df, test_cols=ORIGINAL_MEASUREMENT_COLS
        )

    def remove_tests_outside_interval(
        self,
        original_df: pd.DataFrame,
        admission_col: str = ADMISSION_COL,
        discharge_col: str = DISCHARGE_COL,
        id_col: str = "ID",
    ) -> pd.DataFrame:
        """
        Remove test measurements outside hospital stay

        Args:
            original_df(pd.DataFrame): original dataframe
            admission_col(str): admission date column
            discharge_col(str): discharge date column
            id_col(str): patient identifier column

        Returns:
            pd.DataFrame: Dataframe with test measurements outside hospital stay removed
        """
        logging.info("Removing test before admission and after discharge")

        original_df, _ = self.data_preparation.remove_tests_outside_hospital_stay(
            df=original_df,
            date_in_col=admission_col,
            date_out_col=discharge_col,
            id_col=id_col,
            return_removed_values=True,
        )

        return original_df

    def remove_patients_without_tests(self, original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove patients who have no valid tests values in any measurement column

        Args:
            original_df(pd.DataFrame): original dataframe
            id_col(str): patient identifier column

        Returns:
            pd.DataFrame: Data frame woth patients without test removed
        """
        logging.info("Removing patients without a test measurements")

        test_cols = [c for c in ORIGINAL_MEASUREMENT_COLS if c in original_df.columns]

        if not test_cols:
            logging.warning("No test measurement columns found in dataframe")
            return original_df

        no_tests_mask = original_df[test_cols].replace("", pd.NA).isna().all(axis=1)

        removed_count = int(no_tests_mask.sum())

        logging.info(
            "Removed %d patients without any test measurements (wave=%d)",
            removed_count,
            original_df["Wave"].iloc[0],
        )

        return original_df.loc[~no_tests_mask].copy()

    def prepare_original_data(self, original_df: pd.DataFrame, wave_name: str) -> pd.DataFrame:
        """
        Main function that prepares original data
        - add wave identifier
        - drop unnecessary columns
        - rename chosen columns
        - convert gender to numeric
        - convert admission and discharge to datetime
        - calculate duration
        - clear invalid test values
        - sort test measurements chronologically
        - normalize test measurement timestamps to days since admission
        - remove test measurements outside hospital stay
        - remove patients without any test measurements

        Args:
            original_df(pd.DataFrame): original dataframe

        Returns:
            pd.DataFrame: Preprocessed original dataframe
        """
        logging.info("--- ORIGINAL DATASET PREPROCESSING HAS STARTED ---")

        original_df = self._add_wave(original_df, wave_name)
        original_df = self.drop_original_cols(original_df)
        original_df = self.rename_original_cols(original_df)
        original_df = self.convert_original_gender_to_numbers(original_df)
        original_df = self.convert_original_admission_and_discharge_to_datetime(original_df)
        original_df = self.calculate_duration(original_df)

        wave = original_df["Wave"].iloc[0]

        original_df = self._snapshot_step(
            df=original_df,
            step_fn=self.clear_original_tests,
            step_name="clear_invalid_tests_measurements",
            test_cols=ORIGINAL_MEASUREMENT_COLS,
            wave=wave,
        )
        original_df = self.sort_original_test_measurements(original_df)
        original_df = self.normalize_original_test_measurements(original_df)
        original_df = self._snapshot_step(
            df=original_df,
            step_fn=self.remove_tests_outside_interval,
            step_name="remove_tests_outside_interval",
            test_cols=ORIGINAL_MEASUREMENT_COLS,
            wave=wave,
        )
        original_df = self.remove_patients_without_tests(original_df)

        return original_df

    def load_prepared_data_dict(self) -> dict[str, pd.DataFrame]:
        """
        Load prepared data

        Args:
            None

        Returns:
            dict[str, pd.DataFrame]: Prepared dataframe in dict with key
        """
        logging.info("Loading prepared data")
        return self.data_loader.load_prepared_waves()

    def align_prepared_to_original(
        self, original_df: pd.DataFrame, prepared_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Ensure prepared_df contains only patients present in original_df
        and that they are perfectly aligned.

        Args:
            original_df (pd.DataFrame): Original dataframe with patient IDs.
            prepared_df (pd.DataFrame): Prepared dataframe.

        Returns:
            pd.DataFrame: Prepared dataframe filtered and aligned to original_df patients.
        """
        logging.info("Aligning prepared dataframe to original")
        prepared_df = prepared_df.copy()

        # Handle ID in prepared dataframe
        if "Poradie" in prepared_df.columns:
            logging.info("Using 'Poradie' as ID in prepared data")
            prepared_df = prepared_df.rename(columns={"Poradie": "ID"})
            prepared_df["ID"] = prepared_df["ID"].astype(str)

        if "ID" not in prepared_df.columns:
            # Fallback
            logging.warning("No ID found in prepared data! Attempting to align by index (RISKY)")
            if len(prepared_df) != len(original_df):
                if len(prepared_df) > len(original_df):
                    logging.warning(
                        "Prepared data is longer than original. Slicing to match length."
                    )
                    prepared_df = prepared_df.iloc[: len(original_df)].reset_index(drop=True)
                else:
                    raise ValueError(
                        f"Prepared length {len(prepared_df)} != Original length "
                        f"{len(original_df)} and no ID column found!"
                    )

            prepared_df["ID"] = original_df["ID"].values

        # Align rows to match original_df
        # Filter prepared to keep only IDs that are in original (e.g. after filtering duration <=0)
        original_ids = original_df["ID"].unique()

        # Check if all original IDs are in prepared
        ids_set = set(prepared_df["ID"])
        missing_ids = [pid for pid in original_ids if pid not in ids_set]

        if missing_ids:
            raise ValueError(
                f"Missing {len(missing_ids)} IDs in prepared data (e.g. {missing_ids[:3]}...)"
            )

        # Filter prepared to only relevant IDs
        prepared_df = prepared_df[prepared_df["ID"].isin(original_ids)]

        # Sort/Align prepared dataframe to match the order of original_df
        prepared_df = prepared_df.set_index("ID").reindex(original_ids).reset_index()

        # Ensure lengths match
        assert len(prepared_df) == len(original_df), "Alignment failed: lengths mismtach"
        assert (
            prepared_df["ID"].reset_index(drop=True) == original_df["ID"].reset_index(drop=True)
        ).all(), "Alignment failed: IDs mismatch"

        return prepared_df

    def rename_prepared_cols(self, prepared_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename chosen columns in prepared dataset

        Args:
            prepared_df(pd.DataFrame): prepared dataframe

        Returns:
            pd.DataFrame: Dataframe with renamed columns
        """
        logging.info("Renaming chosen attributes in prepared dataset")
        return prepared_df.rename(mapper=PREPARED_RENAME_MAP, axis=1)

    def select_prepared_columns(self, prepared_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select only chosen columns

        Args:
            prepared_df(pd.DataFrame): prepared dataframe

        Returns:
            pd.DataFrame: Dataframe with selected columns
        """
        logging.info("Selecting chosen columns")

        cols = [c for c in PREPARED_COLS_TO_USE if c in prepared_df.columns]

        if "ID" in prepared_df.columns:
            cols = ["ID"] + cols

        missing = set(PREPARED_COLS_TO_USE) - set(cols)
        if missing:
            logging.warning("Missing prepared columns: %s", sorted(missing))

        return prepared_df.loc[:, cols]

    def convert_prepared_binary_to_bool(self, prepared_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert binary attributes to bool data type

        Args:
            prepared_df(pd.DataFrame): prepared dataframe

        Returns:
            pd.DataFrame: Dataframe with converted binary columns
        """
        logging.info("Convert binary column values to bool data type")

        bool_cols = prepared_df.select_dtypes(include="bool").columns
        prepared_df[bool_cols] = prepared_df[bool_cols].replace({True: 1, False: 0})

        return prepared_df

    def convert_prepared_target_to_binary(self, prepared_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert target variable values to 2 categories
        Before:
            1 - Discharged to home care or social facility
            2 - Transferred to another department
            3 - Exitus (death)
        After:
            0 - Censored (1, 2)
            1 - Died (3)

        Args:
            prepared_df(pd.DataFrame): Prepared dataframe

        Returns:
            pd.DataFrame: Dataframe with converted target variable
        """
        logging.info("Converting target variable to binary categories")

        if EVENT_COL not in prepared_df.columns:
            raise ValueError(f"Event column {EVENT_COL} not found")

        assert EVENT_COL not in ORIGINAL_MEASUREMENT_COLS

        df = prepared_df.copy()
        df[EVENT_COL] = df[EVENT_COL].apply(lambda x: 0 if x in (1, 2) else 1)

        return df

    def prepare_prepared_data(
        self,
        original_df: pd.DataFrame,
        prepared_df: pd.DataFrame,
        wave_name: str,
    ) -> pd.DataFrame:
        """
        Main function that prepares prepared data
        - add wave identifer
        - rename chosen columns
        - select only chosen columns
        - convert binary attributes to bool data type
        - convert target variable to binary categories

        Args:
            prepared_df(pd.DataFrame): Prepared dataframe.
            wave_name(str): Wave name.

        Returns:
            pd.DataFrame: Preprocessed prepared dataframe.
        """
        logging.info("--- PREPARED DATASET PREPROCESSING HAS STARTED ---")

        if "Typ vakcíny" in prepared_df.columns:
            prepared_df = prepared_df.drop(columns="Typ vakcíny", axis=1)

        prepared_df = self._add_wave(prepared_df, wave_name)
        prepared_df = self.align_prepared_to_original(original_df, prepared_df)
        prepared_df = self.rename_prepared_cols(prepared_df)
        prepared_df = self.select_prepared_columns(prepared_df)
        prepared_df = self.convert_prepared_binary_to_bool(prepared_df)
        prepared_df = self.convert_prepared_target_to_binary(prepared_df)

        return prepared_df

    def merge_original_and_prepared(
        self, original_df: pd.DataFrame, prepared_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge original and prepared dataframes on patient ID

        Args:
            original_df(pd.DataFrame): Original dataframe
            prepared_df(pd.DataFrame): Prepared dataframe

        Returns:
            Inner merged dataframe (original_df + prepared_df)
        """
        logging.info("Merging original and prepared dataframes")

        original_df = original_df.copy()
        prepared_df = prepared_df.copy()

        merged = pd.merge(
            left=original_df,
            right=prepared_df,
            on="ID",
            how="inner",
            suffixes=("_orig", "_prep"),
        )

        logging.info(
            "Patients in original: %d, prepared: %d, merged: %d",
            original_df["ID"].nunique(),
            prepared_df["ID"].nunique(),
            merged["ID"].nunique(),
        )

        return merged

    def parse_merged(
        self,
        merged_df: pd.DataFrame,
        parsing_cols: list[str] = ORIGINAL_MEASUREMENT_COLS,
        static_cols: list[str] = PARSING_STATIC_COLS,
        id_col: str = "ID",
    ) -> pd.DataFrame:
        """
        Parse tests columns and attach static patient attributes

        Args:
            merged_df(pd.DataFrame): Merged dataframe
            parsing_cols(list[str]): list of columns to parse
            static_cols(list[str]): list of static columns to attach
            id_col(str): Patient identifier column

        Returns:
            pd.DataFrame: Parsed dataframe
        """
        logging.info("Parsing merged dataframe")

        # Ensure duration is included in static columns if present
        if "duration" in merged_df.columns:
            static_cols = static_cols + ["duration"]

        # Parse merged dataframe
        return self.data_preparation.parse(
            merged_df=merged_df,
            parsing_cols=parsing_cols,
            static_cols=static_cols,
            id_col=id_col,
        )

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def pivot_parsed_with_static(
        self,
        parsed_df: pd.DataFrame,
        static_cols: list[str] = PIVOT_STATIC_COLS,
        index: list[str] = ["ID", "time"],
        tests_col: str = "variable",
        test_vals: str = "value",
        aggfunc: str = "last",
    ) -> pd.DataFrame:
        """
        Pivot parsed dataframe and attach static attributes.

        Args:
            parsed_df(pd.DataFrame): Parsed dataframe
            static_cols(list[str]): list of static columns to attach
            index(list[str]): list of columns to use as index
            tests_col(str): Tests column name
            test_vals(str): Test values column name
            aggfunc(str): Aggregation function for pivoting

        Returns:
            pd.DataFrame: Pivoted dataframe with attached static attributes
        """
        logging.info("Pivoting parsed dataframe")

        # Ensure duration is included in static columns if present
        if "duration" in parsed_df.columns:
            static_cols = static_cols + ["duration"]

        # Pivot parsed dataframe
        pivoted = self.data_preparation.pivot(
            parsed_df=parsed_df,
            index=index,
            tests_col=tests_col,
            test_vals=test_vals,
            aggfunc=aggfunc,
        )

        # Attach static attributes, TODO: add it as optional
        pivoted = self.data_preparation.attach_static_attributes(
            pivoted_df=pivoted, parsed_df=parsed_df, static_cols=static_cols
        )

        logging.info("Pivoted len: %s", len(pivoted))

        return pivoted

    def _assert_no_patient_leakage(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, patient_id: str = "ID"
    ) -> None:
        """
        Check that there is no patient leakage between train and test sets.

        Args:
            train_df (pd.DataFrame): Training dataframe.
            test_df (pd.DataFrame): Testing dataframe.
            patient_id (str): Patient identifier column.

        Raises:
            AssertionError: If patient leakage is detected.
        """
        overlap = set(train_df[patient_id]) & set(test_df[patient_id])
        if overlap:
            raise AssertionError(
                f"Patient leakage detected: {len(overlap)} patients between train and test sets"
            )

    def split_pivoted_to_train_test(
        self, pivoted_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split pivoted dataframe into train and test set before imputation

        Args:
            pivoted_df(pd.DataFrame): Pivoted dataframe with missing values

        Returns:
            tuple:
                - pivoted_train_df(pd.DataFrame): Pivoted train set
                - pivoted_test_df(pd.DataFrame): Pivoted test set
        """
        logging.info("Splitting pivoted dataframe with missing values")

        # Split using GroupShuffleSplit to avoid patient leakage
        groups = pivoted_df[GROUP_COL]
        cv = StratifiedGroupKFold(
            n_splits=self.config.train.n_splits,
            shuffle=self.config.train.shuffle,
            random_state=self.config.base.seed,
        )
        y_strat = pivoted_df[EVENT_COL]

        # Get train and test indices
        train_idx, test_idx = next(cv.split(X=pivoted_df, y=y_strat, groups=groups))
        pivoted_train_df = pivoted_df.iloc[train_idx].reset_index(drop=True)
        pivoted_test_df = pivoted_df.iloc[test_idx].reset_index(drop=True)

        # Ensure no patient leakage
        self._assert_no_patient_leakage(
            train_df=pivoted_train_df, test_df=pivoted_test_df, patient_id=GROUP_COL
        )

        return pivoted_train_df, pivoted_test_df

    def split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Split dataframe into x and y.

        Args:
            df(pd.DataFrame): Dataframe to split from.

        Returns:
            tuple:
                - x(pd.DataFrame): Features
                - y(pd.DataFrame): Event and duration
                - groups(pd.Series): Group identifiers
        """

        missing = {GROUP_COL, EVENT_COL, DURATION_COL, *self.feature_cols} - set(df.columns)

        assert not missing, f"Missing columns: {missing}"

        forbidden = {GROUP_COL, EVENT_COL, DURATION_COL}
        overlap = forbidden & set(self.feature_cols)
        assert not overlap, f"Forbidden columns in features: {overlap}"

        # Define feature columns and target columns
        x = df[self.feature_cols].copy()

        y = df[[EVENT_COL, DURATION_COL]].copy()

        groups = df[GROUP_COL].copy()

        return x, y, groups
