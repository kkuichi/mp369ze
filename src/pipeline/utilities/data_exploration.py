"""
Module for data exploration utilities.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

import re

import pandas as pd
from IPython.display import display
from src.pipeline.constants import ADMISSION_COL, DISCHARGE_COL, EVENT_COL, GROUP_COL


class DataExploration:
    """Class for exploring dataframes."""

    COMORBIDITY_TO_BIOMARKERS = {
        "Hypertenzia": ["S-Na", "S-K", "S-Urea", "S-Kreat", "S-PBNP"],
        "Diabetes mellitus": ["S-Gluk", "S-Urea", "S-Kreat", "HGB", "S-CRP", "S-FER"],
        "Kardiovaskulárne ochorenia": [
            "S-CK",
            "S-CK-MB",
            "S-LD",
            "S-PBNP",
            "PT (INR)",
            "D-dimér HS",
        ],
        "Chronické respiračné ochorenia": ["P-Laktát", "SatO2 %", "S-CRP", "S-IL6"],
        "Renálne ochorenia": ["S-Kreat", "S-Urea", "S-Na", "S-K", "S-CL", "S-Alb"],
        "Pečeňové ochorenia": [
            "S-ALT",
            "S-AST",
            "S-GMT",
            "S-ALP",
            "S-Bil-T",
            "S-Bil-D",
            "S-Alb",
            "S-Chol",
        ],
        "Onkologické ochorenia": [
            "HGB",
            "WBC",
            "PLT",
            "S-CRP",
            "S-FER",
            "S-LD",
            "CD3+",
            "CD4+",
            "CD8+",
            "NK",
        ],
        "Imunosupresia": [
            "S-IgG",
            "S-IgA",
            "S-Ig M",
            "CD3+",
            "CD4+",
            "CD8+",
            "CD4+/CD8+",
            "CD19+",
            "NK",
            "S-VITD",
            "S-CRP",
            "S-IL6",
        ],
    }

    MEASUREMENT_REGEX = re.compile(r"\|\s*([\d,]+)")

    def head_tail(self, df: pd.DataFrame, n: int) -> None:
        """
        Shows the first and last n rows of a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to explore.
            n (int): Number of rows to display from the start and end.

        Returns:
            None
        """
        print(f"First {n} rows:")
        display(df.head(n=n))

        print(f"Last {n} rows:")
        display(df.tail(n=n))

    def show_nan(self, df: pd.DataFrame) -> None:
        """
        Displays a summary of NaN values in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.

        Returns:
            None
        """
        nan_counts = df.isna().sum()
        total_rows = df.shape[0]
        nan_columns = nan_counts[nan_counts > 0]

        if not nan_columns.empty:
            summary = pd.DataFrame(
                {
                    "Number of missing values": nan_columns,
                    "Percent (%)": (nan_columns / total_rows * 100).round(2),
                }
            ).sort_values(by="Percent (%)", ascending=False)

            display(summary)
        else:
            print("No missing values.")

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def select_biomarkers(
        self,
        df: pd.DataFrame,
        comorbidity: str,
        id_col: str = "ID",
        additional_cols: list[str] | None = None,
        extract_only_positives: bool = True,
    ) -> pd.DataFrame:
        """
        Selects biomarkers related to a specific comorbidity from the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            comorbidity (str): The comorbidity to filter by.
            id_col (str): The column name for the identifier.
            additional_cols ([list[str]]): Additional columns to include.
            extract_only_positives (bool): Whether to extract only positive cases.

        Returns:
            pd.DataFrame: A DataFrame containing the selected biomarkers.
        """
        if comorbidity not in self.COMORBIDITY_TO_BIOMARKERS:
            raise ValueError(f"Unknown comorbidity: {comorbidity}")

        biomarkers = self.COMORBIDITY_TO_BIOMARKERS[comorbidity]
        cols = [id_col, comorbidity] + biomarkers
        if additional_cols:
            cols += additional_cols

        mask = df[comorbidity] == (1 if extract_only_positives else 0)
        subset = df.loc[mask, cols]

        return subset

    # ruff: noqa: PLR0913
    def audit_daily_measurements(
        self,
        parsed_df: pd.DataFrame,
        pivoted_df: pd.DataFrame,
        id_col: str = GROUP_COL,
        test_col: str = "variable",
        measurement_time_col: str = "time",
        event_col: str = EVENT_COL,
        admit_col: str = ADMISSION_COL,
        discharge_col: str = DISCHARGE_COL,
    ) -> pd.DataFrame:
        """
        Raw daily measurements vs values kept after pivoting.

        Args:
            parsed_df (pd.DataFrame): long-format dataframe with raw measurements.
            pivoted_df (pd.DataFrame): pivoted dataframe with measurements.
            id_col (str): patient ID column.
            test_col (str): test name column in parsed_df.
            measurement_time_col (str): datetime column in both dataframes.
            event_col (str): event column in both dataframes.
            admit_col (str): admission date column in both dataframes.
            discharge_col (str): discharge date column in both dataframes.

        Returns:
            pd.DataFrame: audit dataframe with lost measurements.
        """

        # Raw counts
        raw = parsed_df.copy()
        raw["date"] = pd.to_datetime(raw[measurement_time_col]).dt.normalize()

        raw_counts = (
            raw.groupby(
                [id_col, event_col, admit_col, discharge_col, test_col, "date"],
                dropna=False,
            )
            .size()
            .to_frame("n_parsed_measurements")
            .reset_index()
        )

        # Pivot counts
        pivot = pivoted_df.copy()
        pivot["date"] = pd.to_datetime(pivot[measurement_time_col]).dt.normalize()

        pivot_long = pivot.melt(
            id_vars=[id_col, "date", event_col, admit_col, discharge_col],
            var_name=test_col,
            value_name="pivot_value",
        )

        pivot_long["n_in_pivot"] = pivot_long["pivot_value"].notna().astype(int)

        pivot_counts = (
            pivot_long.groupby([id_col, test_col, "date"], dropna=False)["n_in_pivot"]
            .sum()
            .reset_index()
        )

        # Merge
        audit = raw_counts.merge(pivot_counts, on=[id_col, test_col, "date"], how="left")

        audit["n_in_pivot"] = audit["n_in_pivot"].fillna(0).astype(int)
        audit["n_lost_measurements"] = audit["n_parsed_measurements"] - audit["n_in_pivot"]

        return audit

    def analyze_measurement_intensity(
        self,
        audit_df: pd.DataFrame,
        *,
        id_col: str = "ID",
        test_col: str = "variable",
        event_col: str = "Závažnosť priebehu ochorenia",
    ) -> dict[str, pd.DataFrame]:
        """
        Analyze relationship between daily measurement intensity and outcomes.

        Returns dictionary of analysis tables.
        """

        df = audit_df.copy()

        daily_max = df.groupby([id_col, "date"])["n_parsed_measurements"].max().reset_index()

        patient_max = (
            daily_max.groupby(id_col)["n_parsed_measurements"]
            .max()
            .reset_index(name="max_tests_per_day")
        )

        # attach outcome
        outcomes = df[[id_col, event_col]].drop_duplicates(subset=[id_col])

        patient_max = patient_max.merge(outcomes, on=id_col, how="left")

        # bucketize
        patient_max["tests_per_day_group"] = pd.cut(
            patient_max["max_tests_per_day"],
            bins=[0, 1, 2, 3, float("inf")],
            labels=["1", "2", "3", "4+"],
            right=True,
            include_lowest=True,
        )

        mortality_by_intensity = (
            patient_max.groupby("tests_per_day_group")[event_col]
            .agg(n_patients="count", n_deaths="sum")
            .reset_index()
        )

        mortality_by_intensity["patients_pct"] = (
            mortality_by_intensity["n_patients"] / mortality_by_intensity["n_patients"].sum() * 100
        ).round(2)

        mortality_by_intensity["death_pct"] = (
            mortality_by_intensity["n_deaths"] / mortality_by_intensity["n_patients"] * 100
        ).round(2)

        repeated_tests = df[df["n_parsed_measurements"] > 1]

        test_frequency = (
            repeated_tests.groupby(test_col)
            .agg(
                days_repeated=("date", "count"),
                total_measurements=("n_parsed_measurements", "sum"),
                avg_per_day=("n_parsed_measurements", "mean"),
            )
            .reset_index()
            .sort_values("total_measurements", ascending=False)
        )

        total_repeated_days = repeated_tests.shape[0]
        test_frequency["pct_of_repeated_days"] = (
            test_frequency["days_repeated"] / total_repeated_days * 100
        ).round(2)

        patient_repeated = (
            repeated_tests.groupby(id_col)
            .agg(
                n_days_repeated=("date", "nunique"),
                max_tests_per_day=("n_parsed_measurements", "max"),
            )
            .reset_index()
            .merge(outcomes, on=id_col, how="left")
            .sort_values("max_tests_per_day", ascending=False)
        )

        test_outcome = (
            repeated_tests.groupby([test_col, event_col])
            .size()
            .reset_index(name="n_days")
            .pivot(index=test_col, columns=event_col, values="n_days")
            .fillna(0)
            .reset_index()
        )

        return {
            "mortality_by_measurement_intensity": mortality_by_intensity,
            "repeated_test_frequency": test_frequency,
            "patients_with_repeated_tests": patient_repeated,
            "test_vs_outcome": test_outcome,
        }

    def count_measurements_per_test(self, df: pd.DataFrame, test_cols: list[str]) -> pd.DataFrame:
        """
        Count the number of measurements in each test column and return a summary dataset.

        Each measurement is a single entry separated by ';' inside a cell.
        This method is intended to be called before and after preprocessing steps
        that remove measurements, to track how many measurements were removed.

        Args:
            df (pd.DataFrame): The DataFrame containing test columns.
            test_cols (list[str]): List of test column names to count measurements in.

        Returns:
            pd.DataFrame: A DataFrame with columns:
                - test: test column name
                - n_measurements: total number of measurements across all patients
                - n_patients_with_test: number of patients who have at least one measurement
                - avg_per_patient: average number of measurements per patient (over all patients)
        """
        n_patients = len(df)
        rows = []

        for col in test_cols:
            if col not in df.columns:
                continue

            total = 0
            patients_with_test = 0

            for cell in df[col]:
                if pd.isna(cell):
                    continue
                parts = [p for p in str(cell).split(";") if p.strip()]
                if parts:
                    total += len(parts)
                    patients_with_test += 1

            rows.append(
                {
                    "test": col,
                    "n_measurements": total,
                    "n_patients_with_test": patients_with_test,
                    "avg_per_patient": (round(total / n_patients, 2) if n_patients > 0 else 0.0),
                }
            )

        return pd.DataFrame(rows)
