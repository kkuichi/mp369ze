"""
Module for data preparation utilities in the pipeline.

This module provides:
- Identification and removal of invalid test values.
- Sorting and normalization of datetime values.
- Removal of test entries outside hospital stay.

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

from __future__ import annotations

import logging
import re
from re import Pattern

import pandas as pd
from src.pipeline.constants import GROUP_COL


class DataPreparation:
    """Class of helper methods for data preprocessing."""

    DEFAULT_VALID_PATTERN = re.compile(r"^\d+(,\d+)?$")
    DEFAULT_EXTRACT_PATTERN = r"\|\s*([^;]*);"

    @staticmethod
    def _split_cell(cell: object) -> list[str]:
        """Split cell content into individual measurement parts."""
        if pd.isna(cell):
            return []
        return [p.strip() for p in re.split(r";|_x000D_", str(cell)) if p.strip()]

    @staticmethod
    def _parse_cell_to_datetime_value(cell: object) -> list[tuple[pd.Timestamp, str]]:
        """
        Parse cell into (datetime, value) tuples.

        Args:
            cell (object): Raw cell value.

        Returns:
            list of (datetime, value) tuples.
        """
        parsed: list[tuple[pd.Timestamp, str]] = []

        for part in DataPreparation._split_cell(cell):
            if "|" not in part:
                continue

            t, val = part.split("|", 1)
            dt = pd.to_datetime(t.strip(), dayfirst=True, errors="coerce")
            if pd.notna(dt):
                parsed.append((dt, val.strip()))

        return parsed

    def identify_invalid_test_values(
        self,
        df: pd.DataFrame,
        columns: list[str],
        valid_pattern: Pattern = DEFAULT_VALID_PATTERN,
        extract_pattern: str = DEFAULT_EXTRACT_PATTERN,
    ) -> pd.DataFrame:
        """
        Identify invalid test values in selected columns.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (list[str]): Columns to scan.
            valid_pattern (Pattern): Regex defining valid values.
            extract_pattern (str): Regex for value extraction.

        Returns:
            pd.DataFrame with invalid value records.
        """
        logging.info("Identifying invalid test values.")

        invalid_entries: list[dict] = []

        for col in columns:
            for idx, cell in df[col].items():
                if not isinstance(cell, str):
                    continue

                for line in cell.splitlines():
                    match = re.search(extract_pattern, line)
                    if match:
                        value = match.group(1).strip()
                        if not valid_pattern.match(value):
                            invalid_entries.append(
                                {
                                    "index": idx,
                                    "column": col,
                                    "invalid_value": value,
                                    "original_cell": line,
                                }
                            )

        return pd.DataFrame(invalid_entries)

    @staticmethod
    def _clean_cell(
        cell: object, valid_pattern: Pattern, extract_pattern: str, empty_as_na: bool
    ) -> tuple[object, list[str]]:
        """
        Remove invalid measurement rows from a single cell and return removed values.

        Args:
            cell (object): Raw cell value.
            valid_pattern (Pattern): Regex defining valid values.
            extract_pattern (str): Regex for extraction.
            empty_as_na (bool): Replace empty cells with pd.NA.

        Returns:
            tuple:
                - Cleaned cell value
                - list of removed invalid lines
        """
        if not isinstance(cell, str):
            return cell, []

        cleaned_lines: list[str] = []
        removed_lines: list[str] = []

        for line in cell.splitlines():
            match = re.search(extract_pattern, line)
            if match:
                value = match.group(1).strip()
                if valid_pattern.match(value):
                    cleaned_lines.append(line)
                else:
                    removed_lines.append(line)
            else:
                cleaned_lines.append(line)

        cleaned_cell = "\n".join(cleaned_lines) if cleaned_lines else (pd.NA if empty_as_na else "")
        return cleaned_cell, removed_lines

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def clean_invalid_rows_in_cells(
        self,
        df: pd.DataFrame,
        columns: list[str],
        valid_pattern: Pattern = DEFAULT_VALID_PATTERN,
        extract_pattern: str = DEFAULT_EXTRACT_PATTERN,
        empty_as_na: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """
        Clean invalid measurement rows across selected columns.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (list[str]): Columns to clean.
            valid_pattern (Pattern): Regex defining valid values.
            extract_pattern (str): Regex for extraction.
            empty_as_na (bool): Replace empty cells with pd.NA.

        Returns:
            tuple:
                - Cleaned DataFrame
                - Dictionary of removed invalid values per column
        """
        df = df.copy()
        removed_values_dict: dict[str, list[str]] = {col: [] for col in columns}

        for col in columns:
            cleaned_cells = []
            for cell in df[col]:
                cleaned_cell, removed_lines = self._clean_cell(
                    cell, valid_pattern, extract_pattern, empty_as_na
                )
                cleaned_cells.append(cleaned_cell)
                removed_values_dict[col].extend(
                    [r.split(" | ")[1] for r in removed_lines if " | " in r]
                )
            df[col] = cleaned_cells

        return df, removed_values_dict

    def sort_datetime_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort time-varying measurements inside cells by datetime.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame
        """
        df = df.copy()

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].apply(self._sort_cell_by_datetime)

        return df

    @staticmethod
    def _sort_cell_by_datetime(cell: object) -> object:
        """Sort a single cell by datetime."""
        parsed = DataPreparation._parse_cell_to_datetime_value(cell)
        if not parsed:
            return cell

        parsed.sort(key=lambda x: x[0])

        return "; ".join(f"{dt.strftime('%d.%m.%Y %H:%M:%S')} | {val}" for dt, val in parsed)

    def normalize_test_datetimes_to_days(
        self, df: pd.DataFrame, test_cols: list[str]
    ) -> pd.DataFrame:
        """
        Normalize test datetimes to day precision.

        Args:
            df (pd.DataFrame): Input DataFrame.
            test_cols (list[str]): Test columns.

        Returns:
            pd.DataFrame
        """
        df = df.copy()

        for col in test_cols:
            df[col] = df[col].apply(self._normalize_cell_to_day)

        return df

    @staticmethod
    def _normalize_cell_to_day(cell: object) -> object:
        """Normalize datetimes inside a cell to midnight."""
        parsed = DataPreparation._parse_cell_to_datetime_value(cell)
        if not parsed:
            return cell

        normalized = [
            f"{dt.normalize().strftime('%d.%m.%Y %H:%M:%S')} | {val}" for dt, val in parsed
        ]

        return "; ".join(normalized) if normalized else pd.NA

    # ruff:noqa: PLR0913
    # pylint: disable=too-many-locals
    def remove_tests_outside_hospital_stay(
        self,
        df: pd.DataFrame,
        date_in_col: str = "Dátum príjmu",
        date_out_col: str = "Dátum prepustenia",
        id_col: str = "ID",
        save_removed_path: str | None = None,
        return_removed_values: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
        """
        Remove test entries outside hospital stay.

        Args:
            df (pd.DataFrame): Input DataFrame.
            date_in_col (str): Admission date column.
            date_out_col (str): Discharge date column.
            id_col (str): Patient ID column.
            save_removed_path (str | None): CSV path for removed logs.

        Returns:
            pd.DataFrame
        """
        df = df.copy()
        removed_logs: list[dict] = []
        removed_values_per_test: dict = {
            c: []
            for c in df.select_dtypes(include="object").columns
            if c not in {date_in_col, date_out_col, id_col}
        }

        test_cols = list(removed_values_per_test.keys())

        for idx, row in df.iterrows():
            d_in = pd.to_datetime(row[date_in_col], dayfirst=True, errors="coerce")
            d_out = pd.to_datetime(row[date_out_col], dayfirst=True, errors="coerce")
            pid = row.get(id_col, idx)

            if pd.isna(d_in) or pd.isna(d_out):
                continue

            for col in test_cols:
                kept, removed = self._filter_by_date_range(row[col], d_in, d_out)

                if removed:
                    removed_logs.append(
                        {
                            "ID": pid,
                            "column": col,
                            "removed_count": len(removed),
                            "removed_entries": "; ".join(removed),
                        }
                    )
                    removed_values_per_test[col].extend([r.split(" | ")[1] for r in removed])

                df.at[idx, col] = "; ".join(kept) if kept else pd.NA

        if save_removed_path and removed_logs:
            pd.DataFrame(removed_logs).to_csv(save_removed_path, index=False, encoding="utf-8-sig")

        if return_removed_values:
            return df, removed_values_per_test
        return df

    @staticmethod
    def _filter_by_date_range(
        cell: object, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[list[str], list[str]]:
        """Filter parsed test entries by date range."""
        kept, removed = [], []

        for dt, val in DataPreparation._parse_cell_to_datetime_value(cell):
            formatted = f"{dt.strftime('%d.%m.%Y %H:%M:%S')} | {val}"
            if start <= dt <= end:
                kept.append(formatted)
            else:
                removed.append(formatted)

        return kept, removed

    # pylint: disable=dangerous-default-value
    def parse(
        self,
        merged_df: pd.DataFrame,
        parsing_cols: list[str],
        static_cols: list[str],
        id_col: str = GROUP_COL,
    ) -> pd.DataFrame:
        """
        Parse time-series test columns into long format and
        attach static patient attributes to each measurement.

        Args:
            merged_df (pd.DataFrame): Merged (original + prepared) dataframe.
            parsing_cols (list[str]): Columns to parse.
            static_cols (list[str]): Static attribute columns.
            id_col (str): Patient ID column.

        Returns:
            pd.DataFrame in long format.
        """

        parsed_rows = []

        for col in parsing_cols:
            if col not in merged_df.columns:
                continue

            sub = merged_df[[id_col, col] + static_cols].dropna(subset=[col]).copy()
            sub[id_col] = sub[id_col].astype(int)

            # Split cell into individual measurements
            sub["raw"] = sub[col].astype(str).str.replace("_x000D_", "", regex=False).str.split(";")

            sub = sub.explode("raw", ignore_index=True)
            sub["raw"] = sub["raw"].str.strip()
            sub = sub[sub["raw"] != ""]

            # Split time | value
            split = sub["raw"].str.split("|", n=1, expand=True)
            sub["time"] = pd.to_datetime(split[0].str.strip(), dayfirst=True, errors="coerce")
            sub["value"] = pd.to_numeric(
                split[1].str.strip().str.replace(",", ".", regex=False), errors="coerce"
            )

            sub["variable"] = col

            # Drop only helper columns
            sub = sub.drop(columns=[col, "raw"])

            parsed_rows.append(sub)

        parsed_df = pd.concat(parsed_rows, ignore_index=True)

        return parsed_df

    def pivot(
        self,
        parsed_df: pd.DataFrame,
        index: list[str],
        tests_col: str,
        test_vals: str,
        aggfunc: str,
    ) -> pd.DataFrame:
        """
        Pivot table

        Args:
            parsed_df (pd.DataFrame): Parsed dataframe.
            index (list[str]): Index columns.
            tests_col (str): Column with test names.
            test_vals (str): Column with test values.
            aggfunc (str): Aggregation function.

        Returns:
            pd.DataFrame: Pivoted dataframe.
        """
        pivoted_df = parsed_df.pivot_table(
            index=index, columns=tests_col, values=test_vals, aggfunc=aggfunc
        ).reset_index()

        return pivoted_df

    def attach_static_attributes(
        self,
        pivoted_df: pd.DataFrame,
        parsed_df: pd.DataFrame,
        static_cols: list[str],
        id_col: str = GROUP_COL,
    ) -> pd.DataFrame:
        """
        Attach static (time-invariant) attributes to pivoted dataframe.

        Args:
            pivoted_df (pd.DataFrame): Pivoted dataframe.
            parsed_df (pd.DataFrame): Parsed dataframe.
            static_cols (list[str]): Static attribute columns.
            id_col (str): Patient ID column.

        Returns:
            pd.DataFrame: Dataframe with attached static attributes.
        """

        static_df = (
            parsed_df[[id_col] + static_cols].drop_duplicates(subset=[id_col]).set_index(id_col)
        )

        out = pivoted_df.copy()
        out = out.merge(static_df, left_on=id_col, right_index=True, how="left")

        return out
