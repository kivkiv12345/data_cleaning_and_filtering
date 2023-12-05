from __future__ import annotations

from typing import Sequence

import pandas as pd
from pandas import DataFrame


def parse_excel(file_path: str, sheet: int | str, columns: Sequence = None, **kwargs) -> DataFrame:
    """
    Parse an Excel file and filter columns based on specified conditions.

    Parameters:
    - file_path (str): Path to the Excel file.
    - sheet (int | str): Which sheet to parse in the specified file.
    - columns (Sequence): List of column names to return. If None, return all columns.
    - **kwargs: Key-value pairs for filtering column values.

    Returns:
    - pd.DataFrame: A DataFrame containing the specified columns and filtered data.
    """
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path, header=None, sheet_name=sheet)

    # Find the header row by checking for the specified columns in **kwargs
    header_row = 0
    while not set(kwargs.keys()).issubset(df.loc[header_row].values):
        header_row += 1

    # Read the Excel file again with the correct header row
    df = pd.read_excel(file_path, header=header_row, sheet_name=sheet)

    # Filter columns if specified
    if columns:
        df = df[columns]

    # Apply filters based on kwargs
    for column, value in kwargs.items():
        df = df[df[column] == value]

    return df
