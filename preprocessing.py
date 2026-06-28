"""
preprocessing.py — Simple Data Cleaning Module
================================================
Handles missing values in the uploaded CSV before it goes to the model.

Kept intentionally simple (no advanced imputation, no ML-based filling):
  • Numeric columns   -> fill missing with the column MEDIAN
  • Text/categorical   -> fill missing with the column MODE (most frequent value)
  • Date column        -> drop rows where Date itself is missing (can't fix a missing date)
  • Fully empty cols   -> dropped (median/mode would be meaningless)

Usage:
    from preprocessing import clean_data
    df, report = clean_data(df)
"""

import pandas as pd


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean missing values in the dataframe using simple rules.

    Returns
    -------
    df      : cleaned DataFrame
    report  : dict summary of what was done (useful for logging / flash messages)
    """
    df = df.copy()
    report = {
        "rows_before": len(df),
        "missing_before": int(df.isna().sum().sum()),
        "dropped_empty_columns": [],
        "dropped_rows_missing_date": 0,
        "filled_numeric_median": {},
        "filled_categorical_mode": {},
    }

    # ── 1. Drop columns that are completely empty (median/mode impossible) ──
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        report["dropped_empty_columns"] = empty_cols

    # ── 2. Drop rows with a missing Date (can't forecast without a date) ───
    if "Date" in df.columns:
        before = len(df)
        df = df.dropna(subset=["Date"])
        report["dropped_rows_missing_date"] = before - len(df)

    # ── 3. Fill remaining missing values ────────────────────────────────────
    for col in df.columns:
        if col == "Date":
            continue
        if df[col].isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)
            report["filled_numeric_median"][col] = round(float(fill_value), 2)
        else:
            mode_series = df[col].mode()
            fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)
            report["filled_categorical_mode"][col] = fill_value

    df = df.reset_index(drop=True)
    report["rows_after"] = len(df)
    report["missing_after"] = int(df.isna().sum().sum())

    return df, report
