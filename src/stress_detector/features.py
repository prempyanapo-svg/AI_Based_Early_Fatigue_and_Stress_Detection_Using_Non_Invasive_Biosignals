import numpy as np
import pandas as pd


def numeric_feature_columns(df: pd.DataFrame, exclude: list) -> list:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in exclude]


def compute_personal_baseline(df: pd.DataFrame, participant_col: str, condition_col: str, neutral_code: int,
                              base_feature_cols: list) -> pd.DataFrame:
    """Compute per-participant baseline means using neutral condition rows."""
    neutral = df[df[condition_col] == neutral_code]
    if neutral.empty:
        return pd.DataFrame()
    return neutral.groupby(participant_col)[base_feature_cols].mean(numeric_only=True)


def add_delta_from_baseline(df: pd.DataFrame, baseline: pd.DataFrame, participant_col: str,
                            base_feature_cols: list, suffix: str = "_delta") -> pd.DataFrame:
    """Add delta features: feature - participant_baseline(feature)."""
    df = df.copy()
    for col in base_feature_cols:
        df[col + suffix] = np.nan

    if baseline is None or baseline.empty:
        return df.fillna(0)

    def _apply(row):
        pid = row[participant_col]
        if pid in baseline.index:
            base = baseline.loc[pid]
            for col in base_feature_cols:
                row[col + suffix] = row[col] - base[col]
        return row

    return df.apply(_apply, axis=1).fillna(0)


def make_labels(df: pd.DataFrame, condition_col: str, neutral_code: int, stress_codes: tuple) -> pd.Series:
    """Binary label: 0 neutral, 1 stressed."""
    return df[condition_col].apply(lambda c: 0 if c == neutral_code else (1 if c in stress_codes else 1))
