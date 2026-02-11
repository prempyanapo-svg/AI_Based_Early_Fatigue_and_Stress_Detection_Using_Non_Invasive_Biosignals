import pandas as pd


def load_swell_csv(path: str) -> pd.DataFrame:
    """Load SWELL-KW physiology feature CSV."""
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame, raw_participant_col: str, participant_col: str,
                        raw_condition_col: str, condition_col: str) -> pd.DataFrame:
    """Rename common SWELL-KW columns to consistent names."""
    ren = {}
    if raw_participant_col in df.columns and participant_col not in df.columns:
        ren[raw_participant_col] = participant_col
    if raw_condition_col in df.columns and condition_col not in df.columns:
        ren[raw_condition_col] = condition_col
    return df.rename(columns=ren) if ren else df


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().reset_index(drop=True)
