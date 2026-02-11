import pandas as pd


def load_swell_csv(path: str) -> pd.DataFrame:
    """Load SWELL-KW physiology feature CSV.

    Fixes for common exports:
    - Treat 999 as missing value (NaN)
    - If file loads as a single column, retry with semicolon separator
    - Drop trailing-comma columns (Unnamed: x) and columns that are fully empty
    """
    df = pd.read_csv(path, na_values=[999, 999.0, "999"])
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=';', na_values=[999, 999.0, "999"])

    # Remove columns created by trailing commas
    df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed')]
    # Remove columns that are completely empty
    df = df.dropna(axis=1, how='all')
    return df


def standardize_columns(df: pd.DataFrame,
                        raw_participant_col: str, participant_col: str,
                        raw_condition_col: str, condition_col: str) -> pd.DataFrame:
    """Rename columns to a consistent schema.

    Your CSV example:
    - PP = participant id
    - C = numeric condition code (1/2/3)
    - Condition = text label (e.g., N/R)

    Standardization:
    - PP -> Participant
    - Condition (text) -> Condition_text
    - C -> Condition (numeric, used for ML label)
    """
    ren = {}

    # Participant column variants
    if participant_col not in df.columns:
        if raw_participant_col in df.columns:
            ren[raw_participant_col] = participant_col
        elif 'PP' in df.columns:
            ren['PP'] = participant_col
        elif 'P' in df.columns:
            ren['P'] = participant_col
        elif 'subject' in df.columns:
            ren['subject'] = participant_col

    # Condition conflict: keep numeric code as Condition
    if 'C' in df.columns and 'Condition' in df.columns:
        ren['Condition'] = 'Condition_text'
        ren['C'] = condition_col
    else:
        if raw_condition_col in df.columns and condition_col not in df.columns:
            ren[raw_condition_col] = condition_col

    return df.rename(columns=ren) if ren else df


def drop_missing(df: pd.DataFrame, required_cols=None) -> pd.DataFrame:
    """Drop rows only when critical columns are missing."""
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")
        df = df.dropna(subset=required_cols)
    return df.reset_index(drop=True)
