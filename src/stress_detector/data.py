
import pandas as pd


def load_swell_csv(path: str) -> pd.DataFrame:
    """
    Load SWELL-KW CSV and handle:
    - 999 as missing value (NaN)
    - trailing commas -> Unnamed columns
    - semicolon-separated files (fallback)
    """
    df = pd.read_csv(path, na_values=[999, 999.0, "999"])

    # If it loads as a single column, retry with semicolon
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";", na_values=[999, 999.0, "999"])

    # Remove columns created by trailing commas (Unnamed: x)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.dropna(axis=1, how="all")

    return df


def standardize_columns(df: pd.DataFrame,
                        raw_participant_col: str, participant_col: str,
                        raw_condition_col: str, condition_col: str) -> pd.DataFrame:
    """
    Make the dataset consistent for the pipeline:
    - PP -> Participant
    - If both C (numeric) and Condition (text) exist:
        Condition -> Condition_text
        C -> Condition
    """
    ren = {}

    # Participant column variants
    if participant_col not in df.columns:
        if raw_participant_col in df.columns:
            ren[raw_participant_col] = participant_col
        elif "PP" in df.columns:
            ren["PP"] = participant_col
        elif "P" in df.columns:
            ren["P"] = participant_col
        elif "subject" in df.columns:
            ren["subject"] = participant_col

    # Handle the common conflict: both 'C' and 'Condition' exist
    # We want numeric C as the ML label called 'Condition'
    if "C" in df.columns and "Condition" in df.columns:
        ren["Condition"] = "Condition_text"
        ren["C"] = condition_col  # condition_col should be "Condition"
    else:
        # Normal renaming case
        if raw_condition_col in df.columns and condition_col not in df.columns:
            ren[raw_condition_col] = condition_col

    return df.rename(columns=ren) if ren else df


def drop_missing(df: pd.DataFrame, required_cols=None) -> pd.DataFrame:
    """
    Drop rows only if critical columns are missing.
    """
    if required_cols:
        # Safety check: show clear error if columns don't exist
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

        df = df.dropna(subset=required_cols)

    return df.reset_index(drop=True)
