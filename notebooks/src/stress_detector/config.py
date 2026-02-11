from dataclasses import dataclass

@dataclass
class Config:
    # Path to the SWELL-KW physiology features CSV
    data_path: str = "data/D - Physiology features (HR_HRV_SCL - final).csv"

    # Column names used in this SWELL-KW export
    # CSV uses PP (participant id) and C (numeric condition code)
    participant_col: str = "Participant"      # internal standardized name
    raw_participant_col: str = "PP"           # observed in your CSV

    condition_col: str = "Condition"          # internal standardized name (numeric)
    raw_condition_col: str = "C"              # observed in your CSV

    # Conditions
    neutral_code: int = 1
    stress_codes: tuple = (2, 3)

    # Early-warning settings
    consecutive_threshold: int = 3

    # Model settings
    random_state: int = 42
    n_estimators: int = 300
