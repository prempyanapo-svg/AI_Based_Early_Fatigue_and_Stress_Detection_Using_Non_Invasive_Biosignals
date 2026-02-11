from dataclasses import dataclass

@dataclass

class Config:
    data_path: str = "data/D - Physiology features (HR_HRV_SCL - final).csv"

    # Your CSV uses PP and C
    participant_col: str = "Participant"
    raw_participant_col: str = "PP"

    # Use numeric block column for the label
    condition_col: str = "Condition"
    raw_condition_col: str = "C"

    neutral_code: int = 1
    stress_codes: tuple = (2, 3)

    consecutive_threshold: int = 3
    random_state: int = 42
    n_estimators: int = 300
