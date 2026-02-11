from dataclasses import dataclass

@dataclass
class Config:
    # Path to the SWELL-KW physiology features CSV
    data_path: str = "data/D - Physiology features (HR_HRV_SCL - final).csv"

    # Column names used in SWELL-KW feature file
    participant_col: str = "Participant"
    raw_participant_col: str = "subject"  # original name in many SWELL-KW files
    condition_col: str = "Condition"
    raw_condition_col: str = "C"          # block/condition code: 1 neutral, 2 time pressure, 3 interruptions

    # Conditions
    neutral_code: int = 1
    stress_codes: tuple = (2, 3)

    # Early-warning settings
    consecutive_threshold: int = 3  # warn if >= this many consecutive stress predictions

    # Model settings
    random_state: int = 42
    n_estimators: int = 300
