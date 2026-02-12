# AI-Based Early Fatigue & Stress Detection Using Non-Invasive Biosignals (SWELL-KW)

This repository implements a Python machine-learning pipeline to detect **stress / early fatigue risk** using **non-invasive physiological features** (HR/HRV/SCL).



## Precomputed outputs included
This repo can include precomputed results in `outputs/reports/` for easy review.
If you run the code, files in `outputs/reports/` may be overwritten (regenerated).

## Quick Start Windows

### Install
```bash
pip install -r requirements.txt
pip install -e .
```

### Train
```bash
python -m stress_detector.train --data "data/D - Physiology features (HR_HRV_SCL - final).csv" --out outputs
```

### Predict (creates predictions.csv)
```bash
python -m stress_detector.predict --model outputs/models/rf_stress_model.joblib   --data "data/D - Physiology features (HR_HRV_SCL - final).csv"   --save outputs/reports/predictions.csv
```

## Quick Start (macOS / Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python3 -m stress_detector.train --data "data/D - Physiology features (HR_HRV_SCL - final).csv" --out outputs
python3 -m stress_detector.predict --model "outputs/models/rf_stress_model.joblib"   --data "data/D - Physiology features (HR_HRV_SCL - final).csv"   --save "outputs/reports/predictions.csv"
```

### VS Code Tip (Play ▶️ button)
You can run the project using the **Play ▶️ button** in VS Code:

1. Open `run_train.py`
2. Click **Run Python File ▶️** (top-right)

Then do the same for:
- `run_predict.py`

### VS Code Windows Run Guide
See `VSCode_Windows_Run_Guide` for more info

## Dataset (SWELL-KW)

This project uses the **SWELL Knowledge Work (SWELL-KW)** dataset (physiology feature file: `D - Physiology features (HR_HRV_SCL - final).csv`).

### Where to get the dataset
Download from the official archive (DANS Data Station):
- DOI: https://ssh.datastations.nl/file.xhtml?fileId=189473&version=4.1

> The dataset is **not included** in this repository. Please download it from the official source.

### License / redistribution
The dataset is distributed under **CC BY-NC-SA 4.0** (Attribution, Non-Commercial, Share-Alike).

### How to place the dataset locally
```text
data/D - Physiology features (HR_HRV_SCL - final).csv
```

### CSV notes (your export)
- Participant column is `PP` (standardized to `Participant`).
- Numeric condition column is `C` (standardized to `Condition`).
- Text condition column `Condition` (e.g., N/R) is renamed to `Condition_text`.
- Value `999` is treated as missing (NaN).

### Citation
Kraaij, W., Koldijk, S., & Sappelli, M. (2014). *The SWELL Knowledge Work Dataset for Stress and User Modeling Research*. DANS Data Station Social Sciences and Humanities. https://doi.org/10.17026/DANS-X55-69ZP , https://ssh.datastations.nl/file.xhtml?fileId=189473&version=4.1 
