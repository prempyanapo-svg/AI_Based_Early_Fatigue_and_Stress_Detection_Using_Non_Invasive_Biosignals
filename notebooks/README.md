# AI-Based Early Fatigue & Stress Detection Using Non-Invasive Biosignals (SWELL-KW)

This repo trains a stress detector using SWELL-KW physiology features (HR/HRV/SCL). It includes:
- Personalized baseline (per participant)
- Early-warning trend detection
- Simple explanations

## Dataset
Place your CSV here (not included in repo):

```
AI-Based-Early-Fatigue-Stress-Detection-GitHub-PP999/data/D - Physiology features (HR_HRV_SCL - final).csv
```

### Important dataset notes
- Participant column in your export is `PP` (standardized to `Participant`).
- Numeric condition column is `C` (standardized to `Condition`).
- Text condition column `Condition` (e.g., N/R) is renamed to `Condition_text`.
- Value `999` is treated as **missing (NaN)** automatically.

## Install
```bash
pip install -r requirements.txt
pip install -e .
```

Windows: if `python` is not recognized, try the launcher:
```bash
py -m pip install -r requirements.txt
py -m pip install -e .
```

## Train
```bash
python -m stress_detector.train --data "data/D - Physiology features (HR_HRV_SCL - final).csv" --out outputs
```

Or:
```bash
python run_train.py --data "data/D - Physiology features (HR_HRV_SCL - final).csv" --out outputs
```

## Predict (Early warning + explanations)
```bash
python -m stress_detector.predict --model outputs/models/rf_stress_model.joblib   --data "data/D - Physiology features (HR_HRV_SCL - final).csv"   --save outputs/reports/predictions.csv
```

## Outputs
- `outputs/models/rf_stress_model.joblib`
- `outputs/reports/metrics.json`
- `outputs/reports/confusion_matrix.png`
- `outputs/reports/predictions.csv`
