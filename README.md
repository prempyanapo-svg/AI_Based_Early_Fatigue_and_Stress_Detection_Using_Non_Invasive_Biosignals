# AI-Based Early Fatigue & Stress Detection Using Non-Invasive Biosignals

**Author:** Premkumar Natarajan  
**Date:** 25/01/2026

This repository implements a Python machine-learning pipeline to detect **stress / early fatigue risk** using **non-invasive physiological features** from the **SWELL-KW** dataset, with three project additions:

- **(C) Personalized baseline calibration** (each participant’s “normal” is different)
- **(A) Early-warning trend detection** (warn if stress stays high for multiple windows)
- **(B) Simple explanations** (e.g., “HRV decreased vs baseline”)

## Dataset
Place the SWELL-KW physiology feature file here:

```text
data/D - Physiology features (HR_HRV_SCL - final).csv
```

The condition code is stored in `C` (renamed to `Condition` by the code):
- `1` = neutral
- `2` = time pressure
- `3` = interruptions

## Install
```bash
pip install -r requirements.txt
```

## Train
```bash
python -m stress_detector.train --data "data/D - Physiology features (HR_HRV_SCL - final).csv" --out outputs
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

## Folder structure
```text
src/        → python package code
notebooks/  → EDA + experiments
docs/       → report summary
outputs/    → trained model + figures + results
```
