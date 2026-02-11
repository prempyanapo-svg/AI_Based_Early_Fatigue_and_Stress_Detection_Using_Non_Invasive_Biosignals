# Running the Project in VS Code (Windows)

1) Open the project folder in VS Code.

2) Put the dataset CSV here:
```
data/D - Physiology features (HR_HRV_SCL - final).csv
```

3) Install:
```powershell
py -m pip install -r requirements.txt
py -m pip install -e .
```

If `py` is not available, use your python.exe path (example):
```powershell
$pyexe = "C:\Users\<you>\AppData\Local\Microsoft\WindowsApps\python3.13.exe"
& $pyexe -m pip install -r requirements.txt
& $pyexe -m pip install -e .
```

4) Train:
```powershell
py -m stress_detector.train --data "data/D - Physiology features (HR_HRV_SCL - final).csv" --out outputs
```

5) Predict:
```powershell
py -m stress_detector.predict --model outputs/models/rf_stress_model.joblib --data "data/D - Physiology features (HR_HRV_SCL - final).csv" --save outputs/reports/predictions.csv
```

Tip: Use Run & Debug with `.vscode/launch.json` for one-click Train/Predict.
