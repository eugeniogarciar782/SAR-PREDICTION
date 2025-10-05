# Cerro Prieto Dam Fill Predictor

This small project trains a RandomForest model to predict monthly dam fill percentage for Presa Cerro Prieto using monthly meteorological inputs.

Files:
- `dam_fill_predictor.py`: main script to train and predict
- `data/cerro_prieto_sample.csv`: small sample dataset (1958 monthly)
- `requirements.txt`: Python dependencies

Quick start (Windows PowerShell):


1. Create a Python environment and install requirements (use the Windows launcher `py -3`):

```powershell
py -3 -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Train model using the sample data:

```powershell
py -3 dam_fill_predictor.py --train data\cerro_prieto_sample.csv --model model.joblib
```

3. Predict (using the same file as example):

```powershell
py -3 dam_fill_predictor.py --predict data\cerro_prieto_sample.csv --model model.joblib --output preds.csv
```

Single-sample prediction (enter year, month and meteorological inputs):

```powershell
py -3 dam_fill_predictor.py --single --year 2025 --month 8 --evap_mm 26 --precip_mm 60 --max_temp 33 --min_temp 20
```

Notes:
- The sample dataset is synthetic and provided for quick testing. Replace with your full historical CSV (monthly rows since 1958) that includes columns: `year, month, evap_mm, precip_mm, max_temp, min_temp` and optionally `fill_pct` for training.
- The script uses simple feature engineering (mean temp, net mm, lag 1) and a RandomForest regressor. Improve by adding reservoir storage-area relations or hydrological routing for better accuracy.
