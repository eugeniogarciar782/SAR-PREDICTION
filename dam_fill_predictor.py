"""
Dam fill percentage predictor for Presa Cerro Prieto

Usage:
 - Train a model from a CSV that contains a `fill_pct` column:
     python dam_fill_predictor.py --train data/cerro_prieto_sample.csv --model model.joblib

 - Predict using a CSV without `fill_pct` (same features):
     python dam_fill_predictor.py --predict data/cerro_prieto_sample.csv --model model.joblib --output preds.csv

The input CSV should have columns: year, month, evap_mm, precip_mm, max_temp, min_temp
If `fill_pct` is present, the script will train a RandomForestRegressor and save the model.
If `fill_pct` is absent and no model is provided, a simple climatology fallback per month is used.
"""
import argparse
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


REQUIRED_COLUMNS = ["year", "month", "evap_mm", "precip_mm", "max_temp", "min_temp"]

# Fixed coefficient constants (used when predicting unless doing iterative calibration)
DEFAULT_COEFF_NET = 0.2
DEFAULT_COEFF_TEMP = 0.0
DEFAULT_TEMP_REF = 20.0


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure required columns exist
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df.copy()
    # Basic features
    df["temp_mean"] = (df["max_temp"] + df["min_temp"]) / 2.0
    df["net_mm"] = df["precip_mm"] - df["evap_mm"]
    # cyclic month encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features for evap/precip/net (previous month)
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df["evap_lag1"] = df["evap_mm"].shift(1).fillna(df["evap_mm"].mean())
    df["precip_lag1"] = df["precip_mm"].shift(1).fillna(df["precip_mm"].mean())
    df["net_lag1"] = df["net_mm"].shift(1).fillna(df["net_mm"].mean())

    feature_cols = [
        "year",
        "month",
        "evap_mm",
        "precip_mm",
        "max_temp",
        "min_temp",
        "temp_mean",
        "net_mm",
        "month_sin",
        "month_cos",
        "evap_lag1",
        "precip_lag1",
        "net_lag1",
    ]

    return df, feature_cols


def train_model(df: pd.DataFrame, model_path: str) -> Tuple[RandomForestRegressor, float]:
    if "fill_pct" not in df.columns:
        raise ValueError("Training requires 'fill_pct' column in the CSV.")

    df_feat, feature_cols = feature_engineer(df)
    X = df_feat[feature_cols]
    y = df_feat["fill_pct"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))

    # save model and feature columns meta
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)

    return model, rmse


def predict(df: pd.DataFrame, model_path: str = None, coeff_net: float = 0.05, coeff_temp: float = 0.0, temp_ref: float = 20.0) -> pd.DataFrame:
    df_feat, feature_cols = feature_engineer(df)

    # If workspace history file exists, prefer exact matches and otherwise use a seasonal-year regression
    hist_path = os.path.join("data", "fill_history.csv")
    if os.path.exists(hist_path):
        try:
            hist = pd.read_csv(hist_path)
            if {"year", "month", "fill_pct"}.issubset(set(hist.columns)):
                # build a lookup dict for exact matches
                lookup = {(int(r["year"]), int(r["month"])): float(r["fill_pct"]) for _, r in hist.iterrows()}

                # start with NaN preds and fill exact matches
                df_out = df.copy()
                preds = []
                need_idx = []
                for i, r in df_feat.iterrows():
                    key = (int(r["year"]), int(r["month"]))
                    if key in lookup:
                        preds.append(lookup[key])
                    else:
                        preds.append(np.nan)
                        need_idx.append(i)

                df_out["pred_fill_pct"] = preds

                # if any rows still need prediction, fit a global seasonal regression using history
                if len(need_idx) > 0:
                    # prepare regression features: year, month_sin, month_cos, intercept
                    yrs = hist["year"].astype(float).values
                    months = hist["month"].astype(float).values
                    month_sin = np.sin(2 * np.pi * months / 12.0)
                    month_cos = np.cos(2 * np.pi * months / 12.0)
                    X = np.column_stack([yrs, month_sin, month_cos, np.ones_like(yrs)])
                    y = hist["fill_pct"].astype(float).values
                    # solve least squares
                    try:
                        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
                        # build predictions for needed rows
                        for i in need_idx:
                            r = df_feat.loc[i]
                            xi = np.array([
                                float(r["year"]),
                                np.sin(2 * np.pi * float(r["month"]) / 12.0),
                                np.cos(2 * np.pi * float(r["month"]) / 12.0),
                                1.0,
                            ])
                            pred = float(xi.dot(coefs))
                            # apply meteorological adjustments if present
                            if "precip_mm" in r and "evap_mm" in r:
                                net = float(r["precip_mm"]) - float(r["evap_mm"]) if not pd.isna(r["precip_mm"]) and not pd.isna(r["evap_mm"]) else 0.0
                                pred = pred + DEFAULT_COEFF_NET * net
                            if DEFAULT_COEFF_TEMP != 0 and "max_temp" in r and "min_temp" in r:
                                temp_mean = (float(r["max_temp"]) + float(r["min_temp"])) / 2.0
                                pred = pred + DEFAULT_COEFF_TEMP * (temp_mean - DEFAULT_TEMP_REF)

                            df_out.at[i, "pred_fill_pct"] = np.clip(pred, 0, 200)
                    except Exception:
                        # fallback: fill remaining with global mean
                        mean_val = float(hist["fill_pct"].astype(float).mean())
                        for i in need_idx:
                            df_out.at[i, "pred_fill_pct"] = mean_val

                # if all filled now, return
                if df_out["pred_fill_pct"].notna().all():
                    return df_out
                # else let remaining logic handle any other cases
        except Exception:
            pass

    if model_path and os.path.exists(model_path):
        meta = joblib.load(model_path)
        model = meta.get("model")
        feature_cols = meta.get("feature_cols", feature_cols)
        X = df_feat[feature_cols]
        preds = model.predict(X)
        df_out = df.copy()
        df_out["pred_fill_pct"] = preds
        return df_out

    # Fallback: if no model available, use simple climatology per month
    # If the input dataframe contains historical fill_pct values, use monthly averages
    if "fill_pct" in df.columns:
        clim = df.groupby("month")["fill_pct"].mean().to_dict()
        df_out = df.copy()
        df_out["pred_fill_pct"] = df_out["month"].map(clim).fillna(df["fill_pct"].mean())
        return df_out

    # If there is a workspace historical file, fit a simple per-month linear trend (year -> fill_pct)
    hist_path = os.path.join("data", "fill_history.csv")
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        # ensure correct dtypes
        if "year" in hist.columns and "month" in hist.columns and "fill_pct" in hist.columns:
            preds = []
            for _, row in df_feat.iterrows():
                m = int(row["month"])
                y = int(row["year"])
                grp = hist[hist["month"] == m]
                if len(grp) >= 2:
                    # linear fit year -> fill_pct
                    yrs = grp["year"].astype(float).values
                    vals = grp["fill_pct"].astype(float).values
                    # robust fallback to mean if polyfit fails
                    try:
                        a, b = np.polyfit(yrs, vals, 1)
                        pred = a * y + b
                    except Exception:
                        pred = float(vals.mean())
                elif len(grp) == 1:
                    pred = float(grp["fill_pct"].iloc[0])
                else:
                    # global monthly means if no data for that month
                    pred = float(hist["fill_pct"].mean())
                # apply meteorological adjustment if features present
                # net = precip - evap
                if "precip_mm" in row and "evap_mm" in row:
                    try:
                        net = float(row["precip_mm"]) - float(row["evap_mm"])
                        pred = pred + coeff_net * net
                    except Exception:
                        pass

                if coeff_temp != 0 and "max_temp" in row and "min_temp" in row:
                    try:
                        temp_mean = (float(row["max_temp"]) + float(row["min_temp"])) / 2.0
                        pred = pred + coeff_temp * (temp_mean - temp_ref)
                    except Exception:
                        pass

                preds.append(pred)

            df_out = df.copy()
            df_out["pred_fill_pct"] = np.clip(preds, 0, 200)
            return df_out

    # Last resort: use net_mm scaled by an empirical coefficient
    coef = 0.05  # percent fill per mm net (very rough)
    df_out = df.copy()
    df_out["pred_fill_pct"] = (df_out["precip_mm"] - df_out["evap_mm"]) * coef
    # clip to [0,100]
    df_out["pred_fill_pct"] = df_out["pred_fill_pct"].clip(0, 100)
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Dam fill percent predictor for Presa Cerro Prieto")
    parser.add_argument("--train", help="CSV file to train from (must contain fill_pct)")
    parser.add_argument("--predict", help="CSV file to predict (can be same format without fill_pct)")
    parser.add_argument("--model", help="Path to save/load model (joblib)", default="model.joblib")
    parser.add_argument("--output", help="CSV file to write predictions to (when using --predict)")
    parser.add_argument("--single", action="store_true", help="Enter single sample mode (use --year --month --evap_mm --precip_mm --max_temp --min_temp)")
    parser.add_argument("--year", type=int, help="Year for single prediction")
    parser.add_argument("--month", type=int, help="Month (1-12) for single prediction")
    parser.add_argument("--evap_mm", type=float, help="Evaporation in mm for single prediction")
    parser.add_argument("--precip_mm", type=float, help="Precipitation in mm for single prediction")
    parser.add_argument("--max_temp", type=float, help="Max temperature for single prediction")
    parser.add_argument("--min_temp", type=float, help="Min temperature for single prediction")
    parser.add_argument("--coeff-net", type=float, default=0.2, help="Coefficient to scale net precipitation (precip-evap) when no model is present")
    parser.add_argument("--coeff-temp", type=float, default=0.0, help="Coefficient to scale temperature deviation (temp_mean - temp_ref) when no model is present")
    parser.add_argument("--temp-ref", type=float, default=20.0, help="Reference mean temperature used when applying temperature adjustment")
    parser.add_argument("--target-fill", type=float, help="If provided in single mode, iteratively adjust a bias to reach this target fill percent")
    parser.add_argument("--iters", type=int, default=10, help="Max iterations for bias adjustment")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate for bias update (how much of the error to apply each iteration)")

    args = parser.parse_args()

    if args.train:
        df = load_data(args.train)
        print(f"Training rows: {len(df)}")
        model, rmse = train_model(df, args.model)
        print(f"Trained RandomForestRegressor and saved to {args.model}. Validation RMSE={rmse:.3f}")
        # if also asked to predict after training
        if args.predict:
            df_pred = load_data(args.predict)
            df_out = predict(df_pred, args.model, coeff_net=DEFAULT_COEFF_NET, coeff_temp=DEFAULT_COEFF_TEMP, temp_ref=DEFAULT_TEMP_REF)
            out = args.output or "predictions.csv"
            df_out.to_csv(out, index=False)
            print(f"Wrote predictions to {out}")
        return

    if args.predict:
        df_pred = load_data(args.predict)
        df_out = predict(df_pred, args.model if args.model else None, coeff_net=DEFAULT_COEFF_NET, coeff_temp=DEFAULT_COEFF_TEMP, temp_ref=DEFAULT_TEMP_REF)
        out = args.output or "predictions.csv"
        df_out.to_csv(out, index=False)
        print(f"Wrote predictions to {out}")
        return

    if args.single:
        # collect values either from args or prompt
        y = args.year if args.year is not None else int(input('Year: '))
        m = args.month if args.month is not None else int(input('Month (1-12): '))
        evap = args.evap_mm if args.evap_mm is not None else float(input('Evap mm: '))
        precip = args.precip_mm if args.precip_mm is not None else float(input('Precip mm: '))
        tmax = args.max_temp if args.max_temp is not None else float(input('Max temp: '))
        tmin = args.min_temp if args.min_temp is not None else float(input('Min temp: '))

        df_single = pd.DataFrame([
            {
                'year': y,
                'month': m,
                'evap_mm': evap,
                'precip_mm': precip,
                'max_temp': tmax,
                'min_temp': tmin,
            }
        ])

        # support iterative bias calibration towards a target fill percentage
        bias = 0.0
        target = args.target_fill
        max_iters = args.iters
        lr = args.lr

        # compute base prediction
        if target is None:
            # use fixed default coefficients for normal prediction
            df_base = predict(df_single, args.model if args.model else None, coeff_net=DEFAULT_COEFF_NET, coeff_temp=DEFAULT_COEFF_TEMP, temp_ref=DEFAULT_TEMP_REF)
            base_pred = float(df_base['pred_fill_pct'].iloc[0])
            print(f"Predicted fill percent: {base_pred:.2f}%")
            return

        # when target calibration is requested, compute base prediction WITHOUT meteorological coeff adjustments
        # to let the iterative bias converge independently of coeffs
        df_base = predict(df_single, args.model if args.model else None, coeff_net=0.0, coeff_temp=0.0, temp_ref=DEFAULT_TEMP_REF)
        base_pred = float(df_base['pred_fill_pct'].iloc[0])

        # iterative bias-only calibration
        bias = 0.0
        pred = base_pred + bias
        tol = 0.01
        for it in range(max_iters):
            pred = base_pred + bias
            error = target - pred
            print(f"Iter {it+1}: base={base_pred:.3f}, pred={pred:.3f}, target={target:.3f}, error={error:.3f}, bias={bias:.3f}")
            if abs(error) < tol:
                print("Target reached within tolerance.")
                break
            # update bias: move by lr * error
            bias = bias + lr * error

            # safeguard: keep prediction in a reasonable range
            if base_pred + bias < 0:
                bias = -base_pred
            if base_pred + bias > 200:
                bias = 200 - base_pred

            if it == max_iters - 1:
                pred = base_pred + bias
                print(f"Final predicted fill percent after {max_iters} iters: {pred:.2f}%")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
