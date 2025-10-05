"""Compute monthly climatology from fill history and produce predictions CSV."""
import pandas as pd


def main():
    path = "data/fill_history.csv"
    df = pd.read_csv(path)

    # ensure month is numeric 1-12
    # here file uses numeric months already
    df["month"] = df["month"].astype(int)

    clim = df.groupby("month")["fill_pct"].agg(["mean", "count"]).reset_index()
    clim = clim.rename(columns={"mean": "mean_fill_pct", "count": "obs_count"})

    # write climatology
    clim.to_csv("data/monthly_climatology.csv", index=False)

    # produce predictions for each year-month present using climatology
    df_out = df.merge(clim[["month", "mean_fill_pct"]], on="month", how="left")
    df_out = df_out.rename(columns={"mean_fill_pct": "pred_fill_pct"})
    df_out.to_csv("data/preds_fill_climatology.csv", index=False)

    print("Wrote data/monthly_climatology.csv and data/preds_fill_climatology.csv")


if __name__ == "__main__":
    main()
