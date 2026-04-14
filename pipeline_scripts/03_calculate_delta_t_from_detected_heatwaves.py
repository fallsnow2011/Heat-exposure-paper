#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CITIES = ["London", "Birmingham", "Bristol", "Manchester", "Newcastle"]


def load_flagged_full_year(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"date", "Tmax_C", "is_heatwave"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {csv_path}")
    df["date"] = pd.to_datetime(df["date"])
    df["Tmax_C"] = pd.to_numeric(df["Tmax_C"], errors="coerce")
    # tolerate True/False as strings
    if df["is_heatwave"].dtype == object:
        df["is_heatwave"] = df["is_heatwave"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["is_heatwave"] = df["is_heatwave"].astype(bool)
    df = df.dropna(subset=["date", "Tmax_C"]).sort_values("date").reset_index(drop=True)
    return df


def compute_delta_t_for_city(df_full_year: pd.DataFrame, city: str, year: int) -> tuple[dict, pd.DataFrame]:
    summer = df_full_year[(df_full_year["date"].dt.year == year) & (df_full_year["date"].dt.month.isin([6, 7, 8]))].copy()
    if summer.empty:
        raise ValueError(f"No summer rows for {city} in year {year}.")

    heat = summer[summer["is_heatwave"]]
    typical = summer[~summer["is_heatwave"]]
    if heat.empty:
        raise ValueError(f"No heatwave days flagged for {city} in summer {year}.")
    if typical.empty:
        raise ValueError(f"No typical (non-heatwave) days left for {city} in summer {year}.")

    t_heatwave = float(heat["Tmax_C"].mean())
    t_typ_mean = float(typical["Tmax_C"].mean())
    t_typ_median = float(typical["Tmax_C"].median())

    result = {
        "city": city,
        "T_heatwave_mean": t_heatwave,
        "T_typical_mean": t_typ_mean,
        "T_typical_median": t_typ_median,
        "delta_t_mean": t_heatwave - t_typ_mean,
        "delta_t_median": t_heatwave - t_typ_median,
        "T_summer_mean": float(summer["Tmax_C"].mean()),
        "T_summer_max": float(summer["Tmax_C"].max()),
        "T_summer_min": float(summer["Tmax_C"].min()),
        "n_heatwave_days": int(heat.shape[0]),
        "n_typical_days": int(typical.shape[0]),
    }

    return result, summer


def main():
    parser = argparse.ArgumentParser(description="Compute 螖T from detected heatwaves (Scheme A).")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument(
        "--detection-dir",
        default="GEE_LST_Baseline/heatwave_detection",
        help="Folder created by scripts/gee_heatwave_pipeline.py",
    )
    parser.add_argument("--out-dir", default="GEE_LST_Baseline/delta_t")
    args = parser.parse_args()

    detection_dir = Path(args.detection_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for city in CITIES:
        src = detection_dir / f"ERA5Land_daily_Tmax_{args.year}_{city}_full_year_with_flags.csv"
        if not src.exists():
            raise SystemExit(
                f"Missing input for {city}: {src}\n"
                "Run first: python scripts/gee_heatwave_pipeline.py --year 2022"
            )
        df_full = load_flagged_full_year(src)
        res, df_summer = compute_delta_t_for_city(df_full, city=city, year=args.year)
        results.append(res)

        # Write the summer subset (kept compatible with FigS2_temperature_validation.py)
        df_out = df_summer.copy()
        df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
        df_out.to_csv(out_dir / f"daily_tmax_with_heatwave_flag_{city}.csv", index=False)

    summary = pd.DataFrame(results)
    summary.to_csv(out_dir / "delta_t_summary_all_cities.csv", index=False)

    print(f"Saved: {out_dir / 'delta_t_summary_all_cities.csv'}")
    print(f"Saved: {out_dir / 'daily_tmax_with_heatwave_flag_<City>.csv'}")


if __name__ == "__main__":
    main()




