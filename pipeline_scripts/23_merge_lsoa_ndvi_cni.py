#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCENARIO_FILES = {
    "heatwave": "lsoa_hei_summary_heatwave.csv",
    "typical_day": "lsoa_hei_summary_typical_day.csv",
}


def add_quintiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add NDVI and CNI quintile labels (1-5) using ranked qcut."""
    df = df.copy()
    valid = df["ndvi_mean"].notna() & df["cni_28"].notna()
    df.loc[valid, "ndvi_quintile"] = pd.qcut(
        df.loc[valid, "ndvi_mean"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype("Int64")
    df.loc[valid, "cni_quintile"] = pd.qcut(
        df.loc[valid, "cni_28"].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype("Int64")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LSOA HEI summary with NDVI values for a given scenario."
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_FILES.keys()),
        default="heatwave",
        help="Scenario to merge (default: heatwave).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override results directory (default: <repo>/results).",
    )
    parser.add_argument(
        "--ndvi",
        default=None,
        help="Override NDVI summary CSV (default: results/ndvi_analysis/lsoa_ndvi_values.csv).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output path (default: results/ndvi_analysis/lsoa_ndvi_cni_merged_<scenario>.csv).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir) if args.results_dir else base_dir / "results"

    lsoa_path = results_dir / "inequality_analysis" / SCENARIO_FILES[args.scenario]
    ndvi_path = Path(args.ndvi) if args.ndvi else results_dir / "ndvi_analysis" / "lsoa_ndvi_values.csv"
    output_path = (
        Path(args.output)
        if args.output
        else results_dir / "ndvi_analysis" / f"lsoa_ndvi_cni_merged_{args.scenario}.csv"
    )

    if not lsoa_path.exists():
        raise FileNotFoundError(f"LSOA summary not found: {lsoa_path}")
    if not ndvi_path.exists():
        raise FileNotFoundError(f"NDVI summary not found: {ndvi_path}")

    lsoa_df = pd.read_csv(lsoa_path)
    ndvi_df = pd.read_csv(ndvi_path)

    merged = lsoa_df.merge(ndvi_df, on="lsoa11cd", how="left")
    merged = add_quintiles(merged)
    merged["scenario"] = args.scenario

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    missing_ndvi = merged["ndvi_mean"].isna().sum()
    print(f"Saved: {output_path}")
    print(f"Rows: {len(merged)} | Missing NDVI: {missing_ndvi}")


if __name__ == "__main__":
    main()



