#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INEQ_DIR = BASE_DIR / "results" / "inequality_analysis"
OUT_PATH = BASE_DIR / "results" / "sensitivity_analysis" / "london_exclusion_summary_20260208.csv"


def pop_weighted_gap(df: pd.DataFrame, hei_col: str = "hei_mean") -> float:
    sub = df.dropna(subset=[hei_col, "IMD_Decile", "TotPop"]).copy()
    poor = sub[sub["IMD_Decile"].isin([1, 2, 3])]
    rich = sub[sub["IMD_Decile"].isin([8, 9, 10])]
    if poor.empty or rich.empty:
        return np.nan
    poor_mean = np.average(poor[hei_col].to_numpy(dtype=float), weights=poor["TotPop"].to_numpy(dtype=float))
    rich_mean = np.average(rich[hei_col].to_numpy(dtype=float), weights=rich["TotPop"].to_numpy(dtype=float))
    return float(poor_mean - rich_mean)


def build_group_rows(typ: pd.DataFrame, hw: pd.DataFrame, label: str) -> dict:
    g_typ = pop_weighted_gap(typ)
    g_hw = pop_weighted_gap(hw)
    amp = g_hw / g_typ if pd.notna(g_typ) and g_typ != 0 else np.nan
    return {
        "group": label,
        "gap_typical_pop_weighted": g_typ,
        "gap_heatwave_pop_weighted": g_hw,
        "amplification_ratio_hw_over_typ": amp,
        "delta_gap_hw_minus_typ": g_hw - g_typ if pd.notna(g_hw) and pd.notna(g_typ) else np.nan,
        "n_lsoa_typical": int(len(typ)),
        "n_lsoa_heatwave": int(len(hw)),
    }


def main():
    typ = pd.read_csv(INEQ_DIR / "lsoa_hei_summary_typical_day.csv")
    hw = pd.read_csv(INEQ_DIR / "lsoa_hei_summary_heatwave.csv")

    rows = [
        build_group_rows(typ, hw, "All_cities"),
        build_group_rows(typ[typ["city"] != "London"], hw[hw["city"] != "London"], "Excluding_London"),
        build_group_rows(typ[typ["city"] == "London"], hw[hw["city"] == "London"], "London_only"),
    ]
    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()



