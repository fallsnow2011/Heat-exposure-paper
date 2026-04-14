#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parents[1]
IN_PATH = BASE_DIR / "results" / "ndvi_analysis" / "lsoa_ndvi_cni_merged_heatwave.csv"
OUT_PATH = BASE_DIR / "results" / "mechanism" / "ndvi_mechanism_controls_20260208.csv"
IMD_GPKG = BASE_DIR / "city_boundaries" / "Indices_of_Multiple_Deprivation_(IMD)_2019_8404722932957776031.gpkg"


def fit_and_extract(df: pd.DataFrame, formula: str, model_name: str) -> pd.DataFrame:
    mod = smf.ols(formula, data=df).fit(cov_type="HC3")
    ci = mod.conf_int()
    rows = []
    for term in mod.params.index:
        rows.append(
            {
                "model": model_name,
                "formula": formula,
                "term": term,
                "coef": float(mod.params[term]),
                "std_err": float(mod.bse[term]),
                "p_value": float(mod.pvalues[term]),
                "ci_low": float(ci.loc[term, 0]),
                "ci_high": float(ci.loc[term, 1]),
                "n_obs": int(mod.nobs),
                "r2": float(mod.rsquared),
                "adj_r2": float(mod.rsquared_adj),
            }
        )
    return pd.DataFrame(rows)


def load_lsoa_area_km2() -> pd.DataFrame:
    gdf = gpd.read_file(IMD_GPKG)
    lsoa_col = "lsoa11cd" if "lsoa11cd" in gdf.columns else "LSOA11CD"
    if lsoa_col not in gdf.columns:
        raise RuntimeError("Cannot find LSOA code column in IMD geometry.")
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=27700)
    gdf = gdf.to_crs(epsg=27700)
    out = gdf[[lsoa_col, "geometry"]].copy()
    out = out.rename(columns={lsoa_col: "lsoa11cd"})
    out["lsoa11cd"] = out["lsoa11cd"].astype(str)
    out["area_km2"] = out.geometry.area / 1_000_000.0
    return out[["lsoa11cd", "area_km2"]]


def main():
    df = pd.read_csv(IN_PATH)
    req = ["lsoa11cd", "ndvi_mean", "total_length", "IMD_Decile", "city", "shadow_mean", "cni_28"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    work = df.dropna(subset=req).copy()
    work["lsoa11cd"] = work["lsoa11cd"].astype(str)
    area_df = load_lsoa_area_km2()
    work = work.merge(area_df, on="lsoa11cd", how="left")
    work = work.dropna(subset=["area_km2"]).copy()
    # Match Methods definition: total road length per LSOA area (m km^-2).
    work["road_density_m_per_km2"] = work["total_length"] / (work["area_km2"] + 1e-9)
    work["IMD_Decile"] = pd.to_numeric(work["IMD_Decile"], errors="coerce")
    work = work.dropna(subset=["IMD_Decile"]).copy()

    # Model A: street shade mechanism
    f1 = "shadow_mean ~ ndvi_mean + road_density_m_per_km2 + IMD_Decile + C(city)"
    r1 = fit_and_extract(work, f1, "Model_A_shadow")

    # Model B: connectivity mechanism
    f2 = "cni_28 ~ ndvi_mean + road_density_m_per_km2 + IMD_Decile + C(city)"
    r2 = fit_and_extract(work, f2, "Model_B_cni28")

    out = pd.concat([r1, r2], ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()



