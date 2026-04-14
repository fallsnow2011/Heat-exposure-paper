#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
HEAT_DIR = BASE_DIR / "results" / "heat_exposure"
INEQ_DIR = BASE_DIR / "results" / "inequality_analysis"
OUT_DIR = BASE_DIR / "results" / "policy_network"
OUT_FIG = BASE_DIR / "paper-draft-tex" / "Supplementary_Fig_S6_CoolShare_vs_CNI_curves.pdf"
IMD_GPKG = BASE_DIR / "city_boundaries" / "Indices_of_Multiple_Deprivation_(IMD)_2019_8404722932957776031.gpkg"

CITIES = ["London", "Birmingham", "Manchester", "Bristol", "Newcastle"]
THRESHOLDS = np.linspace(20, 45, 51)
ALPHA_B = 0.6
ALPHA_V = 0.8
DELTA_T_V = 2.0
VEG_ADD = 0.10

LONDON_BOROUGHS = [
    "City of London",
    "Barking and Dagenham",
    "Barnet",
    "Bexley",
    "Brent",
    "Bromley",
    "Camden",
    "Croydon",
    "Ealing",
    "Enfield",
    "Greenwich",
    "Hackney",
    "Hammersmith and Fulham",
    "Haringey",
    "Harrow",
    "Havering",
    "Hillingdon",
    "Hounslow",
    "Islington",
    "Kensington and Chelsea",
    "Kingston upon Thames",
    "Lambeth",
    "Lewisham",
    "Merton",
    "Newham",
    "Redbridge",
    "Richmond upon Thames",
    "Southwark",
    "Sutton",
    "Tower Hamlets",
    "Waltham Forest",
    "Wandsworth",
    "Westminster",
]

CITY_LADS = {
    "London": LONDON_BOROUGHS,
    "Birmingham": ["Birmingham"],
    "Bristol": ["Bristol, City of"],
    "Manchester": ["Manchester"],
    "Newcastle": ["Newcastle upon Tyne"],
}


def load_cni_function():
    path = BASE_DIR / "scripts" / "25_b_lite_robustness_checks.py"
    spec = importlib.util.spec_from_file_location("b_lite", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    # Register module before execution to avoid dataclass type-resolution issues.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod.calculate_cni_curve


def curves_from_roads(roads: gpd.GeoDataFrame, hei_col: str, calc_cni_curve):
    cool, cni, tcni, n_valid = calc_cni_curve(
        roads_gdf=roads,
        values=roads[hei_col].to_numpy(dtype=float),
        thresholds=THRESHOLDS,
        snap_m=1.0,
    )
    return cool, cni, float(tcni), int(n_valid)


def load_imd_with_area() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(IMD_GPKG)
    lsoa_col = "lsoa11cd" if "lsoa11cd" in gdf.columns else "LSOA11CD"
    lad_col = "LADnm" if "LADnm" in gdf.columns else "LADNM"
    gdf = gdf.rename(columns={lsoa_col: "lsoa11cd", lad_col: "LADnm"})
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=27700)
    gdf = gdf.to_crs(epsg=27700)
    gdf["area_km2"] = gdf.geometry.area / 1_000_000.0
    gdf["city"] = None
    for city, lads in CITY_LADS.items():
        gdf.loc[gdf["LADnm"].isin(lads), "city"] = city
    return gdf[gdf["city"].notna()][["lsoa11cd", "city", "area_km2", "geometry"]]


def compute_baseline_curves(calc_cni_curve):
    rows = []
    for city in CITIES:
        for scenario in ("typical_day", "heatwave"):
            path = HEAT_DIR / f"{city}_roads_hei_improved_{scenario}.gpkg"
            roads = gpd.read_file(path)
            cool, cni, tcni, n_valid = curves_from_roads(roads, "hei_improved", calc_cni_curve)
            for t, cs, cn in zip(THRESHOLDS, cool, cni):
                rows.append(
                    {
                        "city": city,
                        "scenario": scenario,
                        "threshold": float(t),
                        "coolshare": float(cs),
                        "cni": float(cn),
                        "tcni": tcni,
                        "n_valid_roads": n_valid,
                    }
                )
    return pd.DataFrame(rows)


def assign_road_lsoa(roads: gpd.GeoDataFrame, imd_city: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    roads = roads.copy()
    cent = roads.geometry.centroid
    cent_gdf = gpd.GeoDataFrame({"road_idx": roads.index}, geometry=cent, crs=roads.crs)
    joined = gpd.sjoin(cent_gdf, imd_city[["lsoa11cd", "geometry"]], how="left", predicate="within")
    roads["lsoa11cd"] = joined.set_index("road_idx")["lsoa11cd"].astype(str)
    return roads


def compute_policy_network(calc_cni_curve):
    lsoa = pd.read_csv(INEQ_DIR / "lsoa_hei_summary_heatwave.csv")
    imd = load_imd_with_area()
    lsoa["lsoa11cd"] = lsoa["lsoa11cd"].astype(str)
    lsoa = lsoa.merge(imd[["lsoa11cd", "area_km2"]], on="lsoa11cd", how="left")
    lsoa["road_density"] = lsoa["total_length"] / (lsoa["area_km2"] + 1e-6)

    hei_med = float(lsoa["hei_mean"].median())
    veg_med = float(lsoa["shadow_vegetation_mean"].median())
    dens_q75 = float(lsoa["road_density"].quantile(0.75))

    targets = {
        "baseline": set(),
        "S1_citywide": set(lsoa["lsoa11cd"].astype(str)),
        "S2_corridors": set(lsoa.loc[lsoa["road_density"] >= dens_q75, "lsoa11cd"].astype(str)),
        "S3_equity_first": set(
            lsoa.loc[
                (lsoa["IMD_Decile"].isin([1, 2, 3]))
                & (lsoa["hei_mean"] > hei_med)
                & (lsoa["shadow_vegetation_mean"] < veg_med),
                "lsoa11cd",
            ].astype(str)
        ),
    }

    rows = []
    for city in CITIES:
        roads = gpd.read_file(HEAT_DIR / f"{city}_roads_hei_improved_heatwave.gpkg")
        imd_city = imd[imd["city"] == city].copy()
        roads = assign_road_lsoa(roads, imd_city)

        _, cni_base, tcni_base, _ = curves_from_roads(roads, "hei_improved", calc_cni_curve)
        cni35_base = float(cni_base[np.argmin(np.abs(THRESHOLDS - 35.0))])

        for scenario, target_lsoas in targets.items():
            if scenario == "baseline":
                hei_new = roads["hei_improved"].to_numpy(dtype=float)
            else:
                sv = roads["shadow_vegetation_avg"].to_numpy(dtype=float)
                sb = roads["shadow_building_avg"].to_numpy(dtype=float)
                lst = roads["lst"].to_numpy(dtype=float)
                in_target = roads["lsoa11cd"].astype(str).isin(target_lsoas).to_numpy()
                sv_new = np.clip(sv + VEG_ADD * in_target.astype(float), 0.0, 1.0)
                hei_new = lst * (1.0 - ALPHA_B * sb - ALPHA_V * sv_new) - DELTA_T_V * sv_new
            roads_tmp = roads.copy()
            roads_tmp["hei_new"] = hei_new
            _, cni_new, tcni_new, n_valid = curves_from_roads(roads_tmp, "hei_new", calc_cni_curve)
            cni35_new = float(cni_new[np.argmin(np.abs(THRESHOLDS - 35.0))])
            rows.append(
                {
                    "city": city,
                    "scenario": scenario,
                    "n_valid_roads": n_valid,
                    "tcni_baseline": tcni_base,
                    "tcni_after": tcni_new,
                    "tcni_change_pct": ((tcni_new - tcni_base) / tcni_base * 100.0) if tcni_base != 0 else np.nan,
                    "cni35_baseline": cni35_base,
                    "cni35_after": cni35_new,
                }
            )
    out = pd.DataFrame(rows)
    weighted_rows = []
    for scenario, grp in out.groupby("scenario"):
        w = grp["n_valid_roads"].to_numpy(dtype=float)
        weighted_rows.append(
            {
                "city": "All_cities_weighted",
                "scenario": scenario,
                "n_valid_roads": int(w.sum()),
                "tcni_baseline": float(np.average(grp["tcni_baseline"], weights=w)),
                "tcni_after": float(np.average(grp["tcni_after"], weights=w)),
                "tcni_change_pct": float(
                    (np.average(grp["tcni_after"], weights=w) - np.average(grp["tcni_baseline"], weights=w))
                    / np.average(grp["tcni_baseline"], weights=w)
                    * 100.0
                ),
                "cni35_baseline": float(np.average(grp["cni35_baseline"], weights=w)),
                "cni35_after": float(np.average(grp["cni35_after"], weights=w)),
            }
        )
    return pd.concat([out, pd.DataFrame(weighted_rows)], ignore_index=True)


def plot_s7(curves_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, city in enumerate(CITIES):
        ax = axes[i]
        for scen, color in [("typical_day", "#1f77b4"), ("heatwave", "#d62728")]:
            sub = curves_df[(curves_df["city"] == city) & (curves_df["scenario"] == scen)].sort_values("threshold")
            ax.plot(sub["threshold"], sub["coolshare"], "--", color=color, lw=1.2, label=f"{scen} CoolShare")
            ax.plot(sub["threshold"], sub["cni"], "-", color=color, lw=1.3, label=f"{scen} CNI")
        ax.set_title(city)
        ax.grid(alpha=0.2, lw=0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].axis("off")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.supxlabel("HEI threshold (deg C)", y=0.065)
    fig.supylabel("Fraction of roads", x=0.02)
    fig.suptitle("CoolShare vs CNI by threshold across cities", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.16, wspace=0.05, hspace=0.12)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calc_cni_curve = load_cni_function()

    curves = compute_baseline_curves(calc_cni_curve)
    curves_out = OUT_DIR / "coolshare_cni_curves_all_cities.csv"
    curves.to_csv(curves_out, index=False)

    tsel = curves[curves["threshold"].isin([28.0, 35.0])].copy()
    tsel = tsel.sort_values(["city", "scenario", "threshold"])
    table_s19_out = OUT_DIR / "table_s19_coolshare_vs_cni.csv"
    tsel.to_csv(table_s19_out, index=False)

    policy = compute_policy_network(calc_cni_curve)
    policy_out = OUT_DIR / "policy_network_response_heatwave.csv"
    policy.to_csv(policy_out, index=False)

    s9_ext = policy[policy["city"] == "All_cities_weighted"][
        ["scenario", "tcni_baseline", "tcni_after", "tcni_change_pct", "cni35_baseline", "cni35_after"]
    ].copy()
    s9_ext_out = OUT_DIR / "table_s9_network_extension.csv"
    s9_ext.to_csv(s9_ext_out, index=False)

    plot_s7(curves)
    print(f"Saved: {curves_out}")
    print(f"Saved: {table_s19_out}")
    print(f"Saved: {policy_out}")
    print(f"Saved: {s9_ext_out}")
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()



