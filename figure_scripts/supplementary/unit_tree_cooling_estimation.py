#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "geopandas is required to read LSOA area from the IMD geopackage."
    ) from exc


BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results" / "inequality_analysis"
SUPPLEMENT_DIR = BASE_DIR / "paper" / "06_supplement"

IMD_GPKG = (
    BASE_DIR
    / "city_boundaries"
    / "Indices_of_Multiple_Deprivation_(IMD)_2019_8404722932957776031.gpkg"
)

# === HEI response parameters (match scripts/21_policy_scenarios_fixed.py) ===
ALPHA_B = 0.6
ALPHA_V = 0.8

# Policy-scenario design: +10 percentage points vegetation shade in target LSOAs
DELTA_S_V = 0.10

# Optional: vegetation-specific extra cooling term used in `scripts/12_hei_sensitivity_analysis.py`:
#   HEI = LST * (1 - 伪_b*S_b - 伪_v*S_v) - 螖T_veg * S_v
# Set to 0.0 to disable.
DELTA_T_VEG = 2.0

# === Tree-to-shade conversion assumptions (user-tunable) ===
ROAD_WIDTH_M = 8.0
F_OVERLAP = 0.4
F_TIME = 0.6


@dataclass(frozen=True)
class TreeType:
    label: str
    crown_diameter_m: float
    source: str

    @property
    def crown_area_m2(self) -> float:
        return math.pi * (self.crown_diameter_m / 2) ** 2

    @property
    def effective_shade_area_m2(self) -> float:
        return self.crown_area_m2 * F_OVERLAP * F_TIME


TREE_TYPES: list[TreeType] = [
    TreeType(
        label="Small (4.4 m crown)",
        crown_diameter_m=4.4,
        source="McPherson & Muchnick (2005), crape myrtle example",
    ),
    TreeType(
        label="Medium (8.0 m crown)",
        crown_diameter_m=8.0,
        source="Assumed mid-range mature street tree",
    ),
    TreeType(
        label="Large (13.7 m crown)",
        crown_diameter_m=13.7,
        source="McPherson & Muchnick (2005), Chinese hackberry example",
    ),
]


def clip_shadow(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def calculate_hei_response(lst: np.ndarray, s_b: np.ndarray, s_v: np.ndarray) -> np.ndarray:
    """
    Shade-only component of the HEI response function:
      HEI_shade = LST * (1 - 伪_b*S_b - 伪_v*S_v)
    with a safeguard that total shaded effect does not exceed 1.

    Note: the policy-scenario simulations and the main paper use the *full* HEI
    definition that additionally includes an evapotranspiration term
    (see `calculate_hei_response_with_veg_extra`).
    """
    s_b = clip_shadow(s_b)
    s_v = clip_shadow(s_v)

    total_shadow = ALPHA_B * s_b + ALPHA_V * s_v
    scaling = np.ones_like(total_shadow)
    over = total_shadow > 1
    scaling[over] = 1 / total_shadow[over]
    return lst * (1 - total_shadow * scaling)


def calculate_hei_response_with_veg_extra(
    lst: np.ndarray, s_b: np.ndarray, s_v: np.ndarray, *, delta_t_veg: float
) -> np.ndarray:
    """
    Full HEI response function used in the paper/policy scenarios:
      HEI = HEI_shade - 螖T_veg * S_v
    """
    hei_shade_only = calculate_hei_response(lst, s_b, s_v)
    return hei_shade_only - float(delta_t_veg) * clip_shadow(s_v)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float((values * weights).sum() / weights.sum())


def load_area_km2() -> pd.DataFrame:
    gdf = gpd.read_file(IMD_GPKG)
    area_df = gdf[["lsoa11cd", "st_areasha"]].copy()
    area_df["area_km2"] = area_df["st_areasha"] / 1e6
    return area_df[["lsoa11cd", "area_km2"]]


def build_target_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Replicates the selection logic in scripts/21_policy_scenarios_fixed.py, but returns
    masks keyed by the *Figure 5 labels*.
    """
    hei_median = df["hei_mean"].median()
    shadow_median = df["shadow_vegetation_mean"].median()

    mask_equity = (
        (df["IMD_Decile"].isin([1, 2, 3]))
        & (df["hei_mean"] > hei_median)
        & (df["shadow_vegetation_mean"] < shadow_median)
    )

    if "area_km2" in df.columns and df["area_km2"].notna().any():
        road_density = df["total_length"] / (df["area_km2"] + 0.001)
        density_75 = road_density.quantile(0.75)
        mask_corridors = road_density >= density_75
    else:  # pragma: no cover
        mask_corridors = df["TotPop"] >= df["TotPop"].quantile(0.75)

    mask_citywide = pd.Series(True, index=df.index)

    # Note: policy-scenario IDs are now canonical (baseline / S1_citywide / S2_corridors / S3_equity_first).
    # This appendix keeps figure-facing labels for readability.
    return {
        "S3 Equity First": mask_equity,
        "S2 Corridors": mask_corridors,
        "S1 Citywide": mask_citywide,
    }


def scenario_shade_to_cooling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the empirical slope (掳C per 1pp vegetation shade) for each scenario using
    only the scenario HEI response function (no mixing with hei_mean formula variants).
    """
    lst = df["lst_mean"].to_numpy()
    s_b = df["shadow_building_mean"].to_numpy()
    s_v = df["shadow_vegetation_mean"].to_numpy()
    w = df["TotPop"].to_numpy()

    hei0 = calculate_hei_response(lst, s_b, s_v)
    hei0_with_veg = calculate_hei_response_with_veg_extra(
        lst, s_b, s_v, delta_t_veg=DELTA_T_VEG
    )
    masks = build_target_masks(df)

    out = []
    for scenario, mask in masks.items():
        s_v_new = s_v.copy()
        s_v_new[mask.to_numpy()] = clip_shadow(s_v_new[mask.to_numpy()] + DELTA_S_V)
        hei1 = calculate_hei_response(lst, s_b, s_v_new)
        hei1_with_veg = calculate_hei_response_with_veg_extra(
            lst, s_b, s_v_new, delta_t_veg=DELTA_T_VEG
        )

        mask_np = mask.to_numpy()
        w_t = w[mask_np]

        cooling_target = weighted_mean(hei0[mask_np] - hei1[mask_np], w_t)
        cooling_all = weighted_mean(hei0 - hei1, w)

        cooling_target_with_veg = weighted_mean(
            hei0_with_veg[mask_np] - hei1_with_veg[mask_np], w_t
        )
        cooling_all_with_veg = weighted_mean(hei0_with_veg - hei1_with_veg, w)

        pop_pct = (df.loc[mask, "TotPop"].sum() / df["TotPop"].sum()) * 100
        road_km = df.loc[mask, "total_length"].sum() / 1000

        slope_pp = cooling_target / (DELTA_S_V * 100)  # 掳C per 1pp vegetation shade
        slope_pp_with_veg = cooling_target_with_veg / (DELTA_S_V * 100)

        out.append(
            {
                "scenario": scenario,
                "target_pop_pct": pop_pct,
                "target_road_km": road_km,
                "delta_t_veg_C": DELTA_T_VEG,
                "cooling_target_C_at_10pp": cooling_target,
                "cooling_target_C_at_10pp_with_veg_extra": cooling_target_with_veg,
                "cooling_target_C_at_10pp_veg_extra_only": cooling_target_with_veg
                - cooling_target,
                "cooling_overall_C_at_10pp": cooling_all,
                "cooling_overall_C_at_10pp_with_veg_extra": cooling_all_with_veg,
                "cooling_overall_C_at_10pp_veg_extra_only": cooling_all_with_veg
                - cooling_all,
                "slope_C_per_pp": slope_pp,
                "slope_C_per_pp_with_veg_extra": slope_pp_with_veg,
                "slope_C_per_pp_veg_extra_only": slope_pp_with_veg - slope_pp,
            }
        )

    return pd.DataFrame(out).sort_values("scenario")


def trees_to_shade_pp(n_trees: int, road_km: float, a_eff_m2: float) -> float:
    road_m = road_km * 1000
    a_road = road_m * ROAD_WIDTH_M
    return 100.0 * (n_trees * a_eff_m2) / a_road


def trees_needed_for_10pp(road_km: float, a_eff_m2: float) -> float:
    road_m = road_km * 1000
    a_road = road_m * ROAD_WIDTH_M
    return (DELTA_S_V * a_road) / a_eff_m2


def build_unit_tree_table(
    scenario_table: pd.DataFrame, *, n_trees: int = 10_000
) -> pd.DataFrame:
    rows = []
    for _, srow in scenario_table.iterrows():
        road_km = float(srow["target_road_km"])
        pop_pct = float(srow["target_pop_pct"])
        slope = float(srow["slope_C_per_pp"])
        slope_with_veg = float(srow["slope_C_per_pp_with_veg_extra"])

        for tree in TREE_TYPES:
            shade_pp = trees_to_shade_pp(n_trees=n_trees, road_km=road_km, a_eff_m2=tree.effective_shade_area_m2)
            cooling_target = shade_pp * slope
            cooling_target_with_veg = shade_pp * slope_with_veg
            cooling_population = cooling_target * (pop_pct / 100.0)
            cooling_population_with_veg = cooling_target_with_veg * (pop_pct / 100.0)

            rows.append(
                {
                    "scenario": srow["scenario"],
                    "tree_type": tree.label,
                    "tree_source": tree.source,
                    "crown_diameter_m": tree.crown_diameter_m,
                    "effective_shade_area_m2": tree.effective_shade_area_m2,
                    "n_trees": n_trees,
                    "target_road_km": road_km,
                    "target_pop_pct": pop_pct,
                    "delta_t_veg_C": float(srow["delta_t_veg_C"]),
                    "shade_increase_pp": shade_pp,
                    "cooling_target_mean_C": cooling_target,
                    "cooling_population_mean_C": cooling_population,
                    "cooling_target_mean_C_with_veg_extra": cooling_target_with_veg,
                    "cooling_population_mean_C_with_veg_extra": cooling_population_with_veg,
                    "cooling_target_mean_C_veg_extra_only": cooling_target_with_veg
                    - cooling_target,
                    "cooling_population_mean_C_veg_extra_only": cooling_population_with_veg
                    - cooling_population,
                    "trees_needed_for_10pp_total": trees_needed_for_10pp(
                        road_km=road_km, a_eff_m2=tree.effective_shade_area_m2
                    ),
                }
            )
    return pd.DataFrame(rows)


def run(time_scenario: str) -> None:
    df = pd.read_csv(RESULTS_DIR / f"lsoa_hei_summary_{time_scenario}.csv")
    df = df.merge(load_area_km2(), on="lsoa11cd", how="left")

    scenario_table = scenario_shade_to_cooling(df)
    unit_tree = build_unit_tree_table(scenario_table, n_trees=10_000)

    scenario_out = SUPPLEMENT_DIR / f"unit_tree_cooling_scenario_metrics_{time_scenario}.csv"
    unit_out = SUPPLEMENT_DIR / f"unit_tree_cooling_per_10k_{time_scenario}.csv"

    scenario_table.to_csv(scenario_out, index=False)
    unit_tree.to_csv(unit_out, index=False)

    print(f"\n=== {time_scenario} ===")
    print(f"Saved: {scenario_out}")
    print(f"Saved: {unit_out}")
    print("\nScenario metrics:")
    print(scenario_table.to_string(index=False))


if __name__ == "__main__":
    SUPPLEMENT_DIR.mkdir(parents=True, exist_ok=True)
    run("heatwave")
    run("typical_day")


