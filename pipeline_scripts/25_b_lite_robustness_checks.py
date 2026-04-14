#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_HEAT_EXPOSURE_DIR = BASE_DIR / "results" / "heat_exposure"
RESULTS_INEQUALITY_DIR = BASE_DIR / "results" / "inequality_analysis"
RESULTS_SENS_DIR = BASE_DIR / "results" / "sensitivity_analysis"
RESULTS_SENS_DIR.mkdir(parents=True, exist_ok=True)

HEATWAVE_DETECTION_DIR = BASE_DIR / "GEE_LST_Baseline" / "heatwave_detection"
DELTA_T_DIR = BASE_DIR / "GEE_LST_Baseline" / "delta_t"

PAPER_DRAFT_TEX_DIR = BASE_DIR / "paper-draft-tex"

CITIES = ["London", "Birmingham", "Bristol", "Manchester", "Newcastle"]

# HEI coefficients (must match scripts/13_recalculate_hei_improved.py)
ALPHA_BUILDING = 0.6
ALPHA_VEGETATION = 0.8
DELTA_T_VEGETATION = 2.0

TEMPERATURE_THRESHOLDS = np.linspace(20, 45, 51)
GCC_SNAP_METERS = 1.0


@dataclass(frozen=True)
class DeltaTAlternative:
    delta_t_base: float
    delta_t_alt: float
    delta_extra: float
    t_heatwave_mean_base: float
    t_heatwave_mean_alt: float
    t_typical_median_base: float


def _snap_xy(xy: tuple[float, float], snap_m: float = 1.0) -> tuple[int, int]:
    x, y = xy
    return (int(round(x / snap_m)), int(round(y / snap_m)))


def _representative_linestring(geom):
    """
    Extract a representative segment from a LineString or MultiLineString for
    endpoint-based connectivity /
    浠庯紙LineString | MultiLineString锛変腑鎻愬彇浠ｈ〃绾挎锛岀敤浜庣鐐硅繛閫氭€с€?
    This avoids introducing an extra dependency on `shapely.ops.linemerge`;
    using the longest component as a fallback is sufficient here /
    閬垮厤寮曞叆 `shapely.ops.linemerge` 渚濊禆锛涙寜鏈€闀垮瓙绾挎鍏滃簳鍗冲彲銆?    """
    if geom is None or geom.is_empty:
        return None

    gtype = geom.geom_type
    if gtype == "LineString":
        return geom
    if gtype == "MultiLineString":
        geoms = list(getattr(geom, "geoms", []))
        if len(geoms) == 0:
            return None
        if len(geoms) == 1:
            return geoms[0]
        return max(geoms, key=lambda g: g.length)
    return None


def calculate_cni_curve(
    roads_gdf,
    values: np.ndarray,
    thresholds: np.ndarray = TEMPERATURE_THRESHOLDS,
    snap_m: float = GCC_SNAP_METERS,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Compute CoolShare(胃), CNI(胃), and TCNI=鈭獵NI d胃 using the same conventions
    as scripts/13_recalculate_hei_improved.py:

    - CoolShare(胃) = #(valid roads with value<=胃) / #(valid roads)
    - CNI(胃)      = #(valid roads in GCC of cool subgraph at 胃) / #(valid roads)
    - Denominator is the count of valid road segments (no length weighting).
    """
    values = np.asarray(values, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    u_nodes: list[tuple[int, int] | None] = []
    v_nodes: list[tuple[int, int] | None] = []
    for geom in roads_gdf.geometry:
        line = _representative_linestring(geom)
        if line is None or line.is_empty:
            u_nodes.append(None)
            v_nodes.append(None)
            continue
        coords = list(line.coords)
        if len(coords) < 2:
            u_nodes.append(None)
            v_nodes.append(None)
            continue
        u_nodes.append(_snap_xy(coords[0], snap_m=snap_m))
        v_nodes.append(_snap_xy(coords[-1], snap_m=snap_m))

    has_uv = np.fromiter(
        (u is not None and v is not None for u, v in zip(u_nodes, v_nodes)),
        dtype=bool,
        count=len(values),
    )
    valid_edge = np.isfinite(values) & has_uv
    edge_indices = np.flatnonzero(valid_edge)
    if edge_indices.size == 0:
        coolshare_curve = np.full_like(thresholds, np.nan, dtype=float)
        cni_curve = np.full_like(thresholds, np.nan, dtype=float)
        return coolshare_curve, cni_curve, np.nan, 0

    vals = values[edge_indices]
    u_list = [u_nodes[i] for i in edge_indices]
    v_list = [v_nodes[i] for i in edge_indices]

    node_set = set(u_list).union(set(v_list))
    node_id = {n: i for i, n in enumerate(node_set)}
    u_id = np.fromiter((node_id[n] for n in u_list), dtype=np.int64, count=len(u_list))
    v_id = np.fromiter((node_id[n] for n in v_list), dtype=np.int64, count=len(v_list))

    order = np.argsort(vals)
    vals_sorted = vals[order]
    u_sorted = u_id[order]
    v_sorted = v_id[order]

    thresholds_sorted = np.sort(thresholds)
    total_edges = int(len(vals_sorted))
    n_nodes = int(len(node_id))

    parent = np.arange(n_nodes, dtype=np.int64)
    node_size = np.ones(n_nodes, dtype=np.int64)
    edge_size = np.zeros(n_nodes, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a: int, b: int) -> int:
        ra, rb = find(a), find(b)
        if ra == rb:
            edge_size[ra] += 1
            return ra
        if node_size[ra] < node_size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        node_size[ra] += node_size[rb]
        edge_size[ra] += edge_size[rb] + 1
        node_size[rb] = 0
        edge_size[rb] = 0
        return ra

    coolshare_curve = np.zeros(len(thresholds_sorted), dtype=float)
    cni_curve = np.zeros(len(thresholds_sorted), dtype=float)

    ptr = 0
    gcc_root = 0 if n_nodes > 0 else None
    gcc_nodes = int(node_size[gcc_root]) if gcc_root is not None else 0

    for i, t in enumerate(thresholds_sorted):
        while ptr < total_edges and vals_sorted[ptr] <= t:
            root = union(int(u_sorted[ptr]), int(v_sorted[ptr]))
            root = find(root)
            if node_size[root] > gcc_nodes:
                gcc_root = root
                gcc_nodes = int(node_size[root])
            ptr += 1

        coolshare_curve[i] = ptr / total_edges
        gcc_root_now = find(int(gcc_root)) if gcc_root is not None else None
        cni_curve[i] = (edge_size[gcc_root_now] / total_edges) if gcc_root_now is not None else 0.0

    tcni = float(np.trapezoid(cni_curve, thresholds_sorted))
    return coolshare_curve, cni_curve, tcni, total_edges


def calculate_hei_improved(
    lst_values: np.ndarray,
    shadow_building: np.ndarray,
    shadow_vegetation: np.ndarray,
    alpha_b: float = ALPHA_BUILDING,
    alpha_v: float = ALPHA_VEGETATION,
    delta_t_veg: float = DELTA_T_VEGETATION,
) -> np.ndarray:
    shadow_cooling = alpha_b * shadow_building + alpha_v * shadow_vegetation
    hei_base = lst_values * (1 - shadow_cooling)
    vegetation_cooling = delta_t_veg * shadow_vegetation
    return hei_base - vegetation_cooling


def compute_fallback_attribution_stats() -> pd.DataFrame:
    """
    Measure:
    - fallback segment share: building_ratio==0.5 and vegetation_ratio==0.5
    - fallback contribution to total cooling (length-weighted)
    - vegetation cooling share overall and within fallback
    """
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"geopandas import failed: {exc}") from exc

    rows: list[dict] = []
    for city in CITIES:
        path = RESULTS_HEAT_EXPOSURE_DIR / f"{city}_roads_hei_improved_heatwave.gpkg"
        if not path.exists():
            raise FileNotFoundError(f"Missing heatwave roads GPKG: {path}")

        gdf = gpd.read_file(path)
        if "building_ratio" not in gdf.columns or "vegetation_ratio" not in gdf.columns:
            raise ValueError(f"Missing ratio columns in {path}")

        b = gdf["building_ratio"].astype(float)
        v = gdf["vegetation_ratio"].astype(float)
        fallback = (np.abs(b - 0.5) < 1e-9) & (np.abs(v - 0.5) < 1e-9)

        length = gdf.geometry.length.astype(float)
        cooling_total = gdf["cooling_total"].astype(float)
        cooling_veg = gdf["cooling_vegetation"].astype(float)

        w_all = float((length * cooling_total).sum())
        w_fb = float((length[fallback] * cooling_total[fallback]).sum())

        fallback_seg_pct = float(fallback.mean() * 100.0)
        fallback_cooling_pct = float(w_fb / w_all * 100.0) if w_all else float("nan")
        veg_cooling_share_all_pct = float((length * cooling_veg).sum() / w_all * 100.0) if w_all else float("nan")
        veg_cooling_share_fallback_pct = float((length[fallback] * cooling_veg[fallback]).sum() / w_fb * 100.0) if w_fb else float("nan")

        rows.append(
            {
                "city": city,
                "fallback_segment_pct": fallback_seg_pct,
                "fallback_segments_n": int(fallback.sum()),
                "segments_n": int(len(gdf)),
                "fallback_cooling_contribution_pct": fallback_cooling_pct,
                "vegetation_cooling_share_all_pct": veg_cooling_share_all_pct,
                "vegetation_cooling_share_within_fallback_pct": veg_cooling_share_fallback_pct,
            }
        )

    df = pd.DataFrame(rows).sort_values("city").reset_index(drop=True)
    return df


def compute_london_delta_t_alternative() -> DeltaTAlternative:
    """
    Alternative 螖T for London by including the 2-day July 2022 extreme (2022-07-18/19)
    as an additional event, while keeping the original typical-day median fixed.

    - Base 螖T: from paper0128/GEE_LST_Baseline/delta_t/delta_t_summary_all_cities.csv
    - Alt heatwave mean: mean(Tmax_C) over (flagged heatwave days + 2022-07-18/19) in summer
    - Typical median: median(Tmax_C) over non-heatwave summer days (base definition)
    """
    dt_path = DELTA_T_DIR / "delta_t_summary_all_cities.csv"
    if not dt_path.exists():
        raise FileNotFoundError(f"Missing 螖T summary: {dt_path}")
    dt = pd.read_csv(dt_path)
    base_row = dt.loc[dt["city"] == "London"]
    if base_row.empty:
        raise ValueError("螖T summary missing London row")
    delta_t_base = float(base_row["delta_t_median"].iloc[0])

    full_year = HEATWAVE_DETECTION_DIR / "ERA5Land_daily_Tmax_2022_London_full_year_with_flags.csv"
    if not full_year.exists():
        raise FileNotFoundError(f"Missing London full-year Tmax with flags: {full_year}")
    df = pd.read_csv(full_year)
    required = {"date", "Tmax_C", "is_heatwave"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {full_year}")

    df["date"] = pd.to_datetime(df["date"])
    df["Tmax_C"] = pd.to_numeric(df["Tmax_C"], errors="coerce")
    df["is_heatwave"] = df["is_heatwave"].astype(bool)

    summer = df[df["date"].dt.month.isin([6, 7, 8])].dropna(subset=["Tmax_C"]).copy()
    heat_base = summer[summer["is_heatwave"]]
    typical_base = summer[~summer["is_heatwave"]]
    if heat_base.empty or typical_base.empty:
        raise ValueError("London summer heatwave/typical split is empty; check flags and inputs.")

    t_heatwave_mean_base = float(heat_base["Tmax_C"].mean())
    t_typical_median_base = float(typical_base["Tmax_C"].median())

    july_2day = {pd.Timestamp("2022-07-18"), pd.Timestamp("2022-07-19")}
    heat_alt_mask = summer["is_heatwave"] | summer["date"].isin(july_2day)
    heat_alt = summer[heat_alt_mask]
    t_heatwave_mean_alt = float(heat_alt["Tmax_C"].mean())

    delta_t_alt = float(t_heatwave_mean_alt - t_typical_median_base)
    delta_extra = float(delta_t_alt - delta_t_base)

    return DeltaTAlternative(
        delta_t_base=delta_t_base,
        delta_t_alt=delta_t_alt,
        delta_extra=delta_extra,
        t_heatwave_mean_base=t_heatwave_mean_base,
        t_heatwave_mean_alt=t_heatwave_mean_alt,
        t_typical_median_base=t_typical_median_base,
    )


def compute_population_weighted_gap(df: pd.DataFrame, hei_col: str = "hei_mean") -> float:
    """
    Population-weighted mean HEI difference: (Deprived D1-3) - (Affluent D8-10).
    """
    needed = {"IMD_Decile", "TotPop", hei_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for gap calculation: {sorted(missing)}")

    valid = df[np.isfinite(df[hei_col]) & np.isfinite(df["TotPop"])].copy()
    deprived = valid[valid["IMD_Decile"].isin([1, 2, 3])]
    affluent = valid[valid["IMD_Decile"].isin([8, 9, 10])]
    if deprived.empty or affluent.empty:
        return float("nan")

    def wmean(x: pd.DataFrame) -> float:
        w = x["TotPop"].astype(float)
        v = x[hei_col].astype(float)
        return float((w * v).sum() / w.sum())

    return float(wmean(deprived) - wmean(affluent))


def build_london_sensitivity_outputs() -> tuple[pd.DataFrame, Path]:
    """
    Compute:
    - London CNI/TCNI (heatwave) under base vs 2-day alternative 螖T
    - Population-weighted gaps: (all cities pooled) and (London only)
    - Save curve + summary CSVs
    - Save figure PDF for SI
    """
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"geopandas import failed: {exc}") from exc

    alt = compute_london_delta_t_alternative()

    # Load London curves (base) from existing outputs for consistent plotting
    curves_typ = pd.read_csv(RESULTS_HEAT_EXPOSURE_DIR / "London_cni_curves_typical_day.csv")
    curves_hw_base = pd.read_csv(RESULTS_HEAT_EXPOSURE_DIR / "London_cni_curves_heatwave.csv")

    # Load roads (heatwave) and compute alt HEI
    roads_hw_path = RESULTS_HEAT_EXPOSURE_DIR / "London_roads_hei_improved_heatwave.gpkg"
    roads_hw = gpd.read_file(roads_hw_path)

    lst_base = roads_hw["lst"].to_numpy(dtype=float)
    sb = roads_hw["shadow_building_avg"].to_numpy(dtype=float)
    sv = roads_hw["shadow_vegetation_avg"].to_numpy(dtype=float)

    lst_alt = lst_base + alt.delta_extra
    hei_alt = calculate_hei_improved(lst_alt, sb, sv)

    # Recompute curves (base from roads to validate; alt for reporting)
    _, cni_alt, tcni_alt, n_valid = calculate_cni_curve(roads_hw, hei_alt, thresholds=TEMPERATURE_THRESHOLDS, snap_m=GCC_SNAP_METERS)

    # Base tcni from summary file (avoid numerical drift)
    summary = pd.read_csv(RESULTS_HEAT_EXPOSURE_DIR / "hei_cni_tcni_summary_improved.csv")
    london_typ_tcni = float(summary[(summary["city"] == "London") & (summary["scenario"] == "typical_day")]["tcni"].iloc[0])
    london_hw_tcni = float(summary[(summary["city"] == "London") & (summary["scenario"] == "heatwave")]["tcni"].iloc[0])

    # Save alt curve CSV
    curve_out = RESULTS_SENS_DIR / "London_cni_curves_heatwave_london2day_20260201.csv"
    out_df = pd.DataFrame(
        {
            "threshold": np.sort(TEMPERATURE_THRESHOLDS),
            "cni_hei": cni_alt,
        }
    )
    out_df.to_csv(curve_out, index=False)

    # Inequality: pooled + London-only gaps (population-weighted)
    lsoa_hw_path = RESULTS_INEQUALITY_DIR / "lsoa_hei_summary_heatwave.csv"
    if not lsoa_hw_path.exists():
        raise FileNotFoundError(f"Missing LSOA summary (heatwave): {lsoa_hw_path}")
    lsoa_hw = pd.read_csv(lsoa_hw_path)

    gap_pooled_base = compute_population_weighted_gap(lsoa_hw, hei_col="hei_mean")
    gap_london_base = compute_population_weighted_gap(lsoa_hw[lsoa_hw["city"] == "London"], hei_col="hei_mean")

    # Adjust London HEI by delta_extra * (1 - alpha_b*Sb - alpha_v*Sv)
    lsoa_hw_alt = lsoa_hw.copy()
    is_london = lsoa_hw_alt["city"].astype(str).eq("London")
    sb_mean = lsoa_hw_alt.loc[is_london, "shadow_building_mean"].astype(float)
    sv_mean = lsoa_hw_alt.loc[is_london, "shadow_vegetation_mean"].astype(float)
    delta_hei_london = alt.delta_extra * (1 - ALPHA_BUILDING * sb_mean - ALPHA_VEGETATION * sv_mean)
    lsoa_hw_alt.loc[is_london, "hei_mean"] = lsoa_hw_alt.loc[is_london, "hei_mean"].astype(float) + delta_hei_london

    gap_pooled_alt = compute_population_weighted_gap(lsoa_hw_alt, hei_col="hei_mean")
    gap_london_alt = compute_population_weighted_gap(lsoa_hw_alt[lsoa_hw_alt["city"] == "London"], hei_col="hei_mean")

    summary_out = RESULTS_SENS_DIR / "london_two_day_extreme_sensitivity_20260201.csv"
    summary_df = pd.DataFrame(
        [
            {
                "delta_t_base_median": alt.delta_t_base,
                "delta_t_alt_median_keep_typical": alt.delta_t_alt,
                "delta_extra": alt.delta_extra,
                "t_heatwave_mean_base": alt.t_heatwave_mean_base,
                "t_heatwave_mean_alt": alt.t_heatwave_mean_alt,
                "t_typical_median_base": alt.t_typical_median_base,
                "tcni_typical": london_typ_tcni,
                "tcni_heatwave_base": london_hw_tcni,
                "tcni_heatwave_alt": tcni_alt,
                "tcni_collapse_base_pct": (london_hw_tcni - london_typ_tcni) / london_typ_tcni * 100.0,
                "tcni_collapse_alt_pct": (tcni_alt - london_typ_tcni) / london_typ_tcni * 100.0,
                "gap_pooled_heatwave_base": gap_pooled_base,
                "gap_pooled_heatwave_alt": gap_pooled_alt,
                "gap_london_heatwave_base": gap_london_base,
                "gap_london_heatwave_alt": gap_london_alt,
                "n_valid_roads_london": n_valid,
            }
        ]
    )
    summary_df.to_csv(summary_out, index=False)

    # Create SI figure (consistent with main figure style)
    fig_path = PAPER_DRAFT_TEX_DIR / "Supplementary_Fig_S7_london_two_day_sensitivity.pdf"
    _plot_london_sensitivity_figure(
        curves_typ=curves_typ,
        curves_hw_base=curves_hw_base,
        curves_hw_alt=out_df,
        summary_row=summary_df.iloc[0].to_dict(),
        out_path=fig_path,
    )

    return summary_df, fig_path


def _plot_london_sensitivity_figure(
    curves_typ: pd.DataFrame,
    curves_hw_base: pd.DataFrame,
    curves_hw_alt: pd.DataFrame,
    summary_row: dict,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    # Nature Cities standard style (consistent with other SI figures)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["axes.labelsize"] = 7
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 6
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5

    # Okabe-Ito colorblind-friendly colors (consistent with other SI figures)
    london_blue = "#0173B2"
    london_blue_light = "#56B4E9"
    grey = "#808080"

    fig = plt.figure(figsize=(7.5, 3.0))  # Adjusted size for better spacing
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.2], wspace=0.35)  # Balanced proportions

    # Panel a: CNI curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        curves_typ["threshold"],
        curves_typ["cni_hei"],
        color=grey,
        lw=1.0,  # Reduced line width
        label="Typical day",
        alpha=0.9,
    )
    ax1.plot(
        curves_hw_base["threshold"],
        curves_hw_base["cni_hei"],
        color=london_blue_light,
        lw=1.0,
        label="Heatwave (base 螖T)",
        alpha=0.95,
    )
    ax1.plot(
        curves_hw_alt["threshold"],
        curves_hw_alt["cni_hei"],
        color=london_blue,
        lw=1.0,
        ls="--",
        label="Heatwave (+2-day extreme)",
        alpha=0.95,
    )
    ax1.set_xlim(20, 45)
    ax1.set_ylim(0, max(0.05, float(curves_typ["cni_hei"].max()) * 1.05))
    ax1.set_xlabel("HEI threshold (掳C)", fontsize=7)
    ax1.set_ylabel("CNI(胃)", fontsize=7)
    ax1.set_title("a  London percolation curves (CNI)", fontsize=7, pad=6)
    ax1.grid(True, alpha=0.2, lw=0.4)
    ax1.legend(frameon=False, loc="upper left", fontsize=6)  # Legend on left side

    # Panel b: summary bars
    ax2 = fig.add_subplot(gs[0, 1])
    tcni_typ = float(summary_row["tcni_typical"])
    tcni_hw_base = float(summary_row["tcni_heatwave_base"])
    tcni_hw_alt = float(summary_row["tcni_heatwave_alt"])

    gap_base = float(summary_row["gap_pooled_heatwave_base"])
    gap_alt = float(summary_row["gap_pooled_heatwave_alt"])

    x = np.arange(2)
    w = 0.34
    ax2.bar(x[0] - w / 2, tcni_hw_base, width=w, color=london_blue_light,
            label="Base 螖T", edgecolor='none')
    ax2.bar(x[0] + w / 2, tcni_hw_alt, width=w, color=london_blue,
            label="+2-day", edgecolor='none')
    ax2.bar(x[1] - w / 2, gap_base, width=w, color=london_blue_light, edgecolor='none')
    ax2.bar(x[1] + w / 2, gap_alt, width=w, color=london_blue, edgecolor='none')

    ax2.set_xticks(x)
    ax2.set_xticklabels(["TCNI\n(heatwave)", "Gap\n(pooled)"], fontsize=7)
    ax2.set_title("b  Sensitivity summary", fontsize=7, pad=6)
    ax2.grid(True, axis="y", alpha=0.2, lw=0.4)

    # 螖T annotation - moved inside panel b (top-right corner) to avoid title overlap
    dt_base = float(summary_row["delta_t_base_median"])
    dt_alt = float(summary_row["delta_t_alt_median_keep_typical"])
    ax2.text(
        0.98,  # Right side
        0.95,  # Top inside
        f"螖T: {dt_base:.2f} 鈫?{dt_alt:.2f}掳C",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=6,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                 edgecolor='gray', linewidth=0.5, alpha=0.9)
    )

    # Tighten y-lims
    ymax = max(tcni_hw_alt, tcni_hw_base, gap_alt, gap_base) * 1.25
    ax2.set_ylim(0, max(0.5, float(ymax)))
    ax2.legend(frameon=False, loc="upper left", fontsize=6)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)  # Increased padding


def main() -> None:
    # 1) Fallback attribution stats
    fallback_stats = compute_fallback_attribution_stats()
    fallback_out = RESULTS_SENS_DIR / "fallback_attribution_stats_20260201.csv"
    fallback_stats.to_csv(fallback_out, index=False)

    # 2) London 2-day extreme sensitivity + figure
    london_summary, fig_path = build_london_sensitivity_outputs()

    print("Saved:")
    print(f"- {fallback_out}")
    print(f"- {RESULTS_SENS_DIR / 'London_cni_curves_heatwave_london2day_20260201.csv'}")
    print(f"- {RESULTS_SENS_DIR / 'london_two_day_extreme_sensitivity_20260201.csv'}")
    print(f"- {fig_path}")


if __name__ == "__main__":
    main()



