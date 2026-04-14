#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.ops import linemerge
from scipy import stats
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_NDVI_PATH = BASE_DIR / "GEE_NDVI_Exports" / "London_NDVI_Max_2022_10m.tif"
HEI_DIR = BASE_DIR / "results" / "heat_exposure"
OUTPUT_DIR = BASE_DIR / "results" / "ndvi_analysis"


def _snap_xy(xy: tuple[float, float], snap_m: float = 1.0) -> tuple[int, int]:
    x, y = xy
    return (int(round(x / snap_m)), int(round(y / snap_m)))


def _representative_linestring(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return geom
    if geom.geom_type == "MultiLineString":
        if len(geom.geoms) == 1:
            return geom.geoms[0]
        try:
            merged = linemerge(geom)
            if merged.geom_type == "LineString":
                return merged
        except Exception:
            pass
        return max(list(geom.geoms), key=lambda g: g.length)
    return None


def sample_ndvi_at_centroids(roads: gpd.GeoDataFrame, ndvi_path: Path) -> np.ndarray:
    with rasterio.open(ndvi_path) as src:
        if roads.crs != src.crs:
            roads = roads.to_crs(src.crs)
        centroids = roads.geometry.centroid
        coords = list(zip(centroids.x, centroids.y))
        sampled = list(src.sample(coords, masked=True))

    out = np.full(len(sampled), np.nan, dtype=float)
    for i, v in enumerate(sampled):
        val = v[0]
        if np.ma.is_masked(val):
            continue
        val = float(val)
        if not np.isfinite(val):
            continue
        out[i] = val
    return out


def gcc_membership_by_hei(
    roads: gpd.GeoDataFrame,
    hei: np.ndarray,
    threshold: float,
    snap_m: float = 1.0,
) -> np.ndarray:
    """
    Return a boolean mask (len=roads) indicating whether each road segment is
    part of the city-wide GCC of the cool subgraph at the given HEI threshold.
    """
    hei = np.asarray(hei, dtype=float)
    in_gcc = np.zeros(len(hei), dtype=bool)

    # Endpoints (snapped)
    u_nodes: list[tuple[int, int] | None] = []
    v_nodes: list[tuple[int, int] | None] = []
    for geom in roads.geometry:
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

    has_uv = np.fromiter((u is not None and v is not None for u, v in zip(u_nodes, v_nodes)), dtype=bool, count=len(hei))
    valid = np.isfinite(hei) & has_uv
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return in_gcc

    # Only cool edges participate in the cool subgraph
    cool_mask = hei[idx] <= threshold
    idx = idx[cool_mask]
    if idx.size == 0:
        return in_gcc

    u_list = [u_nodes[i] for i in idx]
    v_list = [v_nodes[i] for i in idx]

    parents: dict[object, object] = {}
    sizes: dict[object, int] = {}

    def find(x):
        parents.setdefault(x, x)
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        sa, sb = sizes.get(ra, 1), sizes.get(rb, 1)
        if sa < sb:
            ra, rb = rb, ra
            sa, sb = sb, sa
        parents[rb] = ra
        sizes[ra] = sa + sb
        sizes[rb] = 0

    nodes = set(u_list).union(set(v_list))
    for n in nodes:
        parents[n] = n
        sizes[n] = 1

    for u, v in zip(u_list, v_list):
        union(u, v)

    gcc_root, gcc_size = max(sizes.items(), key=lambda kv: kv[1])
    if gcc_size <= 1:
        return in_gcc

    roots_u = [find(u) for u in u_list]
    roots_v = [find(v) for v in v_list]
    in_comp = np.fromiter((ru == gcc_root and rv == gcc_root for ru, rv in zip(roots_u, roots_v)), dtype=bool, count=len(u_list))
    in_gcc[idx[in_comp]] = True
    return in_gcc


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    if method == "spearman":
        return float(stats.spearmanr(x[mask], y[mask]).correlation)
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def run_for_scenario(
    scenario: str,
    ndvi_path: Path,
    thresholds: list[float],
    snap_m: float,
    city: str = "London",
    out_dir: Path = OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    roads_path = HEI_DIR / f"{city}_roads_hei_improved_{scenario}.gpkg"
    if not roads_path.exists():
        raise FileNotFoundError(f"Missing roads file: {roads_path}")

    roads = gpd.read_file(roads_path)
    if "hei_improved" not in roads.columns:
        raise ValueError(f"Missing 'hei_improved' in {roads_path}")

    ndvi = sample_ndvi_at_centroids(roads, ndvi_path)
    hei = roads["hei_improved"].to_numpy(dtype=float)

    valid = np.isfinite(ndvi) & np.isfinite(hei)
    df = pd.DataFrame({"ndvi": ndvi, "hei": hei})
    for t in thresholds:
        df[f"cool_{int(t)}"] = hei <= t
        df[f"in_gcc_{int(t)}"] = gcc_membership_by_hei(roads, hei, threshold=t, snap_m=snap_m)

    df = df[valid].copy()

    # Correlations (NDVI vs HEI; NDVI vs cool indicator; NDVI vs GCC membership)
    corr_rows = []
    corr_rows.append(
        {
            "scenario": scenario,
            "pair": "NDVI~HEI",
            "pearson_r": _safe_corr(df["ndvi"].to_numpy(), df["hei"].to_numpy(), "pearson"),
            "spearman_r": _safe_corr(df["ndvi"].to_numpy(), df["hei"].to_numpy(), "spearman"),
            "n": int(len(df)),
        }
    )
    for t in thresholds:
        cool = df[f"cool_{int(t)}"].astype(float).to_numpy()
        gcc = df[f"in_gcc_{int(t)}"].astype(float).to_numpy()
        corr_rows.append(
            {
                "scenario": scenario,
                "pair": f"NDVI~CoolShare@{int(t)}",
                "pearson_r": _safe_corr(df["ndvi"].to_numpy(), cool, "pearson"),
                "spearman_r": _safe_corr(df["ndvi"].to_numpy(), cool, "spearman"),
                "n": int(len(df)),
            }
        )
        corr_rows.append(
            {
                "scenario": scenario,
                "pair": f"NDVI~GCCmembership@{int(t)}",
                "pearson_r": _safe_corr(df["ndvi"].to_numpy(), gcc, "pearson"),
                "spearman_r": _safe_corr(df["ndvi"].to_numpy(), gcc, "spearman"),
                "n": int(len(df)),
            }
        )

    corr_df = pd.DataFrame(corr_rows)

    # Quintiles summary
    df["ndvi_q"] = pd.qcut(df["ndvi"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    rows = []
    for q, sub in df.groupby("ndvi_q", observed=True):
        row = {
            "scenario": scenario,
            "ndvi_q": str(q),
            "n_roads": int(len(sub)),
            "ndvi_mean": float(sub["ndvi"].mean()),
            "hei_mean": float(sub["hei"].mean()),
        }
        for t in thresholds:
            coolshare = float(sub[f"cool_{int(t)}"].mean())
            cni = float(sub[f"in_gcc_{int(t)}"].mean())
            row[f"coolshare_{int(t)}"] = coolshare
            row[f"cni_{int(t)}"] = cni
            row[f"conn_given_cool_{int(t)}"] = (cni / coolshare) if coolshare > 0 else float("nan")
        rows.append(row)

    quint_df = pd.DataFrame(rows).sort_values("ndvi_q")

    out_dir.mkdir(parents=True, exist_ok=True)
    corr_out = out_dir / f"london_ndvi_cni_correlations_{scenario}.csv"
    quint_out = out_dir / f"london_ndvi_cni_quintiles_{scenario}.csv"
    corr_df.to_csv(corr_out, index=False)
    quint_df.to_csv(quint_out, index=False)

    # Simple plot (quintiles)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax0, ax1 = axes

    # Panel A: HEI by NDVI quintile
    ax0.plot(quint_df["ndvi_q"], quint_df["hei_mean"], marker="o", color="#E76F51", linewidth=2)
    ax0.set_xlabel("NDVI quintile (Q1=lowest greenness)")
    ax0.set_ylabel("Mean HEI (掳C)")
    ax0.set_title(f"A. Local cooling vs greenness ({scenario})")
    ax0.grid(True, alpha=0.3)

    # Panel B: CoolShare vs CNI at the highest threshold in list (usually 35掳C)
    t_ref = int(max(thresholds))
    x = np.arange(len(quint_df))
    width = 0.38
    ax1.bar(x - width / 2, quint_df[f"coolshare_{t_ref}"], width, label=f"CoolShare@{t_ref}掳C", color="#2A9D8F", alpha=0.85)
    ax1.bar(x + width / 2, quint_df[f"cni_{t_ref}"], width, label=f"CNI@{t_ref}掳C (GCC)", color="#264653", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(quint_df["ndvi_q"])
    ax1.set_ylim(0, max(0.01, float(quint_df[f"coolshare_{t_ref}"].max()) * 1.15))
    ax1.set_xlabel("NDVI quintile")
    ax1.set_ylabel("Road-segment share")
    ax1.set_title(f"B. Connectivity paradox ({scenario})")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    fig.suptitle("London: NDVI vs Connected Cool Network (new CNI definition)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig_path = out_dir / f"london_ndvi_cni_quintiles_{scenario}.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return corr_df, quint_df


def main():
    parser = argparse.ArgumentParser(description="Recompute London NDVI vs CNI correlation (new definitions).")
    parser.add_argument("--ndvi", type=str, default=str(DEFAULT_NDVI_PATH), help="Path to London NDVI raster (EPSG:27700).")
    parser.add_argument("--scenario", type=str, default="both", choices=["typical_day", "heatwave", "both"])
    parser.add_argument("--thresholds", type=str, default="28,35", help="Comma-separated HEI thresholds (e.g., '28,35').")
    parser.add_argument("--snap-m", type=float, default=1.0, help="Endpoint snapping grid in meters.")
    args = parser.parse_args()

    ndvi_path = Path(args.ndvi)
    if not ndvi_path.exists():
        raise FileNotFoundError(f"NDVI raster not found: {ndvi_path}")

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    if not thresholds:
        raise ValueError("No thresholds provided.")

    scenarios = ["typical_day", "heatwave"] if args.scenario == "both" else [args.scenario]

    for scen in scenarios:
        run_for_scenario(
            scenario=scen,
            ndvi_path=ndvi_path,
            thresholds=thresholds,
            snap_m=float(args.snap_m),
        )

    print(f"[OK] Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()



