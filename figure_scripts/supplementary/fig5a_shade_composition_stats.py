#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_CSV = BASE_DIR / "results" / "inequality_analysis" / "lsoa_hei_summary_heatwave.csv"
OUTPUT_MD = BASE_DIR / "paper" / "06_supplement" / "TableS_fig5a_shade_composition_stats.md"
OUTPUT_SAMPLES_CSV = BASE_DIR / "paper" / "06_supplement" / "TableS_fig5a_shade_composition_bootstrap_samples.csv"


@dataclass(frozen=True)
class GroupStats:
    building_mean: float
    vegetation_mean: float
    total_mean: float
    building_share: float
    vegetation_share: float


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.sum(weights))
    if denom <= 0:
        return float("nan")
    return float(np.sum(values * weights) / denom)


def point_estimates(sb: np.ndarray, sv: np.ndarray, w: np.ndarray) -> GroupStats:
    b = _weighted_mean(sb, w)
    v = _weighted_mean(sv, w)
    total = b + v
    share_b = b / total if total > 0 else float("nan")
    share_v = 1.0 - share_b if np.isfinite(share_b) else float("nan")
    return GroupStats(
        building_mean=b,
        vegetation_mean=v,
        total_mean=total,
        building_share=share_b,
        vegetation_share=share_v,
    )


def bootstrap_group(
    sb: np.ndarray,
    sv: np.ndarray,
    w: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    chunk_size: int = 500,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(sb)

    out = {
        "building_mean": np.empty(n_boot, dtype=float),
        "vegetation_mean": np.empty(n_boot, dtype=float),
        "total_mean": np.empty(n_boot, dtype=float),
        "building_share": np.empty(n_boot, dtype=float),
        "vegetation_share": np.empty(n_boot, dtype=float),
    }

    cursor = 0
    while cursor < n_boot:
        m = min(chunk_size, n_boot - cursor)
        idx = rng.integers(0, n, size=(m, n), endpoint=False)
        w_s = w[idx]
        denom = np.sum(w_s, axis=1)

        b = np.sum(sb[idx] * w_s, axis=1) / denom
        v = np.sum(sv[idx] * w_s, axis=1) / denom
        total = b + v

        share_b = np.where(total > 0, b / total, np.nan)
        share_v = 1.0 - share_b

        out["building_mean"][cursor : cursor + m] = b
        out["vegetation_mean"][cursor : cursor + m] = v
        out["total_mean"][cursor : cursor + m] = total
        out["building_share"][cursor : cursor + m] = share_b
        out["vegetation_share"][cursor : cursor + m] = share_v

        cursor += m

    return out


def ci95(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_pvalue_two_sided(diff_samples: np.ndarray) -> float:
    diff_samples = diff_samples[np.isfinite(diff_samples)]
    if diff_samples.size == 0:
        return float("nan")
    p = 2.0 * min(np.mean(diff_samples <= 0), np.mean(diff_samples >= 0))
    return float(min(1.0, p))


def fmt_ci(value: float, lo: float, hi: float, *, digits: int = 2) -> str:
    return f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = ["IMD_Decile", "TotPop", "shadow_building_mean", "shadow_vegetation_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    df = df.dropna(subset=required).copy()

    poor = df[df["IMD_Decile"].isin([1, 2, 3])].copy()
    rich = df[df["IMD_Decile"].isin([8, 9, 10])].copy()

    def to_arrays(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sb = sub["shadow_building_mean"].to_numpy(dtype=float)
        sv = sub["shadow_vegetation_mean"].to_numpy(dtype=float)
        w = sub["TotPop"].to_numpy(dtype=float)
        return sb, sv, w

    sb_p, sv_p, w_p = to_arrays(poor)
    sb_r, sv_r, w_r = to_arrays(rich)

    n_boot = 10000
    seed = 20260122

    pe_p = point_estimates(sb_p, sv_p, w_p)
    pe_r = point_estimates(sb_r, sv_r, w_r)

    boot_p = bootstrap_group(sb_p, sv_p, w_p, n_boot=n_boot, seed=seed + 1)
    boot_r = bootstrap_group(sb_r, sv_r, w_r, n_boot=n_boot, seed=seed + 2)

    metrics = [
        ("Total shade coverage (% of roads)", "total_mean", 100.0, 1),
        ("Building shade coverage (% of roads)", "building_mean", 100.0, 1),
        ("Vegetation shade coverage (% of roads)", "vegetation_mean", 100.0, 1),
        ("Building share of total shade (%)", "building_share", 100.0, 0),
        ("Vegetation share of total shade (%)", "vegetation_share", 100.0, 0),
    ]

    rows: list[dict[str, object]] = []
    for label, key, scale, digits in metrics:
        p_val = getattr(pe_p, key) * scale
        r_val = getattr(pe_r, key) * scale

        p_lo, p_hi = ci95(boot_p[key] * scale)
        r_lo, r_hi = ci95(boot_r[key] * scale)

        diff = (boot_p[key] - boot_r[key]) * scale
        d_hat = (p_val - r_val)
        d_lo, d_hi = ci95(diff)
        p_diff = bootstrap_pvalue_two_sided(diff)

        rows.append(
            {
                "Metric": label,
                "Deprived (D1-D3), pop-weighted mean [95% CI]": fmt_ci(p_val, p_lo, p_hi, digits=digits),
                "Affluent (D8-D10), pop-weighted mean [95% CI]": fmt_ci(r_val, r_lo, r_hi, digits=digits),
                "Diff (Deprived - Affluent) [95% CI]": fmt_ci(d_hat, d_lo, d_hi, digits=digits),
                "p (two-sided, bootstrap)": f"{p_diff:.3f}" if np.isfinite(p_diff) else "n/a",
            }
        )

    out_df = pd.DataFrame(rows)

    # Optional: keep a lightweight trace of bootstrap draws for audit/debug.
    samples_df = pd.DataFrame(
        {
            "poor_total_shade_pct": boot_p["total_mean"] * 100.0,
            "rich_total_shade_pct": boot_r["total_mean"] * 100.0,
            "diff_total_shade_pct": (boot_p["total_mean"] - boot_r["total_mean"]) * 100.0,
            "poor_building_share_pct": boot_p["building_share"] * 100.0,
            "rich_building_share_pct": boot_r["building_share"] * 100.0,
            "diff_building_share_pct": (boot_p["building_share"] - boot_r["building_share"]) * 100.0,
        }
    )

    # Write markdown (SI-ready)
    header = [
        "# Supplementary Table S13. Shade composition in deprived vs affluent areas (Fig. 5a)",
        "",
        "**Data**: `results/inequality_analysis/lsoa_hei_summary_heatwave.csv` (all cities).",
        "",
        "**Grouping**: IMD deciles 1-3 (deprived) vs 8-10 (affluent).",
        "",
        "**Estimator**: population-weighted mean across LSOAs (weights = `TotPop`).",
        "",
        f"**Uncertainty**: nonparametric bootstrap over LSOAs within each group (n={n_boot}, seed={seed}).",
        "",
        f"**Sample sizes**: deprived n={len(poor):,} LSOAs (pop={poor['TotPop'].sum():,.0f}), "
        f"affluent n={len(rich):,} LSOAs (pop={rich['TotPop'].sum():,.0f}).",
        "",
        "Notes:",
        "- 'Shade coverage' is the mean fraction of road segments in shade (daily average), averaged over LSOAs.",
        "- Building/vegetation components are an attribution of total shade (they sum to total by construction).",
        "",
    ]

    def df_to_markdown_table(df_: pd.DataFrame) -> str:
        headers = [str(c).replace("|", "\\|") for c in df_.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in df_.itertuples(index=False, name=None):
            cells = [str(v).replace("|", "\\|") for v in row]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    table_md = df_to_markdown_table(out_df)
    OUTPUT_MD.write_text("\n".join(header) + table_md + "\n", encoding="utf-8")
    samples_df.to_csv(OUTPUT_SAMPLES_CSV, index=False)

    print(f"Wrote: {OUTPUT_MD}")
    print(f"Wrote: {OUTPUT_SAMPLES_CSV}")


if __name__ == "__main__":
    main()


