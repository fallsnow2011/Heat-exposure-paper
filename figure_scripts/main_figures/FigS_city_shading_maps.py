#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / "results" / "heat_exposure"
OUT_DIR = BASE_DIR / "paper" / "06_supplement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ["London", "Birmingham", "Manchester", "Bristol", "Newcastle"]
SCENARIOS = {
    "typical_day": "Typical Day",
    "heatwave": "Heatwave",
}

SHADING_COLS = {
    "building": "shadow_building_avg",
    "vegetation": "shadow_vegetation_avg",
}

CMAPS = {
    "building": LinearSegmentedColormap.from_list(
        "building_greys",
        ["#c5c5c5", "#9a9a9a", "#6b6b6b", "#3d3d3d", "#111111"],
    ),
    "vegetation": LinearSegmentedColormap.from_list(
        "vegetation_ylgn",
        ["#f7fcb4", "#addd8e", "#41ab5d", "#006837", "#004529"],  # 澧炲己瀵规瘮搴?    ),
}

ROW_LABELS = {
    "building": "Building shading",
    "vegetation": "Vegetation shading",
}


def _load_city_data(city: str, scenario: str) -> gpd.GeoDataFrame:
    path = RESULTS_DIR / f"{city}_roads_hei_improved_{scenario}.gpkg"
    gdf = gpd.read_file(path)
    keep_cols = ["geometry"] + list(SHADING_COLS.values())
    return gdf[keep_cols]


def _compute_vmax(data_by_city, col, percentile=99):
    values = []
    for gdf in data_by_city.values():
        vals = gdf[col].to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            values.append(vals)
    if not values:
        return 1.0
    all_vals = np.concatenate(values)
    vmax = float(np.nanpercentile(all_vals, percentile))
    return max(vmax, 0.05)


def add_north_arrow(ax, x=0.9, y=0.9, size=0.08):
    """Small north arrow in axes fraction coordinates."""
    ax.annotate(
        "N",
        xy=(x, y),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
    )
    ax.annotate(
        "",
        xy=(x, y - 0.02),
        xycoords="axes fraction",
        xytext=(x, y - size),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#222222", lw=1.0),
    )


def add_scalebar(ax, length_m=5000, x_offset_frac=0.0, y_offset_frac=0.0):
    """Add a simple scalebar based on current axis limits (meters)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    if width <= 0 or height <= 0:
        return

    length = int(length_m)
    if width < length * 1.3:
        length = int(max(1000, width * 0.25) // 1000) * 1000
    if length <= 0:
        return

    pad_x = width * 0.06
    pad_y = height * 0.06
    bar_x0 = min(x0, x1) + pad_x + width * x_offset_frac
    bar_y0 = min(y0, y1) + pad_y + height * y_offset_frac
    # Clamp to keep the scalebar inside the axes area
    bar_x0 = max(bar_x0, min(x0, x1) + width * 0.02)
    bar_y0 = max(bar_y0, min(y0, y1) + height * 0.02)
    bar_x1 = bar_x0 + length

    box_w = length + width * 0.04
    box_h = height * 0.06
    rect = mpl.patches.Rectangle(
        (bar_x0 - width * 0.02, bar_y0 - height * 0.02),
        box_w,
        box_h,
        facecolor="white",
        edgecolor="none",
        alpha=0.75,
        zorder=2,
    )
    ax.add_patch(rect)

    ax.plot([bar_x0, bar_x1], [bar_y0, bar_y0], color="#222222", lw=1.2, zorder=3)
    ax.plot([bar_x0, bar_x0], [bar_y0, bar_y0 + height * 0.01], color="#222222", lw=1.2, zorder=3)
    ax.plot([bar_x1, bar_x1], [bar_y0, bar_y0 + height * 0.01], color="#222222", lw=1.2, zorder=3)

    ax.text(
        (bar_x0 + bar_x1) / 2,
        bar_y0 + height * 0.02,
        f"{int(length / 1000)} km",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#222222",
        zorder=3,
    )


def set_bounds_with_padding(ax, gdf, pad_frac=0.08):
    minx, miny, maxx, maxy = gdf.total_bounds
    width = maxx - minx
    height = maxy - miny
    pad_x = width * pad_frac
    pad_y = height * pad_frac
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)


def plot_scenario(scenario: str) -> None:
    data_by_city = {city: _load_city_data(city, scenario) for city in CITIES}

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(20.0, 7.6),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.18, wspace=0.12, hspace=0.12)

    for col_idx, city in enumerate(CITIES):
        gdf = data_by_city[city]
        for row_idx, key in enumerate(["building", "vegetation"]):
            ax = axes[row_idx, col_idx]
            col = SHADING_COLS[key]

            vmin = 0.0
            vmax = 0.6
            norm = PowerNorm(gamma=0.6, vmin=vmin, vmax=vmax)

            gdf.plot(
                column=col,
                ax=ax,
                cmap=CMAPS[key],
                norm=norm,
                linewidth=0.7,
                alpha=1.0,
            )

            if row_idx == 0:
                ax.set_title(city, fontsize=11, fontweight="bold")

            ax.set_facecolor("#ffffff")
            ax.set_axis_off()
            ax.set_aspect("equal")
            set_bounds_with_padding(ax, gdf, pad_frac=0.09)
            if row_idx == 0:
                if city in {"Birmingham", "Manchester"}:
                    add_scalebar(ax, x_offset_frac=-0.20, y_offset_frac=-0.08)
                else:
                    add_scalebar(ax)

    # Row labels - 澧炲ぇ瀛椾綋
    fig.text(0.015, 0.74, ROW_LABELS["building"], rotation=90,
             va="center", ha="left", fontsize=12, fontweight="bold")
    fig.text(0.015, 0.28, ROW_LABELS["vegetation"], rotation=90,
             va="center", ha="left", fontsize=12, fontweight="bold")

    # Two horizontal colorbars at the bottom
    cbar_y = 0.07
    cbar_h = 0.018
    cbar_w = 0.34
    gap = 0.04
    left_x = 0.18
    right_x = left_x + cbar_w + gap

    sm_building = mpl.cm.ScalarMappable(
        cmap=CMAPS["building"], norm=PowerNorm(gamma=0.6, vmin=0.0, vmax=0.6)
    )
    sm_building.set_array([])
    cax_b = fig.add_axes([left_x, cbar_y, cbar_w, cbar_h])
    cbar_b = fig.colorbar(sm_building, cax=cax_b, orientation="horizontal")
    cbar_b.set_ticks([0.0, 0.3, 0.6])
    cbar_b.ax.tick_params(labelsize=8)
    cbar_b.set_label("Building Shading Factor (0鈥?.6)", fontsize=9, labelpad=3)

    sm_veg = mpl.cm.ScalarMappable(
        cmap=CMAPS["vegetation"], norm=PowerNorm(gamma=0.6, vmin=0.0, vmax=0.6)
    )
    sm_veg.set_array([])
    cax_v = fig.add_axes([right_x, cbar_y, cbar_w, cbar_h])
    cbar_v = fig.colorbar(sm_veg, cax=cax_v, orientation="horizontal")
    cbar_v.set_ticks([0.0, 0.3, 0.6])
    cbar_v.ax.tick_params(labelsize=8)
    cbar_v.set_label("Vegetation Shading Factor (0鈥?.6)", fontsize=9, labelpad=3)

    fig.suptitle(
        f"Road-level shading by city ({SCENARIOS[scenario]})",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    out_png = OUT_DIR / f"FigS_city_shading_maps_{scenario}.png"
    out_pdf = OUT_DIR / f"FigS_city_shading_maps_{scenario}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {out_png.name}, {out_pdf.name}")


def main():
    for scenario in SCENARIOS:
        plot_scenario(scenario)


if __name__ == "__main__":
    main()


