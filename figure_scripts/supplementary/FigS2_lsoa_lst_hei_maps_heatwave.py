from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    # Use the main repo's geojson file (not in paper0128)
    main_repo = Path(__file__).resolve().parents[3]

    lsoa_geojson = main_repo / "paper" / "06_supplement" / "imd_lsoa.geojson"
    lsoa_summary_csv = (
        repo_root / "results" / "inequality_analysis" / "lsoa_hei_summary_heatwave.csv"
    )

    out_dir = repo_root / "paper" / "final-SI" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "Supplementary_Fig_S2_LST_HEI_by_city_heatwave.png"
    out_pdf = out_dir / "Supplementary_Fig_S2_LST_HEI_by_city_heatwave.pdf"

    cities = ["London", "Birmingham", "Bristol", "Manchester", "Newcastle"]

    df = pd.read_csv(lsoa_summary_csv, usecols=["lsoa11cd", "city", "lst_mean", "hei_mean"])
    df = df[df["city"].isin(cities)].copy()

    gdf = gpd.read_file(lsoa_geojson)[["lsoa11cd", "geometry"]]
    gdf = gdf.merge(df, on="lsoa11cd", how="inner")

    lst_vmin, lst_vmax = 40.0, 52.0
    hei_vmin, hei_vmax = 25.0, 50.0
    cmap = "RdYlBu_r"

    fig = plt.figure(figsize=(17, 7), dpi=200)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=6,
        width_ratios=[1, 1, 1, 1, 1, 0.05],
        wspace=0.02,
        hspace=0.04,
    )

    for col, city in enumerate(cities):
        sub = gdf[gdf["city"] == city]

        ax_lst = fig.add_subplot(gs[0, col])
        ax_hei = fig.add_subplot(gs[1, col])

        sub.plot(
            column="lst_mean",
            ax=ax_lst,
            cmap=cmap,
            vmin=lst_vmin,
            vmax=lst_vmax,
            linewidth=0,
        )
        sub.plot(
            column="hei_mean",
            ax=ax_hei,
            cmap=cmap,
            vmin=hei_vmin,
            vmax=hei_vmax,
            linewidth=0,
        )

        ax_lst.set_title(city, fontsize=12, pad=2)
        ax_lst.set_axis_off()
        ax_hei.set_axis_off()

    cax_lst = fig.add_subplot(gs[0, 5])
    cax_hei = fig.add_subplot(gs[1, 5])

    fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=lst_vmin, vmax=lst_vmax), cmap=cmap),
        cax=cax_lst,
        label="LST (掳C)",
    )
    fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=hei_vmin, vmax=hei_vmax), cmap=cmap),
        cax=cax_hei,
        label="HEI (掳C)",
    )

    fig.text(
        0.01,
        0.75,
        "LST (heatwave)",
        rotation=90,
        va="center",
        ha="left",
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.01,
        0.25,
        "HEI (heatwave)",
        rotation=90,
        va="center",
        ha="left",
        fontsize=12,
        weight="bold",
    )

    fig.suptitle(
        "Supplementary Fig. S2 | City-wide LSOA-scale LST and HEI (heatwave)",
        y=0.98,
        fontsize=14,
        weight="bold",
    )

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()



