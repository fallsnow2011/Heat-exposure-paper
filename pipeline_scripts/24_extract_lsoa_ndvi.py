#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask


CITY_RASTERS = {
    "London": "London_NDVI_Max_2022_10m.tif",
    "Birmingham": "Birmingham_NDVI_Max_2022_10m.tif",
    "Bristol": "Bristol_NDVI_Max_2022_10m.tif",
    "Manchester": "Manchester_NDVI_Max_2022_10m.tif",
    "Newcastle": "Newcastle_NDVI_Max_2022_10m.tif",
}


def _extract_stats(geom, src) -> tuple[float, float, int]:
    if geom is None or geom.is_empty:
        return (np.nan, np.nan, 0)

    try:
        data, _ = mask(src, [geom], crop=True, filled=False)
    except ValueError:
        # Geometry outside raster bounds
        return (np.nan, np.nan, 0)

    arr = data[0]
    if hasattr(arr, "mask"):
        vals = arr[~arr.mask]
    else:
        vals = arr.ravel()

    vals = vals[np.isfinite(vals)]
    if src.nodata is not None:
        vals = vals[vals != src.nodata]

    # NDVI valid range
    vals = vals[(vals >= -1.0) & (vals <= 1.0)]

    if len(vals) == 0:
        return (np.nan, np.nan, 0)

    return (float(np.mean(vals)), float(np.std(vals)), int(len(vals)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LSOA NDVI zonal statistics.")
    parser.add_argument(
        "--lsoa-list",
        default=None,
        help="CSV with lsoa11cd + city columns (default: lsoa_hei_summary_heatwave.csv).",
    )
    parser.add_argument(
        "--imd-gpkg",
        default=None,
        help="IMD LSOA boundary GPKG (default: city_boundaries/...gpkg).",
    )
    parser.add_argument(
        "--ndvi-dir",
        default=None,
        help="Directory containing city NDVI rasters (default: GEE_NDVI_Exports).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV (default: results/ndvi_analysis/lsoa_ndvi_values.csv).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    results_dir = base_dir / "results"

    lsoa_list_path = (
        Path(args.lsoa_list)
        if args.lsoa_list
        else results_dir / "inequality_analysis" / "lsoa_hei_summary_heatwave.csv"
    )
    imd_gpkg = (
        Path(args.imd_gpkg)
        if args.imd_gpkg
        else base_dir
        / "city_boundaries"
        / "Indices_of_Multiple_Deprivation_(IMD)_2019_8404722932957776031.gpkg"
    )
    ndvi_dir = Path(args.ndvi_dir) if args.ndvi_dir else base_dir / "GEE_NDVI_Exports"
    output_path = (
        Path(args.output)
        if args.output
        else results_dir / "ndvi_analysis" / "lsoa_ndvi_values.csv"
    )

    if not lsoa_list_path.exists():
        raise FileNotFoundError(f"LSOA list not found: {lsoa_list_path}")
    if not imd_gpkg.exists():
        raise FileNotFoundError(f"IMD GPKG not found: {imd_gpkg}")

    lsoa_list = pd.read_csv(lsoa_list_path, usecols=["lsoa11cd", "city"]).drop_duplicates()
    imd = gpd.read_file(imd_gpkg)
    imd = imd[["lsoa11cd", "geometry"]]
    imd = imd[imd["lsoa11cd"].isin(lsoa_list["lsoa11cd"])].copy()
    imd = imd.merge(lsoa_list, on="lsoa11cd", how="left")

    results = []

    for city, raster_name in CITY_RASTERS.items():
        raster_path = ndvi_dir / raster_name
        if not raster_path.exists():
            raise FileNotFoundError(f"NDVI raster not found: {raster_path}")

        city_gdf = imd[imd["city"] == city].copy()
        if city_gdf.empty:
            print(f"Skip {city}: no LSOAs")
            continue

        with rasterio.open(raster_path) as src:
            if city_gdf.crs != src.crs:
                city_gdf = city_gdf.to_crs(src.crs)

            for _, row in city_gdf.iterrows():
                mean_val, std_val, count = _extract_stats(row.geometry, src)
                results.append(
                    {
                        "lsoa11cd": row["lsoa11cd"],
                        "ndvi_mean": mean_val,
                        "ndvi_std": std_val,
                        "ndvi_pixels": count,
                    }
                )

        print(f"Processed {city}: {len(city_gdf)} LSOAs")

    out_df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    missing = out_df["ndvi_mean"].isna().sum()
    print(f"Saved: {output_path}")
    print(f"Rows: {len(out_df)} | Missing NDVI: {missing}")


if __name__ == "__main__":
    main()



