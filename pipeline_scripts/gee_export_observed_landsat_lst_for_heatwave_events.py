#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    import ee  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: earthengine-api\n"
        "Install with: pip install earthengine-api\n"
        "Then authenticate: earthengine authenticate\n"
    ) from exc


def mask_landsat_c2(img: ee.Image) -> ee.Image:
    qa = img.select("QA_PIXEL")
    cloud = qa.bitwiseAnd(1 << 3).eq(0)  # bit 3
    shadow = qa.bitwiseAnd(1 << 4).eq(0)  # bit 4
    mask = cloud.And(shadow)
    return img.updateMask(mask)


def add_lst_c(img: ee.Image) -> ee.Image:
    # ST_B10 is scaled to Kelvin in Landsat C2 L2 products:
    # ST(K) = DN * 0.00341802 + 149.0
    lst_k = img.select("ST_B10").multiply(0.00341802).add(149.0)
    lst_c = lst_k.subtract(273.15).rename("LST_C")
    return img.addBands(lst_c)


def main():
    parser = argparse.ArgumentParser(description="Export observed Landsat LST composites for heatwave events.")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument(
        "--project",
        default=None,
        help=(
            "Google Cloud project id for Earth Engine API calls. "
            "If omitted, uses your default EE project (set via `earthengine set_project`)."
        ),
    )
    parser.add_argument(
        "--events-csv",
        default=None,
        help="Path to heatwave_events_<YEAR>.csv (defaults to GEE_LST_Baseline/heatwave_detection).",
    )
    parser.add_argument(
        "--aoi-source",
        choices=["geojson", "gaul"],
        default="geojson",
        help="AOI source (must be consistent with your detection pipeline).",
    )
    parser.add_argument(
        "--drive-folder",
        default="GEE_LST_Observed_Heatwave",
        help="Google Drive folder to export GeoTIFFs into.",
    )
    parser.add_argument(
        "--cloud-cover-max",
        type=float,
        default=None,
        help="Optional scene-level filter: keep scenes with CLOUD_COVER < this value (e.g., 80).",
    )
    parser.add_argument(
        "--cities",
        default=None,
        help="Comma-separated city filter (e.g., London,Bristol). Default: all in events table.",
    )
    parser.add_argument(
        "--event-ids",
        default=None,
        help="Comma-separated event_id filter (e.g., 1,2). Default: all.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Safety limit: export at most N events (after filtering).",
    )
    args = parser.parse_args()

    events_csv = (
        Path(args.events_csv)
        if args.events_csv
        else Path("GEE_LST_Baseline") / "heatwave_detection" / f"heatwave_events_{args.year}.csv"
    )
    if not events_csv.exists():
        raise SystemExit(f"Missing events CSV: {events_csv}\nRun: python scripts/gee_heatwave_pipeline.py --year {args.year}")

    df_events = pd.read_csv(events_csv)
    required = {"city", "event_id", "start_date", "end_date"}
    missing = required - set(df_events.columns)
    if missing:
        raise SystemExit(f"Events CSV missing columns: {sorted(missing)} ({events_csv})")

    if args.cities:
        allowed = {c.strip() for c in args.cities.split(",") if c.strip()}
        df_events = df_events[df_events["city"].isin(sorted(allowed))].copy()

    if args.event_ids:
        allowed_ids = {int(x.strip()) for x in args.event_ids.split(",") if x.strip()}
        df_events = df_events[df_events["event_id"].astype(int).isin(sorted(allowed_ids))].copy()

    df_events = df_events.sort_values(["city", "event_id"]).reset_index(drop=True)
    if df_events.empty:
        raise SystemExit("No events left after filtering.")

    if args.max_events is not None:
        df_events = df_events.head(int(args.max_events)).copy()

    # Import AOI helpers from the detection script to avoid divergence
    from gee_heatwave_pipeline import (  # type: ignore
        load_city_geometry_from_gaul,
        load_city_geometry_from_geojson,
    )

    if args.project:
        ee.Initialize(project=args.project)
    else:
        ee.Initialize()

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    ls = l8.merge(l9)

    tasks = []

    for _, row in df_events.iterrows():
        city = str(row["city"])
        event_id = int(row["event_id"])
        start = datetime.strptime(str(row["start_date"]), "%Y-%m-%d").date()
        end = datetime.strptime(str(row["end_date"]), "%Y-%m-%d").date()
        end_exclusive = end + timedelta(days=1)

        if args.aoi_source == "geojson":
            aoi = load_city_geometry_from_geojson(city)
        else:
            aoi = load_city_geometry_from_gaul(city)

        col = (
            ls.filterBounds(aoi)
            .filterDate(start.isoformat(), end_exclusive.isoformat())
        )
        if args.cloud_cover_max is not None:
            col = col.filter(ee.Filter.lt("CLOUD_COVER", float(args.cloud_cover_max)))

        col = col.map(mask_landsat_c2).map(add_lst_c).select("LST_C")

        n_images = int(col.size().getInfo())
        label = f"{city}_event{event_id}_{start}_{end}"
        if n_images == 0:
            print(f"[SKIP] {label}: no Landsat scenes after filters")
            continue

        img = col.median().clip(aoi)

        desc = f"LST_observed_heatwave_{args.year}_{city}_event{event_id}_{start}_{end}_30m"
        task = ee.batch.Export.image.toDrive(
            image=img.toFloat(),
            description=desc,
            folder=args.drive_folder,
            region=aoi,
            scale=30,
            crs="EPSG:27700",
            maxPixels=1e13,
        )
        task.start()
        tasks.append(task)
        print(f"[EXPORT] {label}: scenes={n_images} -> Drive/{args.drive_folder}/{desc}.tif (task started)")

    print(f"\nStarted {len(tasks)} export task(s). Check Earth Engine Tasks / Drive folder.")


if __name__ == "__main__":
    main()



