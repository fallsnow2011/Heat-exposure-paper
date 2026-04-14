#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import ee  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: earthengine-api\n"
        "Install with: pip install earthengine-api\n"
        "Then authenticate: earthengine authenticate\n"
    ) from exc


THRESHOLDS_C = {
    "London": 28,
    "Birmingham": 26,
    "Bristol": 26,
    "Manchester": 25,
    "Newcastle": 25,
}

GAUL_CONFIGS = {
    # Mirrors the legacy JS workflow (02b_gee_heatwave_detection.js)
    "London": {"collection": "level1", "coords": (-0.1276, 51.5072)},
    "Birmingham": {"collection": "level2", "coords": (-1.8904, 52.4862)},
    "Bristol": {"collection": "level2", "coords": (-2.5879, 51.4545)},
    "Manchester": {"collection": "level2", "coords": (-2.2426, 53.4808)},
    "Newcastle": {"collection": "level2", "coords": (-1.6178, 54.9783)},
}


@dataclass(frozen=True)
class HeatwaveEvent:
    city: str
    event_id: int
    start_date: date
    end_date: date
    n_days: int


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _daterange(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def load_city_geometry_from_geojson(city: str) -> ee.Geometry:
    boundary_path = Path("city_boundaries") / f"{city}_boundary.geojson"
    if not boundary_path.exists():
        raise FileNotFoundError(f"City boundary GeoJSON not found: {boundary_path}")
    geo = json.loads(boundary_path.read_text(encoding="utf-8"))
    if "features" not in geo or not geo["features"]:
        raise ValueError(f"Invalid GeoJSON (no features): {boundary_path}")
    geom = geo["features"][0].get("geometry")
    if not geom:
        raise ValueError(f"Invalid GeoJSON (missing geometry): {boundary_path}")
    ee_geom = ee.Geometry(geom)
    # simplify for faster reduceRegion (ERA5 is coarse anyway)
    return ee_geom.simplify(1000)


def load_city_geometry_from_gaul(city: str) -> ee.Geometry:
    """Load an AOI using FAO GAUL (legacy approach used in scripts/02b_gee_heatwave_detection.js)."""
    if city not in GAUL_CONFIGS:
        raise ValueError(f"Unknown city for GAUL AOI: {city}")
    cfg = GAUL_CONFIGS[city]

    level1 = ee.FeatureCollection("FAO/GAUL/2015/level1")
    level2 = ee.FeatureCollection("FAO/GAUL/2015/level2")
    collection = level1 if cfg["collection"] == "level1" else level2

    point = ee.Geometry.Point(cfg["coords"])
    feature = collection.filterBounds(point).first()

    # If GAUL selection fails, fall back to a small buffer to avoid hard-crash.
    geom = ee.Geometry(
        ee.Algorithms.If(
            feature,
            ee.Feature(feature).geometry().simplify(1000),
            point.buffer(5000),
        )
    )
    return geom


def build_daily_tmax_series(
    geom: ee.Geometry,
    start: str,
    end: str,
    scale_m: int = 11132,
) -> ee.FeatureCollection:
    era = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .select("temperature_2m_max")
        .filterDate(start, end)
        .filterBounds(geom)
    )

    def _to_feature(img: ee.Image) -> ee.Feature:
        date_str = img.date().format("YYYY-MM-dd")
        tmax_k = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=scale_m,
            maxPixels=1e9,
            bestEffort=True,
        ).get("temperature_2m_max")
        tmax_c = ee.Number(tmax_k).subtract(273.15)
        return ee.Feature(None, {"date": date_str, "Tmax_C": tmax_c})

    fc = ee.FeatureCollection(era.map(_to_feature))
    # keep only valid rows (avoid nulls)
    return fc.filter(ee.Filter.notNull(["Tmax_C"]))


def fc_to_dataframe(fc: ee.FeatureCollection) -> pd.DataFrame:
    info = fc.getInfo()
    rows: list[dict] = []
    for feat in info.get("features", []):
        props = feat.get("properties", {})
        rows.append({"date": props.get("date"), "Tmax_C": props.get("Tmax_C")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["Tmax_C"] = pd.to_numeric(df["Tmax_C"], errors="coerce")
    df = df.dropna(subset=["date", "Tmax_C"]).sort_values("date").reset_index(drop=True)
    return df


def compute_p95(df: pd.DataFrame, window: str) -> float:
    if df.empty:
        return float("nan")
    if window == "full_year":
        series = df["Tmax_C"]
    elif window == "jja":
        series = df[df["date"].dt.month.isin([6, 7, 8])]["Tmax_C"]
    else:
        raise ValueError(f"Unknown p95 window: {window}")
    return float(series.quantile(0.95))


def find_consecutive_events(dates: list[date], min_len: int = 3) -> list[list[date]]:
    if len(dates) < min_len:
        return []
    dates_sorted = sorted(dates)
    events: list[list[date]] = []
    cur = [dates_sorted[0]]
    for d in dates_sorted[1:]:
        if (d - cur[-1]).days == 1:
            cur.append(d)
        else:
            if len(cur) >= min_len:
                events.append(cur)
            cur = [d]
    if len(cur) >= min_len:
        events.append(cur)
    return events


def mark_heatwave_days(df: pd.DataFrame, threshold_c: float, p95: float) -> tuple[pd.DataFrame, list[HeatwaveEvent]]:
    df = df.copy()
    df["date_only"] = df["date"].dt.date
    df["meets_threshold"] = df["Tmax_C"] >= threshold_c
    df["meets_p95"] = df["Tmax_C"] >= p95
    df["is_hot_day_raw"] = df["meets_threshold"] & df["meets_p95"]

    hot_dates = sorted(df.loc[df["is_hot_day_raw"], "date_only"].tolist())
    seqs = find_consecutive_events(hot_dates, min_len=3)

    heatwave_dates: set[date] = set()
    events: list[HeatwaveEvent] = []
    for idx, seq in enumerate(seqs, start=1):
        heatwave_dates.update(seq)
        events.append(
            HeatwaveEvent(
                city="",
                event_id=idx,
                start_date=seq[0],
                end_date=seq[-1],
                n_days=len(seq),
            )
        )

    df["is_heatwave"] = df["date_only"].isin(heatwave_dates)
    return df.drop(columns=["date_only"]), events


def write_events(events: list[HeatwaveEvent], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["city", "event_id", "start_date", "end_date", "n_days"])
        for ev in events:
            w.writerow([ev.city, ev.event_id, ev.start_date.isoformat(), ev.end_date.isoformat(), ev.n_days])


def main():
    parser = argparse.ArgumentParser(description="GEE heatwave detection + ERA5 export (Scheme A).")
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
        "--aoi-source",
        choices=["geojson", "gaul"],
        default="geojson",
        help="AOI source: local city_boundaries/*_boundary.geojson (recommended) or FAO/GAUL (legacy).",
    )
    parser.add_argument(
        "--p95-window",
        choices=["full_year", "jja"],
        default="full_year",
        help="How to compute local P95 (full year vs JJA only). Must match your Methods.",
    )
    parser.add_argument(
        "--out-dir",
        default="GEE_LST_Baseline/heatwave_detection",
        help="Local output folder for CSVs.",
    )
    parser.add_argument(
        "--scale-m",
        type=int,
        default=11132,
        help="ReduceRegion scale in meters (ERA5-Land is ~11km).",
    )
    args = parser.parse_args()

    year = args.year
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.project:
            ee.Initialize(project=args.project)
        else:
            ee.Initialize()
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Failed to initialize Earth Engine.\n"
            "- If you see 'no project found', set a default project:\n"
            "    earthengine set_project <YOUR_PROJECT_ID>\n"
            "  or pass it explicitly:\n"
            "    python scripts/gee_heatwave_pipeline.py --project <YOUR_PROJECT_ID> ...\n"
        ) from exc

    all_events: list[HeatwaveEvent] = []
    summary_rows: list[dict] = []

    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"

    for city, thr in THRESHOLDS_C.items():
        print(f"\n=== {city} ===")
        if args.aoi_source == "geojson":
            geom = load_city_geometry_from_geojson(city)
        else:
            geom = load_city_geometry_from_gaul(city)

        fc = build_daily_tmax_series(geom, start=start, end=end, scale_m=args.scale_m)
        df = fc_to_dataframe(fc)
        if df.empty:
            raise SystemExit(f"No ERA5 data returned for city={city}. Check AOI and EE auth.")

        p95 = compute_p95(df, window=args.p95_window)
        df_flagged, events = mark_heatwave_days(df, threshold_c=thr, p95=p95)
        events_city = [
            HeatwaveEvent(city=city, event_id=e.event_id, start_date=e.start_date, end_date=e.end_date, n_days=e.n_days)
            for e in events
        ]
        all_events.extend(events_city)

        # output daily series
        daily_out = out_dir / f"ERA5Land_daily_Tmax_{year}_{city}_full_year_with_flags.csv"
        df_flagged.to_csv(daily_out, index=False)

        heatwave_days = int(df_flagged["is_heatwave"].sum())
        hot_days_raw = int(df_flagged["is_hot_day_raw"].sum())
        summary_rows.append(
            {
                "city": city,
                "met_office_threshold_c": thr,
                "p95_window": args.p95_window,
                "p95_c": p95,
                "n_hot_days_raw": hot_days_raw,
                "n_heatwave_days": heatwave_days,
            }
        )
        print(f"threshold={thr}C, p95={p95:.2f}C, raw_hot_days={hot_days_raw}, heatwave_days={heatwave_days}")
        for ev in events_city:
            print(f"  event {ev.event_id}: {ev.start_date} ~ {ev.end_date} ({ev.n_days} days)")

    # outputs
    summary_out = out_dir / f"heatwave_detection_summary_{year}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_out, index=False)

    events_out = out_dir / f"heatwave_events_{year}.csv"
    write_events(all_events, events_out)

    print("\nSaved:")
    print(f"  - {summary_out}")
    print(f"  - {events_out}")
    print(f"  - {out_dir}/ERA5Land_daily_Tmax_{year}_<City>_full_year_with_flags.csv")


if __name__ == "__main__":
    main()



