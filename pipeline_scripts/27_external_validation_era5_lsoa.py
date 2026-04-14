#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import cdsapi
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import xarray as xr
from rasterio.transform import from_origin
from rasterstats import zonal_stats
from scipy.stats import pearsonr, spearmanr


BASE_DIR = Path(__file__).resolve().parents[1]
LSOA_HEI_PATH = BASE_DIR / "results" / "inequality_analysis" / "lsoa_hei_summary_heatwave.csv"
EVENTS_PATH = BASE_DIR / "GEE_LST_Baseline" / "heatwave_detection" / "heatwave_events_2022.csv"
IMD_GPKG = BASE_DIR / "city_boundaries" / "Indices_of_Multiple_Deprivation_(IMD)_2019_8404722932957776031.gpkg"
OUT_DIR = BASE_DIR / "results" / "validation"
OUT_FIG = BASE_DIR / "paper-draft-tex" / "Supplementary_Fig_S6_HEI_ERA5_validation.pdf"
ARCO_ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

CITY_LADS = {
    "London": [
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
    ],
    "Birmingham": ["Birmingham"],
    "Bristol": ["Bristol, City of"],
    "Manchester": ["Manchester"],
    "Newcastle": ["Newcastle upon Tyne"],
}


@dataclass
class CityEventGrid:
    city: str
    lsoa11cd: str
    era5_tmax_event_mean: float


def _detect_col(cols: Iterable[str], candidates: Iterable[str]) -> str:
    cols_set = {c: c.lower() for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
    for col, low in cols_set.items():
        if low in {c.lower() for c in candidates}:
            return col
    raise RuntimeError(f"Cannot find required column. Tried: {list(candidates)}")


def load_lsoa_geometry() -> gpd.GeoDataFrame:
    if not IMD_GPKG.exists():
        raise FileNotFoundError(f"Missing IMD geometry: {IMD_GPKG}")
    gdf = gpd.read_file(IMD_GPKG)
    if gdf.empty:
        raise RuntimeError("IMD geometry is empty.")
    lsoa_col = _detect_col(gdf.columns, ["lsoa11cd", "LSOA11CD"])
    lad_col = _detect_col(gdf.columns, ["LADnm", "LADNM"])
    gdf = gdf.rename(columns={lsoa_col: "lsoa11cd", lad_col: "LADnm"})
    gdf["city"] = None
    for city, lads in CITY_LADS.items():
        gdf.loc[gdf["LADnm"].isin(lads), "city"] = city
    gdf = gdf[gdf["city"].notna()].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=27700)
    return gdf[["lsoa11cd", "city", "geometry"]]


def build_event_dates() -> dict[str, list[pd.Timestamp]]:
    events = pd.read_csv(EVENTS_PATH)
    events["start_date"] = pd.to_datetime(events["start_date"])
    events["end_date"] = pd.to_datetime(events["end_date"])
    out: dict[str, list[pd.Timestamp]] = {}
    for city, grp in events.groupby("city"):
        dates: list[pd.Timestamp] = []
        for _, row in grp.iterrows():
            dates.extend(pd.date_range(row["start_date"], row["end_date"], freq="D").to_list())
        uniq = sorted(set(dates))
        out[city] = uniq
    return out


def _format_days_for_cds(dates: list[pd.Timestamp]) -> dict[str, list[str]]:
    years = sorted({f"{d.year:04d}" for d in dates})
    months = sorted({f"{d.month:02d}" for d in dates})
    days = sorted({f"{d.day:02d}" for d in dates})
    return {"year": years, "month": months, "day": days}


def _download_city_era5(city: str, city_lsoas_4326: gpd.GeoDataFrame, dates: list[pd.Timestamp], out_nc: Path) -> Path:
    bounds = city_lsoas_4326.total_bounds  # minx, miny, maxx, maxy
    west, south, east, north = [float(v) for v in bounds]
    cds_dates = _format_days_for_cds(dates)
    client = cdsapi.Client(quiet=False, timeout=120)
    req = {
        "variable": "2m_temperature",
        "year": cds_dates["year"],
        "month": cds_dates["month"],
        "day": cds_dates["day"],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [north, west, south, east],  # N, W, S, E
    }
    client.retrieve("reanalysis-era5-land", req, str(out_nc))
    return out_nc


def _event_mean_tmax_array(nc_path: Path) -> tuple[np.ndarray, object, str]:
    ds = xr.open_dataset(nc_path)
    var_name = "t2m" if "t2m" in ds.data_vars else list(ds.data_vars)[0]
    da = ds[var_name]
    if "time" not in da.dims:
        if "valid_time" in da.dims:
            da = da.rename({"valid_time": "time"})
        else:
            raise RuntimeError(f"No time dimension found in {nc_path}; dims={da.dims}")
    da_c = da - 273.15
    daily = da_c.resample(time="1D").max(skipna=True)
    event_mean = daily.mean(dim="time", skipna=True)
    lats = event_mean["latitude"].values
    lons = event_mean["longitude"].values
    arr = np.asarray(event_mean.values, dtype=float)
    # rasterstats expects row 0 = north. If latitude asc, flip vertically.
    if lats[0] < lats[-1]:
        arr = np.flipud(arr)
        lats = lats[::-1]
    xres = abs(float(lons[1] - lons[0]))
    yres = abs(float(lats[1] - lats[0]))
    west = float(np.min(lons) - xres / 2.0)
    north = float(np.max(lats) + yres / 2.0)
    transform = from_origin(west, north, xres, yres)
    return arr, transform, "EPSG:4326"


def _event_mean_tmax_array_from_da(da: xr.DataArray, event_dates: list[pd.Timestamp]) -> tuple[np.ndarray, object, str]:
    da_c = da - 273.15
    daily = da_c.resample(time="1D").max(skipna=True)
    event_days = pd.DatetimeIndex(pd.to_datetime(event_dates)).normalize().unique()
    keep = pd.DatetimeIndex(daily["time"].values).normalize().isin(event_days)
    daily_sel = daily.isel(time=np.where(keep)[0])
    if daily_sel.sizes.get("time", 0) == 0:
        raise RuntimeError("No overlap between requested heatwave dates and ERA5 time coverage.")
    event_mean = daily_sel.mean(dim="time", skipna=True)
    lats = event_mean["latitude"].values
    lons = event_mean["longitude"].values
    if len(lats) < 2 or len(lons) < 2:
        raise RuntimeError("ERA5 spatial subset too small (<2 grid cells per axis).")
    arr = np.asarray(event_mean.values, dtype=float)
    if lats[0] < lats[-1]:
        arr = np.flipud(arr)
        lats = lats[::-1]
    xres = abs(float(lons[1] - lons[0]))
    yres = abs(float(lats[1] - lats[0]))
    west = float(np.min(lons) - xres / 2.0)
    north = float(np.max(lats) + yres / 2.0)
    transform = from_origin(west, north, xres, yres)
    return arr, transform, "EPSG:4326"


def _sample_nearest(arr: np.ndarray, transform, x: float, y: float) -> float:
    col_f, row_f = (~transform) * (x, y)
    row = int(np.clip(np.round(row_f), 0, arr.shape[0] - 1))
    col = int(np.clip(np.round(col_f), 0, arr.shape[1] - 1))
    val = arr[row, col]
    return float(val) if np.isfinite(val) else np.nan


def _lon_to_360(lon: float) -> float:
    return (lon + 360.0) % 360.0


def zonal_mean_for_city(city: str, lsoas_city: gpd.GeoDataFrame, dates: list[pd.Timestamp]) -> list[CityEventGrid]:
    city_dir = OUT_DIR / "cds_downloads"
    city_dir.mkdir(parents=True, exist_ok=True)
    nc_path = city_dir / f"era5land_hourly_2m_temperature_event_days_{city}_2022.nc"
    lsoas_4326 = lsoas_city.to_crs(epsg=4326)
    if nc_path.exists():
        nc = nc_path
    else:
        nc = _download_city_era5(city, lsoas_4326, dates, nc_path)
    arr, transform, _ = _event_mean_tmax_array(nc)
    stats = zonal_stats(
        lsoas_4326,
        arr,
        affine=transform,
        stats=["mean"],
        nodata=np.nan,
    )
    out: list[CityEventGrid] = []
    for (_, row), st in zip(lsoas_4326.iterrows(), stats):
        mean_val = float(st["mean"]) if st["mean"] is not None else np.nan
        if not np.isfinite(mean_val):
            rp = row.geometry.representative_point()
            mean_val = _sample_nearest(arr, transform, float(rp.x), float(rp.y))
        out.append(
            CityEventGrid(
                city=city,
                lsoa11cd=str(row["lsoa11cd"]),
                era5_tmax_event_mean=mean_val,
            )
        )
    return out


def zonal_mean_for_city_arco(
    city: str,
    lsoas_city: gpd.GeoDataFrame,
    dates: list[pd.Timestamp],
    ds_arco: xr.Dataset,
) -> list[CityEventGrid]:
    lsoas_4326 = lsoas_city.to_crs(epsg=4326)
    west, south, east, north = [float(v) for v in lsoas_4326.total_bounds]
    pad = 0.25
    west -= pad
    east += pad
    south -= pad
    north += pad

    start = pd.to_datetime(min(dates)).normalize()
    end = pd.to_datetime(max(dates)).normalize() + pd.Timedelta(days=1) - pd.Timedelta(hours=1)

    da_base = ds_arco["2m_temperature"].sel(
        time=slice(start, end),
        latitude=slice(north, south),  # ERA5 latitude is descending.
    )
    west360 = _lon_to_360(west)
    east360 = _lon_to_360(east)
    if west360 <= east360:
        da = da_base.sel(longitude=slice(west360, east360))
    else:
        da_w = da_base.sel(longitude=slice(west360, 359.75))
        da_e = da_base.sel(longitude=slice(0.0, east360))
        da = xr.concat([da_w, da_e], dim="longitude")
    # Normalize longitudes only after spatial subsetting (small array).
    da = da.assign_coords(longitude=((da["longitude"] + 180.0) % 360.0) - 180.0).sortby("longitude")

    arr, transform, _ = _event_mean_tmax_array_from_da(da, dates)
    stats = zonal_stats(
        lsoas_4326,
        arr,
        affine=transform,
        stats=["mean"],
        nodata=np.nan,
    )
    out: list[CityEventGrid] = []
    for (_, row), st in zip(lsoas_4326.iterrows(), stats):
        mean_val = float(st["mean"]) if st["mean"] is not None else np.nan
        if not np.isfinite(mean_val):
            rp = row.geometry.representative_point()
            mean_val = _sample_nearest(arr, transform, float(rp.x), float(rp.y))
        out.append(
            CityEventGrid(
                city=city,
                lsoa11cd=str(row["lsoa11cd"]),
                era5_tmax_event_mean=mean_val,
            )
        )
    return out


def fallback_city_mean(lsoa_hei: pd.DataFrame, dates_by_city: dict[str, list[pd.Timestamp]]) -> pd.DataFrame:
    rows = []
    for city, grp in lsoa_hei.groupby("city"):
        full = BASE_DIR / "GEE_LST_Baseline" / "heatwave_detection" / f"ERA5Land_daily_Tmax_2022_{city}_full_year_with_flags.csv"
        dfx = pd.read_csv(full)
        dfx["date"] = pd.to_datetime(dfx["date"])
        use_dates = set(dates_by_city[city])
        val = float(dfx[dfx["date"].isin(use_dates)]["Tmax_C"].mean())
        for lsoa in grp["lsoa11cd"].astype(str):
            rows.append({"city": city, "lsoa11cd": lsoa, "era5_tmax_event_mean": val})
    out = pd.DataFrame(rows)
    out["source"] = "fallback_city_mean"
    return out


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    base = df.dropna(subset=["hei_mean", "era5_tmax_event_mean"]).copy()
    for city, grp in base.groupby("city"):
        if len(grp) < 3:
            continue
        r, p = pearsonr(grp["hei_mean"], grp["era5_tmax_event_mean"])
        rho, p_s = spearmanr(grp["hei_mean"], grp["era5_tmax_event_mean"])
        records.append(
            {
                "scope": city,
                "n_lsoa": int(len(grp)),
                "pearson_r": float(r),
                "pearson_p": float(p),
                "spearman_rho": float(rho),
                "spearman_p": float(p_s),
            }
        )
    if len(base) >= 3:
        r, p = pearsonr(base["hei_mean"], base["era5_tmax_event_mean"])
        rho, p_s = spearmanr(base["hei_mean"], base["era5_tmax_event_mean"])
        records.append(
            {
                "scope": "All_cities_pooled",
                "n_lsoa": int(len(base)),
                "pearson_r": float(r),
                "pearson_p": float(p),
                "spearman_rho": float(rho),
                "spearman_p": float(p_s),
            }
        )
        model = smf.ols("hei_mean ~ era5_tmax_event_mean + C(city)", data=base).fit()
        records.append(
            {
                "scope": "All_cities_pooled_cityFE",
                "n_lsoa": int(len(base)),
                "pearson_r": float(model.params.get("era5_tmax_event_mean", np.nan)),
                "pearson_p": float(model.pvalues.get("era5_tmax_event_mean", np.nan)),
                "spearman_rho": float(model.rsquared),
                "spearman_p": np.nan,
            }
        )
    return pd.DataFrame(records)


def plot_validation(df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["hei_mean", "era5_tmax_event_mean"]).copy()
    colors = {
        "London": "#1f77b4",
        "Birmingham": "#d62728",
        "Bristol": "#2ca02c",
        "Manchester": "#9467bd",
        "Newcastle": "#ff7f0e",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for city, grp in plot_df.groupby("city"):
        ax.scatter(
            grp["era5_tmax_event_mean"],
            grp["hei_mean"],
            s=8,
            alpha=0.55,
            color=colors.get(city, "#666666"),
            label=city,
        )
    ax.set_xlabel("ERA5-Land event-mean daily Tmax (deg C)")
    ax.set_ylabel("LSOA mean HEI (heatwave, deg C)")
    ax.set_title("a  HEI vs ERA5-Land (LSOA)")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    # Residual-on-residual after city fixed effects.
    fe1 = smf.ols("hei_mean ~ C(city)", data=plot_df).fit()
    fe2 = smf.ols("era5_tmax_event_mean ~ C(city)", data=plot_df).fit()
    plot_df["hei_resid"] = fe1.resid
    plot_df["era5_resid"] = fe2.resid
    rr_r, rr_p = pearsonr(plot_df["era5_resid"], plot_df["hei_resid"])
    ax2 = axes[1]
    ax2.scatter(plot_df["era5_resid"], plot_df["hei_resid"], s=8, alpha=0.55, color="#444444")
    ax2.axhline(0, lw=0.7, c="#999999")
    ax2.axvline(0, lw=0.7, c="#999999")
    ax2.set_xlabel("ERA5 residual (city FE removed)")
    ax2.set_ylabel("HEI residual (city FE removed)")
    ax2.set_title(f"b  Residual relationship (r={rr_r:.3f}, p={rr_p:.3g})")

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lsoa_hei = pd.read_csv(LSOA_HEI_PATH, usecols=["lsoa11cd", "city", "hei_mean"]).copy()
    lsoa_hei["lsoa11cd"] = lsoa_hei["lsoa11cd"].astype(str)
    lsoas = load_lsoa_geometry()
    lsoas["lsoa11cd"] = lsoas["lsoa11cd"].astype(str)
    lsoas = lsoas[lsoas["lsoa11cd"].isin(set(lsoa_hei["lsoa11cd"]))].copy()
    dates_by_city = build_event_dates()

    era5_df: pd.DataFrame | None = None
    source = "cds_era5_land"
    try:
        rows: list[CityEventGrid] = []
        for city in sorted(lsoa_hei["city"].unique()):
            city_lsoas = lsoas[lsoas["city"] == city].copy()
            if city_lsoas.empty:
                continue
            print(f"[INFO] Running ERA5 zonal aggregation for city={city}")
            rows.extend(zonal_mean_for_city(city, city_lsoas, dates_by_city[city]))
        era5_df = pd.DataFrame([r.__dict__ for r in rows])
        era5_df["source"] = source
    except Exception as cds_exc:
        print(f"[WARN] CDS retrieval failed. Trying ARCO ERA5: {cds_exc}")
        if os.environ.get("ENABLE_ARCO_ERA5", "0") == "1":
            try:
                ds_arco = xr.open_zarr(ARCO_ZARR_URL, consolidated=True, storage_options={"token": "anon"})
                rows = []
                try:
                    for city in sorted(lsoa_hei["city"].unique()):
                        city_lsoas = lsoas[lsoas["city"] == city].copy()
                        if city_lsoas.empty:
                            continue
                        rows.extend(zonal_mean_for_city_arco(city, city_lsoas, dates_by_city[city], ds_arco))
                finally:
                    ds_arco.close()
                era5_df = pd.DataFrame([r.__dict__ for r in rows])
                source = "arco_era5_zarr"
                era5_df["source"] = source
            except Exception as arco_exc:
                print(f"[WARN] ARCO ERA5 failed. Falling back to city mean Tmax: {arco_exc}")
                source = "fallback_city_mean"
                era5_df = fallback_city_mean(lsoa_hei, dates_by_city)
        else:
            print("[WARN] ENABLE_ARCO_ERA5 is not set; using fallback city-mean ERA5.")
            source = "fallback_city_mean"
            era5_df = fallback_city_mean(lsoa_hei, dates_by_city)

    merged = lsoa_hei.merge(era5_df, on=["city", "lsoa11cd"], how="left")
    merged["source"] = source
    stats_df = compute_stats(merged)
    stats_df["source"] = source

    out_validation = OUT_DIR / "hei_era5_lsoa_validation.csv"
    out_stats = OUT_DIR / "hei_era5_validation_stats.csv"
    merged.to_csv(out_validation, index=False)
    stats_df.to_csv(out_stats, index=False)
    plot_validation(merged, stats_df)
    print(f"Saved: {out_validation}")
    print(f"Saved: {out_stats}")
    print(f"Saved: {OUT_FIG}")


if __name__ == "__main__":
    main()



