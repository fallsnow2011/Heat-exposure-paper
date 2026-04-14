#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests


BOUNDARY_SERVICE = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/ArcGIS/rest/services/"
    "Lower_layer_Super_Output_Areas_Dec_2011_Boundaries_Full_Clipped_BFC_EW_V3_2022/"
    "FeatureServer/0/query"
)
IMD_SERVICE = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/ArcGIS/rest/services/"
    "Index_of_Multiple_Deprivation_Dec_2019_Lookup_in_England_2022/FeatureServer/0/query"
)

OUT_DIR = Path("city_boundaries")
OUT_GPKG = OUT_DIR / "Indices_of_Multiple_Deprivation_(IMD)_2019_8404722932957776031.gpkg"
CHUNK_SIZE = 2000
QUERY_CODE_CHUNK = 50

LONDON_BOROUGHS = [
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
]

CITY_LADS = {
    "London": LONDON_BOROUGHS,
    "Birmingham": ["Birmingham"],
    "Bristol": ["Bristol, City of"],
    "Manchester": ["Manchester"],
    "Newcastle": ["Newcastle upon Tyne"],
}


def _request(url: str, params: dict) -> dict:
    r = requests.post(url, data=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"ArcGIS error at {url}: {data['error']}")
    return data


def _count(url: str) -> int:
    data = _request(url, {"where": "1=1", "returnCountOnly": "true", "f": "json"})
    return int(data["count"])


def _fetch_geojson(url: str, offset: int, n: int, out_sr: int | None = None) -> gpd.GeoDataFrame:
    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "resultOffset": str(offset),
        "resultRecordCount": str(n),
    }
    if out_sr is not None:
        params["outSR"] = str(out_sr)
    data = _request(url, params)
    feats = data.get("features", [])
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")


def _fetch_table(url: str, offset: int, n: int) -> pd.DataFrame:
    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "false",
        "f": "json",
        "resultOffset": str(offset),
        "resultRecordCount": str(n),
    }
    data = _request(url, params)
    feats = data.get("features", [])
    if not feats:
        return pd.DataFrame()
    return pd.DataFrame([f.get("attributes", {}) for f in feats])


def _fetch_geojson_by_codes(url: str, codes: list[str], out_sr: int | None = None) -> gpd.GeoDataFrame:
    where = "LSOA11CD IN (" + ",".join(f"'{c}'" for c in codes) + ")"
    params = {
        "where": where,
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
    }
    if out_sr is not None:
        params["outSR"] = str(out_sr)
    data = _request(url, params)
    feats = data.get("features", [])
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")


def _fetch_table_by_codes(url: str, codes: list[str]) -> pd.DataFrame:
    where = "LSOA11CD IN (" + ",".join(f"'{c}'" for c in codes) + ")"
    params = {
        "where": where,
        "outFields": "*",
        "returnGeometry": "false",
        "f": "json",
    }
    data = _request(url, params)
    feats = data.get("features", [])
    if not feats:
        return pd.DataFrame()
    return pd.DataFrame([f.get("attributes", {}) for f in feats])


def load_target_codes() -> list[str]:
    src = Path("results/inequality_analysis/lsoa_hei_summary_heatwave.csv")
    df = pd.read_csv(src, usecols=["lsoa11cd"])
    codes = sorted(df["lsoa11cd"].astype(str).unique().tolist())
    if not codes:
        raise RuntimeError("No target LSOA codes found.")
    return codes


def download_boundary(codes: list[str]) -> gpd.GeoDataFrame:
    print(f"Boundary target rows: {len(codes)}")
    chunks = []
    for i in range(0, len(codes), QUERY_CODE_CHUNK):
        part = codes[i : i + QUERY_CODE_CHUNK]
        print(f"Boundary fetch code chunk {i}..{i+len(part)-1}")
        chunks.append(_fetch_geojson_by_codes(BOUNDARY_SERVICE, part))
    gdf = gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True), crs=chunks[0].crs)
    if "LSOA11CD" not in gdf.columns:
        raise RuntimeError("LSOA11CD missing in boundary dataset.")
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    return gdf


def download_imd_lookup(codes: list[str]) -> pd.DataFrame:
    print(f"IMD target rows: {len(codes)}")
    chunks = []
    for i in range(0, len(codes), QUERY_CODE_CHUNK):
        part = codes[i : i + QUERY_CODE_CHUNK]
        print(f"IMD fetch code chunk {i}..{i+len(part)-1}")
        chunks.append(_fetch_table_by_codes(IMD_SERVICE, part))
    df = pd.concat(chunks, ignore_index=True)
    required = {"LSOA11CD", "LAD19NM", "IMD19"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"IMD lookup missing columns: {required - set(df.columns)}")
    return df


def build_merged_gpkg(boundary: gpd.GeoDataFrame, imd_lookup: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf = boundary.merge(imd_lookup[["LSOA11CD", "LSOA11NM", "LAD19NM", "IMD19"]], on="LSOA11CD", how="left")
    gdf = gdf.rename(
        columns={
            "LSOA11CD": "lsoa11cd",
            "LSOA11NM_x": "lsoa11nm",
            "LSOA11NM_y": "lsoa11nm_lookup",
            "LAD19NM": "LADnm",
            "IMD19": "IMD_Rank",
        }
    )
    if "lsoa11nm" not in gdf.columns and "lsoa11nm_lookup" in gdf.columns:
        gdf["lsoa11nm"] = gdf["lsoa11nm_lookup"]

    rank = pd.to_numeric(gdf["IMD_Rank"], errors="coerce")
    # IMD rank: 1 is most deprived, higher is less deprived.
    gdf["IMD_Decile"] = pd.qcut(rank.rank(method="first"), 10, labels=False) + 1
    gdf["IMDScore"] = rank
    gdf["IncDec"] = gdf["IMD_Decile"]
    gdf["EmpDec"] = gdf["IMD_Decile"]
    gdf["EnvDec"] = gdf["IMD_Decile"]
    gdf["st_areasha"] = gdf.to_crs(epsg=27700).geometry.area
    gdf["TotPop"] = np.nan
    return gdf


def save_city_boundaries(gdf: gpd.GeoDataFrame) -> None:
    for city, lads in CITY_LADS.items():
        sub = gdf[gdf["LADnm"].isin(lads)].copy()
        if sub.empty:
            print(f"Skip city boundary {city}: no rows")
            continue
        boundary = sub.dissolve()
        boundary = boundary[["geometry"]].copy()
        boundary["city"] = city
        out = OUT_DIR / f"{city}_boundary.geojson"
        boundary.to_crs(epsg=4326).to_file(out, driver="GeoJSON")
        print(f"Saved: {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    codes = load_target_codes()
    boundary = download_boundary(codes)
    imd_lookup = download_imd_lookup(codes)
    merged = build_merged_gpkg(boundary, imd_lookup)
    merged.to_crs(epsg=27700).to_file(OUT_GPKG, driver="GPKG")
    print(f"Saved: {OUT_GPKG}")
    save_city_boundaries(merged)
    print("Done.")


if __name__ == "__main__":
    main()



