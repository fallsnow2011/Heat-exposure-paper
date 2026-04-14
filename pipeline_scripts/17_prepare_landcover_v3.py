from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio


CITIES = ["London", "Birmingham", "Bristol", "Manchester", "Newcastle"]


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}\n"
        )


def _write_boundary_27700(city: str, boundary_geojson: Path, boundary_out: Path) -> None:
    boundary_out.parent.mkdir(parents=True, exist_ok=True)
    if boundary_out.exists():
        return

    gdf = gpd.read_file(boundary_geojson)
    gdf_27700 = gdf.to_crs(epsg=27700)
    gdf_27700.to_file(boundary_out, driver="GPKG", layer="boundary")


def _clip_osopen_layer(
    *,
    osopen_gpkg: Path,
    layer: str,
    boundary_27700_gpkg: Path,
    out_gpkg: Path,
    geometry_type: str = "MULTIPOLYGON",
) -> None:
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if out_gpkg.exists():
        return

    cmd = [
        "ogr2ogr",
        "-f",
        "GPKG",
        str(out_gpkg),
        str(osopen_gpkg),
        layer,
        "-clipsrc",
        str(boundary_27700_gpkg),
        "-clipsrclayer",
        "boundary",
        "-nlt",
        geometry_type,
        "-lco",
        "SPATIAL_INDEX=YES",
    ]
    _run(cmd)


def _gdal_rasterize_like(
    *,
    vector_path: Path,
    vector_layer: str,
    reference_raster: Path,
    out_raster: Path,
    burn_value: int = 1,
    dtype: str = "Byte",
    nodata: int = 0,
) -> None:
    out_raster.parent.mkdir(parents=True, exist_ok=True)
    if out_raster.exists():
        return

    with rasterio.open(reference_raster) as ref:
        left, bottom, right, top = ref.bounds
        width, height = ref.width, ref.height

    cmd = [
        "gdal_rasterize",
        "-burn",
        str(burn_value),
        "-init",
        "0",
        "-ot",
        dtype,
        "-a_nodata",
        str(nodata),
        "-a_srs",
        "EPSG:27700",
        "-te",
        str(left),
        str(bottom),
        str(right),
        str(top),
        "-ts",
        str(width),
        str(height),
        "-co",
        "COMPRESS=LZW",
        "-co",
        "TILED=YES",
        "-co",
        "BIGTIFF=YES",
        "-l",
        vector_layer,
        str(vector_path),
        str(out_raster),
    ]
    _run(cmd)


def _gdalwarp_match_reference(
    *, src_raster: Path, reference_raster: Path, out_raster: Path, resampling: str = "near", nodata: int = 0
) -> None:
    out_raster.parent.mkdir(parents=True, exist_ok=True)
    if out_raster.exists():
        return

    with rasterio.open(reference_raster) as ref:
        left, bottom, right, top = ref.bounds
        width, height = ref.width, ref.height

    cmd = [
        "gdalwarp",
        "-overwrite",
        "-t_srs",
        "EPSG:27700",
        "-r",
        resampling,
        "-te",
        str(left),
        str(bottom),
        str(right),
        str(top),
        "-ts",
        str(width),
        str(height),
        "-dstnodata",
        str(nodata),
        "-co",
        "COMPRESS=LZW",
        "-co",
        "TILED=YES",
        "-co",
        "BIGTIFF=YES",
        str(src_raster),
        str(out_raster),
    ]
    _run(cmd)


def _write_green_union_10m(*, ndvi_binary_10m: Path, os_greenspace_10m: Path, out_green_10m: Path) -> None:
    out_green_10m.parent.mkdir(parents=True, exist_ok=True)
    if out_green_10m.exists():
        return

    with rasterio.open(ndvi_binary_10m) as ndvi_src:
        ndvi = ndvi_src.read(1)
        profile = ndvi_src.profile.copy()

    with rasterio.open(os_greenspace_10m) as gs_src:
        gs = gs_src.read(1)

    green = ((ndvi == 1) | (gs == 1)).astype("uint8")

    profile.update(dtype="uint8", nodata=0, compress="lzw", tiled=True, BIGTIFF="YES")
    with rasterio.open(out_green_10m, "w", **profile) as dst:
        dst.write(green, 1)


def _write_heights_blockwise(
    *,
    ndsm_2m: Path,
    building_mask_2m: Path,
    green_final_2m: Path,
    out_building_height: Path,
    out_vegetation_height: Path,
) -> dict:
    out_building_height.parent.mkdir(parents=True, exist_ok=True)
    out_vegetation_height.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ndsm_2m) as ndsm_src, rasterio.open(building_mask_2m) as b_src, rasterio.open(
        green_final_2m
    ) as g_src:
        profile = ndsm_src.profile.copy()
        profile.update(nodata=0, compress="lzw", tiled=True, BIGTIFF="YES")

        ndsm_nodata = ndsm_src.nodata

        stats = {
            "building_pixels": 0,
            "building_sum": 0.0,
            "building_max": 0.0,
            "vegetation_pixels": 0,
            "vegetation_sum": 0.0,
            "vegetation_max": 0.0,
        }

        with rasterio.open(out_building_height, "w", **profile) as b_dst, rasterio.open(
            out_vegetation_height, "w", **profile
        ) as v_dst:
            for _, window in ndsm_src.block_windows(1):
                ndsm = ndsm_src.read(1, window=window)
                bmask = b_src.read(1, window=window) == 1
                gmask = g_src.read(1, window=window) == 1

                valid = ndsm != ndsm_nodata

                b_where = valid & bmask & (ndsm > 0)
                v_where = valid & gmask & (~bmask) & (ndsm > 0)

                building_height = np.where(valid & bmask, ndsm, 0).astype("float32")
                vegetation_height = np.where(valid & gmask & (~bmask), ndsm, 0).astype("float32")

                b_dst.write(building_height, 1, window=window)
                v_dst.write(vegetation_height, 1, window=window)

                if b_where.any():
                    vals = ndsm[b_where].astype("float64")
                    stats["building_pixels"] += int(vals.size)
                    stats["building_sum"] += float(vals.sum())
                    stats["building_max"] = max(stats["building_max"], float(vals.max()))

                if v_where.any():
                    vals = ndsm[v_where].astype("float64")
                    stats["vegetation_pixels"] += int(vals.size)
                    stats["vegetation_sum"] += float(vals.sum())
                    stats["vegetation_max"] = max(stats["vegetation_max"], float(vals.max()))

    stats["building_mean"] = stats["building_sum"] / stats["building_pixels"] if stats["building_pixels"] else 0.0
    stats["vegetation_mean"] = (
        stats["vegetation_sum"] / stats["vegetation_pixels"] if stats["vegetation_pixels"] else 0.0
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--osopen-gpkg",
        default="/mnt/f/mike/Lidar-data-England/OS_Open_Zoomstack.gpkg",
        help="Path to OS Open Zoomstack GPKG",
    )
    parser.add_argument("--ndvi-dir", default="GEE_NDVI_Exports")
    parser.add_argument("--cities", nargs="*", default=CITIES)
    args = parser.parse_args()

    version = args.version
    osopen_gpkg = Path(args.osopen_gpkg)
    ndvi_dir = Path(args.ndvi_dir)

    results_dir = Path("results/landcover")
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"landcover_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logger = logging.getLogger("landcover_v3")

    logger.info("v3 landcover preparation")
    logger.info(f"version={version}")
    logger.info(f"osopen_gpkg={osopen_gpkg}")
    logger.info(f"ndvi_dir={ndvi_dir}")
    logger.info(f"log={log_path}")

    cache_dir = Path("cache") / version
    boundary_dir = cache_dir / "boundaries_27700"
    boundary_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"version": version, "generated_at": datetime.now().isoformat(), "cities": {}}

    for city in args.cities:
        logger.info("=" * 80)
        logger.info(f"City: {city}")

        boundary_geojson = Path("city_boundaries") / f"{city}_boundary.geojson"
        if not boundary_geojson.exists():
            raise FileNotFoundError(boundary_geojson)

        ndsm_2m = Path("city_lidar") / city / f"{city}_nDSM_2m.tif"
        if not ndsm_2m.exists():
            raise FileNotFoundError(ndsm_2m)

        ndvi_binary_10m = ndvi_dir / f"{city}_Greenspace_Binary_2022_10m.tif"
        if not ndvi_binary_10m.exists():
            raise FileNotFoundError(ndvi_binary_10m)

        boundary_27700 = boundary_dir / f"{city}_boundary_27700.gpkg"
        _write_boundary_27700(city, boundary_geojson, boundary_27700)

        buildings_clip = Path("city_boundaries") / f"{city}_buildings_osopen_{version}.gpkg"
        greenspace_clip = Path("city_boundaries") / f"{city}_greenspace_osopen_{version}.gpkg"

        logger.info("Clipping OS Open layers...")
        _clip_osopen_layer(
            osopen_gpkg=osopen_gpkg,
            layer="local_buildings",
            boundary_27700_gpkg=boundary_27700,
            out_gpkg=buildings_clip,
        )
        _clip_osopen_layer(
            osopen_gpkg=osopen_gpkg,
            layer="greenspace",
            boundary_27700_gpkg=boundary_27700,
            out_gpkg=greenspace_clip,
        )

        # Build 10m greenspace raster aligned to NDVI binary
        os_greenspace_10m = cache_dir / "greenspace_10m" / f"{city}_os_greenspace_10m_{version}.tif"
        logger.info("Rasterizing OS greenspace to NDVI 10m grid...")
        _gdal_rasterize_like(
            vector_path=greenspace_clip,
            vector_layer="greenspace",
            reference_raster=ndvi_binary_10m,
            out_raster=os_greenspace_10m,
            burn_value=1,
            dtype="Byte",
            nodata=0,
        )

        green_final_10m = Path("city_lidar") / city / f"{city}_Green_final_10m_{version}.tif"
        logger.info("Union: NDVI-binary OR OS greenspace (10m)...")
        _write_green_union_10m(
            ndvi_binary_10m=ndvi_binary_10m,
            os_greenspace_10m=os_greenspace_10m,
            out_green_10m=green_final_10m,
        )

        # Warp final 10m green mask to match nDSM grid (2m)
        green_final_2m = Path("city_lidar") / city / f"{city}_Green_final_2m_{version}.tif"
        logger.info("Warping Green_final_10m -> Green_final_2m (match nDSM grid)...")
        _gdalwarp_match_reference(src_raster=green_final_10m, reference_raster=ndsm_2m, out_raster=green_final_2m)

        # Rasterize buildings to nDSM grid (2m)
        building_mask_2m = Path("city_lidar") / city / f"{city}_building_mask_2m_{version}.tif"
        logger.info("Rasterizing buildings to nDSM 2m grid...")
        _gdal_rasterize_like(
            vector_path=buildings_clip,
            vector_layer="local_buildings",
            reference_raster=ndsm_2m,
            out_raster=building_mask_2m,
            burn_value=1,
            dtype="Byte",
            nodata=0,
        )

        out_building_height = Path("city_lidar") / city / f"{city}_building_height_2m_{version}.tif"
        out_vegetation_height = Path("city_lidar") / city / f"{city}_vegetation_height_2m_{version}.tif"
        logger.info("Computing building/vegetation height rasters (blockwise)...")
        stats = _write_heights_blockwise(
            ndsm_2m=ndsm_2m,
            building_mask_2m=building_mask_2m,
            green_final_2m=green_final_2m,
            out_building_height=out_building_height,
            out_vegetation_height=out_vegetation_height,
        )

        logger.info(
            "Done. building pixels=%s mean=%.2f max=%.2f | vegetation pixels=%s mean=%.2f max=%.2f",
            f"{stats['building_pixels']:,}",
            stats["building_mean"],
            stats["building_max"],
            f"{stats['vegetation_pixels']:,}",
            stats["vegetation_mean"],
            stats["vegetation_max"],
        )

        manifest["cities"][city] = {
            "boundary_27700": str(boundary_27700),
            "buildings_clip": str(buildings_clip),
            "greenspace_clip": str(greenspace_clip),
            "ndvi_binary_10m": str(ndvi_binary_10m),
            "os_greenspace_10m": str(os_greenspace_10m),
            "green_final_10m": str(green_final_10m),
            "green_final_2m": str(green_final_2m),
            "building_mask_2m": str(building_mask_2m),
            "building_height_2m": str(out_building_height),
            "vegetation_height_2m": str(out_vegetation_height),
            "stats": stats,
        }

    manifest_path = results_dir / f"landcover_manifest_{version}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("=" * 80)
    logger.info(f"All done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()




