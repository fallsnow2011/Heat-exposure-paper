import subprocess
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Paths / 璺緞閰嶇疆
LIDAR_DIR = Path('/mnt/f/mike/Lidar-data-England')
BOUNDARY_DIR = Path('city_boundaries')
OUTPUT_DIR = Path('city_lidar')
OUTPUT_DIR.mkdir(exist_ok=True)

# Source datasets / 婧愭暟鎹?
DSM_FILE = LIDAR_DIR / 'england_dsm_merged.tif'
DTM_FILE = LIDAR_DIR / 'england_lidar_dtm_2m.tif'
GREEN_FILE = LIDAR_DIR / 'final_green_england.tif'

CITIES = ['London', 'Birmingham', 'Bristol', 'Manchester', 'Newcastle']

def get_boundary_bbox_bng(city_name):
    """Get the city-boundary bounding box in British National Grid coordinates / 鑾峰彇鍩庡競杈圭晫鐨?BNG 鍧愭爣 bbox"""
    boundary = gpd.read_file(BOUNDARY_DIR / f'{city_name}_boundary.geojson')
    # Reproject to British National Grid (EPSG:27700) / 杞崲鍒?British National Grid锛圗PSG:27700锛?    boundary_bng = boundary.to_crs(epsg=27700)
    bounds = boundary_bng.total_bounds  # [minx, miny, maxx, maxy]
    return bounds

def clip_raster_gdal(input_file, output_file, bounds, nodata=None):
    """Clip a raster with GDAL using the boundary bounding box / 浣跨敤 GDAL 瑁佸壀鏍呮牸"""
    minx, miny, maxx, maxy = bounds

    cmd = [
        'gdalwarp',
        '-te', str(minx), str(miny), str(maxx), str(maxy),
        '-co', 'COMPRESS=LZW',
        '-co', 'TILED=YES',
        '-co', 'BIGTIFF=YES',
        '-overwrite'
    ]

    if nodata is not None:
        cmd.extend(['-dstnodata', str(nodata)])

    cmd.extend([str(input_file), str(output_file)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error / 閿欒: {result.stderr}")
        return False
    return True

def calculate_ndsm(dsm_file, dtm_file, output_file):
    """Compute nDSM as DSM minus DTM / 璁＄畻 nDSM = DSM - DTM"""
    with rasterio.open(dsm_file) as dsm_src:
        dsm = dsm_src.read(1)
        profile = dsm_src.profile.copy()
        dsm_nodata = dsm_src.nodata

    with rasterio.open(dtm_file) as dtm_src:
        dtm = dtm_src.read(1)
        dtm_nodata = dtm_src.nodata

    # Compute nDSM / 璁＄畻 nDSM
    ndsm = dsm - dtm

    # Handle NoData values / 澶勭悊 NoData
    nodata_mask = (dsm == dsm_nodata) | (dtm == dtm_nodata)
    ndsm[nodata_mask] = -9999

    # Set negative values to 0 because DTM can be slightly higher than DSM in some cells / 灏嗚礋鍊艰涓?0锛堟煇浜涘尯鍩?DTM 鍙兘鐣ラ珮浜?DSM锛?
    ndsm[(ndsm < 0) & (~nodata_mask)] = 0

    # Save output / 淇濆瓨
    profile.update(nodata=-9999)
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(ndsm, 1)

    return ndsm[~nodata_mask]

def main():
    print("="*60)
    print("Clip city LiDAR datasets / 瑁佸壀鍩庡競 LiDAR 鏁版嵁")
    print("="*60)

    for city in CITIES:
        print(f"\n{'='*60}")
        print(f"Processing {city} / 澶勭悊 {city}")
        print("="*60)

        city_dir = OUTPUT_DIR / city
        city_dir.mkdir(exist_ok=True)

        # Get the boundary / 鑾峰彇杈圭晫
        bounds = get_boundary_bbox_bng(city)
        print(f"  Boundary in BNG / 杈圭晫锛圔NG锛? [{bounds[0]:.0f}, {bounds[1]:.0f}, {bounds[2]:.0f}, {bounds[3]:.0f}]")

        # Clip the DSM raster / 瑁佸壀 DSM
        dsm_out = city_dir / f'{city}_DSM_2m.tif'
        if not dsm_out.exists():
            print(f"  Clipping DSM / 瑁佸壀 DSM...")
            clip_raster_gdal(DSM_FILE, dsm_out, bounds)
        else:
            print(f"  DSM already exists, skipping / DSM 宸插瓨鍦紝璺宠繃")

        # Clip the DTM raster / 瑁佸壀 DTM
        dtm_out = city_dir / f'{city}_DTM_2m.tif'
        if not dtm_out.exists():
            print(f"  Clipping DTM / 瑁佸壀 DTM...")
            clip_raster_gdal(DTM_FILE, dtm_out, bounds)
        else:
            print(f"  DTM already exists, skipping / DTM 宸插瓨鍦紝璺宠繃")

        # Clip the green-cover raster / 瑁佸壀 Green
        green_out = city_dir / f'{city}_Green_10m.tif'
        if not green_out.exists():
            print(f"  Clipping Green raster / 瑁佸壀 Green...")
            clip_raster_gdal(GREEN_FILE, green_out, bounds)
        else:
            print(f"  Green raster already exists, skipping / Green 宸插瓨鍦紝璺宠繃")

        # Compute nDSM / 璁＄畻 nDSM
        ndsm_out = city_dir / f'{city}_nDSM_2m.tif'
        if dsm_out.exists() and dtm_out.exists():
            if not ndsm_out.exists():
                print(f"  Computing nDSM (DSM - DTM) / 璁＄畻 nDSM锛圖SM - DTM锛?..")
                ndsm_values = calculate_ndsm(dsm_out, dtm_out, ndsm_out)
                print(f"    nDSM range / nDSM 鑼冨洿: {ndsm_values.min():.2f} - {ndsm_values.max():.2f} m")
                print(f"    Mean height / 骞冲潎楂樺害: {ndsm_values.mean():.2f} m")
            else:
                print(f"  nDSM already exists, skipping / nDSM 宸插瓨鍦紝璺宠繃")

        # Report output file sizes / 妫€鏌ユ枃浠跺ぇ灏?        for f in [dsm_out, dtm_out, green_out, ndsm_out]:
            if f.exists():
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"  {f.name}: {size_mb:.1f} MB")

    print("\n" + "="*60)
    print("All cities processed / 鎵€鏈夊煄甯傚鐞嗗畬鎴愶紒")
    print("="*60)

if __name__ == '__main__':
    main()



