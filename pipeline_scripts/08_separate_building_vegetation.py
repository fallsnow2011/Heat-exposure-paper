import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.enums import Resampling
from rasterio.warp import reproject
import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths / 璺緞閰嶇疆
LIDAR_DIR = Path('city_lidar')
BOUNDARY_DIR = Path('city_boundaries')
OUTPUT_DIR = Path('city_lidar')

CITIES = ['London', 'Birmingham', 'Bristol', 'Manchester', 'Newcastle']

def resample_green_to_2m(green_file, ndsm_file, output_file):
    """Resample the 10 m green raster to 2 m to match the nDSM grid / 灏?10 m Green 鏁版嵁閲嶉噰鏍峰埌 2 m 浠ュ尮閰?nDSM"""
    with rasterio.open(ndsm_file) as ndsm_src:
        target_transform = ndsm_src.transform
        target_shape = (ndsm_src.height, ndsm_src.width)
        target_crs = ndsm_src.crs
        target_profile = ndsm_src.profile.copy()

    with rasterio.open(green_file) as green_src:
        dst_nodata = green_src.nodata if green_src.nodata is not None else 0
        green_resampled = np.full(target_shape, dst_nodata, dtype=green_src.dtypes[0])
        reproject(
            source=rasterio.band(green_src, 1),
            destination=green_resampled,
            src_transform=green_src.transform,
            src_crs=green_src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
            src_nodata=green_src.nodata,
            dst_nodata=dst_nodata,
        )

    target_profile.update(dtype='uint8', nodata=dst_nodata)
    with rasterio.open(output_file, 'w', **target_profile) as dst:
        dst.write(green_resampled.astype('uint8'), 1)

    return green_resampled

def rasterize_buildings(buildings_file, ndsm_file, output_file):
    """Rasterize building footprints into a 2 m mask / 灏嗗缓绛?footprint 鏍呮牸鍖栦负 2 m 鍒嗚鲸鐜囨帺鑶?""
    # Read building footprints / 璇诲彇寤虹瓚
    buildings = gpd.read_file(buildings_file)

    with rasterio.open(ndsm_file) as ndsm_src:
        transform = ndsm_src.transform
        shape = (ndsm_src.height, ndsm_src.width)
        crs = ndsm_src.crs
        profile = ndsm_src.profile.copy()

    # Ensure CRS consistency / 纭繚 CRS 涓€鑷?
    if buildings.crs != crs:
        buildings = buildings.to_crs(crs)

    # Rasterize footprints so building cells = 1 and non-building cells = 0 / 鏍呮牸鍖栵細寤虹瓚鍖哄煙 = 1锛岄潪寤虹瓚 = 0
    building_mask = rasterize(
        [(geom, 1) for geom in buildings.geometry],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    profile.update(dtype='uint8', nodata=255)
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(building_mask, 1)

    return building_mask

def separate_building_vegetation(ndsm_file, green_2m_file, building_mask_file, output_dir, city):
    """Separate building and vegetation heights on the common 2 m grid / 鍒嗙寤虹瓚鍜屾琚珮搴?""
    with rasterio.open(ndsm_file) as src:
        ndsm = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    with rasterio.open(green_2m_file) as src:
        green = src.read(1)

    with rasterio.open(building_mask_file) as src:
        building_mask = src.read(1)

    # Valid-data mask based on nDSM nodata / 鏈夋晥鏁版嵁鎺╄啘
    valid_mask = ndsm != nodata

    # Building height: nDSM values inside building footprints / 寤虹瓚楂樺害锛氬湪寤虹瓚 footprint 鍐?    building_height = np.where(
        (building_mask == 1) & valid_mask,
        ndsm,
        0
    )

    # Vegetation height: nDSM values in green cells and outside buildings / 妞嶈楂樺害锛氬湪缁垮湴鍐呬笖涓嶅湪寤虹瓚鍐?    vegetation_height = np.where(
        (green == 1) & (building_mask == 0) & valid_mask,
        ndsm,
        0
    )

    # Save output / 淇濆瓨寤虹瓚楂樺害
    profile.update(nodata=0)
    building_out = output_dir / f'{city}_building_height_2m.tif'
    with rasterio.open(building_out, 'w', **profile) as dst:
        dst.write(building_height.astype('float32'), 1)

    # Save output / 淇濆瓨妞嶈楂樺害
    vegetation_out = output_dir / f'{city}_vegetation_height_2m.tif'
    with rasterio.open(vegetation_out, 'w', **profile) as dst:
        dst.write(vegetation_height.astype('float32'), 1)

    # Statistics / 缁熻
    building_valid = building_height[building_height > 0]
    vegetation_valid = vegetation_height[vegetation_height > 0]

    return {
        'building_pixels': len(building_valid),
        'building_mean_height': building_valid.mean() if len(building_valid) > 0 else 0,
        'building_max_height': building_valid.max() if len(building_valid) > 0 else 0,
        'vegetation_pixels': len(vegetation_valid),
        'vegetation_mean_height': vegetation_valid.mean() if len(vegetation_valid) > 0 else 0,
        'vegetation_max_height': vegetation_valid.max() if len(vegetation_valid) > 0 else 0
    }

def main():
    print("="*60)
    print("Separate building and vegetation heights / 鍒嗙寤虹瓚鍜屾琚珮搴?)
    print("="*60)

    for city in CITIES:
        print(f"\n{'='*60}")
        print(f"Processing {city} / 澶勭悊 {city}")
        print("="*60)

        city_dir = LIDAR_DIR / city

        # Input files / 杈撳叆鏂囦欢
        ndsm_file = city_dir / f'{city}_nDSM_2m.tif'
        green_file = city_dir / f'{city}_Green_10m.tif'
        buildings_file = BOUNDARY_DIR / f'{city}_buildings_clipped.gpkg'

        # Intermediate files / 涓棿鏂囦欢
        green_2m_file = city_dir / f'{city}_Green_2m.tif'
        building_mask_file = city_dir / f'{city}_building_mask_2m.tif'

        # Check required inputs / 妫€鏌ヨ緭鍏?        if not ndsm_file.exists():
            print(f"  Skip: nDSM file not found / 璺宠繃锛歯DSM 鏂囦欢涓嶅瓨鍦?)
            continue

        # 1. Resample the green raster to 2 m / 閲嶉噰鏍?Green 鍒?2 m
        if not green_2m_file.exists():
            print("  Resampling Green raster to 2 m / 閲嶉噰鏍?Green 鍒?2 m...")
            resample_green_to_2m(green_file, ndsm_file, green_2m_file)
        else:
            print("  Green 2 m raster already exists / Green 2 m 宸插瓨鍦?)

        # 2. Rasterize building footprints / 鏍呮牸鍖栧缓绛?footprint
        if not building_mask_file.exists():
            print("  Rasterizing building footprints / 鏍呮牸鍖栧缓绛?footprint...")
            rasterize_buildings(buildings_file, ndsm_file, building_mask_file)
        else:
            print("  Building mask already exists / 寤虹瓚鎺╄啘宸插瓨鍦?)

        # 3. Separate building and vegetation heights / 鍒嗙寤虹瓚鍜屾琚珮搴?        building_height_file = city_dir / f'{city}_building_height_2m.tif'
        vegetation_height_file = city_dir / f'{city}_vegetation_height_2m.tif'

        if not building_height_file.exists() or not vegetation_height_file.exists():
            print("  Separating building and vegetation heights / 鍒嗙寤虹瓚鍜屾琚珮搴?..")
            stats = separate_building_vegetation(
                ndsm_file, green_2m_file, building_mask_file,
                city_dir, city
            )

            print(f"  Building statistics / 寤虹瓚缁熻:")
            print(f"    Pixel count / 鍍忕礌鏁? {stats['building_pixels']:,}")
            print(f"    Mean height / 骞冲潎楂樺害: {stats['building_mean_height']:.2f} m")
            print(f"    Maximum height / 鏈€澶ч珮搴? {stats['building_max_height']:.2f} m")

            print(f"  Vegetation statistics / 妞嶈缁熻:")
            print(f"    Pixel count / 鍍忕礌鏁? {stats['vegetation_pixels']:,}")
            print(f"    Mean height / 骞冲潎楂樺害: {stats['vegetation_mean_height']:.2f} m")
            print(f"    Maximum height / 鏈€澶ч珮搴? {stats['vegetation_max_height']:.2f} m")
        else:
            print("  Building/vegetation height rasters already exist / 寤虹瓚/妞嶈楂樺害宸插瓨鍦?)

        # Report output file sizes / 鏂囦欢澶у皬
        for f in [building_height_file, vegetation_height_file]:
            if f.exists():
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"  {f.name}: {size_mb:.1f} MB")

    print("\n" + "="*60)
    print("All cities processed / 鎵€鏈夊煄甯傚鐞嗗畬鎴愶紒")
    print("="*60)

if __name__ == '__main__':
    main()



