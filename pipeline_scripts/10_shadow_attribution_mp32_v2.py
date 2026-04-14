import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from shapely import wkt
from pathlib import Path
import argparse
import warnings
import logging
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time

warnings.filterwarnings('ignore')

# Logging setup / 璁剧疆鏃ュ織
LOG_DIR = Path('results/shadow_attribution')
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f'shadow_attribution_mp32_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration / 閰嶇疆
CITIES = ['London', 'Birmingham', 'Bristol', 'Manchester', 'Newcastle']
LIDAR_DIR = Path('city_lidar')
BOUNDARY_DIR = Path('city_boundaries')
SHADOW_DIR = Path('shadow_maps')
OUTPUT_DIR = Path('results/shadow_attribution')

BUFFER_DISTANCE = 15.0  # metres, road buffer distance / 绫筹紝閬撹矾 buffer 璺濈
HEATWAVE_DATES = ['2022-08-11', '2022-08-12']
HOURS = list(range(9, 18))  # 09:00 - 17:00
N_WORKERS = 32  # use 32 worker processes / 浣跨敤 32 涓繘绋?
CHUNK_SIZE = 100  # chunk size handled by each worker / 姣忎釜 worker 澶勭悊鐨?chunk 澶у皬

# Global variables for multiprocessing - land-cover calculation / 鍏ㄥ眬鍙橀噺鐢ㄤ簬澶氳繘绋嬶細land-cover 璁＄畻
_building_path = None
_vegetation_path = None
_buffer_dist = None

# Global variables for multiprocessing - shadow sampling / 鍏ㄥ眬鍙橀噺鐢ㄤ簬澶氳繘绋嬶細闃村奖閲囨牱
_shadow_path = None
_sample_step = None
_shadow_nodata = None


def init_worker_landcover(building_path, vegetation_path, buffer_dist):
    """Initialize process-wide globals for land-cover calculations / 鍒濆鍖?worker 杩涚▼鐨勫叏灞€鍙橀噺锛歭and-cover 璁＄畻"""
    global _building_path, _vegetation_path, _buffer_dist
    _building_path = building_path
    _vegetation_path = vegetation_path
    _buffer_dist = buffer_dist


def init_worker_shadow(shadow_path, sample_step):
    """Initialize process-wide globals for shade sampling / 鍒濆鍖?worker 杩涚▼鐨勫叏灞€鍙橀噺锛氶槾褰遍噰鏍?""
    global _shadow_path, _sample_step, _shadow_nodata
    _shadow_path = shadow_path
    _sample_step = sample_step
    with rasterio.open(_shadow_path) as src:
        _shadow_nodata = src.nodata if src.nodata is not None else 255


def process_single_road_landcover(args):
    """Compute building and vegetation proportions within one road buffer / 澶勭悊鍗曟潯閬撹矾锛岃绠?buffer 鍐呭缓绛?妞嶈鍗犳瘮"""
    idx, geometry_wkt = args

    geometry = wkt.loads(geometry_wkt)

    try:
        # Create a buffer / 鍒涘缓 buffer
        buffer_geom = geometry.buffer(_buffer_dist)
        geom_json = [mapping(buffer_geom)]

        # Extract building data / 鎻愬彇寤虹瓚鏁版嵁
        try:
            with rasterio.open(_building_path) as src:
                building_crop, _ = mask(src, geom_json, crop=True, nodata=0)
            building_pixels = building_crop[0]
            building_count = np.sum(building_pixels > 0)
        except:
            building_count = 0

        # Extract vegetation data / 鎻愬彇妞嶈鏁版嵁
        try:
            with rasterio.open(_vegetation_path) as src:
                vegetation_crop, _ = mask(src, geom_json, crop=True, nodata=0)
            vegetation_pixels = vegetation_crop[0]
            vegetation_count = np.sum(vegetation_pixels > 0)
        except:
            vegetation_count = 0

        total_count = building_count + vegetation_count

        if total_count > 0:
            building_ratio = building_count / total_count
            vegetation_ratio = vegetation_count / total_count
        else:
            building_ratio = 0.5
            vegetation_ratio = 0.5

        return idx, building_ratio, vegetation_ratio

    except Exception as e:
        return idx, 0.5, 0.5


def process_single_road_shadow(args):
    """Sample one shadow raster along a road geometry / 澶勭悊鍗曟潯閬撹矾锛岄噰鏍烽槾褰卞€?""
    idx, geometry_wkt = args

    geometry = wkt.loads(geometry_wkt)

    try:
        length = geometry.length

        if length < _sample_step:
            # If the road is too short, sample only the midpoint / 閬撹矾澶煭鏃剁洿鎺ラ噰鏍蜂腑鐐?
            point = geometry.interpolate(0.5, normalized=True)
            coords = [(point.x, point.y)]
        else:
            # Sample uniformly along each road / 娌块亾璺潎鍖€閲囨牱
            n_samples = max(2, int(length / _sample_step))
            coords = []
            for j in range(n_samples):
                frac = j / (n_samples - 1) if n_samples > 1 else 0.5
                point = geometry.interpolate(frac, normalized=True)
                coords.append((point.x, point.y))

        # Sample from the raster / 浠庢爡鏍奸噰鏍?
        with rasterio.open(_shadow_path) as src:
            values = list(src.sample(coords))
        values = [v[0] for v in values]

        # Filter NoData values / 杩囨护 NoData
        nodata_value = _shadow_nodata if _shadow_nodata is not None else 255
        valid_values = [v for v in values if v != nodata_value]

        if len(valid_values) > 0:
            # In the shadow mask, 0 = shade and 1 = illuminated / shadow_mask 涓?0 = 闃村奖锛? = 鍏夌収
            # Shadow ratio = number of shaded samples divided by the total sample count / 闃村奖姣斾緥 = 闃村奖鍍忕礌鏁?/ 鎬诲儚绱犳暟
            shadow_count = sum(1 for v in valid_values if v == 0)
            shadow_ratio = shadow_count / len(valid_values)
            return idx, shadow_ratio
        else:
            return idx, np.nan

    except Exception as e:
        return idx, np.nan


def calculate_buffer_landcover_ratio_parallel(roads_gdf, building_height_path, vegetation_height_path,
                                              buffer_dist=15.0, n_workers=32, chunk_size=100):
    """
    Calculate building and vegetation pixel ratios within each road buffer in parallel / 浣跨敤澶氳繘绋嬪苟琛岃绠楁瘡鏉￠亾璺?buffer 鍐呯殑寤虹瓚鍜屾琚儚绱犲崰姣?    """
    logger.info(f"Computing land-cover ratios with {n_workers} workers / 浣跨敤 {n_workers} 涓繘绋嬪苟琛岃绠?landcover 鍗犳瘮...")

    # Prepare arguments by passing geometries as WKT strings so they remain serializable / 鍑嗗鍙傛暟锛氫娇鐢?WKT 瀛楃涓蹭紶閫掑嚑浣曚綋锛屼互渚垮簭鍒楀寲
    args_list = [(i, geom.wkt) for i, geom in enumerate(roads_gdf.geometry)]

    n_roads = len(roads_gdf)
    building_ratios = np.zeros(n_roads)
    vegetation_ratios = np.zeros(n_roads)

    start_time = time.time()

    # Use a process pool / 浣跨敤杩涚▼姹?
    with Pool(
        processes=n_workers,
        initializer=init_worker_landcover,
        initargs=(str(building_height_path), str(vegetation_height_path), buffer_dist)
    ) as pool:
        # Use imap_unordered for better throughput while keeping tqdm progress reporting / 浣跨敤 imap_unordered 鎻愰珮鏁堢巼锛屽苟閰嶅悎 tqdm 鏄剧ず杩涘害
        results = list(tqdm(
            pool.imap_unordered(process_single_road_landcover, args_list, chunksize=chunk_size),
            total=n_roads,
            desc="Computing land-cover ratios / 璁＄畻 landcover 鍗犳瘮锛?2 杩涚▼锛?
        ))

    # Organize results / 鏁寸悊缁撴灉
    for idx, b_ratio, v_ratio in results:
        building_ratios[idx] = b_ratio
        vegetation_ratios[idx] = v_ratio

    elapsed = time.time() - start_time
    speed = n_roads / elapsed
    logger.info(f"Land-cover calculation complete / landcover 璁＄畻瀹屾垚: {n_roads} 鏉￠亾璺? 鑰楁椂 {elapsed:.1f} 绉? 閫熷害 {speed:.1f} 鏉?绉?)

    return building_ratios, vegetation_ratios


def sample_shadow_to_roads_parallel(roads_gdf, shadow_path, sample_step=2.0, n_workers=32, chunk_size=100):
    """
    Sample shade values from a raster onto roads in parallel / 浣跨敤澶氳繘绋嬪苟琛岄噰鏍烽槾褰辨爡鏍煎€煎埌閬撹矾

    Parameters / 鍙傛暟锛?    - roads_gdf: road GeoDataFrame / 閬撹矾 GeoDataFrame
    - shadow_path: path to the shadow raster / 闃村奖鏍呮牸璺緞
    - sample_step: sampling interval in metres / 閲囨牱姝ラ暱锛堢背锛?    - n_workers: number of worker processes / 杩涚▼鏁?    - chunk_size: number of roads processed per worker chunk / 姣忎釜 worker 澶勭悊鐨?chunk 澶у皬

    Returns / 杩斿洖锛?    - shadow_ratios: shade fraction for each road (0 = fully sunlit, 1 = fully shaded) / 姣忔潯閬撹矾鐨勯槾褰辨瘮渚嬶紙0 = 鍏ㄥ厜鐓э紝1 = 鍏ㄩ槾褰憋級
    """
    # Prepare arguments by passing geometries as WKT strings / 鍑嗗鍙傛暟锛氫娇鐢?WKT 瀛楃涓蹭紶閫掑嚑浣曚綋
    args_list = [(i, geom.wkt) for i, geom in enumerate(roads_gdf.geometry)]

    n_roads = len(roads_gdf)
    shadow_ratios = np.full(n_roads, np.nan)

    # Use a process pool / 浣跨敤杩涚▼姹?
    with Pool(
        processes=n_workers,
        initializer=init_worker_shadow,
        initargs=(str(shadow_path), sample_step)
    ) as pool:
        results = list(pool.imap_unordered(process_single_road_shadow, args_list, chunksize=chunk_size))

    # Organize results / 鏁寸悊缁撴灉
    for idx, shadow_ratio in results:
        shadow_ratios[idx] = shadow_ratio

    return shadow_ratios


def process_city(city_name):
    """Process all heatwave shade-attribution steps for a single city / 澶勭悊鍗曚釜鍩庡競"""
    logger.info(f"{'='*60}")
    logger.info(f"Processing {city_name} / 澶勭悊 {city_name}")
    logger.info(f"{'='*60}")

    city_start_time = time.time()

    # Paths / 璺緞
    roads_path = BOUNDARY_DIR / f'{city_name}_roads_OS.gpkg'
    suffix = f"_{VERSION}" if VERSION else ""
    building_height_path = LIDAR_DIR / city_name / f'{city_name}_building_height_2m{suffix}.tif'
    vegetation_height_path = LIDAR_DIR / city_name / f'{city_name}_vegetation_height_2m{suffix}.tif'

    # Check required inputs / 妫€鏌ユ枃浠?    if not roads_path.exists():
        logger.error(f"Skip: road file not found / 璺宠繃锛氶亾璺枃浠朵笉瀛樺湪 {roads_path}")
        return None
    if not building_height_path.exists():
        logger.error(f"Skip: building-height file not found / 璺宠繃锛氬缓绛戦珮搴︽枃浠朵笉瀛樺湪 {building_height_path}")
        return None
    if not vegetation_height_path.exists():
        logger.error(f"Skip: vegetation-height file not found / 璺宠繃锛氭琚珮搴︽枃浠朵笉瀛樺湪 {vegetation_height_path}")
        return None

    # Read the road network / 璇诲彇閬撹矾
    logger.info("Reading road network / 璇诲彇閬撹矾缃戠粶...")
    roads = gpd.read_file(roads_path)
    logger.info(f"Road count / 閬撹矾鏁? {len(roads)}")

    # Ensure CRS consistency / 纭繚 CRS 涓€鑷?
    with rasterio.open(building_height_path) as src:
        raster_crs = src.crs
    if roads.crs != raster_crs:
        logger.info(f"Reprojecting roads from {roads.crs} to {raster_crs} / 杞崲閬撹矾 CRS 浠?{roads.crs} 鍒?{raster_crs}")
        roads = roads.to_crs(raster_crs)

    # Compute land-cover ratios within each road buffer / 浣跨敤澶氳繘绋嬭绠?buffer 鍐呭缓绛?妞嶈鍗犳瘮
    logger.info(f"Computing building/vegetation ratios in road buffers (buffer={BUFFER_DISTANCE} m, workers={N_WORKERS}) / 璁＄畻 buffer 鍐呭缓绛?妞嶈鍗犳瘮 (buffer={BUFFER_DISTANCE}m, workers={N_WORKERS})...")
    building_ratios, vegetation_ratios = calculate_buffer_landcover_ratio_parallel(
        roads, building_height_path, vegetation_height_path,
        BUFFER_DISTANCE, n_workers=N_WORKERS, chunk_size=CHUNK_SIZE
    )
    roads['building_ratio'] = building_ratios
    roads['vegetation_ratio'] = vegetation_ratios

    logger.info(f"Mean building ratio / 骞冲潎寤虹瓚鍗犳瘮: {np.mean(building_ratios):.2%}")
    logger.info(f"Mean vegetation ratio / 骞冲潎妞嶈鍗犳瘮: {np.mean(vegetation_ratios):.2%}")

    # Loop through each date and hour using parallel shade sampling / 澶勭悊姣忎釜鏃ユ湡鍜屽皬鏃剁殑闃村奖锛屼娇鐢ㄥ杩涚▼骞惰
    for date_str in HEATWAVE_DATES:
        shadow_date_dir = SHADOW_DIR / city_name / date_str

        if not shadow_date_dir.exists():
            logger.warning(f"Skip date {date_str}: shadow directory not found / 璺宠繃鏃ユ湡 {date_str}锛氶槾褰辩洰褰曚笉瀛樺湪")
            continue

        logger.info(f"Processing {date_str} with parallel shade sampling / 澶勭悊 {date_str}锛堝杩涚▼闃村奖閲囨牱锛?..")

        for hour in HOURS:
            shadow_file = shadow_date_dir / f"shadow_{date_str}_{hour:02d}00.tif"

            if not shadow_file.exists():
                logger.warning(f"Skip {hour:02d}:00 - file not found / 璺宠繃 {hour:02d}:00锛屾枃浠朵笉瀛樺湪")
                continue

            # Sample shade values to roads in parallel / 澶氳繘绋嬮噰鏍烽槾褰卞埌閬撹矾
            hour_start = time.time()
            shadow_ratios = sample_shadow_to_roads_parallel(
                roads, shadow_file, sample_step=2.0,
                n_workers=N_WORKERS, chunk_size=CHUNK_SIZE
            )
            hour_elapsed = time.time() - hour_start

            # Allocate building and vegetation shade contributions / 鍒嗛厤寤虹瓚鍜屾琚础鐚?            # S_building = S_total 脳 building_ratio
            # S_vegetation = S_total 脳 vegetation_ratio
            col_shadow = f'shadow_{hour:02d}'
            col_shadow_building = f'shadow_building_{hour:02d}'
            col_shadow_vegetation = f'shadow_vegetation_{hour:02d}'

            roads[col_shadow] = shadow_ratios
            roads[col_shadow_building] = shadow_ratios * building_ratios
            roads[col_shadow_vegetation] = shadow_ratios * vegetation_ratios

            logger.info(f"  {hour:02d}:00 finished / 瀹屾垚, 鑰楁椂 {hour_elapsed:.1f} 绉? 骞冲潎闃村奖 {np.nanmean(shadow_ratios):.2%}")

    # Compute daily means across hourly rasters / 璁＄畻鏃ュ潎鍊?    shadow_cols = [c for c in roads.columns if c.startswith('shadow_') and len(c) == 9]  # shadow_HH
    if shadow_cols:
        roads['shadow_daily_avg'] = roads[shadow_cols].mean(axis=1)

        building_cols = [c for c in roads.columns if c.startswith('shadow_building_')]
        vegetation_cols = [c for c in roads.columns if c.startswith('shadow_vegetation_')]

        if building_cols:
            roads['shadow_building_avg'] = roads[building_cols].mean(axis=1)
        if vegetation_cols:
            roads['shadow_vegetation_avg'] = roads[vegetation_cols].mean(axis=1)

    # Save output / 淇濆瓨缁撴灉
    output_path = OUTPUT_DIR / f'{city_name}_roads_shadow_attribution_v2{suffix}.gpkg'
    roads.to_file(output_path, driver='GPKG')
    logger.info(f"Saved output / 淇濆瓨: {output_path}")

    city_elapsed = time.time() - city_start_time
    logger.info(f"{city_name} finished / 瀹屾垚, 鎬昏€楁椂: {city_elapsed/60:.1f} 鍒嗛挓")

    # Statistics / 缁熻
    if 'shadow_daily_avg' in roads.columns:
        logger.info("Daily mean shade summary / 鏃ュ潎闃村奖缁熻:")
        logger.info(f"  Total shade / 鎬婚槾褰? {roads['shadow_daily_avg'].mean():.2%}")
        if 'shadow_building_avg' in roads.columns:
            logger.info(f"  Building contribution / 寤虹瓚璐＄尞: {roads['shadow_building_avg'].mean():.2%}")
        if 'shadow_vegetation_avg' in roads.columns:
            logger.info(f"  Vegetation contribution / 妞嶈璐＄尞: {roads['shadow_vegetation_avg'].mean():.2%}")

    return roads


def main():
    global VERSION
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None, help="Optional version suffix for v3 landcover runs")
    args = parser.parse_args()
    VERSION = args.version

    logger.info("="*60)
    logger.info("Road-level shade attribution to building and vegetation sources - 32-process version v2 / 閬撹矾闃村奖寤虹瓚/妞嶈璐＄尞鍒嗛厤璁＄畻锛?2 杩涚▼澶氳繘绋嬬増 v2")
    logger.info("v2 improvement / v2 鏀硅繘: both land-cover calculation and shade sampling use multiprocessing / landcover 璁＄畻 + 闃村奖閲囨牱鍧囦娇鐢ㄥ杩涚▼")
    logger.info("="*60)
    logger.info(f"Log file / 鏃ュ織鏂囦欢: {log_file}")
    logger.info(f"Version / 鐗堟湰: {VERSION or '(none)'}")
    logger.info(f"Worker count / 杩涚▼鏁? {N_WORKERS}")
    logger.info(f"Chunk size / Chunk 澶у皬: {CHUNK_SIZE}")

    total_start = time.time()

    # Create the output directory / 鍒涘缓杈撳嚭鐩綍
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each city / 澶勭悊姣忎釜鍩庡競
    results = {}
    for city in CITIES:
        try:
            result = process_city(city)
            if result is not None:
                results[city] = result
        except Exception as e:
            logger.error(f"Error while processing {city} / 澶勭悊 {city} 鏃跺嚭閿? {e}")
            import traceback
            logger.error(traceback.format_exc())

    total_elapsed = time.time() - total_start

    logger.info("="*60)
    logger.info(f"All processing complete / 鍏ㄩ儴澶勭悊瀹屾垚锛佹€昏€楁椂: {total_elapsed/60:.1f} 鍒嗛挓")
    logger.info("="*60)


if __name__ == '__main__':
    VERSION = None
    main()



