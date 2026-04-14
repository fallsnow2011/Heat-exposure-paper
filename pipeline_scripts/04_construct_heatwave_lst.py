import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ==================== Configuration / 閰嶇疆 ====================
# Data directories aligned with the repository README / 鏁版嵁鐩綍锛堜笌 README 淇濇寔涓€鑷达級
# - baseline: `GEE_LST_Baseline/LST_median_summer_*.tif`
# - 螖T:      `GEE_LST_Baseline/delta_t/delta_t_summary_all_cities.csv`
# - Outputs / 杈撳嚭锛?    `GEE_LST_Baseline/lst_scenarios/`
LST_BASELINE_DIR = Path("GEE_LST_Baseline")
DELTA_T_FILE = Path("GEE_LST_Baseline/delta_t/delta_t_summary_all_cities.csv")
OUTPUT_DIR = Path("GEE_LST_Baseline/lst_scenarios")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Study cities / 鍩庡競鍒楄〃
CITIES = ['London', 'Birmingham', 'Bristol', 'Manchester', 'Newcastle']

# ==================== Core functions / 鏍稿績鍑芥暟 ====================
def load_delta_t(csv_path: Path) -> dict:
    """Load the 螖T summary table / 鍔犺浇 螖T 姹囨€昏〃"""
    df = pd.read_csv(csv_path)
    # Use the median-based 螖T, which is more robust to outliers / 浣跨敤涓綅鏁拌绠楃殑 螖T锛堟洿绋冲仴锛?    delta_t_dict = dict(zip(df['city'], df['delta_t_median']))
    return delta_t_dict

def construct_heatwave_lst(baseline_path: Path, delta_t: float, output_path: Path):
    """
    Construct heatwave LST as baseline + 螖T / 鏋勫缓鐑氮鏃?LST = baseline + 螖T

    Parameters:
    -----------
    baseline_path : Path
        Path to the LST baseline GeoTIFF / LST baseline GeoTIFF 璺緞
    delta_t : float
        City-level warming increment (掳C) / 鍩庡競鍗囨俯閲忥紙掳C锛?    output_path : Path
        Output GeoTIFF path / 杈撳嚭 GeoTIFF 璺緞
    """
    with rasterio.open(baseline_path) as src:
        baseline = src.read(1)
        profile = src.profile.copy()

        # Build the heatwave LST by adding 螖T to valid pixels while preserving NoData / 鏋勫缓鐑氮 LST锛氭湁鏁堝儚绱?+ 螖T锛堜繚鎸?NoData/NaN 涓嶅彉锛?        nodata = src.nodata
        if nodata is not None:
            valid_mask = np.isfinite(baseline) & (baseline != nodata)
        else:
            valid_mask = np.isfinite(baseline)

        heatwave_lst = np.where(valid_mask, baseline + delta_t, baseline)

        # Update the raster profile for float output / 鏇存柊 profile
        profile.update(dtype=rasterio.float32)

        # Save output / 淇濆瓨
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(heatwave_lst.astype(np.float32), 1)

    return heatwave_lst

def create_visualization(city: str, baseline_path: Path, heatwave_path: Path,
                        delta_t: float, output_dir: Path):
    """Create a comparison figure for the baseline and heatwave LST rasters / 鍒涘缓瀵规瘮鍙鍖?""

    with rasterio.open(baseline_path) as src:
        baseline = src.read(1)

    with rasterio.open(heatwave_path) as src:
        heatwave = src.read(1)

    # Mask invalid values for plotting / 澶勭悊 NoData
    baseline = np.where((baseline > 0) & (baseline < 100), baseline, np.nan)
    heatwave = np.where((heatwave > 0) & (heatwave < 100), heatwave, np.nan)

    # Color ramp / 鑹插甫
    cmap = LinearSegmentedColormap.from_list('lst', ['blue', 'cyan', 'yellow', 'orange', 'red'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Typical summer baseline / 鍏稿瀷澶忔棩
    im1 = axes[0].imshow(baseline, cmap=cmap, vmin=20, vmax=45)
    axes[0].set_title(f'{city} - Typical Summer LST', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], shrink=0.7, label='LST (掳C)')

    # Heatwave scenario / 鐑氮鏃?    im2 = axes[1].imshow(heatwave, cmap=cmap, vmin=20, vmax=45)
    axes[1].set_title(f'{city} - Heatwave LST (+{delta_t:.1f}掳C)', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], shrink=0.7, label='LST (掳C)')

    # Difference map, which should approximate a uniform 螖T / 宸紓锛堢悊璁轰笂搴旀帴杩戝潎鍖€鐨?螖T锛?    diff = heatwave - baseline
    im3 = axes[2].imshow(diff, cmap='Reds', vmin=0, vmax=delta_t + 2)
    axes[2].set_title(f'{city} - Difference (螖T)', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], shrink=0.7, label='螖T (掳C)')

    plt.tight_layout()
    plt.savefig(output_dir / f'lst_comparison_{city}.png', dpi=150, bbox_inches='tight')
    plt.close()

def print_statistics(city: str, baseline_path: Path, heatwave_path: Path, delta_t: float):
    """Print summary statistics for the baseline and heatwave rasters / 鎵撳嵃缁熻淇℃伅"""

    with rasterio.open(baseline_path) as src:
        baseline = src.read(1)

    with rasterio.open(heatwave_path) as src:
        heatwave = src.read(1)

    # Valid pixels used for the summary / 鏈夋晥鍍忕礌
    valid = (baseline > 0) & (baseline < 100) & ~np.isnan(baseline)

    print(f"\n{city}:")
    print(f"  螖T applied: {delta_t:.2f} 掳C")
    print(f"  Typical-summer LST / 鍏稿瀷澶忔棩 LST: mean={baseline[valid].mean():.2f}掳C, "
          f"range=[{baseline[valid].min():.2f}, {baseline[valid].max():.2f}]掳C")
    print(f"  Heatwave LST / 鐑氮鏃?LST:   mean={heatwave[valid].mean():.2f}掳C, "
          f"range=[{heatwave[valid].min():.2f}, {heatwave[valid].max():.2f}]掳C")
    print(f"  Valid pixel count / 鏈夋晥鍍忕礌鏁? {valid.sum():,}")

def main():
    """Main entry point / 涓诲嚱鏁?""

    print("=" * 60)
    print("Construct heatwave LST = LST_baseline + 螖T / 鏋勫缓鐑氮鏃?LST = LST_baseline + 螖T")
    print("=" * 60)

    # Load the city-specific 螖T values / 鍔犺浇 螖T
    if not DELTA_T_FILE.exists():
        print(f"鈿?螖T file not found / 螖T 鏂囦欢涓嶅瓨鍦? {DELTA_T_FILE}")
        print("  Run `03_calculate_delta_t.py` first / 璇峰厛杩愯 `03_calculate_delta_t.py`")
        return

    delta_t_dict = load_delta_t(DELTA_T_FILE)
    print(f"\nLoaded 螖T values / 宸插姞杞?螖T 鏁版嵁:")
    for city, dt in delta_t_dict.items():
        print(f"  {city}: {dt:.2f} 掳C")

    # Process each city / 澶勭悊姣忎釜鍩庡競
    for city in CITIES:
        # Check the baseline file and allow alternative year tags / 妫€鏌?baseline 鏂囦欢锛堝吋瀹逛笉鍚屽懡鍚嶏細2018_2023 / 2022 绛夛級
        candidates = sorted(LST_BASELINE_DIR.glob(f"LST_median_summer_*_{city}_30m.tif"))
        baseline_path = candidates[0] if candidates else None

        if baseline_path is None or not baseline_path.exists():
            print(f"\n鈿?{city}: baseline file not found / baseline 鏂囦欢涓嶅瓨鍦?)
            print(f"  Expected pattern / 鏈熸湜鍖归厤: {LST_BASELINE_DIR}/LST_median_summer_*_{city}_30m.tif")
            print("  Export the baseline from GEE first and download it locally / 璇峰厛杩愯 GEE 鑴氭湰瀵煎嚭鏁版嵁骞朵笅杞藉埌鏈湴")
            continue

        if city not in delta_t_dict:
            print(f"\n鈿?{city}: missing 螖T record / 缂哄皯 螖T 鏁版嵁")
            continue

        delta_t = delta_t_dict[city]

        # Output paths / 杈撳嚭璺緞
        typical_path = OUTPUT_DIR / f"LST_typical_summer_{city}_30m.tif"
        heatwave_path = OUTPUT_DIR / f"LST_heatwave_2022_{city}_30m.tif"

        # Copy the baseline raster as the typical-summer scenario / 澶嶅埗 baseline 浣滀负鍏稿瀷澶忔棩
        import shutil
        shutil.copy(baseline_path, typical_path)

        # Build the heatwave raster / 鏋勫缓鐑氮 LST
        construct_heatwave_lst(baseline_path, delta_t, heatwave_path)

        # Statistics / 缁熻
        print_statistics(city, baseline_path, heatwave_path, delta_t)

        # Visualization / 鍙鍖?        create_visualization(city, typical_path, heatwave_path, delta_t, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Output files / 杈撳嚭鏂囦欢:")
    print("=" * 60)
    print(f"Directory / 鐩綍: {OUTPUT_DIR}")
    print("  - LST_typical_summer_<City>_30m.tif  (typical summer / 鍏稿瀷澶忔棩)")
    print("  - LST_heatwave_2022_<City>_30m.tif   (heatwave / 鐑氮鏃?")
    print("  - lst_comparison_<City>.png          (comparison figure / 瀵规瘮鍥?")

    print("\nNext step / 涓嬩竴姝?")
    print("  Use the two LST scenarios to compute HEI 鈫?CNI 鈫?TCNI / 浣跨敤杩欎袱涓満鏅殑 LST 璁＄畻 HEI 鈫?CNI 鈫?TCNI")
    print("  Compare cooling-network differences between heatwave and typical-summer conditions / 瀵规瘮鍒嗘瀽鐑氮 vs 鍏稿瀷澶忔棩鐨勯檷娓╃綉缁滃樊寮?)

# ==================== Example generator for development only / 绀轰緥鐢熸垚锛堜粎鐢ㄤ簬寮€鍙戞祴璇曪紝涓嶅湪榛樿娴佺▼涓嚜鍔ㄨ皟鐢級 ====================
def create_example_baseline():
    """Create a synthetic baseline raster for development tests / 鍒涘缓绀轰緥 baseline 鐢ㄤ簬娴嬭瘯"""
    import numpy as np

    LST_BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    # Synthetic baseline temperatures for the five cities / 鍚勫煄甯傜殑妯℃嫙鍩哄噯娓╁害
    city_params = {
        'London': {'center': (530000, 180000), 'base_temp': 32, 'size': (2000, 1500)},
        'Birmingham': {'center': (410000, 290000), 'base_temp': 30, 'size': (1000, 800)},
        'Bristol': {'center': (360000, 175000), 'base_temp': 29, 'size': (800, 600)},
        'Manchester': {'center': (385000, 400000), 'base_temp': 28, 'size': (1200, 900)},
        'Newcastle': {'center': (425000, 565000), 'base_temp': 26, 'size': (800, 600)}
    }

    for city, params in city_params.items():
        print(f"  Create example raster / 鍒涘缓绀轰緥: {city}")

        rows, cols = params['size']
        base = params['base_temp']

        # Generate a smooth spatial LST pattern / 鐢熸垚绌洪棿鍙樺寲鐨?LST
        np.random.seed(hash(city) % 2**32)
        x = np.linspace(0, 1, cols)
        y = np.linspace(0, 1, rows)
        xx, yy = np.meshgrid(x, y)

        # Simulate an urban heat-island gradient: warmer center, cooler edge / 妯℃嫙鍩庡競鐑矝锛堜腑蹇冪儹锛岃竟缂樺噳锛?        dist = np.sqrt((xx - 0.5)**2 + (yy - 0.5)**2)
        lst = base + 5 * (1 - dist) + np.random.normal(0, 1, (rows, cols))

        # Add several cooler green-space patches / 娣诲姞涓€浜涒€滅豢鍦扳€濆喎鐐?        for _ in range(10):
            cx, cy = np.random.randint(0, cols), np.random.randint(0, rows)
            for i in range(max(0, cy-20), min(rows, cy+20)):
                for j in range(max(0, cx-20), min(cols, cx+20)):
                    if np.sqrt((i-cy)**2 + (j-cx)**2) < 20:
                        lst[i, j] -= 3

        # Write the synthetic GeoTIFF / 鍒涘缓 GeoTIFF
        transform = rasterio.transform.from_bounds(
            params['center'][0] - cols*15,
            params['center'][1] - rows*15,
            params['center'][0] + cols*15,
            params['center'][1] + rows*15,
            cols, rows
        )

        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'width': cols,
            'height': rows,
            'count': 1,
            'crs': 'EPSG:27700',
            'transform': transform,
            'nodata': np.nan
        }

        output_path = LST_BASELINE_DIR / f"LST_median_summer_example_{city}_30m.tif"
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(lst.astype(np.float32), 1)

if __name__ == "__main__":
    if not LST_BASELINE_DIR.exists():
        raise SystemExit(f"Directory not found / 鏈壘鍒扮洰褰? {LST_BASELINE_DIR}锛堣鍏堜笅杞?LST baseline 鍒拌鐩綍锛?)
    if not DELTA_T_FILE.exists():
        raise SystemExit(f"螖T file not found / 鏈壘鍒?螖T 鏂囦欢: {DELTA_T_FILE}锛堣鍏堣繍琛?`scripts/03_calculate_delta_t.py`锛?)

    main()



