import os
import sys
from pathlib import Path

# Add the scripts directory to sys.path / 娣诲姞 scripts 鐩綍鍒拌矾寰?
sys.path.insert(0, str(Path(__file__).parent))

from shadow_engine_taichi import ShadowCalculator, batch_process_day

# Configuration / 閰嶇疆
CITIES = ['London', 'Birmingham', 'Bristol', 'Manchester', 'Newcastle']
LIDAR_DIR = Path('city_lidar')
OUTPUT_BASE = Path('shadow_maps')

# Heatwave dates used for the simulation: 11-12 August 2022 / 鐑氮鏃ユ湡锛?022 骞?8 鏈?11鈥?2 鏃?HEATWAVE_DATES = ['2022-08-11', '2022-08-12']
HOURS = range(9, 18)  # 09:00 - 17:00
MAX_RAY_DIST = 500.0  # metres / 绫?
TIMEZONE = 'Europe/London'


def process_city(city_name, date_str):
    """Run one day of shadow simulation for a single city / 澶勭悊鍗曚釜鍩庡競鍗曞ぉ鐨勯槾褰辫绠?""

    # DSM path / DSM 璺緞
    dsm_path = LIDAR_DIR / city_name / f'{city_name}_DSM_2m.tif'

    if not dsm_path.exists():
        print(f"  Skip {city_name}: DSM file not found / 璺宠繃 {city_name}锛欴SM 鏂囦欢涓嶅瓨鍦?{dsm_path}")
        return False

    # Output directory / 杈撳嚭鐩綍
    output_dir = OUTPUT_BASE / city_name / date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
        print(f"Processing {city_name} - {date_str} / 澶勭悊 {city_name} - {date_str}")
    print(f"{'='*60}")

    # Skip reruns when all hourly outputs already exist / 妫€鏌ユ槸鍚﹀凡瀹屾垚
    expected_files = [output_dir / f"shadow_{date_str}_{h:02d}00.tif" for h in HOURS]
    if all(f.exists() for f in expected_files):
        print(f"  Already finished, skipping / 宸插畬鎴愶紝璺宠繃")
        return True

    # Call the existing batch_process_day function / 璋冪敤鐜版湁鐨?batch_process_day 鍑芥暟
    batch_process_day(
        dsm_path=str(dsm_path),
        output_dir=str(output_dir),
        date_str=date_str,
        hours=HOURS,
        max_ray_dist_meters=MAX_RAY_DIST,
        timezone_str=TIMEZONE
    )

    return True


def main():
    print("="*60)
    print("Batch shadow simulation for five cities / 5 鍩庡競闃村奖璁＄畻鎵瑰鐞?)
    print("="*60)
    print(f"Cities / 鍩庡競: {CITIES}")
    print(f"Dates / 鏃ユ湡: {HEATWAVE_DATES}")
    print(f"Hours / 鏃堕棿: {list(HOURS)}")

    # Ensure the output directory exists / 纭繚杈撳嚭鐩綍瀛樺湪
    OUTPUT_BASE.mkdir(exist_ok=True)

    # Process each city / 澶勭悊姣忎釜鍩庡競鐨勬瘡涓€澶?
    for city in CITIES:
        for date_str in HEATWAVE_DATES:
            try:
                process_city(city, date_str)
            except Exception as e:
                print(f"  Error / 閿欒: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*60)
    print("Shadow simulation complete / 闃村奖璁＄畻瀹屾垚锛?)
    print("="*60)

    # Report the number of generated shadow rasters / 缁熻杈撳嚭
    for city in CITIES:
        city_dir = OUTPUT_BASE / city
        if city_dir.exists():
            tif_files = list(city_dir.rglob("*.tif"))
            print(f"  {city}: {len(tif_files)} shadow files / 涓槾褰辨枃浠?)


if __name__ == '__main__':
    main()



