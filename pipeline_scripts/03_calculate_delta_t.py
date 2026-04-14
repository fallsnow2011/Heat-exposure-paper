import pandas as pd
import numpy as np
from pathlib import Path

# ==================== Configuration / 閰嶇疆 ====================
# Data directories / 鏁版嵁鐩綍
BASE_DIR = Path("GEE_LST_Baseline")
ERA5_ALL_CITIES_CSV = BASE_DIR / "ERA5Land_daily_Tmax_2022_AllCities.csv"
OUTPUT_DIR = BASE_DIR / "delta_t"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Known heatwave dates for each city / 鍚勫煄甯傚凡鐭ョ殑鐑氮鏃ユ湡
HEATWAVE_DATES = {
    'London': ['2022-08-11', '2022-08-12', '2022-08-13', '2022-08-14'],
    'Birmingham': ['2022-07-17', '2022-07-18', '2022-07-19',
                   '2022-08-10', '2022-08-11', '2022-08-12', '2022-08-13', '2022-08-14'],
    'Bristol': ['2022-07-17', '2022-07-18', '2022-07-19',
                '2022-08-10', '2022-08-11', '2022-08-12', '2022-08-13', '2022-08-14'],
    'Manchester': ['2022-07-17', '2022-07-18', '2022-07-19',
                   '2022-08-10', '2022-08-11', '2022-08-12', '2022-08-13', '2022-08-14'],
    'Newcastle': ['2022-07-17', '2022-07-18', '2022-07-19',
                  '2022-08-10', '2022-08-11', '2022-08-12', '2022-08-13', '2022-08-14']
}

# ==================== Processing functions / 澶勭悊鍑芥暟 ====================
def calculate_delta_t(csv_path: Path, city_name: str, heatwave_dates: list) -> dict:
    """
    Compute 螖T for a single city / 璁＄畻鍗曚釜鍩庡競鐨?螖T

    Parameters:
    -----------
    csv_path : Path
        Path to the ERA5 daily Tmax CSV file / ERA5 姣忔棩 Tmax CSV 鏂囦欢璺緞
    city_name : str
        City name / 鍩庡競鍚嶇О
    heatwave_dates : list
        List of heatwave dates in 'YYYY-MM-DD' format / 鐑氮鏃ユ湡鍒楄〃锛堟牸寮忥細`YYYY-MM-DD`锛?
    Returns:
    --------
    dict : Dictionary with summary statistics / 鍖呭惈鍚勭缁熻淇℃伅鐨勫瓧鍏?    """
    # Read the ERA5 table / 璇诲彇鏁版嵁
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    # Flag heatwave dates / 鏍囪鐑氮鏃?    heatwave_dates_dt = pd.to_datetime(heatwave_dates)
    df['is_heatwave'] = df['date'].isin(heatwave_dates_dt)

    # Compute summary statistics / 璁＄畻缁熻
    # Mean temperature on heatwave days / 鐑氮鏃ュ钩鍧囨俯搴?    T_heatwave = df[df['is_heatwave']]['Tmax_C'].mean()

    # Typical summer temperatures from non-heatwave days / 鍏稿瀷澶忔棩锛堥潪鐑氮鏃ワ級娓╁害
    T_typical_mean = df[~df['is_heatwave']]['Tmax_C'].mean()
    T_typical_median = df[~df['is_heatwave']]['Tmax_C'].median()

    # Compute delta T / 螖T 璁＄畻
    delta_t_mean = T_heatwave - T_typical_mean
    delta_t_median = T_heatwave - T_typical_median

    # Whole-summer summary statistics / 鏁翠釜澶忓缁熻
    T_summer_mean = df['Tmax_C'].mean()
    T_summer_max = df['Tmax_C'].max()
    T_summer_min = df['Tmax_C'].min()

    result = {
        'city': city_name,
        'T_heatwave_mean': T_heatwave,
        'T_typical_mean': T_typical_mean,
        'T_typical_median': T_typical_median,
        'delta_t_mean': delta_t_mean,
        'delta_t_median': delta_t_median,
        'T_summer_mean': T_summer_mean,
        'T_summer_max': T_summer_max,
        'T_summer_min': T_summer_min,
        'n_heatwave_days': df['is_heatwave'].sum(),
        'n_typical_days': (~df['is_heatwave']).sum()
    }

    return result, df

def main():
    """Main entry point / 涓诲嚱鏁?""

    results = []
    all_cities_df = None
    if ERA5_ALL_CITIES_CSV.exists():
        all_cities_df = pd.read_csv(ERA5_ALL_CITIES_CSV)

    print("=" * 60)
    print("Compute city-level 螖T (heatwave days - typical summer days) / 璁＄畻鍚勫煄甯?螖T锛堢儹娴棩 - 鍏稿瀷澶忔棩锛?)
    print("=" * 60)

    for city_name, heatwave_dates in HEATWAVE_DATES.items():
        # City-level CSV path following the GEE export naming rule / CSV 鏂囦欢璺緞锛堟牴鎹?GEE 瀵煎嚭鐨勫懡鍚嶈鍒欙級
        csv_path = BASE_DIR / f"ERA5Land_daily_Tmax_2022_{city_name}.csv"

        if csv_path.exists():
            df_city = pd.read_csv(csv_path)
        elif all_cities_df is not None:
            df_city = all_cities_df[all_cities_df["city"] == city_name].copy()
            if df_city.empty:
                print(f"\n鈿?{city_name}: city records not found in the AllCities table / AllCities 琛ㄤ腑鏈壘鍒拌鍩庡競鏁版嵁 ({ERA5_ALL_CITIES_CSV})")
                continue
        else:
            print(f"\n鈿?{city_name}: neither the city CSV nor the AllCities merged table was found / 鏈壘鍒板煄甯?CSV 鎴?AllCities 鍚堝苟琛?)
            print(f"  - City CSV / 鍩庡競 CSV: {csv_path}")
            print(f"  - Merged table / 鍚堝苟琛? {ERA5_ALL_CITIES_CSV}")
            continue

        # Compute delta T for the city / 璁＄畻 螖T
        result, df = calculate_delta_t_from_df(df_city, city_name, heatwave_dates)
        results.append(result)

        # Print the summary for the city / 鎵撳嵃缁撴灉
        print(f"\n{city_name}:")
        print(f"  Heatwave-day count / 鐑氮鏃ユ暟: {result['n_heatwave_days']}")
        print(f"  Mean Tmax on heatwave days / 鐑氮鏃ュ钩鍧?Tmax: {result['T_heatwave_mean']:.2f} 掳C")
        print(f"  Mean Tmax on typical summer days / 鍏稿瀷澶忔棩骞冲潎 Tmax: {result['T_typical_mean']:.2f} 掳C")
        print(f"  Median Tmax on typical summer days / 鍏稿瀷澶忔棩涓綅 Tmax: {result['T_typical_median']:.2f} 掳C")
        print(f"  螖T using the mean baseline / 螖T锛堢敤鍧囧€硷級: {result['delta_t_mean']:.2f} 掳C")
        print(f"  螖T using the median baseline / 螖T锛堢敤涓綅鏁帮級: {result['delta_t_median']:.2f} 掳C")

        # Save output / 淇濆瓨甯︽爣璁扮殑姣忔棩鏁版嵁
        df.to_csv(OUTPUT_DIR / f"daily_tmax_with_heatwave_flag_{city_name}.csv", index=False)

    # Summarize all cities together / 姹囨€绘墍鏈夊煄甯?    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(OUTPUT_DIR / "delta_t_summary_all_cities.csv", index=False)

        print("\n" + "=" * 60)
        print("Summary table / 姹囨€昏〃:")
        print("=" * 60)
        print(summary_df[['city', 'T_heatwave_mean', 'T_typical_median', 'delta_t_median']].to_string(index=False))

        print(f"\nResults saved to / 缁撴灉宸蹭繚瀛樺埌: {OUTPUT_DIR}")
        print("  - delta_t_summary_all_cities.csv")
        print("  - daily_tmax_with_heatwave_flag_<City>.csv")

    return results

def calculate_delta_t_from_df(df: pd.DataFrame, city_name: str, heatwave_dates: list) -> tuple[dict, pd.DataFrame]:
    """Compute 螖T from a DataFrame, mainly for the combined AllCities table / 浠?DataFrame 璁＄畻 螖T锛堢敤浜庢敮鎸?AllCities 鍚堝苟琛級"""
    if 'date' not in df.columns:
        raise ValueError("ERA5 CSV is missing the `date` column / ERA5 CSV 缂哄皯 `date` 鍒?)
    if 'Tmax_C' not in df.columns:
        raise ValueError("ERA5 CSV is missing the `Tmax_C` column (confirm the GEE export was converted to 掳C) / ERA5 CSV 缂哄皯 `Tmax_C` 鍒楋紙璇风‘璁?GEE 瀵煎嚭宸茶浆鎹㈠埌 鈩冿級")
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    heatwave_dates_dt = pd.to_datetime(heatwave_dates)
    df['is_heatwave'] = df['date'].isin(heatwave_dates_dt)

    T_heatwave = df[df['is_heatwave']]['Tmax_C'].mean()
    T_typical_mean = df[~df['is_heatwave']]['Tmax_C'].mean()
    T_typical_median = df[~df['is_heatwave']]['Tmax_C'].median()

    delta_t_mean = T_heatwave - T_typical_mean
    delta_t_median = T_heatwave - T_typical_median

    result = {
        'city': city_name,
        'T_heatwave_mean': T_heatwave,
        'T_typical_mean': T_typical_mean,
        'T_typical_median': T_typical_median,
        'delta_t_mean': delta_t_mean,
        'delta_t_median': delta_t_median,
        'T_summer_mean': df['Tmax_C'].mean(),
        'T_summer_max': df['Tmax_C'].max(),
        'T_summer_min': df['Tmax_C'].min(),
        'n_heatwave_days': int(df['is_heatwave'].sum()),
        'n_typical_days': int((~df['is_heatwave']).sum()),
    }

    return result, df


if __name__ == "__main__":
    if not BASE_DIR.exists():
        raise SystemExit(f"Directory not found / 鏈壘鍒扮洰褰? {BASE_DIR}锛堣鍏堜笅杞?ERA5 CSV 鍒拌鐩綍锛?)
    if not ERA5_ALL_CITIES_CSV.exists() and not any(BASE_DIR.glob("ERA5Land_daily_Tmax_2022_*.csv")):
        raise SystemExit(
            "ERA5 CSV files were not found / 鏈壘鍒?ERA5 CSV锛歕n"
            f"- Merged table / 鍚堝苟琛? {ERA5_ALL_CITIES_CSV}\n"
            f"- Or city-level CSVs / 鎴栧崟鍩庡競琛? {BASE_DIR}/ERA5Land_daily_Tmax_2022_<City>.csv\n"
            "Run the GEE script `scripts/02_gee_era5_daily_tmax.js` first and download the outputs / 璇峰厛杩愯 GEE 鑴氭湰 `scripts/02_gee_era5_daily_tmax.js` 骞朵笅杞借緭鍑恒€?
        )

    main()



