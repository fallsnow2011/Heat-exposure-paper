#!/usr/bin/env python3
"""验证硬编码的热浪日期与 ERA5 数据的匹配情况"""

import pandas as pd
from pathlib import Path

# 硬编码的热浪日期（来自 03_calculate_delta_t.py）
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

# ERA5 数据文件
ERA5_FILE = Path("GEE_LST_Baseline/ERA5Land_daily_Tmax_2022_AllCities.csv")

def verify_city(city_name, heatwave_dates):
    """验证单个城市的热浪日期匹配情况"""

    # 读取 ERA5 数据
    df = pd.read_csv(ERA5_FILE)
    df['date'] = pd.to_datetime(df['date'])

    # 筛选该城市的数据
    city_df = df[df['city'] == city_name].copy()

    # 转换硬编码日期
    heatwave_dates_dt = pd.to_datetime(heatwave_dates)

    # 检查匹配情况
    print(f"\n{'='*60}")
    print(f"城市: {city_name}")
    print(f"{'='*60}")
    print(f"硬编码热浪日期数量: {len(heatwave_dates)}")

    matched = []
    missing = []

    for date in heatwave_dates_dt:
        if date in city_df['date'].values:
            temp = city_df[city_df['date'] == date]['Tmax_C'].values[0]
            matched.append((date.strftime('%Y-%m-%d'), temp))
        else:
            missing.append(date.strftime('%Y-%m-%d'))

    print(f"实际匹配到的日期数量: {len(matched)}")

    if matched:
        print(f"\n[OK] 匹配到的日期 ({len(matched)}天):")
        for date, temp in matched:
            print(f"  {date}: {temp:.2f}C")

    if missing:
        print(f"\n[ERROR] 缺失的日期 ({len(missing)}天):")
        for date in missing:
            print(f"  {date}")
    else:
        print(f"\n[OK] 所有硬编码日期都在 ERA5 数据中")

    return len(matched), len(missing)

if __name__ == "__main__":
    print("验证硬编码热浪日期与 ERA5 数据的匹配情况")
    print("="*60)

    total_matched = 0
    total_missing = 0

    for city, dates in HEATWAVE_DATES.items():
        matched, missing = verify_city(city, dates)
        total_matched += matched
        total_missing += missing

    print(f"\n{'='*60}")
    print(f"总结")
    print(f"{'='*60}")
    print(f"总匹配: {total_matched} 天")
    print(f"总缺失: {total_missing} 天")
