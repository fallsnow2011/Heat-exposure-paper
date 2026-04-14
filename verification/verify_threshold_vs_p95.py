#!/usr/bin/env python3
"""验证硬编码日期是否满足 Met Office 阈值但不满足 P95"""

import pandas as pd
from pathlib import Path

# Met Office 阈值
THRESHOLDS = {
    'London': 28,
    'Birmingham': 26,
    'Bristol': 26,
    'Manchester': 25,
    'Newcastle': 25
}

# 硬编码的热浪日期
HARDCODED_DATES = {
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

# 读取 ERA5 数据
ERA5_FILE = Path("GEE_LST_Baseline/ERA5Land_daily_Tmax_2022_AllCities.csv")
df_all = pd.read_csv(ERA5_FILE)
df_all['date'] = pd.to_datetime(df_all['date'])

print("="*80)
print("验证硬编码日期的筛选标准")
print("="*80)

for city in THRESHOLDS.keys():
    print(f"\n{'='*80}")
    print(f"城市: {city}")
    print(f"{'='*80}")

    # 筛选该城市数据
    df_city = df_all[df_all['city'] == city].copy()

    # 计算 P95
    p95 = df_city['Tmax_C'].quantile(0.95)
    threshold = THRESHOLDS[city]

    print(f"\nMet Office 阈值: {threshold}C")
    print(f"P95: {p95:.2f}C")

    # 硬编码日期
    hardcoded = pd.to_datetime(HARDCODED_DATES[city])

    print(f"\n硬编码的热浪日期 ({len(hardcoded)}天):")

    meet_threshold = 0
    meet_p95 = 0
    meet_both = 0

    for date in hardcoded:
        date_str = date.strftime('%Y-%m-%d')
        temp = df_city[df_city['date'] == date]['Tmax_C'].values[0]

        is_threshold = temp >= threshold
        is_p95 = temp >= p95

        meet_threshold += is_threshold
        meet_p95 += is_p95
        meet_both += (is_threshold and is_p95)

        status = []
        if is_threshold:
            status.append("Met Office")
        if is_p95:
            status.append("P95")

        status_str = " + ".join(status) if status else "NONE"

        print(f"  {date_str}: {temp:.2f}C - [{status_str}]")

    print(f"\n统计:")
    print(f"  满足 Met Office 阈值: {meet_threshold}/{len(hardcoded)} ({meet_threshold/len(hardcoded)*100:.1f}%)")
    print(f"  满足 P95: {meet_p95}/{len(hardcoded)} ({meet_p95/len(hardcoded)*100:.1f}%)")
    print(f"  同时满足两者: {meet_both}/{len(hardcoded)} ({meet_both/len(hardcoded)*100:.1f}%)")

    if meet_threshold == len(hardcoded) and meet_p95 < len(hardcoded):
        print(f"\n[结论] 硬编码使用的是 Met Office 阈值，忽略了 P95 标准")
    elif meet_both == len(hardcoded):
        print(f"\n[结论] 硬编码同时满足 Met Office 阈值和 P95 标准")
    else:
        print(f"\n[结论] 硬编码的标准不明确")

print(f"\n{'='*80}")
print("验证完成")
print(f"{'='*80}")
