#!/usr/bin/env python3
"""验证热浪日期是否符合 Met Office + P95 + ≥3连续天规则"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# Met Office 阈值
THRESHOLDS = {
    'London': 28,
    'Birmingham': 26,
    'Bristol': 26,
    'Manchester': 25,
    'Newcastle': 25
}

# 硬编码的热浪日期（来自 03_calculate_delta_t.py）
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

def find_consecutive_sequences(dates):
    """找出连续≥3天的序列"""
    if len(dates) < 3:
        return []

    dates = sorted(dates)
    sequences = []
    current_seq = [dates[0]]

    for i in range(1, len(dates)):
        if (dates[i] - dates[i-1]).days == 1:
            current_seq.append(dates[i])
        else:
            if len(current_seq) >= 3:
                sequences.append(current_seq)
            current_seq = [dates[i]]

    if len(current_seq) >= 3:
        sequences.append(current_seq)

    return sequences

def detect_heatwave_by_rules(city_name, df):
    """按照 Methods 规则检测热浪日期"""

    threshold = THRESHOLDS[city_name]

    # 计算 P95
    p95 = df['Tmax_C'].quantile(0.95)

    # 筛选满足阈值的日期
    hot_days = df[(df['Tmax_C'] >= threshold) & (df['Tmax_C'] >= p95)].copy()

    if len(hot_days) == 0:
        return [], []

    # 找出连续≥3天的序列
    sequences = find_consecutive_sequences(hot_days['date'].tolist())

    # 合并所有序列中的日期
    all_heatwave_dates = []
    for seq in sequences:
        all_heatwave_dates.extend(seq)

    return sequences, all_heatwave_dates

# 读取 ERA5 数据
ERA5_FILE = Path("GEE_LST_Baseline/ERA5Land_daily_Tmax_2022_AllCities.csv")
df_all = pd.read_csv(ERA5_FILE)
df_all['date'] = pd.to_datetime(df_all['date'])

print("="*80)
print("验证热浪识别规则一致性")
print("="*80)
print("\nMethods 规则: Met Office 阈值 + P95 + 连续≥3天\n")

for city in THRESHOLDS.keys():
    print(f"\n{'='*80}")
    print(f"城市: {city}")
    print(f"{'='*80}")

    # 筛选该城市数据
    df_city = df_all[df_all['city'] == city].copy()

    # 按规则检测
    sequences, detected_dates = detect_heatwave_by_rules(city, df_city)

    # 硬编码日期
    hardcoded = pd.to_datetime(HARDCODED_DATES[city])

    # 对比
    threshold = THRESHOLDS[city]
    p95 = df_city['Tmax_C'].quantile(0.95)

    print(f"\nMet Office 阈值: {threshold}C")
    print(f"P95: {p95:.2f}C")
    print(f"\n按规则检测到的热浪序列:")

    if sequences:
        for i, seq in enumerate(sequences, 1):
            start = seq[0].strftime('%Y-%m-%d')
            end = seq[-1].strftime('%Y-%m-%d')
            print(f"  序列{i}: {start} ~ {end} ({len(seq)}天)")
    else:
        print("  无")

    print(f"\n按规则检测到的总天数: {len(detected_dates)}天")
    print(f"硬编码的总天数: {len(hardcoded)}天")

    # 检查一致性
    detected_set = set([d.strftime('%Y-%m-%d') for d in detected_dates])
    hardcoded_set = set([d.strftime('%Y-%m-%d') for d in hardcoded])

    if detected_set == hardcoded_set:
        print(f"\n[OK] 硬编码日期与规则检测结果一致")
    else:
        print(f"\n[ERROR] 硬编码日期与规则检测结果不一致！")

        only_detected = detected_set - hardcoded_set
        only_hardcoded = hardcoded_set - detected_set

        if only_detected:
            print(f"\n  仅在规则检测中出现:")
            for date in sorted(only_detected):
                temp = df_city[df_city['date'] == date]['Tmax_C'].values[0]
                print(f"    {date}: {temp:.2f}C")

        if only_hardcoded:
            print(f"\n  仅在硬编码中出现:")
            for date in sorted(only_hardcoded):
                temp = df_city[df_city['date'] == date]['Tmax_C'].values[0]
                print(f"    {date}: {temp:.2f}C")

print(f"\n{'='*80}")
print("验证完成")
print(f"{'='*80}")
