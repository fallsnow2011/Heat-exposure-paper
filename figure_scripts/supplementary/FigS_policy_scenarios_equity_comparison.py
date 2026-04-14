#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============ 璺緞璁剧疆 ============
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / 'results' / 'heat_exposure'
INEQUALITY_DIR = BASE_DIR / 'results' / 'inequality_analysis'
BOUNDARY_DIR = BASE_DIR / 'city_boundaries'
FIGURES_DIR = BASE_DIR / 'paper' / '06_supplement'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============ 鍏ㄥ眬鏍峰紡璁剧疆 ============
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ============ 閰嶈壊鏂规 ============
COLOR_DEPRIVED = '#E74C3C'    # 绾㈣壊 - 璐洶鍖?
COLOR_AFFLUENT = '#3498DB'    # 钃濊壊 - 瀵岃鍖?
COLOR_GAP = '#2C3E50'         # 娣辩伆 - Gap

# 鎯呮櫙閰嶈壊
SCENARIO_COLORS = {
    'baseline': '#95a5a6',
    'S1_citywide': '#a6bddb',
    'S2_corridors': '#3690c0',
    'S3_equity_first': '#016c59',
}

SCENARIO_LABELS = {
    'baseline': 'Baseline',
    'S1_citywide': 'S1: Citywide',
    'S2_corridors': 'S2: Corridors',
    'S3_equity_first': 'S3: Equity First',
}

SCENARIO_ORDER = ['baseline', 'S1_citywide', 'S2_corridors', 'S3_equity_first']
CITIES = ['London', 'Birmingham', 'Manchester', 'Bristol', 'Newcastle']

# HEI鍙傛暟
ALPHA_B = 0.6
ALPHA_V = 0.8
DELTA_T_VEG = 2.0
SHADOW_INCREASE = 0.10


def load_roads_data(city, scenario='heatwave'):
    """鍔犺浇閬撹矾HEI鏁版嵁"""
    file_path = RESULTS_DIR / f'{city}_roads_hei_improved_{scenario}.gpkg'
    if not file_path.exists():
        return None
    roads = gpd.read_file(file_path)
    if roads.crs.to_epsg() != 27700:
        roads = roads.to_crs(epsg=27700)
    return roads


def load_lsoa_data(scenario='heatwave'):
    """鍔犺浇LSOA绾у埆鏁版嵁"""
    file_path = INEQUALITY_DIR / f'lsoa_hei_summary_{scenario}.csv'
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)


def load_imd_geometry():
    """鍔犺浇IMD鍑犱綍鏁版嵁"""
    imd_path = BOUNDARY_DIR / 'Indices_of_Multiple_Deprivation_(IMD)_2019_8404722932957776031.gpkg'
    gdf = gpd.read_file(imd_path)
    if gdf.crs.to_epsg() != 27700:
        gdf = gdf.to_crs(epsg=27700)
    return gdf[['lsoa11cd', 'IMD_Decile', 'geometry']]


def assign_roads_to_lsoa(roads_gdf, imd_gdf):
    """灏嗛亾璺垎閰嶅埌LSOA"""
    roads_gdf = roads_gdf.copy()
    roads_gdf['centroid'] = roads_gdf.geometry.centroid
    centroids_gdf = gpd.GeoDataFrame(
        roads_gdf[['hei_improved']],
        geometry=roads_gdf['centroid'],
        crs=roads_gdf.crs
    )
    centroids_gdf['road_idx'] = roads_gdf.index
    joined = gpd.sjoin(centroids_gdf, imd_gdf, how='left', predicate='within')
    roads_gdf['lsoa11cd'] = joined.set_index('road_idx')['lsoa11cd']
    roads_gdf['IMD_Decile'] = joined.set_index('road_idx')['IMD_Decile']
    return roads_gdf


def calculate_hei(lst, shadow_building, shadow_vegetation):
    """璁＄畻HEI"""
    shadow_building = np.clip(shadow_building, 0, 1)
    shadow_vegetation = np.clip(shadow_vegetation, 0, 1)
    shadow_cooling = ALPHA_B * shadow_building + ALPHA_V * shadow_vegetation
    hei_base = lst * (1 - shadow_cooling)
    vegetation_cooling = DELTA_T_VEG * shadow_vegetation
    return hei_base - vegetation_cooling


def apply_scenario(roads_gdf, lsoa_df, scenario_name):
    """搴旂敤鏀跨瓥鎯呮櫙"""
    roads = roads_gdf.copy()

    if scenario_name == 'baseline':
        roads['target'] = False
        roads['hei_new'] = roads['hei_improved']
        return roads

    if scenario_name == 'S3_equity_first':
        hei_median = lsoa_df['hei_mean'].median()
        shadow_veg_median = lsoa_df['shadow_vegetation_mean'].median()
        target_lsoas = lsoa_df[
            (lsoa_df['IMD_Decile'].isin([1, 2, 3])) &
            (lsoa_df['hei_mean'] > hei_median) &
            (lsoa_df['shadow_vegetation_mean'] < shadow_veg_median)
        ]['lsoa11cd'].tolist()
        roads['target'] = roads['lsoa11cd'].isin(target_lsoas)

    elif scenario_name == 'S2_corridors':
        # Match the main policy-scenario definition (top quartile by road-length density).
        if 'total_length' in lsoa_df.columns and 'area_km2' in lsoa_df.columns:
            tmp = lsoa_df[['lsoa11cd', 'total_length', 'area_km2']].copy()
            tmp['road_density'] = tmp['total_length'] / (tmp['area_km2'] + 0.001)  # m / km2
            density_75 = tmp['road_density'].quantile(0.75)
            target_lsoas = tmp[tmp['road_density'] >= density_75]['lsoa11cd'].tolist()
        elif 'n_roads' in lsoa_df.columns:
            # Fallback when area_km2 is not available.
            density_75 = lsoa_df['n_roads'].quantile(0.75)
            target_lsoas = lsoa_df[lsoa_df['n_roads'] >= density_75]['lsoa11cd'].tolist()
        else:
            pop_75 = lsoa_df['TotPop'].quantile(0.75)
            target_lsoas = lsoa_df[lsoa_df['TotPop'] >= pop_75]['lsoa11cd'].tolist()
        roads['target'] = roads['lsoa11cd'].isin(target_lsoas)

    elif scenario_name == 'S1_citywide':
        roads['target'] = True

    roads['shadow_vegetation_new'] = roads['shadow_vegetation_avg'].copy()
    roads.loc[roads['target'], 'shadow_vegetation_new'] = np.clip(
        roads.loc[roads['target'], 'shadow_vegetation_avg'] + SHADOW_INCREASE, 0, 1
    )

    roads['hei_new'] = calculate_hei(
        roads['lst'].values,
        roads['shadow_building_avg'].values,
        roads['shadow_vegetation_new'].values
    )

    return roads


def calculate_city_stats(city, lsoa_df, imd_gdf, time_scenario='heatwave'):
    """璁＄畻鍗曚釜鍩庡競鍦ㄥ悇鎯呮櫙涓嬬殑璐洶鍖?瀵岃鍖虹粺璁?""

    roads = load_roads_data(city, time_scenario)
    if roads is None:
        return None

    roads = assign_roads_to_lsoa(roads, imd_gdf)
    city_lsoas = roads['lsoa11cd'].dropna().unique().tolist()
    city_lsoa_df = lsoa_df[lsoa_df['lsoa11cd'].isin(city_lsoas)].copy()

    # Add LSOA area for road-length density targeting (consistent with scripts/21_policy_scenarios_fixed.py)
    area_df = imd_gdf[['lsoa11cd', 'geometry']].copy()
    area_df['area_km2'] = area_df.geometry.area / 1e6
    city_lsoa_df = city_lsoa_df.merge(area_df[['lsoa11cd', 'area_km2']], on='lsoa11cd', how='left')

    results = []

    for scenario in SCENARIO_ORDER:
        roads_scenario = apply_scenario(roads, city_lsoa_df, scenario)

        # 璐洶鍖?(D1-D3) 鍜?瀵岃鍖?(D8-D10)
        deprived_mask = roads_scenario['IMD_Decile'].isin([1, 2, 3])
        affluent_mask = roads_scenario['IMD_Decile'].isin([8, 9, 10])

        # 璁＄畻缁熻閲?(閬撹矾闀垮害鍔犳潈)
        deprived_roads = roads_scenario[deprived_mask & roads_scenario['hei_new'].notna()]
        affluent_roads = roads_scenario[affluent_mask & roads_scenario['hei_new'].notna()]

        if len(deprived_roads) > 0:
            weights_d = deprived_roads.geometry.length
            hei_baseline_d = np.average(deprived_roads['hei_improved'], weights=weights_d)
            hei_new_d = np.average(deprived_roads['hei_new'], weights=weights_d)
        else:
            hei_baseline_d = hei_new_d = np.nan

        if len(affluent_roads) > 0:
            weights_a = affluent_roads.geometry.length
            hei_baseline_a = np.average(affluent_roads['hei_improved'], weights=weights_a)
            hei_new_a = np.average(affluent_roads['hei_new'], weights=weights_a)
        else:
            hei_baseline_a = hei_new_a = np.nan

        results.append({
            'city': city,
            'scenario': scenario,
            'hei_deprived_baseline': hei_baseline_d,
            'hei_deprived_new': hei_new_d,
            'hei_affluent_baseline': hei_baseline_a,
            'hei_affluent_new': hei_new_a,
            'cooling_deprived': hei_baseline_d - hei_new_d,
            'cooling_affluent': hei_baseline_a - hei_new_a,
            'gap_baseline': hei_baseline_d - hei_baseline_a,
            'gap_new': hei_new_d - hei_new_a,
            'n_roads_deprived': len(deprived_roads),
            'n_roads_affluent': len(affluent_roads),
        })

    return pd.DataFrame(results)


def plot_figure(time_scenario='heatwave'):
    """缁戝埗瀹屾暣鐨勮传瀵屽樊璺濆姣斿浘"""

    print(f"\n{'='*60}")
    print(f"缁樺埗璐洶鍖?vs 瀵岃鍖洪檷娓╁姣斿浘")
    print(f"鍦烘櫙: {time_scenario}")
    print(f"{'='*60}")

    # 鍔犺浇鏁版嵁
    lsoa_df = load_lsoa_data(time_scenario)
    imd_gdf = load_imd_geometry()

    if lsoa_df is None:
        print("鏃犳硶鍔犺浇LSOA鏁版嵁")
        return

    # 璁＄畻鍚勫煄甯傜粺璁?
    print("\n璁＄畻鍚勫煄甯傜粺璁?..")
    all_stats = []
    for city in CITIES:
        print(f"  澶勭悊 {city}...")
        city_stats = calculate_city_stats(city, lsoa_df, imd_gdf, time_scenario)
        if city_stats is not None:
            all_stats.append(city_stats)

    stats_df = pd.concat(all_stats, ignore_index=True)

    # 姹囨€荤粺璁?(5鍩庡競骞冲潎)
    summary = stats_df.groupby('scenario').agg({
        'hei_deprived_new': 'mean',
        'hei_affluent_new': 'mean',
        'cooling_deprived': 'mean',
        'cooling_affluent': 'mean',
        'gap_baseline': 'mean',
        'gap_new': 'mean',
    }).reset_index()

    # 鍒涘缓鍥捐〃
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1],
                          hspace=0.30, wspace=0.25,
                          left=0.08, right=0.95, top=0.90, bottom=0.08)

    # ===== Panel (a): 鍚勬儏鏅笅璐洶鍖?瀵岃鍖虹殑HEI鍧囧€煎姣?=====
    ax_a = fig.add_subplot(gs[0, 0])

    x = np.arange(len(SCENARIO_ORDER))
    width = 0.35

    hei_deprived = [summary[summary['scenario'] == s]['hei_deprived_new'].values[0] for s in SCENARIO_ORDER]
    hei_affluent = [summary[summary['scenario'] == s]['hei_affluent_new'].values[0] for s in SCENARIO_ORDER]

    bars1 = ax_a.bar(x - width/2, hei_deprived, width, label='Deprived (D1-D3)',
                     color=COLOR_DEPRIVED, edgecolor='white', linewidth=0.5)
    bars2 = ax_a.bar(x + width/2, hei_affluent, width, label='Affluent (D8-D10)',
                     color=COLOR_AFFLUENT, edgecolor='white', linewidth=0.5)

    # 鏁板€兼爣绛?
    for i, (d, a) in enumerate(zip(hei_deprived, hei_affluent)):
        ax_a.text(i - width/2, d + 0.3, f'{d:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax_a.text(i + width/2, a + 0.3, f'{a:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_a.set_ylabel('HEI (掳C, length-weighted)', fontsize=10)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER], fontsize=9)
    ax_a.set_title('(a) HEI by Deprivation Group Under Each Scenario', fontsize=11, fontweight='bold', loc='left')
    ax_a.legend(loc='upper right', fontsize=9)
    ax_a.set_ylim(35, 48)
    ax_a.grid(axis='y', linestyle='--', alpha=0.3)

    # ===== Panel (b): 鍚勬儏鏅笅璐洶鍖?瀵岃鍖虹殑闄嶆俯閲忓姣?=====
    ax_b = fig.add_subplot(gs[0, 1])

    cooling_deprived = [summary[summary['scenario'] == s]['cooling_deprived'].values[0] for s in SCENARIO_ORDER]
    cooling_affluent = [summary[summary['scenario'] == s]['cooling_affluent'].values[0] for s in SCENARIO_ORDER]

    # 璺宠繃baseline (闄嶆俯涓?)
    x2 = np.arange(len(SCENARIO_ORDER) - 1)
    scenarios_no_baseline = SCENARIO_ORDER[1:]

    cooling_d = cooling_deprived[1:]
    cooling_a = cooling_affluent[1:]

    bars1 = ax_b.bar(x2 - width/2, cooling_d, width, label='Deprived (D1-D3)',
                     color=COLOR_DEPRIVED, edgecolor='white', linewidth=0.5)
    bars2 = ax_b.bar(x2 + width/2, cooling_a, width, label='Affluent (D8-D10)',
                     color=COLOR_AFFLUENT, edgecolor='white', linewidth=0.5)

    # 鏁板€兼爣绛?
    for i, (d, a) in enumerate(zip(cooling_d, cooling_a)):
        ax_b.text(i - width/2, d + 0.1, f'{d:.1f}掳C', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax_b.text(i + width/2, a + 0.1, f'{a:.1f}掳C', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 鏍囨敞宸紓
    for i, (d, a) in enumerate(zip(cooling_d, cooling_a)):
        diff = d - a
        color = '#27AE60' if diff > 0 else '#E74C3C'
        ax_b.annotate(f'螖={diff:+.1f}掳C', xy=(i, max(d, a) + 0.5),
                     ha='center', fontsize=8, color=color, fontweight='bold')

    ax_b.set_ylabel('Cooling Effect (掳C)', fontsize=10)
    ax_b.set_xticks(x2)
    ax_b.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios_no_baseline], fontsize=9)
    ax_b.set_title('(b) Cooling by Deprivation Group', fontsize=11, fontweight='bold', loc='left')
    ax_b.legend(loc='upper right', fontsize=9)
    ax_b.set_ylim(0, 5)
    ax_b.grid(axis='y', linestyle='--', alpha=0.3)

    # 娣诲姞娉ㄩ噴
    ax_b.text(0.5, 0.95, '螖 > 0: Deprived areas cool more\n(reduces inequality)',
             transform=ax_b.transAxes, fontsize=8, ha='center', va='top',
             color='#27AE60', style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', alpha=0.9, edgecolor='none'))

    # ===== Panel (c): 璐瘜HEI宸窛鍙樺寲 =====
    ax_c = fig.add_subplot(gs[1, 0])

    gap_baseline = summary[summary['scenario'] == 'baseline']['gap_baseline'].values[0]
    gap_new = [summary[summary['scenario'] == s]['gap_new'].values[0] for s in SCENARIO_ORDER]

    colors = [SCENARIO_COLORS[s] for s in SCENARIO_ORDER]
    bars = ax_c.bar(range(len(SCENARIO_ORDER)), gap_new, color=colors, edgecolor='white', linewidth=0.5)

    # Baseline鍙傝€冪嚎
    ax_c.axhline(y=gap_baseline, color='#E74C3C', linestyle='--', linewidth=2, label=f'Baseline Gap ({gap_baseline:.2f}掳C)')

    # 闆剁嚎
    ax_c.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # 涓嶅钩绛夐€嗚浆鍖哄煙
    ax_c.fill_between([-0.5, 3.5], 0, -0.8, color='#e8f5e9', alpha=0.5, zorder=0)
    ax_c.text(3.3, -0.7, 'Inequality\nReversed', fontsize=8, ha='right', va='bottom',
              color='#27AE60', style='italic', fontweight='bold')

    # 鏁板€兼爣绛?
    for i, g in enumerate(gap_new):
        y_pos = g + 0.08 if g >= 0 else g - 0.15
        va = 'bottom' if g >= 0 else 'top'
        ax_c.text(i, y_pos, f'{g:.2f}掳C', ha='center', va=va, fontsize=10, fontweight='bold')

    ax_c.set_ylabel('HEI Gap: Deprived 鈭?Affluent (掳C)', fontsize=10)
    ax_c.set_xticks(range(len(SCENARIO_ORDER)))
    ax_c.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER], fontsize=9)
    ax_c.set_title('(c) Inequality Gap Under Each Scenario', fontsize=11, fontweight='bold', loc='left')
    ax_c.legend(loc='upper right', fontsize=9)
    ax_c.set_ylim(-0.8, 1.8)
    ax_c.grid(axis='y', linestyle='--', alpha=0.3)

    # ===== Panel (d): 鍚勫煄甯傝缁嗗姣?=====
    ax_d = fig.add_subplot(gs[1, 1])

    # 璁＄畻鍚勫煄甯傚悇鎯呮櫙鐨凣ap鍙樺寲
    city_gaps = []
    for city in CITIES:
        city_data = stats_df[stats_df['city'] == city]
        baseline_gap = city_data[city_data['scenario'] == 'baseline']['gap_baseline'].values[0]
        for scenario in SCENARIO_ORDER[1:]:
            new_gap = city_data[city_data['scenario'] == scenario]['gap_new'].values[0]
            gap_change = new_gap - baseline_gap
            city_gaps.append({
                'city': city,
                'scenario': scenario,
                'gap_change': gap_change,
                'gap_reduction_pct': -gap_change / baseline_gap * 100 if baseline_gap != 0 else 0
            })

    city_gaps_df = pd.DataFrame(city_gaps)

    # 缁樺埗鍒嗙粍鏌辩姸鍥?
    x3 = np.arange(len(CITIES))
    width3 = 0.25

    for i, scenario in enumerate(SCENARIO_ORDER[1:]):
        scenario_data = city_gaps_df[city_gaps_df['scenario'] == scenario]
        values = [scenario_data[scenario_data['city'] == c]['gap_change'].values[0] for c in CITIES]
        ax_d.bar(x3 + (i - 1) * width3, values, width3,
                label=SCENARIO_LABELS[scenario], color=SCENARIO_COLORS[scenario],
                edgecolor='white', linewidth=0.5)

    ax_d.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax_d.fill_between([-0.5, 4.5], 0, -1.5, color='#e8f5e9', alpha=0.3, zorder=0)

    ax_d.set_ylabel('Gap Change (掳C)', fontsize=10)
    ax_d.set_xticks(x3)
    ax_d.set_xticklabels(CITIES, fontsize=9)
    ax_d.set_title('(d) Gap Change by City and Scenario', fontsize=11, fontweight='bold', loc='left')
    ax_d.legend(loc='lower left', fontsize=8, ncol=1)
    ax_d.set_ylim(-1.5, 0.5)
    ax_d.grid(axis='y', linestyle='--', alpha=0.3)

    # 娣诲姞娉ㄩ噴
    ax_d.text(0.95, 0.05, 'Negative = Gap reduced\n(less inequality)',
             transform=ax_d.transAxes, fontsize=8, ha='right', va='bottom',
             color='#27AE60', style='italic')

    # 鎬绘爣棰?
    scenario_label = 'Heatwave' if time_scenario == 'heatwave' else 'Typical Day'
    fig.suptitle(f'Policy Scenarios: Deprived vs Affluent Cooling Comparison ({scenario_label})',
                 fontsize=13, fontweight='bold', y=0.96)

    # 淇濆瓨
    output_name = f'FigS_policy_scenarios_equity_comparison_{time_scenario}'
    fig.savefig(FIGURES_DIR / f'{output_name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES_DIR / f'{output_name}.pdf', bbox_inches='tight', facecolor='white')

    print(f"\n鍥捐〃宸蹭繚瀛樿嚦: {FIGURES_DIR}")
    print(f"  - {output_name}.png")
    print(f"  - {output_name}.pdf")

    # 淇濆瓨鏁版嵁
    stats_df.to_csv(FIGURES_DIR / f'policy_scenarios_equity_stats_{time_scenario}.csv', index=False)
    print(f"  - policy_scenarios_equity_stats_{time_scenario}.csv")

    plt.close(fig)

    return stats_df


def main():
    """涓诲嚱鏁?""
    # 缁樺埗鐑氮鏃ュ浘
    plot_figure(time_scenario='heatwave')

    # 缁樺埗鍏稿瀷鏃ュ浘
    plot_figure(time_scenario='typical_day')

    print("\n" + "=" * 60)
    print("瀹屾垚!")
    print("=" * 60)


if __name__ == '__main__':
    main()


