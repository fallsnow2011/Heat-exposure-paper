#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
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
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.3
plt.rcParams['axes.edgecolor'] = '#CCCCCC'

# ============ 閰嶈壊鏂规 ============
COOL_COLOR = '#00AEEF'        # 鐢靛厜钃?- 鍑夌埥閬撹矾
HOT_COLOR = '#F5F5F5'         # 鏋佹祬鐏?- 鐑亾璺?
BOUNDARY_COLOR = '#AAAAAA'    # 杈圭晫绾?
NEW_COOL_COLOR = '#2ECC71'    # 缈犵豢鑹?- 鏂板鍑夌埥閬撹矾

COOL_LINEWIDTH = 0.6
HOT_LINEWIDTH = 0.08
BOUNDARY_LINEWIDTH = 0.5

# 鍩庡競鍒楄〃
CITIES = ['London', 'Birmingham', 'Manchester', 'Bristol', 'Newcastle']

# HEI鍙傛暟 (涓?3_recalculate_hei_improved.py涓€鑷?
ALPHA_B = 0.6
ALPHA_V = 0.8
DELTA_T_VEG = 2.0
SHADOW_INCREASE = 0.10  # 鏀跨瓥鎯呮櫙澧炲姞10%闃村奖

# HEI闃堝€?
THRESHOLD = 35  # 浣跨敤35掳C浣滀负灞曠ず闃堝€?


def load_roads_data(city, scenario='heatwave'):
    """鍔犺浇閬撹矾HEI鏁版嵁"""
    file_path = RESULTS_DIR / f'{city}_roads_hei_improved_{scenario}.gpkg'
    if not file_path.exists():
        print(f"鏂囦欢涓嶅瓨鍦? {file_path}")
        return None

    roads = gpd.read_file(file_path)
    if roads.crs.to_epsg() != 27700:
        roads = roads.to_crs(epsg=27700)
    return roads


def load_lsoa_data(scenario='heatwave'):
    """鍔犺浇LSOA绾у埆鏁版嵁"""
    file_path = INEQUALITY_DIR / f'lsoa_hei_summary_{scenario}.csv'
    if not file_path.exists():
        print(f"鏂囦欢涓嶅瓨鍦? {file_path}")
        return None
    return pd.read_csv(file_path)


def load_imd_geometry():
    """鍔犺浇IMD鍑犱綍鏁版嵁"""
    imd_path = BOUNDARY_DIR / 'Indices_of_Multiple_Deprivation_(IMD)_2019_8404722932957776031.gpkg'
    gdf = gpd.read_file(imd_path)
    if gdf.crs.to_epsg() != 27700:
        gdf = gdf.to_crs(epsg=27700)
    return gdf[['lsoa11cd', 'IMD_Decile', 'geometry']]


def load_city_boundary(city):
    """鍔犺浇鍩庡競杈圭晫"""
    boundary_file = BOUNDARY_DIR / f'{city}_boundary.geojson'
    if boundary_file.exists():
        gdf = gpd.read_file(boundary_file)
        if gdf.crs.to_epsg() != 27700:
            gdf = gdf.to_crs(epsg=27700)
        return gdf
    return None


def calculate_hei(lst, shadow_building, shadow_vegetation):
    """璁＄畻HEI"""
    shadow_building = np.clip(shadow_building, 0, 1)
    shadow_vegetation = np.clip(shadow_vegetation, 0, 1)

    shadow_cooling = ALPHA_B * shadow_building + ALPHA_V * shadow_vegetation
    hei_base = lst * (1 - shadow_cooling)
    vegetation_cooling = DELTA_T_VEG * shadow_vegetation
    hei = hei_base - vegetation_cooling

    return hei


def assign_roads_to_lsoa(roads_gdf, imd_gdf):
    """灏嗛亾璺垎閰嶅埌LSOA (閫氳繃璐ㄥ績绌洪棿杩炴帴)"""
    # 璁＄畻閬撹矾璐ㄥ績
    roads_gdf = roads_gdf.copy()
    roads_gdf['centroid'] = roads_gdf.geometry.centroid

    # 鍒涘缓璐ㄥ績GeoDataFrame
    centroids_gdf = gpd.GeoDataFrame(
        roads_gdf[['hei_improved']],
        geometry=roads_gdf['centroid'],
        crs=roads_gdf.crs
    )
    centroids_gdf['road_idx'] = roads_gdf.index

    # 绌洪棿杩炴帴
    joined = gpd.sjoin(centroids_gdf, imd_gdf, how='left', predicate='within')

    # 灏哃SOA淇℃伅鍚堝苟鍥為亾璺?
    roads_gdf['lsoa11cd'] = joined.set_index('road_idx')['lsoa11cd']
    roads_gdf['IMD_Decile'] = joined.set_index('road_idx')['IMD_Decile']

    return roads_gdf


def apply_scenario(roads_gdf, lsoa_df, scenario_name):
    """
    搴旂敤鏀跨瓥鎯呮櫙锛岃繑鍥炰慨鏀瑰悗鐨勯亾璺暟鎹?
    鎯呮櫙瀹氫箟:
    - S3_equity_first: 璐洶(D1-3) + 楂楬EI + 浣庢琚槾褰辩殑LSOA涓殑閬撹矾
    - S2_corridors: 楂樿矾缃戝瘑搴SOA涓殑閬撹矾锛堜紭鍏堜娇鐢?total_length/area_km2锛?    - S1_citywide: 鎵€鏈夐亾璺?    """
    roads = roads_gdf.copy()

    if scenario_name == 'baseline':
        roads['target'] = False
        roads['hei_new'] = roads['hei_improved']
        return roads

    # 鑾峰彇LSOA绾у埆鐨勭粺璁′俊鎭?
    lsoa_stats = lsoa_df.set_index('lsoa11cd')

    if scenario_name == 'S3_equity_first':
        # 璇嗗埆鐩爣LSOA: 璐洶 + 楂楬EI + 浣庢琚槾褰?        hei_median = lsoa_df['hei_mean'].median()
        shadow_veg_median = lsoa_df['shadow_vegetation_mean'].median()

        target_lsoas = lsoa_df[
            (lsoa_df['IMD_Decile'].isin([1, 2, 3])) &
            (lsoa_df['hei_mean'] > hei_median) &
            (lsoa_df['shadow_vegetation_mean'] < shadow_veg_median)
        ]['lsoa11cd'].tolist()

        roads['target'] = roads['lsoa11cd'].isin(target_lsoas)

    elif scenario_name == 'S2_corridors':
        # 璇嗗埆鐩爣LSOA: 楂樿矾缃戝瘑搴︼紙浼樺厛浣跨敤 total_length/area_km2锛?        if 'total_length' in lsoa_df.columns and 'area_km2' in lsoa_df.columns:
            tmp = lsoa_df[['lsoa11cd', 'total_length', 'area_km2']].copy()
            tmp['road_density'] = tmp['total_length'] / (tmp['area_km2'] + 0.001)  # m / km2
            density_75 = tmp['road_density'].quantile(0.75)
            target_lsoas = tmp[tmp['road_density'] >= density_75]['lsoa11cd'].tolist()
        elif 'n_roads' in lsoa_df.columns:
            # 閫€鍥炲埌閬撹矾娈垫暟浠ｇ悊
            density_75 = lsoa_df['n_roads'].quantile(0.75)
            target_lsoas = lsoa_df[lsoa_df['n_roads'] >= density_75]['lsoa11cd'].tolist()
        else:
            # 閫€鍥炲埌浜哄彛瀵嗗害浠ｇ悊
            pop_75 = lsoa_df['TotPop'].quantile(0.75)
            target_lsoas = lsoa_df[lsoa_df['TotPop'] >= pop_75]['lsoa11cd'].tolist()

        roads['target'] = roads['lsoa11cd'].isin(target_lsoas)

    elif scenario_name == 'S1_citywide':
        # 鎵€鏈夐亾璺?        roads['target'] = True

    # 瀵圭洰鏍囬亾璺鍔犳琚槾褰?
    roads['shadow_vegetation_new'] = roads['shadow_vegetation_avg'].copy()
    roads.loc[roads['target'], 'shadow_vegetation_new'] = np.clip(
        roads.loc[roads['target'], 'shadow_vegetation_avg'] + SHADOW_INCREASE,
        0, 1
    )

    # 閲嶆柊璁＄畻HEI
    roads['hei_new'] = calculate_hei(
        roads['lst'].values,
        roads['shadow_building_avg'].values,
        roads['shadow_vegetation_new'].values
    )

    return roads


def plot_city_scenario(ax, roads, boundary, threshold, scenario_name, city, show_stats=True):
    """缁戝埗鍗曚釜鍩庡競鍗曚釜鎯呮櫙鐨勫湴鍥?""

    valid_roads = roads[roads['hei_new'].notna()].copy()

    # 浣跨敤鍘熷HEI鍜屾柊HEI鍒ゆ柇鍑夌埥鐘舵€?
    cool_baseline = valid_roads['hei_improved'] < threshold
    cool_new = valid_roads['hei_new'] < threshold

    # 鍒嗙被:
    # 1. 濮嬬粓鍑夌埥 (baseline鍜宻cenario閮藉噳鐖?
    # 2. 鏂板鍑夌埥 (baseline鐑絾scenario鍑夌埥)
    # 3. 濮嬬粓鐑?(閮界儹)
    always_cool = valid_roads[cool_baseline & cool_new]
    newly_cool = valid_roads[~cool_baseline & cool_new]
    still_hot = valid_roads[~cool_new]

    # 缁樺埗椤哄簭: 鐑亾璺?-> 濮嬬粓鍑夌埥 -> 鏂板鍑夌埥
    if len(still_hot) > 0:
        still_hot.plot(ax=ax, color=HOT_COLOR, linewidth=HOT_LINEWIDTH, alpha=0.5, zorder=1)

    if len(always_cool) > 0:
        always_cool.plot(ax=ax, color=COOL_COLOR, linewidth=COOL_LINEWIDTH, alpha=0.9, zorder=2)

    if len(newly_cool) > 0:
        newly_cool.plot(ax=ax, color=NEW_COOL_COLOR, linewidth=COOL_LINEWIDTH * 1.2, alpha=0.95, zorder=3)

    # 鍩庡競杈圭晫
    if boundary is not None:
        boundary.boundary.plot(ax=ax, color=BOUNDARY_COLOR, linewidth=BOUNDARY_LINEWIDTH, zorder=4)

    # 缁熻淇℃伅
    if show_stats:
        n_total = len(valid_roads)
        n_cool_baseline = cool_baseline.sum()
        n_cool_new = cool_new.sum()
        n_newly_cool = len(newly_cool)
        pct_baseline = n_cool_baseline / n_total * 100 if n_total > 0 else 0
        pct_new = n_cool_new / n_total * 100 if n_total > 0 else 0
        gain = pct_new - pct_baseline

        # 璁＄畻骞冲潎HEI鍙樺寲
        hei_baseline_mean = valid_roads['hei_improved'].mean()
        hei_new_mean = valid_roads['hei_new'].mean()
        hei_change = hei_new_mean - hei_baseline_mean

        # 鏄剧ず: 鍑夌埥閬撹矾姣斾緥 + 鏂板鏁伴噺 + 骞冲潎闄嶆俯
        stats_text = f'{pct_new:.1f}%'
        if scenario_name != 'baseline':
            if n_newly_cool > 0:
                stats_text += f'\n+{n_newly_cool:,} roads'
            if hei_change < -0.1:
                stats_text += f'\n螖HEI: {hei_change:.1f}掳C'

        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
               fontsize=8, fontweight='bold', va='top', ha='left',
               color='#333333',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='none'))

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_figure(time_scenario='heatwave', threshold=35):
    """缁戝埗瀹屾暣鐨勬斂绛栨儏鏅┖闂村垎甯冨浘"""

    print(f"\n{'='*60}")
    print(f"缁戝埗鏀跨瓥鎯呮櫙鍑夌埥缃戠粶绌洪棿鍒嗗竷鍥?)
    print(f"鍦烘櫙: {time_scenario}, 闃堝€? {threshold}掳C")
    print(f"{'='*60}")

    # 鍔犺浇LSOA鏁版嵁鍜孖MD鍑犱綍
    print("\n鍔犺浇LSOA鏁版嵁...")
    lsoa_df = load_lsoa_data(time_scenario)
    imd_gdf = load_imd_geometry()

    if lsoa_df is None:
        print("鏃犳硶鍔犺浇LSOA鏁版嵁")
        return

    # Add area_km2 for road-length density targeting (consistent with scripts/21_policy_scenarios_fixed.py)
    area_df = imd_gdf[['lsoa11cd', 'geometry']].copy()
    area_df['area_km2'] = area_df.geometry.area / 1e6
    lsoa_df = lsoa_df.merge(area_df[['lsoa11cd', 'area_km2']], on='lsoa11cd', how='left')

    # 鎯呮櫙鍒楄〃
    scenarios = ['baseline', 'S1_citywide', 'S2_corridors', 'S3_equity_first']
    scenario_labels = {
        'baseline': 'Baseline',
        'S1_citywide': 'S1: Citywide (+10%)',
        'S2_corridors': 'S2: Corridors',
        'S3_equity_first': 'S3: Equity First'
    }

    # 鍒涘缓鍥捐〃: 4琛?(鎯呮櫙) 脳 5鍒?(鍩庡競)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.patch.set_facecolor('white')

    # 閬嶅巻鍩庡競鍜屾儏鏅?
    for col_idx, city in enumerate(CITIES):
        print(f"\n澶勭悊 {city}...")

        # 鍔犺浇閬撹矾鏁版嵁
        roads = load_roads_data(city, time_scenario)
        boundary = load_city_boundary(city)

        if roads is None:
            for row_idx in range(4):
                axes[row_idx, col_idx].text(0.5, 0.5, 'No Data', ha='center', va='center',
                                            transform=axes[row_idx, col_idx].transAxes)
                axes[row_idx, col_idx].axis('off')
            continue

        # 灏嗛亾璺垎閰嶅埌LSOA
        print(f"  鍒嗛厤閬撹矾鍒癓SOA...")
        roads = assign_roads_to_lsoa(roads, imd_gdf)

        # 鑾峰彇璇ュ煄甯傜殑LSOA鏁版嵁
        city_lsoas = roads['lsoa11cd'].dropna().unique().tolist()
        city_lsoa_df = lsoa_df[lsoa_df['lsoa11cd'].isin(city_lsoas)].copy()

        for row_idx, scenario in enumerate(scenarios):
            print(f"  搴旂敤鎯呮櫙: {scenario}...")
            ax = axes[row_idx, col_idx]

            # 搴旂敤鎯呮櫙
            roads_scenario = apply_scenario(roads, city_lsoa_df, scenario)

            # 缁樺埗鍦板浘
            plot_city_scenario(ax, roads_scenario, boundary, threshold, scenario, city)

            # 鍩庡競鏍囬 (浠呯涓€琛?
            if row_idx == 0:
                ax.set_title(city, fontsize=11, fontweight='bold', pad=10)

            # 鎯呮櫙鏍囩 (浠呯涓€鍒?
            if col_idx == 0:
                ax.text(-0.15, 0.5, scenario_labels[scenario],
                       transform=ax.transAxes, fontsize=10, fontweight='bold',
                       rotation=90, va='center', ha='center')

    # 鎬绘爣棰?
    scenario_label = 'Heatwave' if time_scenario == 'heatwave' else 'Typical Day'
    fig.suptitle(f'Cool Street Network Under Policy Scenarios ({scenario_label}, 胃={threshold}掳C)',
                 fontsize=14, fontweight='bold', y=0.98)

    # 鍥句緥
    legend_elements = [
        Line2D([0], [0], color=COOL_COLOR, linewidth=3, label=f'Cool roads (HEI < {threshold}掳C)'),
        Line2D([0], [0], color=NEW_COOL_COLOR, linewidth=3, label='Newly cooled roads'),
        Line2D([0], [0], color='#CCCCCC', linewidth=1.5, label='Hot roads'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0.05, 0.03, 1, 0.96])
    plt.subplots_adjust(wspace=0.05, hspace=0.08)

    # 淇濆瓨
    output_name = f'FigS_policy_scenarios_cool_network_{time_scenario}_{threshold}C'
    fig.savefig(FIGURES_DIR / f'{output_name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES_DIR / f'{output_name}.pdf', bbox_inches='tight', facecolor='white')

    print(f"\n鍥捐〃宸蹭繚瀛樿嚦: {FIGURES_DIR}")
    print(f"  - {output_name}.png")
    print(f"  - {output_name}.pdf")

    plt.close(fig)


def main():
    """涓诲嚱鏁?""
    # 缁樺埗鐑氮鏃?35掳C闃堝€?
    plot_figure(time_scenario='heatwave', threshold=35)

    # 缁樺埗鐑氮鏃?28掳C闃堝€?
    plot_figure(time_scenario='heatwave', threshold=28)

    print("\n" + "=" * 60)
    print("瀹屾垚!")
    print("=" * 60)


if __name__ == '__main__':
    main()


