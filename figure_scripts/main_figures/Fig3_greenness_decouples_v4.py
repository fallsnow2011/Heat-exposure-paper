#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import warnings
warnings.filterwarnings('ignore')

# ============ 璺緞璁剧疆 ============
BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / 'results'
NDVI_DIR = BASE_DIR / 'GEE_NDVI_Exports'
FIGURES_DIR = BASE_DIR / 'paper' / '05_figures'
BOUNDARY_DIR = BASE_DIR / 'city_boundaries'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============ 鍏ㄥ眬鏍峰紡璁剧疆 ============
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0

# ============ 閰嶈壊鏂规 ============
BLUE_COOL = '#1f77b4'
GREEN_IDEAL = '#2ca02c'
GREY_HOT = '#7f7f7f'
RED_ISOLATED = '#d62728'

# 鍦板浘鍏冪礌
BUILDING_COLOR = '#e0e0e0'
BUILDING_EDGE = '#a0a0a0'
VEGETATION_COLOR = '#a8ddb5'
VEGETATION_EDGE = '#31a354'
ROAD_COLOR = 'black'


def load_data():
    """鍔犺浇鏁版嵁"""
    merged_df = pd.read_csv(RESULTS_DIR / 'ndvi_analysis' / 'lsoa_ndvi_cni_merged_heatwave.csv')
    imd_gdf = gpd.read_file(BOUNDARY_DIR / 'Indices_of_Multiple_Deprivation_(IMD)_2019_8404722932957776031.gpkg')
    imd_gdf = imd_gdf.to_crs(epsg=27700)
    return merged_df, imd_gdf


def find_quadrant_cases(df):
    """鎵惧埌鍥涗釜璞￠檺鐨勫吀鍨嬫渚?""
    df_filtered = df[df['n_roads'] >= 10].copy()
    df_filtered = df_filtered.dropna(subset=['ndvi_mean', 'shadow_mean'])

    ndvi_q20 = df_filtered['ndvi_mean'].quantile(0.2)
    ndvi_q80 = df_filtered['ndvi_mean'].quantile(0.8)
    shadow_q20 = df_filtered['shadow_mean'].quantile(0.2)
    shadow_q80 = df_filtered['shadow_mean'].quantile(0.8)

    quadrants = {
        'grey_cool': df_filtered[(df_filtered['ndvi_mean'] <= ndvi_q20) & (df_filtered['shadow_mean'] >= shadow_q80)],
        'green_cool': df_filtered[(df_filtered['ndvi_mean'] >= ndvi_q80) & (df_filtered['shadow_mean'] >= shadow_q80)],
        'grey_hot': df_filtered[(df_filtered['ndvi_mean'] <= ndvi_q20) & (df_filtered['shadow_mean'] <= shadow_q20)],
        'green_isolated': df_filtered[(df_filtered['ndvi_mean'] >= ndvi_q80) & (df_filtered['shadow_mean'] <= shadow_q20)],
    }

    cases = {}
    for name, subset in quadrants.items():
        # 涓嶅啀浼樺厛閫夋嫨浼︽暒锛岀洿鎺ヤ粠鎵€鏈夌鍚堟潯浠剁殑 LSOA 涓€夋嫨
        if len(subset) > 0:
            if name == 'grey_cool':
                idx = subset['shadow_mean'].idxmax()
            elif name == 'green_cool':
                idx = (subset['ndvi_mean'] * subset['shadow_mean']).idxmax()
            elif name == 'grey_hot':
                idx = (subset['ndvi_mean'] + subset['shadow_mean']).idxmin()
            else:
                idx = (subset['ndvi_mean'] / (subset['shadow_mean'] + 0.01)).idxmax()

            cases[name] = df_filtered.loc[idx].to_dict()

    return cases


def create_quintile_crosstab(df):
    """鍒涘缓浜斿垎浣嶄氦鍙夎〃"""
    # 杩囨护浣庨亾璺暟 LSOA (n_roads >= 10)
    df = df[df['n_roads'] >= 10].copy()
    df = df.dropna(subset=['ndvi_mean', 'shadow_mean'])
    df['ndvi_q'] = pd.qcut(df['ndvi_mean'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    df['shadow_q'] = pd.qcut(df['shadow_mean'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    crosstab = pd.crosstab(df['ndvi_q'], df['shadow_q'], normalize='all') * 100
    return crosstab


def plot_panel_a(ax, df):
    """Panel (a): Hexbin - NDVI vs Street Shading"""
    # 杩囨护浣庨亾璺暟 LSOA (n_roads >= 10)
    df_filtered = df[df['n_roads'] >= 10].copy()
    df_filtered = df_filtered.dropna(subset=['ndvi_mean', 'shadow_mean'])

    x = df_filtered['ndvi_mean'].values
    y = df_filtered['shadow_mean'].values

    # Hexbin with better colormap (GnBu - green to blue)
    hb = ax.hexbin(x, y, gridsize=35, cmap='GnBu', mincnt=1,
                   extent=[0.1, 1.0, 0, 0.65], linewidths=0.2, edgecolors='white')

    cb = plt.colorbar(hb, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label('Count', fontsize=8)

    # 鍥炲綊绾?(绾㈣壊绮楃嚎)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0.15, 0.95, 100)
    ax.plot(x_line, p(x_line), '-', color=RED_ISOLATED, linewidth=3, zorder=10)

    # 鐩稿叧绯绘暟
    r = np.corrcoef(x, y)[0, 1]

    # 鍙繚鐣欎富瑕佺煕鐩剧殑鏍囩锛堝乏涓?vs 鍙充笅锛?
    ax.text(0.08, 0.92, 'Compact &\nShaded', transform=ax.transAxes,
           fontsize=10, ha='left', va='top', color=BLUE_COOL,
           fontweight='bold')
    ax.text(0.92, 0.08, 'Disconnected\nGreen', transform=ax.transAxes,
           fontsize=10, ha='right', va='bottom', color=RED_ISOLATED,
           fontweight='bold')

    # 缁熻妗嗭紙浣跨敤杩囨护鍚庣殑鏍锋湰鏁帮級
    stats_text = f'r = {r:.2f}\nn = {len(df_filtered):,}'
    ax.text(0.50, 0.92, stats_text, transform=ax.transAxes,
           fontsize=12, ha='center', va='top', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor=RED_ISOLATED, linewidth=2))

    ax.set_xlabel('NDVI (Vegetation Greenness)', fontsize=11)
    ax.set_ylabel('Street Shading (Shadow Coverage)', fontsize=11)
    ax.set_title('(a) Greenness vs Street Shading', fontweight='bold', fontsize=12, loc='left')
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0, 0.65)
    ax.grid(True, linestyle='--', alpha=0.3)


def plot_panel_b(ax, crosstab):
    """Panel (b): Crosstab - 鍙繚鐣欏叧閿珮浜?""

    data = crosstab.values.astype(float)

    # 钃濊壊娓愬彉: 涓?Panel (a) 鐨?GnBu 閰嶈壊涓€鑷?
    cmap = 'GnBu'
    im = ax.imshow(data, cmap=cmap, aspect='equal', vmin=0, vmax=9)

    # 鏁板瓧鏍囨敞 (鍔犵矖)
    for i in range(5):
        for j in range(5):
            value = data[i, j]
            color = 'white' if value > 5 else 'black'
            ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                   fontsize=10, color=color, fontweight='bold')

    # 楂樹寒鍥涜 (涓嶄娇鐢ㄨ櫄绾挎锛屼繚鎸佹竻鏅扮洿瑙?
    # 宸︿笂瑙? Low NDVI + Low Shadow = Grey & Hot
    rect0 = plt.Rectangle((0-0.5, 0-0.5), 1, 1, fill=False,
                          edgecolor=GREY_HOT, linewidth=3)
    ax.add_patch(rect0)

    # 鍙充笂瑙? Low NDVI + High Shadow = Grey but Cool
    rect1 = plt.Rectangle((4-0.5, 0-0.5), 1, 1, fill=False,
                          edgecolor=BLUE_COOL, linewidth=3)
    ax.add_patch(rect1)

    # 宸︿笅瑙? High NDVI + Low Shadow = Green but Isolated
    rect2 = plt.Rectangle((0-0.5, 4-0.5), 1, 1, fill=False,
                          edgecolor=RED_ISOLATED, linewidth=3)
    ax.add_patch(rect2)

    # 鍙充笅瑙? High NDVI + High Shadow = Ideal Synergy
    rect3 = plt.Rectangle((4-0.5, 4-0.5), 1, 1, fill=False,
                          edgecolor=GREEN_IDEAL, linewidth=3)
    ax.add_patch(rect3)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'], fontsize=9)
    ax.set_yticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'], fontsize=9)

    ax.set_xlabel('Street Shading Quintile', fontsize=11)
    ax.set_ylabel('NDVI Quintile', fontsize=11)
    ax.set_title('(b) NDVI 脳 Shading Cross-tabulation', fontweight='bold', fontsize=12, loc='left')

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('% of LSOAs', fontsize=9)

    # 绠€娲佸浘渚?    ax.text(0, -1.0, 'Grey & Hot', fontsize=9, color=GREY_HOT,
            fontweight='bold', ha='center', clip_on=False)
    ax.text(4, -1.0, 'Grey but Cool', fontsize=9, color=BLUE_COOL,
            fontweight='bold', ha='center', clip_on=False)
    ax.text(0, 5.5, 'Green but Isolated', fontsize=9, color=RED_ISOLATED,
            fontweight='bold', ha='center', clip_on=False)
    ax.text(4, 5.5, 'Ideal Synergy', fontsize=9, color=GREEN_IDEAL,
            fontweight='bold', ha='center', clip_on=False)


def plot_lsoa_map(ax, lsoa_code, city, imd_gdf, main_title, subtitle, title_color):
    """缁樺埗LSOA鍦板浘 - 娓呮櫚鐨勫眰绾ф帓鐗?""

    lsoa_geom = imd_gdf[imd_gdf['lsoa11cd'] == lsoa_code]
    if len(lsoa_geom) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
        ax.axis('off')
        return

    bounds = lsoa_geom.total_bounds
    buf = 15

    # 搴曡壊
    lsoa_geom.plot(ax=ax, facecolor='#fafafa', edgecolor='none', zorder=0)

    # 缁垮湴
    try:
        gs = gpd.read_file(BOUNDARY_DIR / f'{city}_greenspace_osopen_v3.gpkg',
                          bbox=(bounds[0]-buf, bounds[1]-buf, bounds[2]+buf, bounds[3]+buf))
        if len(gs) > 0:
            gs_clip = gpd.clip(gs, lsoa_geom)
            if len(gs_clip) > 0:
                gs_clip.plot(ax=ax, facecolor=VEGETATION_COLOR, edgecolor=VEGETATION_EDGE,
                            linewidth=0.3, alpha=0.7, zorder=1)
    except:
        pass

    # 寤虹瓚
    try:
        bld = gpd.read_file(BOUNDARY_DIR / f'{city}_buildings_osopen_v3.gpkg',
                           bbox=(bounds[0]-buf, bounds[1]-buf, bounds[2]+buf, bounds[3]+buf))
        if len(bld) > 0:
            bld_clip = gpd.clip(bld, lsoa_geom)
            if len(bld_clip) > 0:
                bld_clip.plot(ax=ax, facecolor=BUILDING_COLOR, edgecolor=BUILDING_EDGE,
                             linewidth=0.3, zorder=2)
    except:
        pass

    # 璺綉 (绾粦鑹茬粏绾?
    try:
        roads = gpd.read_file(BOUNDARY_DIR / f'{city}_roads_OS.gpkg',
                             bbox=(bounds[0]-buf, bounds[1]-buf, bounds[2]+buf, bounds[3]+buf))
        if len(roads) > 0:
            roads_clip = gpd.clip(roads, lsoa_geom)
            if len(roads_clip) > 0:
                roads_clip.plot(ax=ax, color=ROAD_COLOR, linewidth=0.5, zorder=3)
    except:
        pass

    # LSOA 杈圭晫
    lsoa_geom.boundary.plot(ax=ax, color='black', linewidth=1.5, zorder=4)

    ax.set_xlim(bounds[0]-buf, bounds[2]+buf)
    ax.set_ylim(bounds[1]-buf, bounds[3]+buf)
    ax.set_aspect('equal')
    ax.axis('off')

    # 鏍囬灞傜骇: 涓绘爣棰?(Bold, 涓婃柟) + 鍓爣棰?(Italic, 涓绘爣棰樹笅鏂?
    # 浣跨敤 text 鑰屼笉鏄?set_title 浠ヤ究鏇村ソ鍦版帶鍒朵綅缃?
    ax.text(0.5, 1.12, main_title, transform=ax.transAxes, fontsize=10,
           fontweight='bold', color=title_color, ha='center', va='bottom')
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, fontsize=8,
           ha='center', va='bottom', style='italic', color='#555555')


def plot_lsoa_map_simple(ax, lsoa_code, city, imd_gdf):
    """缁樺埗LSOA鍦板浘 - 绠€鍖栫増锛屼娇鐢?NDVI 鏍呮牸鏄剧ず缁垮害"""

    lsoa_geom = imd_gdf[imd_gdf['lsoa11cd'] == lsoa_code]
    if len(lsoa_geom) == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
        ax.axis('off')
        return

    bounds = lsoa_geom.total_bounds
    buf = 15

    # 搴曡壊
    lsoa_geom.plot(ax=ax, facecolor='#fafafa', edgecolor='none', zorder=0)

    # 浣跨敤 NDVI 鏍呮牸鏁版嵁鏄剧ず缁垮害锛堟浛浠?OS Open greenspace锛?
    try:
        ndvi_file = NDVI_DIR / f'{city}_NDVI_Max_2022_10m.tif'
        if ndvi_file.exists():
            with rasterio.open(ndvi_file) as src:
                # 瑁佸壀鍒?LSOA 杈圭晫
                geom = [mapping(lsoa_geom.geometry.values[0])]
                out_image, out_transform = mask(src, geom, crop=True)
                ndvi_data = out_image[0]

                # 鍒涘缓缁胯壊鑹插浘 (NDVI > 0.3 鏄剧ず涓虹豢鑹?
                ndvi_masked = np.ma.masked_where(ndvi_data < 0.3, ndvi_data)

                # 璁＄畻鏄剧ず鑼冨洿
                rows, cols = ndvi_data.shape
                left = out_transform[2]
                top = out_transform[5]
                right = left + cols * out_transform[0]
                bottom = top + rows * out_transform[4]

                # 鍒涘缓缁胯壊娓愬彉鑹插浘
                green_cmap = LinearSegmentedColormap.from_list('green',
                    ['#c7e9c0', '#74c476', '#31a354', '#006d2c'])

                ax.imshow(ndvi_masked, extent=[left, right, bottom, top],
                         cmap=green_cmap, vmin=0.3, vmax=0.9, alpha=0.8, zorder=1)
    except Exception as e:
        # 濡傛灉 NDVI 鍔犺浇澶辫触锛屽洖閫€鍒?OS Open greenspace
        try:
            gs = gpd.read_file(BOUNDARY_DIR / f'{city}_greenspace_osopen_v3.gpkg',
                              bbox=(bounds[0]-buf, bounds[1]-buf, bounds[2]+buf, bounds[3]+buf))
            if len(gs) > 0:
                gs_clip = gpd.clip(gs, lsoa_geom)
                if len(gs_clip) > 0:
                    gs_clip.plot(ax=ax, facecolor=VEGETATION_COLOR, edgecolor=VEGETATION_EDGE,
                                linewidth=0.3, alpha=0.7, zorder=1)
        except:
            pass

    # 寤虹瓚
    try:
        bld = gpd.read_file(BOUNDARY_DIR / f'{city}_buildings_osopen_v3.gpkg',
                           bbox=(bounds[0]-buf, bounds[1]-buf, bounds[2]+buf, bounds[3]+buf))
        if len(bld) > 0:
            bld_clip = gpd.clip(bld, lsoa_geom)
            if len(bld_clip) > 0:
                bld_clip.plot(ax=ax, facecolor=BUILDING_COLOR, edgecolor=BUILDING_EDGE,
                             linewidth=0.3, zorder=2)
    except:
        pass

    # 璺綉 (绾粦鑹茬粏绾?
    try:
        roads = gpd.read_file(BOUNDARY_DIR / f'{city}_roads_OS.gpkg',
                             bbox=(bounds[0]-buf, bounds[1]-buf, bounds[2]+buf, bounds[3]+buf))
        if len(roads) > 0:
            roads_clip = gpd.clip(roads, lsoa_geom)
            if len(roads_clip) > 0:
                roads_clip.plot(ax=ax, color=ROAD_COLOR, linewidth=0.5, zorder=3)
    except:
        pass

    # LSOA 杈圭晫
    lsoa_geom.boundary.plot(ax=ax, color='black', linewidth=1.5, zorder=4)

    ax.set_xlim(bounds[0]-buf, bounds[2]+buf)
    ax.set_ylim(bounds[1]-buf, bounds[3]+buf)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_panel_c(fig, gs_row, cases, imd_gdf):
    """Panel (c): 鍥涗釜鍏稿瀷LSOA - 娓呮櫚鎺掔増"""

    # 浣跨敤2琛屽瓙缃戞牸: 绗竴琛屾斁鏍囬锛岀浜岃鏀惧湴鍥?    gs_inner = gs_row.subgridspec(2, 4, wspace=0.12, hspace=0.04, height_ratios=[0.22, 1])

    legend_elements = [
        mpatches.Patch(facecolor=BUILDING_COLOR, edgecolor=BUILDING_EDGE, linewidth=0.5, label='Buildings'),
        mpatches.Patch(facecolor=VEGETATION_COLOR, edgecolor=VEGETATION_EDGE, linewidth=0.5, label='Green Space'),
        plt.Line2D([0], [0], color=ROAD_COLOR, linewidth=1, label='Roads'),
    ]

    # 閰嶇疆: (key, main_title, subtitle, color)
    config = [
        ('grey_cool', 'Grey but Cool', 'Dense streets, high shading', BLUE_COOL),
        ('green_cool', 'Ideal Synergy', 'Green and well-connected', GREEN_IDEAL),
        ('grey_hot', 'Grey & Hot', 'Heat island, no shade', GREY_HOT),
        ('green_isolated', 'Green but Isolated', 'Large parks, fragmented paths', RED_ISOLATED),
    ]

    for i, (key, title, subtitle, color) in enumerate(config):
        # 鏍囬鍖哄煙 (涓婃柟)
        ax_title = fig.add_subplot(gs_inner[0, i])
        ax_title.axis('off')
        ax_title.text(0.5, 0.30, title, transform=ax_title.transAxes, fontsize=12,
                     fontweight='bold', color=color, ha='center', va='bottom')
        ax_title.text(0.5, 0.0, subtitle, transform=ax_title.transAxes, fontsize=9,
                     style='italic', color='#555555', ha='center', va='bottom')

        # 鍦板浘鍖哄煙 (涓嬫柟)
        ax = fig.add_subplot(gs_inner[1, i])

        if key in cases:
            row = cases[key]
            # 涓嶅啀浼犻€掓爣棰樺弬鏁?
            plot_lsoa_map_simple(ax, row['lsoa11cd'], row['city'], imd_gdf)

            # 搴曢儴鏁版嵁鏍囩
            info = f"NDVI: {row['ndvi_mean']:.2f} | Shade: {row['shadow_mean']:.2f}"
            ax.text(0.5, -0.05, info, transform=ax.transAxes, fontsize=9,
                    ha='center', va='top', color='#666666')

            # 鍥句緥鏀惧湪绗?4 涓瓙鍥惧彸涓婅锛岄伩鍏嶉伄鎸￠潰鏉挎暣浣?            if i == 3:
                ax.legend(handles=legend_elements, loc='upper right',
                          fontsize=8, framealpha=0.9, borderaxespad=0.2,
                          borderpad=0.25, labelspacing=0.2,
                          handlelength=1.2, handletextpad=0.4)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.axis('off')


def plot_figure3(merged_df, imd_gdf):
    """缁戝埗 Figure 3 v4"""

    print("Finding cases...")
    cases = find_quadrant_cases(merged_df)
    for k, v in cases.items():
        print(f"  {k}: {v['lsoa11cd']} - NDVI={v['ndvi_mean']:.3f}, Shadow={v['shadow_mean']:.3f}")

    print("Creating crosstab...")
    crosstab = create_quintile_crosstab(merged_df)

    # 鍒涘缓鍥捐〃
    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95], width_ratios=[1.2, 1],
                          wspace=0.22, hspace=0.34,
                          left=0.06, right=0.96, top=0.94, bottom=0.06)

    # Panel (a)
    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_a(ax_a, merged_df)

    # Panel (b)
    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel_b(ax_b, crosstab)

    # Panel (c)
    gs_bottom = gs[1, :]
    ax_c = fig.add_subplot(gs_bottom)
    ax_c.axis('off')
    ax_c.set_title('(c) Representative LSOAs: Four Typologies',
                  fontweight='bold', fontsize=13, loc='left', pad=15)

    plot_panel_c(fig, gs_bottom, cases, imd_gdf)

    # 淇濆瓨
    fig.savefig(FIGURES_DIR / 'Fig3_greenness_decouples_v4.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES_DIR / 'Fig3_greenness_decouples_v4.pdf',
                bbox_inches='tight', facecolor='white')

    print(f"\nSaved to: {FIGURES_DIR}")
    plt.close(fig)


if __name__ == '__main__':
    print("Loading data...")
    merged_df, imd_gdf = load_data()
    print(f"Loaded {len(merged_df)} LSOAs")

    print("\nPlotting Figure 3 v4...")
    plot_figure3(merged_df, imd_gdf)
    print("Done!")


