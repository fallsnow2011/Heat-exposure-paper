#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============ 璺緞璁剧疆 ============
BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / 'results' / 'heat_exposure'
FIGURES_DIR = BASE_DIR / 'paper' / '05_figures'

# 纭繚杈撳嚭鐩綍瀛樺湪
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============ 鍏ㄥ眬鏍峰紡璁剧疆 ============
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 9

# ============ 閰嶈壊鏂规 - 鍩庡競娴呰壊绯婚鏍?============
# 鍩庡競棰滆壊 - 娣辫壊鐗堟湰 (Heatwave / Top 10% hottest)
city_colors_dark = {
    'London': '#1f77b4',      # 娣辫摑
    'Birmingham': '#17becf',  # 娣遍潚
    'Manchester': '#ff7f0e',  # 娣辨
    'Bristol': '#9467bd',     # 娣辩传
    'Newcastle': '#d62728'    # 娣辩孩
}

# 鍩庡競棰滆壊 - 娴呰壊鐗堟湰 (Typical Day / Normal streets)
city_colors_light = {
    'London': '#aec7e8',      # 娴呰摑
    'Birmingham': '#9edae5',  # 娴呴潚
    'Manchester': '#ffbb78',  # 娴呮
    'Bristol': '#c5b0d5',     # 娴呯传
    'Newcastle': '#ff9896'    # 娴呯孩
}

# Panel (e) 涓撶敤棰滆壊 - 娴呯伆鑹插拰娴呯豢鑹?
BUILDING_GRAY = '#b0b0b0'      # 娴呯伆鑹?- 寤虹瓚闃村奖
VEGETATION_GREEN = '#98df8a'   # 娴呯豢鑹?- 妞嶈闃村奖

# 鍩庡競椤哄簭 (涓?Figure 1 涓€鑷?
cities = ['London', 'Birmingham', 'Manchester', 'Bristol', 'Newcastle']


def load_data():
    """Load all road-level datasets."""
    print("Loading road-level datasets...")

    data_typical = {}
    data_heatwave = {}

    for city in cities:
        data_typical[city] = gpd.read_file(
            RESULTS_DIR / f'{city}_roads_hei_improved_typical_day.gpkg'
        )
        data_heatwave[city] = gpd.read_file(
            RESULTS_DIR / f'{city}_roads_hei_improved_heatwave.gpkg'
        )

    print(f"Loaded datasets for {len(cities)} cities")
    return data_typical, data_heatwave


def plot_figure2(data_typical, data_heatwave):
    """Create figure 2."""

    # 鍒涘缓鍥捐〃
    fig = plt.figure(figsize=(14, 8))

    # Grid layout: 2 rows
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], width_ratios=[1, 1, 1.3],
                          hspace=0.25, wspace=0.25, left=0.06, right=0.98,
                          top=0.95, bottom=0.20)

    # ============ 鍒涘缓娓╁害鍒嗘暎鑹插浘 (Blue-White-Red) ============
    cmap_diverging = LinearSegmentedColormap.from_list('temp_diverging',
        ['#2166AC', '#67A9CF', '#D1E5F0', '#F7F7F7', '#FDDBC7', '#EF8A62', '#B2182B'])

    # ============ Panel (a): LST Map ============
    ax_a = fig.add_subplot(gs[0, 0])
    london_hw = data_heatwave['London']

    # South London 鍖哄煙
    xmin, xmax = 525000, 530000
    ymin, ymax = 173000, 178000
    london_subset = london_hw.cx[xmin:xmax, ymin:ymax].copy()

    mean_lst = london_subset['lst'].mean()
    vmin, vmax = 36, 50
    vcenter = 43
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    london_subset.plot(ax=ax_a, column='lst', cmap=cmap_diverging, linewidth=1.2,
                       norm=norm, legend=False)
    ax_a.set_xlim(xmin, xmax)
    ax_a.set_ylim(ymin, ymax)
    ax_a.set_aspect('equal')
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.set_title('(a) LST: Surface Temperature\n(South London, Heatwave)', fontweight='bold', fontsize=10)

    # 娣诲姞缁嗛粦杈规
    for spine in ax_a.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')
        spine.set_visible(True)

    # 鍧囧€兼爣娉?
    ax_a.text(0.05, 0.95, f'Mean: {mean_lst:.1f}°C', transform=ax_a.transAxes,
              fontsize=8, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange'))

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_diverging, norm=norm)
    cbar_a = plt.colorbar(sm, ax=ax_a, orientation='horizontal', pad=0.02,
                          fraction=0.046, aspect=20)
    cbar_a.set_label('Street Temperature (°C)', fontsize=8)
    cbar_a.ax.tick_params(labelsize=7)

    # ============ Panel (b): HEI Map ============
    ax_b = fig.add_subplot(gs[0, 1])
    mean_hei = london_subset['hei_improved'].mean()

    london_subset.plot(ax=ax_b, column='hei_improved', cmap=cmap_diverging, linewidth=1.2,
                       norm=norm, legend=False)
    ax_b.set_xlim(xmin, xmax)
    ax_b.set_ylim(ymin, ymax)
    ax_b.set_aspect('equal')
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    ax_b.set_title('(b) HEI: With Shade Cooling\n(South London, Heatwave)', fontweight='bold', fontsize=10)

    # 娣诲姞缁嗛粦杈规
    for spine in ax_b.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')
        spine.set_visible(True)

    # 鍧囧€煎拰闄嶆俯鏍囨敞
    delta = mean_hei - mean_lst
    ax_b.text(0.05, 0.95, f'Mean: {mean_hei:.1f}°C', transform=ax_b.transAxes,
              fontsize=8, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9, edgecolor='dodgerblue'))
    ax_b.text(0.65, 0.95, f'Δ = {delta:.1f}°C', transform=ax_b.transAxes,
              fontsize=8, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='green'))

    # Colorbar
    cbar_b = plt.colorbar(sm, ax=ax_b, orientation='horizontal', pad=0.02,
                          fraction=0.046, aspect=20)
    cbar_b.set_label('Street Temperature (°C)', fontsize=8)
    cbar_b.ax.tick_params(labelsize=7)

    # ============ Panel (c): LST vs HEI Hexbin ============
    ax_c = fig.add_subplot(gs[0, 2])

    # 鏀堕泦鎵€鏈夊煄甯傜殑鐑氮鏁版嵁
    all_lst = []
    all_hei = []
    for city in cities:
        gdf = data_heatwave[city]
        valid = gdf[['lst', 'hei_improved']].dropna()
        all_lst.extend(valid['lst'].values)
        all_hei.extend(valid['hei_improved'].values)

    all_lst = np.array(all_lst)
    all_hei = np.array(all_hei)

    # 鑳屾櫙闃村奖闄嶆俯鍖哄煙 - 瀵硅绾夸互涓嬪叏鍩?(HEI < LST 琛ㄧず閬槾闄嶆俯)
    # x: 35-55, y 涓嬭竟鐣? 30 (鍥捐〃搴曢儴), y 涓婅竟鐣? 瀵硅绾?(y=x)
    ax_c.plot([30, 55], [30, 55], 'k--', linewidth=1, alpha=0.5)
    ax_c.set_facecolor('white')

    # Hexbin 瀵嗗害鍥?
    hb = ax_c.hexbin(all_lst, all_hei, gridsize=40, cmap='YlOrRd', mincnt=1,
                     extent=[35, 55, 30, 55])

    ax_c.set_xlabel('LST (°C)', fontsize=9)
    ax_c.set_ylabel('HEI (°C)', fontsize=9)
    ax_c.set_title('(c) LST vs HEI (All Cities)', fontweight='bold', fontsize=10)
    ax_c.set_xlim(35, 55)
    ax_c.set_ylim(30, 55)

    # "Shade cooling" 鏍囩
    ax_c.text(0.15, 0.85, 'HEI < LST\n(shade cooling)', transform=ax_c.transAxes,
              fontsize=9, color='darkgreen', fontweight='bold', alpha=0.8)

    # Colorbar
    cbar_c = plt.colorbar(hb, ax=ax_c, label='Roads')
    cbar_c.ax.tick_params(labelsize=7)

    # ============ ROW 2: Super-Row (d, e, f) ============

    # ============ Panel (d): Shade Cooling Effect - 鍩庡競娴呰壊绯?============
    ax_d = fig.add_subplot(gs[1, 0])

    # 璁＄畻姣忎釜鍩庡競鐨勫钩鍧囬檷娓╋紙鎸夐亾璺暱搴﹀姞鏉冿級
    cooling_typical = []
    cooling_heatwave = []
    for city in cities:
        gdf_t = data_typical[city].copy()
        gdf_h = data_heatwave[city].copy()
        gdf_t['length'] = gdf_t.geometry.length
        gdf_h['length'] = gdf_h.geometry.length
        # 闀垮害鍔犳潈骞冲潎
        weighted_typical = (gdf_t['cooling_total'] * gdf_t['length']).sum() / gdf_t['length'].sum()
        weighted_heatwave = (gdf_h['cooling_total'] * gdf_h['length']).sum() / gdf_h['length'].sum()
        cooling_typical.append(weighted_typical)
        cooling_heatwave.append(weighted_heatwave)

    y_pos = np.arange(len(cities))
    bar_height = 0.35

    # 鍩庡競涓撳睘娴呰壊/娣辫壊鏉″舰鍥?- 鏃犺竟妗?
    for i, city in enumerate(cities):
        # 娴呰壊鏉?(Typical Day) - 鏃犺竟妗?
        ax_d.barh(y_pos[i] + bar_height/2, cooling_typical[i], bar_height,
                  color=city_colors_light[city], edgecolor='none')
        # 娣辫壊鏉?(Heatwave) - 鏃犺竟妗?
        ax_d.barh(y_pos[i] - bar_height/2, cooling_heatwave[i], bar_height,
                  color=city_colors_dark[city], edgecolor='none')

    # 鏁板€兼爣绛?- 鍏ㄩ儴榛戝瓧鏀惧湪鏉″舰澶?
    for i, (v_light, v_dark) in enumerate(zip(cooling_typical, cooling_heatwave)):
        ax_d.text(v_light + 0.05, i + bar_height/2, f'{v_light:.1f}',
                  va='center', fontsize=8, color='black')
        ax_d.text(v_dark + 0.05, i - bar_height/2, f'{v_dark:.1f}',
                  va='center', fontsize=8, color='black')

    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(cities, fontsize=9)
    for i, label in enumerate(ax_d.get_yticklabels()):
        label.set_color(city_colors_dark[cities[i]])
        label.set_fontweight('bold')

    ax_d.set_xlabel('Temperature Reduction (°C)', fontsize=9)
    ax_d.set_title('(d) Shade Cooling Effect', fontweight='bold', fontsize=10)
    ax_d.set_xlim(0, 4)
    ax_d.invert_yaxis()
    legend_handles_d = [
        mpatches.Patch(facecolor=city_colors_light['London'], edgecolor='none',
                       label='Typical day'),
        mpatches.Patch(facecolor=city_colors_dark['London'], edgecolor='none',
                       label='Heatwave'),
    ]
    ax_d.legend(handles=legend_handles_d, loc='upper center', ncol=2,
                fontsize=8, frameon=True, fancybox=True,
                bbox_to_anchor=(0.5, -0.18), borderaxespad=0.0)


    # ============ Panel (e): Building vs Vegetation Cooling - 娴呯伆鑹?娴呯豢鑹?============
    ax_e = fig.add_subplot(gs[1, 1])

    # 璁＄畻璐＄尞姣斾緥锛堟寜閬撹矾闀垮害鍔犳潈锛?
    building_pct = []
    vegetation_pct = []
    for city in cities:
        gdf = data_heatwave[city].copy()
        gdf['length'] = gdf.geometry.length
        # 闀垮害鍔犳潈鐨勯檷娓╅噺
        total_cooling_weighted = (gdf['cooling_total'] * gdf['length']).sum()
        building_cooling_weighted = (gdf['cooling_building'] * gdf['length']).sum()
        vegetation_cooling_weighted = (gdf['cooling_vegetation'] * gdf['length']).sum()

        if total_cooling_weighted > 0:
            b_pct = (building_cooling_weighted / total_cooling_weighted) * 100
            v_pct = (vegetation_cooling_weighted / total_cooling_weighted) * 100
        else:
            b_pct, v_pct = 50, 50

        building_pct.append(b_pct)
        vegetation_pct.append(v_pct)

    # 娴呯伆鑹?+ 娴呯豢鑹?100% 鍫嗗彔鏉″舰鍥?- 鏃犺竟妗?
    bars_building = ax_e.barh(y_pos, building_pct, bar_height*1.8,
                              color=BUILDING_GRAY, edgecolor='none')
    bars_vegetation = ax_e.barh(y_pos, vegetation_pct, bar_height*1.8,
                                left=building_pct, color=VEGETATION_GREEN, edgecolor='none')

    # 鐧惧垎姣旀爣绛?- 榛戝瓧
    for i, pct in enumerate(building_pct):
        if pct > 10:
            ax_e.text(pct/2, i, f'{pct:.0f}%', ha='center', va='center',
                      fontsize=8, fontweight='bold', color='black')

    for i, (pct, b_pct) in enumerate(zip(vegetation_pct, building_pct)):
        if pct > 10:
            ax_e.text(b_pct + pct/2, i, f'{pct:.0f}%', ha='center', va='center',
                      fontsize=8, fontweight='bold', color='black')

    ax_e.set_yticks(y_pos)
    ax_e.set_yticklabels([''] * len(cities))  # Super-row 璁捐锛屼笉鏄剧ず鏍囩
    ax_e.set_xlabel('Cooling contribution (%)', fontsize=9)
    ax_e.set_title('(e) Cooling contribution (length-weighted)',
                   fontweight='bold', fontsize=10)
    ax_e.set_xlim(0, 100)
    legend_handles_e = [
        mpatches.Patch(facecolor=BUILDING_GRAY, edgecolor='none', label='Building cooling'),
        mpatches.Patch(facecolor=VEGETATION_GREEN, edgecolor='none', label='Vegetation cooling'),
    ]
    ax_e.legend(handles=legend_handles_e, loc='upper center', ncol=2,
                fontsize=8, frameon=True, fancybox=True,
                bbox_to_anchor=(0.5, -0.18), borderaxespad=0.0)

    ax_e.invert_yaxis()

    # ============ Panel (f): Why Hot Streets Stay Hot - 鍩庡競娴呰壊绯?============
    ax_f = fig.add_subplot(gs[1, 2])

    # 璁＄畻 Top 10% 鏈€鐑閬?vs 鍏朵粬琛楅亾鐨勯檷娓╋紙鎸夐亾璺暱搴﹀姞鏉冿級
    cooling_hot = []
    cooling_normal = []

    for city in cities:
        gdf = data_heatwave[city].copy()
        gdf['length'] = gdf.geometry.length
        threshold = gdf['hei_improved'].quantile(0.90)

        hot_streets = gdf[gdf['hei_improved'] >= threshold]
        normal_streets = gdf[gdf['hei_improved'] < threshold]

        # 闀垮害鍔犳潈骞冲潎
        if hot_streets['length'].sum() > 0:
            weighted_hot = (hot_streets['cooling_total'] * hot_streets['length']).sum() / hot_streets['length'].sum()
        else:
            weighted_hot = 0
        weighted_normal = (normal_streets['cooling_total'] * normal_streets['length']).sum() / normal_streets['length'].sum()

        cooling_hot.append(weighted_hot)
        cooling_normal.append(weighted_normal)

    # 鍩庡競涓撳睘娴呰壊/娣辫壊鏉″舰鍥?- 鏃犺竟妗?
    for i, city in enumerate(cities):
        # 娴呰壊鏉?(Normal streets) - 鏃犺竟妗?
        ax_f.barh(y_pos[i] + bar_height/2, cooling_normal[i], bar_height,
                  color=city_colors_light[city], edgecolor='none')
        # 娣辫壊鏉?(Top 10% hottest) - 鏃犺竟妗?
        ax_f.barh(y_pos[i] - bar_height/2, cooling_hot[i], bar_height,
                  color=city_colors_dark[city], edgecolor='none')

    # 鏁板€兼爣绛?- 鍏ㄩ儴榛戝瓧鏀惧湪鏉″舰澶?
    for i, (v_normal, v_hot) in enumerate(zip(cooling_normal, cooling_hot)):
        ax_f.text(v_normal + 0.05, i + bar_height/2, f'{v_normal:.1f}',
                  va='center', fontsize=8, color='black')
        ax_f.text(v_hot + 0.08, i - bar_height/2, f'{v_hot:.1f}',
                  va='center', fontsize=8, color='black')

    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels([''] * len(cities))
    ax_f.set_xlabel('Shade Cooling (°C)', fontsize=9)
    ax_f.set_title('(f) Heatwave only: Other vs Top 10% hottest', fontweight='bold', fontsize=10)
    ax_f.set_xlim(0, 4)
    ax_f.invert_yaxis()

    # 鍏抽敭鍙戠幇娉ㄩ噴
    ax_f.annotate('Top 10% hottest streets:\nNo shade -> No cooling',
                  xy=(0.5, 4), xytext=(2.0, 3.5),
                  fontsize=8, ha='left',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                            edgecolor='orange', alpha=0.95),
                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
                                  color='orange'))
    legend_handles_f = [
        mpatches.Patch(facecolor=city_colors_light['London'], edgecolor='none',
                       label='Other streets'),
        mpatches.Patch(facecolor=city_colors_dark['London'], edgecolor='none',
                       label='Top 10% hottest'),
    ]
    ax_f.legend(handles=legend_handles_f, loc='upper center', ncol=2,
                fontsize=8, frameon=True, fancybox=True,
                bbox_to_anchor=(0.5, -0.18), borderaxespad=0.0)


    # 淇濆瓨鍥捐〃
    fig.savefig(FIGURES_DIR / 'Fig2_street_thermal_exposure.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES_DIR / 'Fig2_street_thermal_exposure.pdf',
                bbox_inches='tight', facecolor='white')

    print(f"`nSaved Figure 2 to: {FIGURES_DIR}")
    print("  - Fig2_street_thermal_exposure.png")
    print("  - Fig2_street_thermal_exposure.pdf")

    # plt.show()  # Disabled for non-interactive execution

    return fig


if __name__ == '__main__':
    # 鍔犺浇鏁版嵁
    data_typical, data_heatwave = load_data()

    # 缁樺埗鍥捐〃
    print("`nRendering Figure 2...")
    fig = plot_figure2(data_typical, data_heatwave)

    print("`nDone.")


