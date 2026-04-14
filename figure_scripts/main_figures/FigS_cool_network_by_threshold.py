#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import geopandas as gpd
import numpy as np
from pathlib import Path

# ============ 璺緞璁剧疆 ============
BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / 'results' / 'heat_exposure'
FIGURES_DIR = BASE_DIR / 'paper' / '06_supplement'
BOUNDARY_DIR = BASE_DIR / 'city_boundaries'

# 纭繚杈撳嚭鐩綍瀛樺湪
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============ 鍏ㄥ眬鏍峰紡璁剧疆 (Nature Style) ============
plt.rcParams['font.family'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.3
plt.rcParams['axes.edgecolor'] = '#CCCCCC'

# ============ 椤跺垔绾ч厤鑹叉柟妗?============
# 鍑夌埥閬撹矾: 鐢靛厜钃?(Electric Azure) - 楂橀ケ鍜屽害锛屾瀬鍏风┛閫忓姏
COOL_COLOR = '#00AEEF'
# 鑳屾櫙鐑矾缃? 鏋佹祬鏆栫伆 - 鍑犱箮闅愬舰锛屽绾稿紶姘村嵃
HOT_COLOR = '#E8E8E8'
# 鍩庡競杈圭晫: 涓伆鑹茬粏绾?
BOUNDARY_COLOR = '#999999'

# ============ 绾挎潯鏉冮噸 (浼樺寲鍚? ============
COOL_LINEWIDTH = 1.0      # 鍑夌埥閬撹矾: 鍔犵矖锛屾洿鏄庢樉
HOT_LINEWIDTH = 0.15      # 鐑亾璺? 鐣ョ矖涓€鐐癸紝鎻愪緵鑳屾櫙鍙傝€?
BOUNDARY_LINEWIDTH = 0.8  # 杈圭晫绾垮姞绮?

# 鍩庡競椤哄簭
CITIES = ['London', 'Birmingham', 'Manchester', 'Bristol', 'Newcastle']

# HEI闃堝€?
THRESHOLDS = [28, 35]


def load_roads_data(city, scenario='heatwave'):
    """鍔犺浇閬撹矾HEI鏁版嵁"""
    if scenario == 'heatwave':
        file_path = RESULTS_DIR / f'{city}_roads_hei_improved_heatwave.gpkg'
    else:
        file_path = RESULTS_DIR / f'{city}_roads_hei_improved_typical_day.gpkg'

    if not file_path.exists():
        print(f"鏂囦欢涓嶅瓨鍦? {file_path}")
        return None

    roads = gpd.read_file(file_path)

    # 纭繚CRS涓€鑷?(British National Grid)
    if roads.crs.to_epsg() != 27700:
        roads = roads.to_crs(epsg=27700)

    return roads


def load_city_boundary(city):
    """鍔犺浇鍩庡競杈圭晫"""
    boundary_file = BOUNDARY_DIR / f'{city}_boundary.geojson'
    if boundary_file.exists():
        gdf = gpd.read_file(boundary_file)
        if gdf.crs.to_epsg() != 27700:
            gdf = gdf.to_crs(epsg=27700)
        return gdf
    return None


def calculate_cool_stats(roads, threshold):
    """璁＄畻鍑夌埥缃戠粶缁熻"""
    valid_roads = roads[roads['hei_improved'].notna()]
    cool_roads = valid_roads[valid_roads['hei_improved'] < threshold]

    n_total = len(valid_roads)
    n_cool = len(cool_roads)
    pct_cool = (n_cool / n_total * 100) if n_total > 0 else 0

    # 璁＄畻閬撹矾鎬婚暱搴?(km)
    total_length_km = valid_roads.geometry.length.sum() / 1000
    cool_length_km = cool_roads.geometry.length.sum() / 1000 if len(cool_roads) > 0 else 0

    return n_total, n_cool, pct_cool, total_length_km, cool_length_km


def add_scale_bar(ax, length_km=5, location='lower right'):
    """娣诲姞绠€娲佺殑姣斾緥灏?(Nature Style)"""
    # 鑾峰彇鍧愭爣鑼冨洿
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 姣斾緥灏洪暱搴?(绫?
    bar_length = length_km * 1000

    # 浣嶇疆璁＄畻
    if location == 'lower right':
        x_start = xlim[1] - (xlim[1] - xlim[0]) * 0.05 - bar_length
        y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    else:  # lower left
        x_start = xlim[0] + (xlim[1] - xlim[0]) * 0.05
        y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.05

    # 缁樺埗姣斾緥灏虹嚎鏉?
    ax.plot([x_start, x_start + bar_length], [y_pos, y_pos],
            color='#333333', linewidth=1.5, solid_capstyle='butt')

    # 娣诲姞鏍囩
    ax.text(x_start + bar_length / 2, y_pos + (ylim[1] - ylim[0]) * 0.02,
            f'{length_km} km', ha='center', va='bottom',
            fontsize=7, color='#333333', fontweight='normal')


def plot_cool_network_figure(scenario='heatwave'):
    """缁戝埗鍑夌埥缃戠粶鍒嗗竷鍥?(Nature Cities 椤跺垔绾у埆)"""

    scenario_label = 'Heatwave' if scenario == 'heatwave' else 'Typical Day'
    print(f"\n姝ｅ湪缁戝埗 {scenario_label} 鍑夌埥缃戠粶鍥?..")

    # 鍒涘缓鍥捐〃: 2琛?脳 5鍒楋紝澧炲姞鍛煎惛鎰?
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    fig.patch.set_facecolor('white')

    # 涓绘爣棰?(绠€娲侊紝Nature Style)
    fig.suptitle(f'Cool Street Network Distribution ({scenario_label})',
                 fontsize=13, fontweight='bold', y=0.97)

    # 閬嶅巻闃堝€煎拰鍩庡競
    for row_idx, threshold in enumerate(THRESHOLDS):
        for col_idx, city in enumerate(CITIES):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('white')

            print(f"  澶勭悊 {city} @ 胃={threshold}掳C...")

            # 鍔犺浇鏁版嵁
            roads = load_roads_data(city, scenario)
            boundary = load_city_boundary(city)

            if roads is None:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='#999999')
                ax.axis('off')
                continue

            # 鍒嗙鍑夌埥鍜岄潪鍑夌埥閬撹矾
            valid_roads = roads[roads['hei_improved'].notna()].copy()
            cool_mask = valid_roads['hei_improved'] < threshold
            cool_roads = valid_roads[cool_mask]
            hot_roads = valid_roads[~cool_mask]

            # 璁＄畻缁熻
            n_total, n_cool, pct_cool, total_km, cool_km = calculate_cool_stats(roads, threshold)

            # ===== 缁樺埗椤哄簭: 搴曞眰鍒伴《灞?=====

            # 1. 缁樺埗闈炲噳鐖介亾璺?(搴曞眰锛屾瀬娴呯伆锛屽嚑涔庨殣褰?
            if len(hot_roads) > 0:
                hot_roads.plot(ax=ax, color=HOT_COLOR, linewidth=HOT_LINEWIDTH,
                              alpha=0.6, zorder=1)

            # 2. 缁樺埗鍑夌埥閬撹矾 (涓婂眰锛岀數鍏夎摑锛屽閫氱數鑸寒璧?
            if len(cool_roads) > 0:
                cool_roads.plot(ax=ax, color=COOL_COLOR, linewidth=COOL_LINEWIDTH,
                              alpha=0.95, zorder=2)

            # 3. 缁樺埗鍩庡競杈圭晫 (鏈€涓婂眰锛岀粏鐏扮嚎)
            if boundary is not None:
                boundary.boundary.plot(ax=ax, color=BOUNDARY_COLOR,
                                       linewidth=BOUNDARY_LINEWIDTH, zorder=3)

            # ===== 鏍囬鍜屾爣绛?=====

            # 鍩庡競鏍囬 (浠呯涓€琛? - 浣跨敤 fig.text 纭繚姘村钩瀵归綈
            # 鏍囬灏嗗湪寰幆澶栫粺涓€娣诲姞

            # 闃堝€兼爣绛?(宸︿晶绗竴鍒?
            if col_idx == 0:
                # 鏇存樉鐪肩殑闃堝€兼爣绛?- 澧炲ぇ瀛椾綋
                threshold_text = f'胃 = {threshold}掳C'
                ax.text(-0.15, 0.5, threshold_text,
                       transform=ax.transAxes, fontsize=14, fontweight='bold',
                       rotation=90, va='center', ha='center', color='#222222')

            # ===== 鐜颁唬鍖栫粺璁℃枃鏈 (浼樺寲鍚? =====
            # 瀛椾綋灞傜骇: 鐧惧垎姣斿姞绮楀ぇ瀛楋紝闀垮害淇℃伅涓瓑鐏拌壊
            stats_text = f'{pct_cool:.1f}%'
            detail_text = f'{cool_km:.0f}/{total_km:.0f} km'

            ax.text(0.03, 0.97, stats_text,
                   transform=ax.transAxes, fontsize=13, fontweight='bold',
                   va='top', ha='left', color='#222222',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            alpha=0.9, edgecolor='none'))
            ax.text(0.03, 0.83, detail_text,
                   transform=ax.transAxes, fontsize=9, fontweight='normal',
                   va='top', ha='left', color='#555555')

            # ===== 鍧愭爣杞磋缃?=====
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

            # 鍘绘帀澶栬竟妗?
            for spine in ax.spines.values():
                spine.set_visible(False)

            # ===== 姣斾緥灏?(姣忚绗竴涓煄甯? =====
            if col_idx == 0:
                # London 鐢?10km锛屽叾浠栧煄甯傜敤 5km
                scale_km = 10 if city == 'London' else 5
                add_scale_bar(ax, length_km=scale_km, location='lower right')

    # ===== 娣诲姞鍩庡競鏍囬 (浣跨敤 fig.text 纭繚姘村钩瀵归綈) =====
    # 璁＄畻姣忓垪鐨勪腑蹇冧綅缃?
    city_title_y = 0.93  # 缁熶竴鐨?y 浣嶇疆
    col_positions = [0.13, 0.31, 0.50, 0.69, 0.88]  # 5鍒楃殑 x 浣嶇疆
    for col_idx, city in enumerate(CITIES):
        fig.text(col_positions[col_idx], city_title_y, city,
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # ===== 鐜颁唬鍖栧浘渚?(Nature Style) =====
    legend_elements = [
        Line2D([0], [0], color=COOL_COLOR, linewidth=4, label='Cool roads (HEI < 胃)'),
        Line2D([0], [0], color='#BBBBBB', linewidth=2, label='Other roads'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=10, frameon=False, handlelength=2.5,
               bbox_to_anchor=(0.5, 0.01))

    # 璋冩暣甯冨眬 (澧炲姞鍛煎惛鎰?
    plt.tight_layout(rect=[0.04, 0.04, 1, 0.95])
    plt.subplots_adjust(wspace=0.08, hspace=0.15)

    # 淇濆瓨鍥捐〃
    output_name = f'FigS_cool_network_{scenario}'
    fig.savefig(FIGURES_DIR / f'{output_name}.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES_DIR / f'{output_name}.pdf',
                bbox_inches='tight', facecolor='white')

    print(f"\n鍥捐〃宸蹭繚瀛樿嚦: {FIGURES_DIR}")
    print(f"  - {output_name}.png")
    print(f"  - {output_name}.pdf")

    plt.close(fig)

    return fig


def main():
    """涓诲嚱鏁?""
    print("=" * 60)
    print("缁戝埗鍑夌埥缃戠粶鍒嗗竷鍥?(Supplementary Figure)")
    print("=" * 60)

    # 缁戝埗鐑氮鏃ュ浘
    plot_cool_network_figure(scenario='heatwave')

    # 缁戝埗鍏稿瀷鏃ュ浘
    plot_cool_network_figure(scenario='typical_day')

    print("\n" + "=" * 60)
    print("瀹屾垚!")
    print("=" * 60)


if __name__ == '__main__':
    main()


