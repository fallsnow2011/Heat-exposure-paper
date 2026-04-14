#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

# ============ 璺緞璁剧疆 ============
BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / 'results' / 'heat_exposure'
FIGURES_DIR = BASE_DIR / 'paper' / '05_figures'
BOUNDARY_DIR = BASE_DIR / 'city_boundaries'

# 纭繚杈撳嚭鐩綍瀛樺湪
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============ 鍏ㄥ眬鏍峰紡璁剧疆 ============
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.labelweight'] = 'normal'

# ============ 閰嶈壊鏂规 (tab20 鏍囧噯鑹叉澘) ============
# 鍩庡競棰滆壊 - 娣辫壊鐗堟湰 (鐢ㄤ簬 Heatwave / 涓昏鏍囪瘑)
city_colors_dark = {
    'London': '#1f77b4',      # Blue
    'Birmingham': '#17becf',  # Cyan/Teal
    'Manchester': '#ff7f0e',  # Orange
    'Bristol': '#9467bd',     # Purple
    'Newcastle': '#d62728'    # Red
}

# 鍩庡競棰滆壊 - 娴呰壊鐗堟湰 (鐢ㄤ簬 Typical Day)
city_colors_light = {
    'London': '#aec7e8',      # Light Blue
    'Birmingham': '#9edae5',  # Light Cyan
    'Manchester': '#ffbb78',  # Light Orange
    'Bristol': '#c5b0d5',     # Light Purple
    'Newcastle': '#ff9896'    # Light Red
}

# 鍩庡競椤哄簭 (涓?Figure 1 panel a 涓€鑷?
cities = ['London', 'Birmingham', 'Manchester', 'Bristol', 'Newcastle']


def load_data():
    """鍔犺浇鎵€鏈夐渶瑕佺殑鏁版嵁"""
    # 鍔犺浇 TCNI 姹囨€绘暟鎹?
    summary_df = pd.read_csv(RESULTS_DIR / 'hei_cni_tcni_summary_improved.csv')

    # 鍔犺浇 CNI 鏇茬嚎鏁版嵁 (浣跨敤鏀硅繘鍙ｅ緞 cni_curves锛岃鍙?cni_hei 鍒?
    cni_curves = {}
    for city in cities:
        typical_df = pd.read_csv(RESULTS_DIR / f'{city}_cni_curves_typical_day.csv')
        heatwave_df = pd.read_csv(RESULTS_DIR / f'{city}_cni_curves_heatwave.csv')
        # 閲嶅懡鍚嶅垪浠ュ吋瀹瑰悗缁唬鐮?
        typical_df['cni'] = typical_df['cni_hei']
        heatwave_df['cni'] = heatwave_df['cni_hei']
        cni_curves[city] = {
            'typical': typical_df,
            'heatwave': heatwave_df
        }

    # 鍔犺浇鍩庡競杈圭晫 (geojson 鏍煎紡)
    city_boundaries = {}
    for city in cities:
        boundary_file = BOUNDARY_DIR / f'{city}_boundary.geojson'
        if boundary_file.exists():
            city_boundaries[city] = gpd.read_file(boundary_file)

    return summary_df, cni_curves, city_boundaries


def prepare_tcni_data(summary_df):
    """鍑嗗 TCNI 缁戝浘鏁版嵁"""
    tcni_data = []

    for city in cities:
        typical = summary_df[(summary_df['city'] == city) & (summary_df['scenario'] == 'typical_day')]
        heatwave = summary_df[(summary_df['city'] == city) & (summary_df['scenario'] == 'heatwave')]

        tcni_typical = typical['tcni'].values[0]
        tcni_heatwave = heatwave['tcni'].values[0]
        change_pct = (tcni_heatwave - tcni_typical) / tcni_typical * 100

        tcni_data.append({
            'city': city,
            'tcni_typical': tcni_typical,
            'tcni_heatwave': tcni_heatwave,
            'change_pct': change_pct,
            'n_roads': typical['n_roads'].values[0]
        })

    return pd.DataFrame(tcni_data)


def plot_figure1(summary_df, cni_curves, city_boundaries):
    """缁戝埗瀹屾暣鐨?Figure 1"""

    # 鍑嗗鏁版嵁
    tcni_df = prepare_tcni_data(summary_df)

    # 鍒涘缓鍥捐〃 - 瀹藉箙甯冨眬
    fig = plt.figure(figsize=(16, 6))

    # 浣跨敤 GridSpec 绮剧‘鎺у埗甯冨眬 - Panel (a) 鍒嗕负涓ら儴鍒嗭細杞粨 + 鍩庡競鍚嶇О
    # 浣跨敤宓屽 GridSpec锛氬乏渚?(a+b)銆佸彸渚?(c)锛宎鍜宐涔嬮棿闂磋窛灏忥紝b鍜宑涔嬮棿闂磋窛澶?
    gs_main = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.5], wspace=0.15,
                               left=0.03, right=0.98, top=0.90, bottom=0.12)
    # 宸︿晶鍐嶅垎涓?Panel (a) 鍜?Panel (b)
    gs_left = gs_main[0].subgridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.02)
    # Panel (a) 鍐呴儴鍒嗕负杞粨鍜屾爣绛句袱閮ㄥ垎
    gs_a = gs_left[0].subgridspec(1, 2, width_ratios=[0.55, 0.45], wspace=0.02)

    # ============ Panel (a): Study Area - 鍩庡競杞粨鍥?============
    ax_a_shapes = fig.add_subplot(gs_a[0])  # 鍩庡競杞粨
    ax_a_labels = fig.add_subplot(gs_a[1])  # 鍩庡競鍚嶇О

    # Y 杞翠綅缃細涓?Panel (b) 鐨?y=0,1,2,3,4 瀵瑰簲锛堜絾 Panel (b) 浼?invert锛?
    # 鎵€浠ヨ繖閲?y=0 瀵瑰簲绗竴涓煄甯?London锛寉=4 瀵瑰簲鏈€鍚庝竴涓煄甯?Newcastle
    y_positions = {
        'London': 0,
        'Birmingham': 1,
        'Manchester': 2,
        'Bristol': 3,
        'Newcastle': 4
    }

    # 缁樺埗姣忎釜鍩庡競鐨勮疆寤?
    for city in cities:
        if city in city_boundaries:
            gdf = city_boundaries[city]
            # 鎶曞奖鍒?British National Grid (EPSG:27700) 閬垮厤鍙樺舰
            gdf_projected = gdf.to_crs(epsg=27700)

            # 鑾峰彇鍩庡競杈圭晫鐨勮川蹇冨拰鑼冨洿
            bounds = gdf_projected.total_bounds  # [minx, miny, maxx, maxy]
            centroid = gdf_projected.geometry.centroid.iloc[0]

            # 璁＄畻缂╂斁鍥犲瓙浣挎墍鏈夊煄甯傚ぇ灏忕浉浼?
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            max_dim = max(width, height)
            scale = 0.9 / max_dim  # 褰掍竴鍖栧埌绾?.9鍗曚綅锛堟洿澶э級

            # 骞崇Щ鍜岀缉鏀惧嚑浣曚綋
            y_pos = y_positions[city]
            translated = gdf_projected.geometry.translate(xoff=-centroid.x, yoff=-centroid.y)
            scaled = translated.scale(xfact=scale, yfact=scale, origin=(0, 0))
            final = scaled.translate(xoff=0.5, yoff=y_pos)

            # 缁樺埗杞粨
            temp_gdf = gpd.GeoDataFrame(geometry=final)
            temp_gdf.plot(ax=ax_a_shapes, color=city_colors_dark[city], edgecolor='white',
                         linewidth=0.5, alpha=0.9)

    # 璁剧疆杞粨鍖哄煙
    ax_a_shapes.set_xlim(-0.1, 1.1)
    ax_a_shapes.set_ylim(-0.5, 4.5)
    ax_a_shapes.invert_yaxis()  # 涓?Panel (b) 涓€鑷?
    ax_a_shapes.axis('off')
    ax_a_shapes.set_title('(a) Study Area', fontweight='bold', fontsize=11, loc='left')

    # 娣诲姞鍩庡競鍚嶇О鍜岄亾璺暟閲忥紙鍦ㄥ崟鐙殑杞翠笂锛?
    for city in cities:
        y_pos = y_positions[city]
        n_roads = tcni_df[tcni_df['city'] == city]['n_roads'].values[0]

        # 鍩庡競鍚嶇О - 涓庤疆寤撲腑蹇冨榻愶紝绋嶅井鍋忎笂
        ax_a_labels.text(0.0, y_pos - 0.15, city, fontsize=10, fontweight='bold',
                        color=city_colors_dark[city], va='center', ha='left')
        # 閬撹矾鏁伴噺 - 鍦ㄥ悕绉颁笅鏂?
        ax_a_labels.text(0.0, y_pos + 0.15, f'({n_roads//1000}k)', fontsize=8,
                        color='gray', va='center', ha='left')

    ax_a_labels.set_xlim(0, 1)
    ax_a_labels.set_ylim(-0.5, 4.5)
    ax_a_labels.invert_yaxis()  # 涓?Panel (b) 涓€鑷?
    ax_a_labels.axis('off')

    # ============ Panel (b): TCNI Collapse - 鐐瑰浘 ============
    ax_b = fig.add_subplot(gs_left[1])

    y_pos = np.arange(len(cities))

    for i, city in enumerate(cities):
        row = tcni_df[tcni_df['city'] == city].iloc[0]

        # 鍏稿瀷鏃?- 绌哄績鍦?
        ax_b.scatter(row['tcni_typical'], i, s=200, facecolors='white',
                    edgecolors=city_colors_dark[city], linewidths=2.5, zorder=5)

        # 鐑氮鏃?- 瀹炲績鍦?
        ax_b.scatter(row['tcni_heatwave'], i, s=200, c=city_colors_dark[city],
                    edgecolors='white', linewidths=1, zorder=5)

        # 杩炴帴绾?
        ax_b.plot([row['tcni_heatwave'], row['tcni_typical']], [i, i],
                 color='gray', linewidth=1.5, linestyle='-', alpha=0.5, zorder=1)

        # 鍙樺寲鐧惧垎姣旀爣娉?- 浣跨敤涓庡煄甯傚搴旂殑棰滆壊
        ax_b.text(row['tcni_heatwave'] - 0.5, i, f'{row["change_pct"]:.0f}%',
                 fontsize=9, color=city_colors_dark[city], fontweight='bold', va='center', ha='right')

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([])  # 闅愯棌Y杞存爣绛撅紝涓嶱anel (a)妯悜瀵归綈

    ax_b.set_xlabel('TCNI (Temperature-integrated Cool Network Index)', fontsize=9)
    ax_b.set_title('(b) TCNI Collapse Under Heatwave', fontweight='bold', fontsize=11, loc='left')
    ax_b.set_xlim(-1, 11)
    ax_b.set_ylim(-0.5, 4.5)  # 涓?Panel (a) 涓€鑷?
    ax_b.invert_yaxis()
    ax_b.grid(axis='x', linestyle='--', alpha=0.3)

    # 绉婚櫎宸﹀彸涓婅竟妗嗙嚎锛屽彧淇濈暀涓嬭竟妗?
    ax_b.spines['left'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.spines['top'].set_visible(False)
    ax_b.tick_params(left=False)  # 绉婚櫎宸︿晶鍒诲害绾?

    # 鍥句緥
    legend_elements = [
        plt.scatter([], [], s=120, facecolors='white', edgecolors='gray',
                   linewidths=2, label='Typical Day'),
        plt.scatter([], [], s=120, c='gray', edgecolors='white',
                   linewidths=1, label='Heatwave')
    ]
    ax_b.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

    # ============ Panel (c): Percolation Curves ============
    ax_c = fig.add_subplot(gs_main[1])

    # 娣诲姞娓╁害鍙傝€冨尯鍩?
    ax_c.axvspan(26, 30, alpha=0.15, color='#3498DB', zorder=0)  # HEI浣庡€煎尯闂?    ax_c.axvspan(33, 37, alpha=0.15, color='#E74C3C', zorder=0)  # HEI楂樺€煎尯闂?
    # 娣诲姞娓╁害鏍囩
    ax_c.text(28, 1.02, 'HEI 28掳C', fontsize=8, ha='center', color='#3498DB', fontweight='bold')
    ax_c.text(35, 1.02, 'HEI 35掳C', fontsize=8, ha='center', color='#E74C3C', fontweight='bold')

    # 缁樺埗鎵€鏈夊煄甯傜殑鏇茬嚎
    for city in cities:
        typical = cni_curves[city]['typical']
        heatwave = cni_curves[city]['heatwave']

        # 鍏稿瀷鏃?- 鐏拌壊缁嗙嚎
        ax_c.plot(typical['threshold'], typical['cni'],
                 '-', color='gray', linewidth=1, alpha=0.4, zorder=1)

        # 鐑氮鏃?- 鍩庡競棰滆壊绮楃嚎
        ax_c.plot(heatwave['threshold'], heatwave['cni'],
                 '-', color=city_colors_dark[city], linewidth=2.5,
                 label=city, zorder=2)

    # 鍦ㄦ洸绾挎湯绔坊鍔犲煄甯傛爣绛?
    for city in cities:
        heatwave = cni_curves[city]['heatwave']
        last_idx = len(heatwave) - 1
        last_threshold = heatwave['threshold'].iloc[last_idx]
        last_cni = heatwave['cni'].iloc[last_idx]

        ax_c.text(last_threshold + 0.5, last_cni, city, fontsize=9,
                 color=city_colors_dark[city], fontweight='bold', va='center')

    ax_c.set_xlabel('Temperature Threshold 胃 (掳C)', fontsize=9)
    ax_c.set_ylabel('CNI (Cool Network Index)', fontsize=9)
    ax_c.set_title('(c) Percolation Curves', fontweight='bold', fontsize=11, loc='left')
    ax_c.set_xlim(20, 45)
    ax_c.set_ylim(0, 1.08)
    ax_c.grid(True, linestyle='--', alpha=0.3)

    # 娣诲姞 "Gray: Typical Day" 娉ㄩ噴
    ax_c.text(22, 0.05, 'Gray: Typical Day', fontsize=8, color='gray', style='italic')

    # 淇濆瓨鍥捐〃
    fig.savefig(FIGURES_DIR / 'Fig1_thermal_connectivity_collapse.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES_DIR / 'Fig1_thermal_connectivity_collapse.pdf',
                bbox_inches='tight', facecolor='white')

    print(f"Figure 1 宸蹭繚瀛樿嚦: {FIGURES_DIR}")
    print("  - Fig1_thermal_connectivity_collapse.png")
    print("  - Fig1_thermal_connectivity_collapse.pdf")

    # plt.show()  # Disabled for non-interactive execution

    return fig


if __name__ == '__main__':
    print("姝ｅ湪鍔犺浇鏁版嵁...")
    summary_df, cni_curves, city_boundaries = load_data()

    print("姝ｅ湪缁戝埗 Figure 1...")
    fig = plot_figure1(summary_df, cni_curves, city_boundaries)

    print("\n瀹屾垚!")


