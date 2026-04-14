#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from pathlib import Path

# ============ 璺緞璁剧疆 ============
BASE_DIR = Path(__file__).resolve().parents[2]
SUPPLEMENT_DIR = BASE_DIR / 'paper' / '06_supplement'
FIGURES_DIR = SUPPLEMENT_DIR

# 纭繚杈撳嚭鐩綍瀛樺湪
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============ 鍏ㄥ眬鏍峰紡璁剧疆 (Nature Style) ============
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ============ 閰嶈壊鏂规 (涓嶧ig5涓€鑷? ============
# 鎯呮櫙棰滆壊 - 鏁堢巼閫掕繘
COLOR_BASELINE = '#d0d0d0'     # 娴呯伆 - Baseline
COLOR_S1 = '#a6bddb'           # 娴呰摑鐏?- S1 Citywide (鏁堢巼鏈€浣?
COLOR_S2 = '#3690c0'           # 涓摑 - S2 Corridors
COLOR_S3 = '#016c59'           # 娣遍潚缁?- S3 Equity First (鏁堢巼鏈€楂?

# 鎯呮櫙閰嶇疆 (CSV鍚嶇О -> 鏄剧ず鏍囩)
# 娉ㄦ剰: Fig5涓殑鍛藉悕鏄犲皠鍏崇郴
# Note: `policy_scenarios_summary_v2.csv` now uses canonical IDs:
# baseline / S1_citywide / S2_corridors / S3_equity_first
SCENARIOS_CONFIG = {
    'baseline': {'label': 'Baseline', 'color': COLOR_BASELINE, 'marker': 'o'},
    'S1_citywide': {'label': 'S1: Citywide (+10%)', 'color': COLOR_S1, 'marker': 's'},
    'S2_corridors': {'label': 'S2: Corridors', 'color': COLOR_S2, 'marker': '^'},
    'S3_equity_first': {'label': 'S3: Equity First', 'color': COLOR_S3, 'marker': 'D'},
}

# 鎯呮櫙椤哄簭 (鎸夋晥鐜囬€掑)
SCENARIO_ORDER = ['baseline', 'S1_citywide', 'S2_corridors', 'S3_equity_first']


def load_data():
    """鍔犺浇鏀跨瓥鎯呮櫙鏁版嵁"""
    summary_df = pd.read_csv(SUPPLEMENT_DIR / 'policy_scenarios_summary_v2.csv')
    decile_df = pd.read_csv(SUPPLEMENT_DIR / 'policy_scenarios_by_decile_v2.csv')
    return summary_df, decile_df


def plot_panel_a(ax, decile_df, time_scenario='heatwave'):
    """Panel (a): 鍚勬儏鏅寜IMD Decile鐨凥EI鍧囧€?(鎶樼嚎鍥?"""

    df = decile_df[decile_df['time_scenario'] == time_scenario].copy()
    deciles = list(range(1, 11))

    for scenario_key in SCENARIO_ORDER:
        config = SCENARIOS_CONFIG[scenario_key]
        scenario_data = df[df['policy_scenario'] == scenario_key]

        if len(scenario_data) == 0:
            continue

        # 鎸塂ecile鎺掑簭
        scenario_data = scenario_data.sort_values('IMD_Decile')

        ax.plot(scenario_data['IMD_Decile'], scenario_data['hei_mean_pop'],
                marker=config['marker'], markersize=6, linewidth=2,
                color=config['color'], label=config['label'],
                markeredgecolor='white', markeredgewidth=0.5)

    ax.set_xlabel('IMD Decile (1=Most Deprived 鈫?10=Least Deprived)', fontsize=9)
    ax.set_ylabel('HEI (掳C, population-weighted)', fontsize=9)
    ax.set_xticks(deciles)
    ax.set_xlim(0.5, 10.5)

    # 娣诲姞鏍囬
    scenario_label = 'Heatwave' if time_scenario == 'heatwave' else 'Typical Day'
    ax.set_title(f'(a) HEI by IMD Decile ({scenario_label})', fontsize=10, fontweight='bold', loc='left')

    ax.legend(loc='upper right', fontsize=7, framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.3)


def plot_panel_b(ax, summary_df):
    """Panel (b): 璐瘜HEI宸窛鍙樺寲 (鍒嗙粍鏌辩姸鍥?"""

    x = np.arange(len(SCENARIO_ORDER))
    width = 0.35

    # 鑾峰彇涓や釜鏃堕棿鍦烘櫙鐨凣ap
    gaps_typical = []
    gaps_heatwave = []

    for scenario_key in SCENARIO_ORDER:
        typical_row = summary_df[(summary_df['time_scenario'] == 'typical_day') &
                                  (summary_df['policy_scenario'] == scenario_key)]
        heatwave_row = summary_df[(summary_df['time_scenario'] == 'heatwave') &
                                   (summary_df['policy_scenario'] == scenario_key)]

        gaps_typical.append(typical_row['gap_pop'].values[0] if len(typical_row) > 0 else 0)
        gaps_heatwave.append(heatwave_row['gap_pop'].values[0] if len(heatwave_row) > 0 else 0)

    # 缁樺埗鏌辩姸鍥?
    bars1 = ax.bar(x - width/2, gaps_typical, width, label='Typical Day',
                   color='#3498DB', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, gaps_heatwave, width, label='Heatwave',
                   color='#E74C3C', edgecolor='white', linewidth=0.5)

    # 闆剁嚎
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # 娣诲姞鏁板€兼爣绛?
    for i, (v1, v2) in enumerate(zip(gaps_typical, gaps_heatwave)):
        y1 = v1 + 0.05 if v1 >= 0 else v1 - 0.12
        y2 = v2 + 0.05 if v2 >= 0 else v2 - 0.12
        va1 = 'bottom' if v1 >= 0 else 'top'
        va2 = 'bottom' if v2 >= 0 else 'top'
        ax.text(i - width/2, y1, f'{v1:.2f}', ha='center', va=va1, fontsize=7, fontweight='bold')
        ax.text(i + width/2, y2, f'{v2:.2f}', ha='center', va=va2, fontsize=7, fontweight='bold')

    # 鏍囩
    labels = [SCENARIOS_CONFIG[s]['label'] for s in SCENARIO_ORDER]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('HEI Gap: Deprived 鈭?Affluent (掳C)', fontsize=9)
    ax.set_title('(b) Inequality Gap by Scenario', fontsize=10, fontweight='bold', loc='left')

    # 娣诲姞"涓嶅钩绛夐€嗚浆"鍖哄煙
    ax.fill_between([-0.5, 3.5], 0, -0.6, color='#e8f5e9', alpha=0.5, zorder=0)
    ax.text(3.3, -0.55, 'Inequality\nreversed', fontsize=7, ha='right', va='bottom',
            color='#2e7d32', style='italic')

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.6, 1.4)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.3)


def plot_panel_c(ax, summary_df, time_scenario='heatwave'):
    """Panel (c): 鎯呮櫙鏁堢巼瀵规瘮 - 瑕嗙洊鐜?vs Gap鍑忓皯"""

    df = summary_df[summary_df['time_scenario'] == time_scenario].copy()

    # 鎺掗櫎baseline (瑕嗙洊鐜囦负0)
    for scenario_key in SCENARIO_ORDER[1:]:  # 璺宠繃baseline
        config = SCENARIOS_CONFIG[scenario_key]
        row = df[df['policy_scenario'] == scenario_key]

        if len(row) == 0:
            continue

        coverage = row['target_pop_pct'].values[0]
        reduction = row['gap_reduction_pop_pct'].values[0]

        ax.scatter(coverage, reduction, s=150, c=config['color'],
                  marker=config['marker'], edgecolors='white', linewidths=1.5,
                  label=config['label'], zorder=5)

        # 鏍囩
        offset_x = 2 if scenario_key != 'S3_equity_first' else -8
        offset_y = 5 if scenario_key != 'S2_corridors' else -10
        ax.annotate(config['label'].split(':')[0], (coverage, reduction),
                   xytext=(offset_x, offset_y), textcoords='offset points',
                   fontsize=8, fontweight='bold', color=config['color'])

    # 娣诲姞鏁堢巼鍙傝€冪嚎 (Gap鍑忓皯/瑕嗙洊鐜?
    ax.plot([0, 100], [0, 100], '--', color='gray', alpha=0.5, linewidth=1,
            label='1:1 efficiency line')

    ax.set_xlabel('Population Coverage (%)', fontsize=9)
    ax.set_ylabel('Gap Reduction (%)', fontsize=9)
    ax.set_title('(c) Scenario Efficiency (Heatwave)', fontsize=10, fontweight='bold', loc='left')
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 160)
    ax.grid(True, linestyle='--', alpha=0.3)

    # 娣诲姞鏁堢巼娉ㄩ噴
    ax.text(50, 140, 'Higher = More efficient\n(more reduction per coverage)',
            fontsize=7, ha='center', va='top', style='italic', color='#666666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))


def plot_panel_d(ax, decile_df, time_scenario='heatwave'):
    """Panel (d): 鍚凞ecile鐨凥EI鍙樺寲閲忕儹鍔涘浘"""

    df = decile_df[decile_df['time_scenario'] == time_scenario].copy()

    # 鑾峰彇baseline鐨凥EI
    baseline_df = df[df['policy_scenario'] == 'baseline'].set_index('IMD_Decile')['hei_mean_pop']

    # 璁＄畻鍚勬儏鏅浉瀵逛簬baseline鐨勫彉鍖?
    scenarios = SCENARIO_ORDER[1:]  # 璺宠繃baseline
    deciles = list(range(1, 11))

    change_matrix = []
    for scenario_key in scenarios:
        scenario_df = df[df['policy_scenario'] == scenario_key].set_index('IMD_Decile')['hei_mean_pop']
        changes = [scenario_df.get(d, 0) - baseline_df.get(d, 0) for d in deciles]
        change_matrix.append(changes)

    change_matrix = np.array(change_matrix)

    # 缁樺埗鐑姏鍥?
    im = ax.imshow(change_matrix, cmap='RdYlGn_r', aspect='auto',
                   vmin=-3.5, vmax=0.5)

    # 娣诲姞鏁板€兼爣绛?
    for i in range(len(scenarios)):
        for j in range(len(deciles)):
            value = change_matrix[i, j]
            color = 'white' if abs(value) > 1.5 else 'black'
            ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                   fontsize=7, color=color, fontweight='bold')

    # 鏍囩
    ax.set_xticks(range(len(deciles)))
    ax.set_xticklabels([f'D{d}' for d in deciles], fontsize=8)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([SCENARIOS_CONFIG[s]['label'] for s in scenarios], fontsize=8)

    ax.set_xlabel('IMD Decile', fontsize=9)
    ax.set_title('(d) HEI Change from Baseline (掳C)', fontsize=10, fontweight='bold', loc='left')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('螖HEI (掳C)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def plot_figure():
    """缁戝埗瀹屾暣鐨勯檮褰曞浘"""

    print("鍔犺浇鏁版嵁...")
    summary_df, decile_df = load_data()

    print("鍒涘缓鍥捐〃...")
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.1, 0.9],
                          hspace=0.28, wspace=0.25,
                          left=0.08, right=0.95, top=0.92, bottom=0.08)

    # Panel (a): HEI by Decile
    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_a(ax_a, decile_df, time_scenario='heatwave')

    # Panel (b): Gap comparison
    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel_b(ax_b, summary_df)

    # Panel (c): Efficiency scatter
    ax_c = fig.add_subplot(gs[1, 0])
    plot_panel_c(ax_c, summary_df, time_scenario='heatwave')

    # Panel (d): Change heatmap
    ax_d = fig.add_subplot(gs[1, 1])
    plot_panel_d(ax_d, decile_df, time_scenario='heatwave')

    # 鎬绘爣棰?
    fig.suptitle('Supplementary Figure: Policy Scenarios Impact on Heat Exposure Distribution',
                 fontsize=12, fontweight='bold', y=0.97)

    # 淇濆瓨
    output_name = 'FigS_policy_scenarios_hei_distribution'
    fig.savefig(FIGURES_DIR / f'{output_name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES_DIR / f'{output_name}.pdf', bbox_inches='tight', facecolor='white')

    print(f"\n鍥捐〃宸蹭繚瀛樿嚦: {FIGURES_DIR}")
    print(f"  - {output_name}.png")
    print(f"  - {output_name}.pdf")

    plt.close(fig)

    return fig


def main():
    """涓诲嚱鏁?""
    print("=" * 60)
    print("缁戝埗鏀跨瓥鎯呮櫙HEI鍒嗗竷闄勫綍鍥?)
    print("=" * 60)

    plot_figure()

    print("\n" + "=" * 60)
    print("瀹屾垚!")
    print("=" * 60)


if __name__ == '__main__':
    main()


