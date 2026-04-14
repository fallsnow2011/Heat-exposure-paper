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

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============ 鍏ㄥ眬鏍峰紡璁剧疆 ============
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ============ 閰嶈壊鏂规 ============
COLOR_S1 = '#a6bddb'
COLOR_S2 = '#3690c0'
COLOR_S3 = '#016c59'
COLOR_TYPICAL = '#3498DB'
COLOR_HEATWAVE = '#E74C3C'

SCENARIOS_CONFIG = {
    'S1_citywide': {'label': 'S1: Citywide', 'color': COLOR_S1, 'marker': 's'},
    'S2_corridors': {'label': 'S2: Corridors', 'color': COLOR_S2, 'marker': '^'},
    'S3_equity_first': {'label': 'S3: Equity First', 'color': COLOR_S3, 'marker': 'D'},
}

SCENARIO_ORDER = ['S1_citywide', 'S2_corridors', 'S3_equity_first']


def load_data():
    """鍔犺浇鏁版嵁"""
    summary_df = pd.read_csv(SUPPLEMENT_DIR / 'policy_scenarios_summary_v2.csv')
    decile_df = pd.read_csv(SUPPLEMENT_DIR / 'policy_scenarios_by_decile_v2.csv')
    return summary_df, decile_df


def plot_panel_a(ax, summary_df):
    """Panel (a): 鏁堢巼鏁ｇ偣鍥?- 涓や釜鍦烘櫙"""

    for time_scenario, time_color, marker_fill in [('typical_day', COLOR_TYPICAL, 'none'),
                                                     ('heatwave', COLOR_HEATWAVE, 'full')]:
        df = summary_df[summary_df['time_scenario'] == time_scenario]

        for scenario_key in SCENARIO_ORDER:
            config = SCENARIOS_CONFIG[scenario_key]
            row = df[df['policy_scenario'] == scenario_key]

            if len(row) == 0:
                continue

            coverage = row['target_pop_pct'].values[0]
            reduction = row['gap_reduction_pop_pct'].values[0]

            if marker_fill == 'none':
                ax.scatter(coverage, reduction, s=120, facecolors='none',
                          edgecolors=config['color'], linewidths=2,
                          marker=config['marker'], zorder=5)
            else:
                ax.scatter(coverage, reduction, s=120, c=config['color'],
                          edgecolors='white', linewidths=1,
                          marker=config['marker'], zorder=5)

    # 1:1鍙傝€冪嚎
    ax.plot([0, 100], [0, 100], '--', color='gray', alpha=0.5, linewidth=1)

    # 鍥句緥
    scenario_handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_S1,
               markersize=8, label='S1: Citywide'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLOR_S2,
               markersize=8, label='S2: Corridors'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLOR_S3,
               markersize=8, label='S3: Equity First'),
    ]
    time_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='gray', markersize=8, markeredgewidth=2, label='Typical Day'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label='Heatwave'),
    ]

    leg1 = ax.legend(handles=scenario_handles, loc='upper left', fontsize=7,
                     title='Scenario', title_fontsize=7, framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=time_handles, loc='lower right', fontsize=7,
              title='Time', title_fontsize=7, framealpha=0.95)

    ax.set_xlabel('Population Coverage (%)', fontsize=9)
    ax.set_ylabel('Gap Reduction (%)', fontsize=9)
    ax.set_title('(a) Scenario Efficiency: Coverage vs Impact', fontsize=10,
                 fontweight='bold', loc='left')
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 200)
    ax.grid(True, linestyle='--', alpha=0.3)


def plot_panel_b(ax, summary_df):
    """Panel (b): 鎴愭湰鏁堢泭鍒嗘瀽 - 姣?瑕嗙洊甯︽潵鐨凣ap鍑忓皯"""

    scenarios = SCENARIO_ORDER
    x = np.arange(len(scenarios))
    width = 0.35

    efficiency_typical = []
    efficiency_heatwave = []

    for scenario_key in scenarios:
        typical_row = summary_df[(summary_df['time_scenario'] == 'typical_day') &
                                  (summary_df['policy_scenario'] == scenario_key)]
        heatwave_row = summary_df[(summary_df['time_scenario'] == 'heatwave') &
                                   (summary_df['policy_scenario'] == scenario_key)]

        # 鏁堢巼 = Gap鍑忓皯% / 瑕嗙洊%
        cov_t = typical_row['target_pop_pct'].values[0]
        red_t = typical_row['gap_reduction_pop_pct'].values[0]
        cov_h = heatwave_row['target_pop_pct'].values[0]
        red_h = heatwave_row['gap_reduction_pop_pct'].values[0]

        efficiency_typical.append(red_t / cov_t if cov_t > 0 else 0)
        efficiency_heatwave.append(red_h / cov_h if cov_h > 0 else 0)

    bars1 = ax.bar(x - width/2, efficiency_typical, width, label='Typical Day',
                   color=COLOR_TYPICAL, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, efficiency_heatwave, width, label='Heatwave',
                   color=COLOR_HEATWAVE, edgecolor='white', linewidth=0.5)

    # 鏁板€兼爣绛?
    for i, (v1, v2) in enumerate(zip(efficiency_typical, efficiency_heatwave)):
        ax.text(i - width/2, v1 + 0.3, f'{v1:.1f}', ha='center', va='bottom',
               fontsize=8, fontweight='bold')
        ax.text(i + width/2, v2 + 0.3, f'{v2:.1f}', ha='center', va='bottom',
               fontsize=8, fontweight='bold')

    labels = [SCENARIOS_CONFIG[s]['label'] for s in scenarios]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Efficiency Ratio\n(Gap Reduction % / Coverage %)', fontsize=8)
    ax.set_title('(b) Cost-Effectiveness by Scenario', fontsize=10, fontweight='bold', loc='left')
    ax.legend(loc='upper left', fontsize=7, framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 娣诲姞娉ㄩ噴
    ax.text(2, max(efficiency_heatwave) * 0.7, 'Higher = More\ncost-effective',
           fontsize=7, ha='center', style='italic', color='#666666',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))


def plot_panel_c(ax, decile_df, summary_df):
    """Panel (c): 鍚凞ecile鐨凥EI鍙樺寲閲?(鐑氮鏃?"""

    df_hw = decile_df[decile_df['time_scenario'] == 'heatwave'].copy()
    baseline_df = df_hw[df_hw['policy_scenario'] == 'baseline'].set_index('IMD_Decile')

    deciles = list(range(1, 11))
    x = np.arange(len(deciles))
    width = 0.25

    for i, scenario_key in enumerate(SCENARIO_ORDER):
        config = SCENARIOS_CONFIG[scenario_key]
        scenario_df = df_hw[df_hw['policy_scenario'] == scenario_key].set_index('IMD_Decile')

        changes = []
        for d in deciles:
            baseline_hei = baseline_df.loc[d, 'hei_mean_pop'] if d in baseline_df.index else 0
            scenario_hei = scenario_df.loc[d, 'hei_mean_pop'] if d in scenario_df.index else 0
            changes.append(scenario_hei - baseline_hei)

        ax.bar(x + (i - 1) * width, changes, width, label=config['label'],
               color=config['color'], edgecolor='white', linewidth=0.3)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in deciles], fontsize=8)
    ax.set_xlabel('IMD Decile (1=Most Deprived)', fontsize=9)
    ax.set_ylabel('螖HEI from Baseline (掳C)', fontsize=9)
    ax.set_title('(c) HEI Reduction by Decile (Heatwave)', fontsize=10, fontweight='bold', loc='left')
    ax.legend(loc='lower right', fontsize=7, framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # 鏍囨敞璐洶鍖?
    ax.axvspan(-0.5, 2.5, alpha=0.1, color=COLOR_HEATWAVE, zorder=0)
    ax.text(1, ax.get_ylim()[0] * 0.9, 'Most\nDeprived', fontsize=7, ha='center',
           color=COLOR_HEATWAVE, style='italic')


def plot_panel_d(ax, summary_df):
    """Panel (d): 缁煎悎鏁堟灉鎬荤粨琛?""

    # 鍒涘缓琛ㄦ牸鏁版嵁
    data = []
    for scenario_key in SCENARIO_ORDER:
        config = SCENARIOS_CONFIG[scenario_key]
        typical_row = summary_df[(summary_df['time_scenario'] == 'typical_day') &
                                  (summary_df['policy_scenario'] == scenario_key)]
        heatwave_row = summary_df[(summary_df['time_scenario'] == 'heatwave') &
                                   (summary_df['policy_scenario'] == scenario_key)]

        data.append([
            config['label'],
            f"{typical_row['target_pop_pct'].values[0]:.1f}%",
            f"{typical_row['gap_pop'].values[0]:.2f}掳C",
            f"{typical_row['gap_reduction_pop_pct'].values[0]:.0f}%",
            f"{heatwave_row['gap_pop'].values[0]:.2f}掳C",
            f"{heatwave_row['gap_reduction_pop_pct'].values[0]:.0f}%",
        ])

    columns = ['Scenario', 'Coverage', 'Gap\n(Typical)', 'Reduction\n(Typical)',
               'Gap\n(Heatwave)', 'Reduction\n(Heatwave)']

    ax.axis('off')

    table = ax.table(cellText=data, colLabels=columns,
                     loc='center', cellLoc='center',
                     colColours=['#f0f0f0'] * len(columns))

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.8)

    # 楂樹寒S3琛?
    for i in range(len(columns)):
        table[(3, i)].set_facecolor('#e8f5e9')

    ax.set_title('(d) Summary Table: Policy Scenarios Performance', fontsize=10,
                 fontweight='bold', loc='left', y=0.95)


def plot_figure():
    """缁樺埗瀹屾暣鍥捐〃"""

    print("鍔犺浇鏁版嵁...")
    summary_df, decile_df = load_data()

    print("鍒涘缓鍥捐〃...")
    fig = plt.figure(figsize=(14, 10))

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                          hspace=0.32, wspace=0.25,
                          left=0.08, right=0.95, top=0.90, bottom=0.08)

    # Panel (a)
    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_a(ax_a, summary_df)

    # Panel (b)
    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel_b(ax_b, summary_df)

    # Panel (c)
    ax_c = fig.add_subplot(gs[1, 0])
    plot_panel_c(ax_c, decile_df, summary_df)

    # Panel (d)
    ax_d = fig.add_subplot(gs[1, 1])
    plot_panel_d(ax_d, summary_df)

    # 鎬绘爣棰?
    fig.suptitle('Supplementary Figure: Policy Scenarios Efficiency Analysis',
                 fontsize=12, fontweight='bold', y=0.96)

    # 淇濆瓨
    output_name = 'FigS_policy_scenarios_efficiency_analysis'
    fig.savefig(FIGURES_DIR / f'{output_name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES_DIR / f'{output_name}.pdf', bbox_inches='tight', facecolor='white')

    print(f"\n鍥捐〃宸蹭繚瀛樿嚦: {FIGURES_DIR}")
    print(f"  - {output_name}.png")
    print(f"  - {output_name}.pdf")

    plt.close(fig)


def main():
    print("=" * 60)
    print("缁戝埗鏀跨瓥鎯呮櫙鏁堢巼鍒嗘瀽鍥?)
    print("=" * 60)

    plot_figure()

    print("\n" + "=" * 60)
    print("瀹屾垚!")
    print("=" * 60)


if __name__ == '__main__':
    main()


