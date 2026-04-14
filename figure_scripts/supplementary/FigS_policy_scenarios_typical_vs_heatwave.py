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
# 鎯呮櫙棰滆壊
COLOR_BASELINE = '#d0d0d0'
COLOR_S1 = '#a6bddb'
COLOR_S2 = '#3690c0'
COLOR_S3 = '#016c59'

# 鏃堕棿鍦烘櫙棰滆壊
COLOR_TYPICAL = '#3498DB'
COLOR_HEATWAVE = '#E74C3C'

SCENARIOS_CONFIG = {
    'baseline': {'label': 'Baseline', 'color': COLOR_BASELINE, 'marker': 'o', 'linestyle': '-'},
    'S1_citywide': {'label': 'S1: Citywide', 'color': COLOR_S1, 'marker': 's', 'linestyle': '--'},
    'S2_corridors': {'label': 'S2: Corridors', 'color': COLOR_S2, 'marker': '^', 'linestyle': '-.'},
    'S3_equity_first': {'label': 'S3: Equity First', 'color': COLOR_S3, 'marker': 'D', 'linestyle': '-'},
}

SCENARIO_ORDER = ['baseline', 'S1_citywide', 'S2_corridors', 'S3_equity_first']


def load_data():
    """鍔犺浇鏁版嵁"""
    # Prefer submission-ready copies bundled in final-SI.
    # Fall back to the working directory if present.
    si_data_dir = BASE_DIR / "paper" / "final-SI" / "data"
    summary_path = si_data_dir / "policy_scenarios_summary_v2.csv"
    decile_path = si_data_dir / "policy_scenarios_by_decile_v2.csv"

    if not summary_path.exists():
        summary_path = SUPPLEMENT_DIR / "policy_scenarios_summary_v2.csv"
    if not decile_path.exists():
        decile_path = SUPPLEMENT_DIR / "policy_scenarios_by_decile_v2.csv"

    summary_df = pd.read_csv(summary_path)
    decile_df = pd.read_csv(decile_path)
    return summary_df, decile_df


def ensure_baseline_rows(decile_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Some exports omit baseline rows in `policy_scenarios_by_decile_v2.csv`.
    For plots requiring 螖HEI relative to baseline, we reconstruct baseline-decile
    means from the SI package LSOA summaries to keep figure logic correct.
    """
    if (decile_df["policy_scenario"] == "baseline").any():
        return decile_df

    base_dir = BASE_DIR / "paper" / "final-SI" / "data"
    typ = pd.read_csv(base_dir / "lsoa_hei_summary_typical_day.csv", usecols=["IMD_Decile", "hei_mean", "TotPop"])
    hw = pd.read_csv(base_dir / "lsoa_hei_summary_heatwave.csv", usecols=["IMD_Decile", "hei_mean", "TotPop"])

    def decile_means(df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for d in range(1, 11):
            sub = df[df["IMD_Decile"] == d]
            if sub.empty:
                continue
            hei_mean_pop = float((sub["hei_mean"] * sub["TotPop"]).sum() / sub["TotPop"].sum())
            hei_mean_lsoa = float(sub["hei_mean"].mean())
            out.append(
                {
                    "IMD_Decile": d,
                    "hei_mean_lsoa": hei_mean_lsoa,
                    "hei_mean_pop": hei_mean_pop,
                }
            )
        return pd.DataFrame(out)

    baseline_typ = decile_means(typ)
    baseline_typ["time_scenario"] = "typical_day"
    baseline_typ["policy_scenario"] = "baseline"
    baseline_typ["scenario_order"] = 0
    baseline_typ["scenario_label"] = "Baseline"

    baseline_hw = decile_means(hw)
    baseline_hw["time_scenario"] = "heatwave"
    baseline_hw["policy_scenario"] = "baseline"
    baseline_hw["scenario_order"] = 0
    baseline_hw["scenario_label"] = "Baseline"

    # Keep required columns consistent with the existing CSV schema.
    baseline = pd.concat([baseline_typ, baseline_hw], ignore_index=True)
    baseline["n_lsoa"] = np.nan
    baseline["total_pop"] = np.nan

    # Align column order with `policy_scenarios_by_decile_v2.csv` if possible.
    cols = list(decile_df.columns)
    for c in baseline.columns:
        if c not in cols:
            cols.append(c)
    baseline = baseline[cols]

    return pd.concat([baseline, decile_df], ignore_index=True)


def plot_hei_by_decile(ax, decile_df, time_scenario, show_legend=True, panel_label='a'):
    """缁樺埗鍚勬儏鏅寜IMD Decile鐨凥EI鍧囧€?""

    df = decile_df[decile_df['time_scenario'] == time_scenario].copy()
    deciles = list(range(1, 11))

    for scenario_key in SCENARIO_ORDER:
        config = SCENARIOS_CONFIG[scenario_key]
        scenario_data = df[df['policy_scenario'] == scenario_key].sort_values('IMD_Decile')

        if len(scenario_data) == 0:
            continue

        ax.plot(scenario_data['IMD_Decile'], scenario_data['hei_mean_pop'],
                marker=config['marker'], markersize=5, linewidth=1.8,
                color=config['color'], label=config['label'],
                linestyle=config['linestyle'],
                markeredgecolor='white', markeredgewidth=0.3)

    ax.set_xlabel('IMD Decile', fontsize=8)
    ax.set_ylabel('HEI (掳C)', fontsize=8)
    ax.set_xticks(deciles)
    ax.set_xlim(0.5, 10.5)

    scenario_label = 'Heatwave' if time_scenario == 'heatwave' else 'Typical Day'
    title_color = COLOR_HEATWAVE if time_scenario == 'heatwave' else COLOR_TYPICAL
    ax.set_title(f'({panel_label}) HEI by Decile - {scenario_label}',
                 fontsize=9, fontweight='bold', loc='left', color=title_color)

    if show_legend:
        ax.legend(loc='upper right', fontsize=6, framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.3)


def plot_change_heatmap(ax, decile_df, time_scenario, panel_label='b'):
    """缁樺埗鍚凞ecile鐨凥EI鍙樺寲閲忕儹鍔涘浘"""

    df = decile_df[decile_df['time_scenario'] == time_scenario].copy()
    baseline_df = df[df['policy_scenario'] == 'baseline'].set_index('IMD_Decile')['hei_mean_pop']

    scenarios = SCENARIO_ORDER[1:]
    deciles = list(range(1, 11))

    change_matrix = []
    for scenario_key in scenarios:
        scenario_df = df[df['policy_scenario'] == scenario_key].set_index('IMD_Decile')['hei_mean_pop']
        changes = [scenario_df.get(d, 0) - baseline_df.get(d, 0) for d in deciles]
        change_matrix.append(changes)

    change_matrix = np.array(change_matrix)

    # 鏍规嵁鍦烘櫙璋冩暣colorbar鑼冨洿
    if time_scenario == 'heatwave':
        vmin, vmax = -4.5, 0.5
    else:
        vmin, vmax = -3.5, 0.5

    im = ax.imshow(change_matrix, cmap='RdYlGn_r', aspect='auto', vmin=vmin, vmax=vmax)

    for i in range(len(scenarios)):
        for j in range(len(deciles)):
            value = change_matrix[i, j]
            color = 'white' if abs(value) > 1.5 else 'black'
            ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                   fontsize=6, color=color, fontweight='bold')

    ax.set_xticks(range(len(deciles)))
    ax.set_xticklabels([f'D{d}' for d in deciles], fontsize=7)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([SCENARIOS_CONFIG[s]['label'] for s in scenarios], fontsize=7)

    ax.set_xlabel('IMD Decile', fontsize=8)

    scenario_label = 'Heatwave' if time_scenario == 'heatwave' else 'Typical Day'
    title_color = COLOR_HEATWAVE if time_scenario == 'heatwave' else COLOR_TYPICAL
    ax.set_title(f'({panel_label}) 螖HEI from Baseline - {scenario_label}',
                 fontsize=9, fontweight='bold', loc='left', color=title_color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('螖HEI (掳C)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def plot_gap_comparison(ax, summary_df):
    """缁樺埗Gap鍙樺寲瀵规瘮 (鍏稿瀷鏃?vs 鐑氮鏃?"""

    scenarios = SCENARIO_ORDER
    x = np.arange(len(scenarios))
    width = 0.35

    gaps_typical = []
    gaps_heatwave = []
    reductions_typical = []
    reductions_heatwave = []

    for scenario_key in scenarios:
        typical_row = summary_df[(summary_df['time_scenario'] == 'typical_day') &
                                  (summary_df['policy_scenario'] == scenario_key)]
        heatwave_row = summary_df[(summary_df['time_scenario'] == 'heatwave') &
                                   (summary_df['policy_scenario'] == scenario_key)]

        gaps_typical.append(typical_row['gap_pop'].values[0] if len(typical_row) > 0 else 0)
        gaps_heatwave.append(heatwave_row['gap_pop'].values[0] if len(heatwave_row) > 0 else 0)
        reductions_typical.append(typical_row['gap_reduction_pop_pct'].values[0] if len(typical_row) > 0 else 0)
        reductions_heatwave.append(heatwave_row['gap_reduction_pop_pct'].values[0] if len(heatwave_row) > 0 else 0)

    bars1 = ax.bar(x - width/2, gaps_typical, width, label='Typical Day',
                   color=COLOR_TYPICAL, edgecolor='white', linewidth=0.5, alpha=0.85)
    bars2 = ax.bar(x + width/2, gaps_heatwave, width, label='Heatwave',
                   color=COLOR_HEATWAVE, edgecolor='white', linewidth=0.5, alpha=0.85)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # 鏁板€兼爣绛?
    for i, (v1, v2, r1, r2) in enumerate(zip(gaps_typical, gaps_heatwave, reductions_typical, reductions_heatwave)):
        y1 = v1 + 0.05 if v1 >= 0 else v1 - 0.1
        y2 = v2 + 0.05 if v2 >= 0 else v2 - 0.1
        va1 = 'bottom' if v1 >= 0 else 'top'
        va2 = 'bottom' if v2 >= 0 else 'top'

        ax.text(i - width/2, y1, f'{v1:.2f}', ha='center', va=va1, fontsize=6, fontweight='bold')
        ax.text(i + width/2, y2, f'{v2:.2f}', ha='center', va=va2, fontsize=6, fontweight='bold')

        # 娣诲姞鍑忓皯鐧惧垎姣?(闈瀊aseline)
        if i > 0 and r2 > 0:
            ax.text(i + width/2, v2/2 if v2 > 0 else v2 + 0.15, f'鈫搟r2:.0f}%',
                   ha='center', va='center', fontsize=5, color='white', fontweight='bold')

    # 涓嶅钩绛夐€嗚浆鍖哄煙
    ax.fill_between([-0.5, 3.5], 0, -0.6, color='#e8f5e9', alpha=0.5, zorder=0)
    ax.text(3.3, -0.52, 'Gap\nreversed', fontsize=6, ha='right', va='bottom',
            color='#2e7d32', style='italic')

    labels = [SCENARIOS_CONFIG[s]['label'] for s in scenarios]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=15, ha='right')
    ax.set_ylabel('HEI Gap (掳C)', fontsize=8)
    ax.set_title('(e) Gap Comparison: Typical vs Heatwave', fontsize=9, fontweight='bold', loc='left')
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.6, 1.4)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.3)


def plot_figure():
    """缁樺埗瀹屾暣鐨勫姣斿浘"""

    print("鍔犺浇鏁版嵁...")
    summary_df, decile_df = load_data()
    decile_df = ensure_baseline_rows(decile_df, summary_df)

    print("鍒涘缓鍥捐〃...")
    fig = plt.figure(figsize=(15, 9))

    # 甯冨眬: 2琛?x 3鍒?
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.2, 1, 0.9],
                          hspace=0.35, wspace=0.30,
                          left=0.06, right=0.96, top=0.90, bottom=0.08)

    # 涓婃帓: 鍏稿瀷鏃?
    ax_a = fig.add_subplot(gs[0, 0])
    plot_hei_by_decile(ax_a, decile_df, 'typical_day', show_legend=True, panel_label='a')

    ax_b = fig.add_subplot(gs[0, 1])
    plot_change_heatmap(ax_b, decile_df, 'typical_day', panel_label='b')

    # 涓嬫帓: 鐑氮鏃?
    ax_c = fig.add_subplot(gs[1, 0])
    plot_hei_by_decile(ax_c, decile_df, 'heatwave', show_legend=False, panel_label='c')

    ax_d = fig.add_subplot(gs[1, 1])
    plot_change_heatmap(ax_d, decile_df, 'heatwave', panel_label='d')

    # 鍙充晶: Gap瀵规瘮 (璺ㄤ袱琛?
    ax_e = fig.add_subplot(gs[:, 2])
    plot_gap_comparison(ax_e, summary_df)

    # 鎬绘爣棰?
    fig.suptitle('Supplementary Figure S7: Policy scenarios - Typical day vs heatwave',
                 fontsize=12, fontweight='bold', y=0.96)

    # 琛屾爣绛?
    fig.text(0.02, 0.72, 'Typical Day', fontsize=10, fontweight='bold',
             rotation=90, va='center', color=COLOR_TYPICAL)
    fig.text(0.02, 0.30, 'Heatwave', fontsize=10, fontweight='bold',
             rotation=90, va='center', color=COLOR_HEATWAVE)

    # 淇濆瓨
    output_name = 'FigS_policy_scenarios_typical_vs_heatwave'
    fig.savefig(FIGURES_DIR / f'{output_name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES_DIR / f'{output_name}.pdf', bbox_inches='tight', facecolor='white')

    print(f"\n鍥捐〃宸蹭繚瀛樿嚦: {FIGURES_DIR}")
    print(f"  - {output_name}.png")
    print(f"  - {output_name}.pdf")

    plt.close(fig)

    return fig


def main():
    print("=" * 60)
    print("缁戝埗鏀跨瓥鎯呮櫙 鍏稿瀷鏃?vs 鐑氮鏃?瀵规瘮鍥?)
    print("=" * 60)

    plot_figure()

    print("\n" + "=" * 60)
    print("瀹屾垚!")
    print("=" * 60)


if __name__ == '__main__':
    main()


