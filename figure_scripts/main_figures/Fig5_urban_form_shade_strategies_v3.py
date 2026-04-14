#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

# ============ 璺緞璁剧疆 ============
BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / 'results' / 'inequality_analysis'
SUPPLEMENT_DIR = BASE_DIR / 'paper' / '06_supplement'
FIGURES_DIR = BASE_DIR / 'paper' / '05_figures'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============ 鍏ㄥ眬鏍峰紡璁剧疆 (Nature Style) ============
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ============ 閰嶈壊鏂规 (鏇村己瀵规瘮) ============
# Panel (a) - 鍐风‖鐏?vs 椴滄椿缁?
COLOR_BUILDING = '#4a4a4a'    # 娣卞喎鐏?- 姘存偿妫灄
COLOR_VEGETATION = '#228b22'   # 妫灄缁?- 鐢熸満

# Panel (b) - 闃剁骇鍒嗗寲
COLOR_BURDEN = '#c0392b'       # 娣辩孩 - 璐熸媴/宸ヤ笟
COLOR_PRIVILEGE = '#27ae60'    # 缈犵豢 - 鐗规潈/鑷劧
COLOR_NS = '#9e9e9e'           # 鐏拌壊 - 涓嶆樉钁?
# Panel (c) - 鏁堢巼閫掕繘
COLOR_BASELINE = '#d0d0d0'     # 娴呯伆
COLOR_S1 = '#a6bddb'           # 娴呰摑鐏?
COLOR_S2 = '#3690c0'           # 涓摑
COLOR_S3 = '#016c59'           # 娣遍潚缁?(Teal) - 涓昏!


def load_data():
    """鍔犺浇鏁版嵁"""
    lsoa_heatwave = pd.read_csv(RESULTS_DIR / 'lsoa_hei_summary_heatwave.csv')
    # Prefer submission-ready SI copies bundled under final-SI/data; fall back to working dir.
    si_data_dir = BASE_DIR / 'paper' / 'final-SI' / 'data'

    lcz_by_imd_path = si_data_dir / 'lcz_distribution_by_imd.csv'
    if not lcz_by_imd_path.exists():
        lcz_by_imd_path = SUPPLEMENT_DIR / 'lcz_distribution_by_imd.csv'

    scenarios_summary_path = si_data_dir / 'policy_scenarios_summary_v2.csv'
    if not scenarios_summary_path.exists():
        scenarios_summary_path = SUPPLEMENT_DIR / 'policy_scenarios_summary_v2.csv'

    lsoa_lcz_dist_path = si_data_dir / 'lsoa_lcz_distribution.csv'
    if not lsoa_lcz_dist_path.exists():
        lsoa_lcz_dist_path = SUPPLEMENT_DIR / 'lsoa_lcz_distribution.csv'

    lcz_by_imd = pd.read_csv(lcz_by_imd_path)
    scenarios_summary = pd.read_csv(scenarios_summary_path)
    lsoa_lcz_dist = pd.read_csv(lsoa_lcz_dist_path)
    return lsoa_heatwave, lcz_by_imd, scenarios_summary, lsoa_lcz_dist


def calculate_shadow_by_deprivation(lsoa_df):
    """璁＄畻璐洶/瀵岃 LSOA 鐨勯槾褰辫础鐚?""
    poor = lsoa_df[lsoa_df['IMD_Decile'].isin([1, 2, 3])].copy()
    rich = lsoa_df[lsoa_df['IMD_Decile'].isin([8, 9, 10])].copy()

    results = {}
    for name, df in [('Deprived\n(D1-D3)', poor), ('Affluent\n(D8-D10)', rich)]:
        weights = df['TotPop'].values
        building_shadow = np.average(df['shadow_building_mean'].values, weights=weights)
        vegetation_shadow = np.average(df['shadow_vegetation_mean'].values, weights=weights)
        total = building_shadow + vegetation_shadow

        results[name] = {
            'building': building_shadow * 100,
            'vegetation': vegetation_shadow * 100,
            'total': total * 100,
            'building_pct': building_shadow / total * 100 if total > 0 else 0
        }
    return results


def plot_panel_a(ax, lsoa_df):
    """Panel (a): Shade composition (100% stacked)"""
    shadow_data = calculate_shadow_by_deprivation(lsoa_df)

    groups = list(shadow_data.keys())
    x = np.arange(len(groups))
    width = 0.55

    building_share = [shadow_data[g]['building_pct'] for g in groups]
    vegetation_share = [100 - s for s in building_share]

    ax.bar(
        x,
        building_share,
        width,
        label='Building shade',
        color=COLOR_BUILDING,
        edgecolor='white',
        linewidth=0.5,
    )
    ax.bar(
        x,
        vegetation_share,
        width,
        bottom=building_share,
        label='Vegetation shade',
        color=COLOR_VEGETATION,
        edgecolor='white',
        linewidth=0.5,
    )

    for i, g in enumerate(groups):
        total = shadow_data[g]['total']
        b = building_share[i]
        v = vegetation_share[i]

        ax.text(
            i,
            b / 2,
            f'Buildings\n{b:.0f}%',
            ha='center',
            va='center',
            fontsize=8,
            color='white',
            fontweight='bold',
        )
        ax.text(
            i,
            b + v / 2,
            f'Vegetation\n{v:.0f}%',
            ha='center',
            va='center',
            fontsize=8,
            color='white',
            fontweight='bold',
        )
        ax.text(
            i,
            103,
            f'Total shade\n{total:.1f}%',
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold',
            color='#212121',
        )

    ax.set_ylabel('Shade composition (% of total shade)', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_xlim(-0.6, 1.6)
    ax.set_yticks([0, 25, 50, 75, 100])

    return shadow_data


def plot_panel_b(ax, ax_mark, lcz_by_imd, lsoa_heatwave, lsoa_lcz_dist):
    """Panel (b): Urban form gap (distribution) + within-IMD association (HEI)"""
    poor_lcz = lcz_by_imd[lcz_by_imd['IMD_Decile'].isin([1, 2, 3])].copy()
    rich_lcz = lcz_by_imd[lcz_by_imd['IMD_Decile'].isin([8, 9, 10])].copy()

    def weighted_avg(df, col):
        return np.average(df[col].values, weights=df['n_lsoa'].values)

    # LCZ categories (match other analyses)
    lcz_categories = ['Sparse/Natural', 'Industry/Large', 'Compact Built', 'Open Built']
    lcz_short = ['Natural\nvegetation', 'Industry &\nwarehouses', 'Compact\nurban', 'Open\nsuburban']

    poor_vals = [weighted_avg(poor_lcz, cat) for cat in lcz_categories]
    rich_vals = [weighted_avg(rich_lcz, cat) for cat in lcz_categories]
    differences = [p - r for p, r in zip(poor_vals, rich_vals)]  # percentage points (pp)

    # ---------- Left: distribution differences (tornado) ----------
    y = np.arange(len(lcz_categories))
    height = 0.72

    bars = ax.barh(y, differences, height, color='#bdbdbd', edgecolor='white', linewidth=0.6)
    ax.axvline(x=0, color='black', linewidth=1.2)

    for i, (diff, _) in enumerate(zip(differences, bars)):
        # Label placement:
        # - Positive diffs: put the label just to the right of the bar.
        # - Small negative diffs (e.g., Open suburban -1.1 pp): put the label just to the left of the bar (outside).
        # - Large negative diffs (e.g., Natural vegetation -5.4 pp): keep the label inside the bar to avoid the y-axis.
        if diff > 0:
            x_pos = diff + 0.25
            ha = "left"
        else:
            if diff <= -4.5:
                x_pos = diff + 0.25
                ha = "left"
            else:
                x_pos = diff - 0.25
                ha = "right"
        ax.text(x_pos, i, f'{diff:+.1f} pp', ha=ha, va='center',
                fontsize=8, fontweight='bold', color='#424242')

    ax.set_yticks(y)
    ax.set_yticklabels(lcz_short, fontsize=8, ha='right')
    ax.set_xlim(-6.5, 4.8)
    ax.set_xlabel('螖 LCZ share (pp) = IMD 1鈥? 鈭?IMD 8鈥?0\nIMD 8鈥?0 higher  鈫?0 鈫? IMD 1鈥? higher', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ---------- Right: within-IMD association markers ----------
    # Harmonise LCZ-share table schema (supports both paper/06_supplement and archive/temp variants)
    if 'lsoa11cd' in lsoa_lcz_dist.columns and set(lcz_categories).issubset(lsoa_lcz_dist.columns):
        lsoa_lcz_std = lsoa_lcz_dist[['lsoa11cd'] + lcz_categories].copy()
    elif 'lsoa_code' in lsoa_lcz_dist.columns and 'group_Compact_Built_pct' in lsoa_lcz_dist.columns:
        rename_map = {
            'lsoa_code': 'lsoa11cd',
            'group_Compact_Built_pct': 'Compact Built',
            'group_Open_Built_pct': 'Open Built',
            'group_Industry_Large_pct': 'Industry/Large',
            'group_Sparse_Natural_pct': 'Sparse/Natural',
        }
        lsoa_lcz_std = (
            lsoa_lcz_dist.rename(columns=rename_map)[['lsoa11cd'] + lcz_categories].copy()
        )
    else:
        raise ValueError("Unrecognised schema for lsoa_lcz_distribution.csv")

    merged = pd.merge(
        lsoa_heatwave[['lsoa11cd', 'IMD_Decile', 'hei_mean']],
        lsoa_lcz_std,
        on='lsoa11cd',
        how='inner'
    )

    imd_groups = [
        ('IMD 1鈥?', [1, 2, 3]),
        ('IMD 4鈥?', [4, 5, 6, 7]),
        ('IMD 8鈥?0', [8, 9, 10]),
    ]

    x_cols = [0, 1, 2]
    ax_mark.set_xlim(-0.6, 2.6)
    ax_mark.set_xticks(x_cols)
    ax_mark.set_xticklabels(['IMD\n1鈥?', 'IMD\n4鈥?', 'IMD\n8鈥?0'], fontsize=7)
    ax_mark.tick_params(axis='y', left=False, labelleft=False)
    ax_mark.tick_params(axis='x', bottom=False, top=False, labelbottom=True)
    ax_mark.text(0.5, 0.98, 'Within-IMD\nassociation', transform=ax_mark.transAxes,
                 ha='center', va='top', fontsize=8, fontweight='bold')

    # subtle column separators
    ax_mark.axvline(0.5, color='#eeeeee', linewidth=1.0, zorder=0)
    ax_mark.axvline(1.5, color='#eeeeee', linewidth=1.0, zorder=0)

    within_imd = {}
    for y_i, lcz_type in enumerate(lcz_categories):
        within_imd[lcz_type] = {}
        for x_i, (imd_label, deciles) in enumerate(imd_groups):
            sub = merged[merged['IMD_Decile'].isin(deciles)]
            r, p = pearsonr(sub[lcz_type].values, sub['hei_mean'].values)
            within_imd[lcz_type][imd_label] = {'r': float(r), 'p': float(p), 'n': int(len(sub))}

            if p < 0.05:
                color = COLOR_BURDEN if r > 0 else COLOR_PRIVILEGE
            else:
                color = COLOR_NS

            ax_mark.scatter(x_cols[x_i], y_i, s=85, marker='o',
                            color=color, edgecolor='white', linewidth=0.7, zorder=3)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax_mark.spines[spine].set_visible(False)

    # compact legend placed on the left axis (avoids cross-panel overlap)
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=COLOR_BURDEN,
                   markeredgecolor='white', markersize=7, label='r>0 (higher HEI)'),
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=COLOR_PRIVILEGE,
                   markeredgecolor='white', markersize=7, label='r<0 (lower HEI)'),
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=COLOR_NS,
                   markeredgecolor='white', markersize=7, label='n.s. (p鈮?.05)'),
    ]
    # Legend: place to the right of the Open suburban row (avoid covering the negative bar/label)
    ax.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.93),
        fontsize=7,
        framealpha=0.9,
        edgecolor='none',
        borderpad=0.3,
        labelspacing=0.3,
        handletextpad=0.4,
    )

    return {'differences': dict(zip(lcz_categories, differences)), 'within_imd': within_imd}


def plot_panel_c(ax, scenarios_summary):
    """Panel (c): The Precision Strike - 閫氬線甯屾湜鐨勯樁姊?
    Note: `policy_scenarios_summary_v2.csv` uses canonical scenario IDs
    (baseline / S1_citywide / S2_corridors / S3_equity_first) and an explicit
    `scenario_order` column for plotting order.
    """
    df_hw = scenarios_summary[scenarios_summary['time_scenario'] == 'heatwave'].copy()

    scenario_display = {
        'baseline': 'Baseline',
        'S1_citywide': 'S1: Citywide',
        'S2_corridors': 'S2: Corridors',
        'S3_equity_first': 'S3: Equity\nFirst',
    }
    scenario_colors = {
        'baseline': COLOR_BASELINE,
        'S1_citywide': COLOR_S1,
        'S2_corridors': COLOR_S2,
        'S3_equity_first': COLOR_S3,
    }

    if 'scenario_order' in df_hw.columns:
        df_hw = df_hw.sort_values('scenario_order')
    else:
        fallback_order = ['baseline', 'S1_citywide', 'S2_corridors', 'S3_equity_first']
        df_hw['scenario_order'] = df_hw['policy_scenario'].map({k: i for i, k in enumerate(fallback_order)})
        df_hw = df_hw.sort_values('scenario_order')

    scenario_ids = df_hw['policy_scenario'].tolist()
    labels = [scenario_display.get(s, s) for s in scenario_ids]
    colors = [scenario_colors.get(s, '#cccccc') for s in scenario_ids]
    gaps = df_hw['gap_pop'].tolist()
    coverages = df_hw['target_pop_pct'].tolist()

    x = np.arange(len(scenario_ids))

    # ===== 绮惧噯鐨?"Gap eliminated / reversed" 鍖哄煙 =====
    # 鍙湪 y<0 鍖哄煙娣诲姞娣＄豢鑹诧紝杈圭紭閿愬埄
    ax.fill_between([x.min() - 0.5, x.max() + 0.5], 0, -0.5, color='#e8f5e9', alpha=0.8, zorder=0)
    ax.text(0.3, -0.38, 'Gap eliminated (added cooling)', fontsize=7, ha='left',
            color='#2e7d32', style='italic')

    # ===== 鑳屾櫙寮曞绠ご: Baseline 鈫?S3 =====
    ax.annotate('', xy=(x.max(), gaps[-1] + 0.15), xytext=(x.min(), gaps[0] + 0.15),
                arrowprops=dict(arrowstyle='->', color='#bdbdbd', lw=3,
                               connectionstyle='arc3,rad=-0.1'),
                zorder=1)

    # 缁樺埗鏉″舰鍥?    for i, (xi, gap, color) in enumerate(zip(x, gaps, colors)):
        is_recommended = scenario_ids[i] == 'S3_equity_first'
        edgecolor = 'black' if is_recommended else 'white'
        linewidth = 1.5 if is_recommended else 0.5
        ax.bar(xi, gap, width=0.65, color=color, edgecolor=edgecolor,
               linewidth=linewidth, zorder=3)

    # 闆剁嚎
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2, zorder=2)

    # 鏁板€兼爣绛?    for i, (gap, coverage) in enumerate(zip(gaps, coverages)):
        if scenario_ids[i] == 'S3_equity_first':  # S3 鐗规畩澶勭悊
            # S3 鐨勬暟鍊兼斁鍦ㄦ煴瀛愬唴閮?            ax.text(i, gap/2, f'{gap:+.2f}掳C', ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
        else:
            va = 'bottom' if gap >= 0 else 'top'
            offset = 0.08 if gap >= 0 else -0.08
            ax.text(i, gap + offset, f'{gap:+.2f}掳C', ha='center', va=va,
                    fontsize=9, fontweight='bold')

    # Gap reduction 鏍囨敞
    improvement = gaps[0] - gaps[-1]
    ax.text((x.min() + x.max()) / 2, 1.5, f'Total reduction: {improvement:.2f}掳C',
            ha='center', fontsize=8, color='#616161',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='#bdbdbd', alpha=0.95))

    ax.set_ylabel('HEI gap: deprived - affluent (掳C)', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(-0.6, 1.7)
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)

    # 鍦ㄥ悇鏌卞瓙涓嬫柟鏍囨敞瑕嗙洊鐜?(浠庢暟鎹姩鎬佽鍙栵紝S3宸叉湁 Recommended 鏍囩)
    for i, cov in enumerate(coverages):
        cov_label = f'{cov:.0f}%' if cov > 0 else '0%'
        if scenario_ids[i] == 'S3_equity_first':
            # 淇濇寔鏍囩绠€鐭紝閬垮厤 PDF 瀵煎嚭鏃跺彸渚ф枃瀛楄瑁佸垏銆?            cov_label = f'{cov_label} 鈽?
            ax.text(i, -0.52, cov_label, ha='center', va='top', fontsize=7,
                    color='white', fontweight='bold')
        else:
            ax.text(i, -0.52, cov_label, ha='center', va='top', fontsize=7, color='#666666')

    return {'gaps': dict(zip(labels, gaps))}


def plot_figure5_v3(lsoa_heatwave, lcz_by_imd, scenarios_summary, lsoa_lcz_dist):
    """缁戝埗 Figure 5 v3"""

    # 鍒涘缓鍥捐〃 - 澧炲姞瀛愬浘闂磋窛
    fig = plt.figure(figsize=(15, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.05, 1.1], wspace=0.32,
                          left=0.05, right=0.98, top=0.85, bottom=0.15)

    # Panel (a)
    ax_a = fig.add_subplot(gs[0])
    shadow_data = plot_panel_a(ax_a, lsoa_heatwave)

    # Panel (b) - nested layout (bars + marker table)
    gs_b = gs[1].subgridspec(1, 2, width_ratios=[3.8, 1.2], wspace=0.05)
    ax_b = fig.add_subplot(gs_b[0])
    ax_b_mark = fig.add_subplot(gs_b[1], sharey=ax_b)
    lcz_data = plot_panel_b(ax_b, ax_b_mark, lcz_by_imd, lsoa_heatwave, lsoa_lcz_dist)

    # Panel (c)
    ax_c = fig.add_subplot(gs[2])
    scenario_data = plot_panel_c(ax_c, scenarios_summary)

    # --- Aligned panel titles (figure-level) ---
    # Using fig.text avoids slight misalignment caused by nested axes in panel (b).
    panel_titles = [
        (ax_a, 'a  Shade composition: deprived areas rely on buildings'),
        (ax_b, 'b  Urban form gap: structural inequality'),
        (ax_c, 'c  Policy scenarios: precision approach wins'),
    ]
    y_title = max(ax.get_position().y1 for ax, _ in panel_titles) + 0.015
    y_title = min(0.99, y_title)
    for ax, title in panel_titles:
        x0 = ax.get_position().x0
        if ax is ax_b:
            x0 += 0.012
        fig.text(
            x0,
            y_title,
            title,
            ha='left',
            va='bottom',
            fontsize=10,
            fontweight='bold',
        )

    # Save with a small pad to avoid any text cropping at the edges.
    fig.savefig(
        FIGURES_DIR / "Fig5_urban_form_shade_strategies_v3.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
        facecolor="white",
    )
    fig.savefig(
        FIGURES_DIR / "Fig5_urban_form_shade_strategies_v3.pdf",
        bbox_inches="tight",
        pad_inches=0.2,
        facecolor="white",
    )

    print(f"Figure 5 v3 宸蹭繚瀛樿嚦: {FIGURES_DIR}")
    print("  - Fig5_urban_form_shade_strategies_v3.png")
    print("  - Fig5_urban_form_shade_strategies_v3.pdf")

    # 鎵撳嵃缁熻
    print("\n===== 鍏抽敭缁熻 =====")
    print("\nPanel (a):")
    for g, d in shadow_data.items():
        print(f"  {g.replace(chr(10), ' ')}: {d['total']:.1f}% (Building {d['building_pct']:.0f}%)")

    print("\nPanel (b) LCZ differences:")
    for cat, diff in lcz_data['differences'].items():
        print(f"  {cat}: {diff:+.1f}%")

    print("\nPanel (c) HEI Gap:")
    for label, gap in scenario_data['gaps'].items():
        print(f"  {label}: {gap:+.2f}掳C")

    return fig


if __name__ == '__main__':
    print("Loading data...")
    lsoa_heatwave, lcz_by_imd, scenarios_summary, lsoa_lcz_dist = load_data()
    print(f"Loaded {len(lsoa_heatwave)} LSOAs")

    print("\nPlotting Figure 5 v3...")
    fig = plot_figure5_v3(lsoa_heatwave, lcz_by_imd, scenarios_summary, lsoa_lcz_dist)
    print("\nDone!")


