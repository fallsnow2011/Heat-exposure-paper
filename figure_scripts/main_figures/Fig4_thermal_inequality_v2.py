#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# ============ 璺緞璁剧疆 ============
# Repo root is 3 levels up from this file.
BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / 'results' / 'inequality_analysis'
FIGURES_DIR = BASE_DIR / 'paper' / '05_figures'

# 纭繚杈撳嚭鐩綍瀛樺湪
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============ 鍏ㄥ眬鏍峰紡璁剧疆 ============
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.labelweight'] = 'normal'

# ============ 閰嶈壊鏂规 ============
COLOR_TYPICAL = '#3498DB'      # 钃濊壊 - 鍏稿瀷鏃?
COLOR_HEATWAVE = '#E74C3C'     # 绾㈣壊 - 鐑氮鏃?
COLOR_TYPICAL_LIGHT = '#AED6F1'  # 娴呰摑鑹?
COLOR_HEATWAVE_LIGHT = '#F5B7B1'  # 娴呯孩鑹?


def load_data():
    """鍔犺浇鎵€鏈夐渶瑕佺殑鏁版嵁"""
    lsoa_typical = pd.read_csv(RESULTS_DIR / 'lsoa_hei_summary_typical_day.csv')
    lsoa_heatwave = pd.read_csv(RESULTS_DIR / 'lsoa_hei_summary_heatwave.csv')
    return lsoa_typical, lsoa_heatwave


def calculate_stats_by_decile(df, column, weight_col='TotPop'):
    """璁＄畻姣忎釜 decile 鐨勪汉鍙ｅ姞鏉冨潎鍊笺€佹爣鍑嗚鍜?95% CI"""
    stats_list = []
    for d in range(1, 11):
        subset = df[df['IMD_Decile'] == d].dropna(subset=[column])
        n = len(subset)
        if n > 0 and weight_col in subset.columns:
            # 浜哄彛鍔犳潈鍧囧€?
            weights = subset[weight_col].values
            values = subset[column].values
            weighted_mean = np.average(values, weights=weights)
            # 鍔犳潈鏍囧噯宸?
            weighted_var = np.average((values - weighted_mean)**2, weights=weights)
            weighted_std = np.sqrt(weighted_var)
            # 鏈夋晥鏍锋湰閲?(Kish's effective sample size)
            n_eff = (weights.sum())**2 / (weights**2).sum()
            se = weighted_std / np.sqrt(n_eff)
        else:
            weighted_mean = subset[column].mean()
            weighted_std = subset[column].std()
            se = weighted_std / np.sqrt(n) if n > 0 else 0
        ci95 = 1.96 * se
        stats_list.append({
            'decile': d,
            'mean': weighted_mean,
            'std': weighted_std,
            'se': se,
            'ci95': ci95,
            'n': n
        })
    return pd.DataFrame(stats_list)


def calculate_concentration_curve(df, health_var, ses_var='IMD_Rank', pop_var='TotPop'):
    """
    璁＄畻闆嗕腑鏇茬嚎

    杩斿洖:
        ci: 闆嗕腑鎸囨暟 (璐熷€艰〃绀?鍧?闆嗕腑浜庤传鍥扮兢浣?
        lorenz_x: 绱浜哄彛姣斾緥 (鎸夎传鍥扮▼搴︽帓搴忥紝鏈€绌峰湪鍓?
        lorenz_y: 绱鐑毚闇叉瘮渚?

    娉ㄦ剰: 濡傛灉鐑毚闇查泦涓簬璐洶缇や綋锛宭orenz_y > lorenz_x锛?
    鍗?Lorenz 鏇茬嚎鍦ㄥ瑙掔嚎涓婃柟锛孡(p) - p > 0
    """
    df_clean = df[[health_var, ses_var, pop_var]].dropna().copy()

    if len(df_clean) == 0:
        return None, None, None

    # 鎸?SES 鎺掑簭锛圛MD_Rank 瓒婁綆瓒婅传鍥帮級
    df_sorted = df_clean.sort_values(ses_var).reset_index(drop=True)

    # 璁＄畻绱浜哄彛姣斾緥
    total_pop = df_sorted[pop_var].sum()
    df_sorted['cum_pop'] = df_sorted[pop_var].cumsum()
    df_sorted['cum_pop_frac'] = df_sorted['cum_pop'] / total_pop

    # 璁＄畻绱鍋ュ悍鍙橀噺姣斾緥
    df_sorted['weighted_health'] = df_sorted[health_var] * df_sorted[pop_var]
    total_health = df_sorted['weighted_health'].sum()
    df_sorted['cum_health'] = df_sorted['weighted_health'].cumsum()
    df_sorted['cum_health_frac'] = df_sorted['cum_health'] / total_health

    # 璁＄畻 CI
    df_sorted['fractional_rank'] = (df_sorted['cum_pop'] - 0.5 * df_sorted[pop_var]) / total_pop
    weighted_health_mean = total_health / total_pop
    cov = np.cov(df_sorted[health_var], df_sorted['fractional_rank'],
                  aweights=df_sorted[pop_var])[0, 1]
    ci = 2 * cov / weighted_health_mean if weighted_health_mean != 0 else np.nan

    # Lorenz 鏇茬嚎鏁版嵁
    lorenz_x = np.concatenate([[0], df_sorted['cum_pop_frac'].values])
    lorenz_y = np.concatenate([[0], df_sorted['cum_health_frac'].values])

    return ci, lorenz_x, lorenz_y


def plot_figure4_v2(lsoa_typical, lsoa_heatwave):
    """缁戝埗閲嶆瀯鍚庣殑 Figure 4"""

    # 鍒涘缓鍥捐〃 - 澧炲姞瀹藉害鍜岄珮搴﹂伩鍏嶉伄鎸?
    fig = plt.figure(figsize=(17, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.32,
                          left=0.045, right=0.98, top=0.85, bottom=0.12)

    deciles = list(range(1, 11))

    # ============ Panel (a): Pointplot with Error Bars ============
    ax_a = fig.add_subplot(gs[0])

    # 璁＄畻缁熻閲?
    stats_typical = calculate_stats_by_decile(lsoa_typical, 'hei_mean')
    stats_heatwave = calculate_stats_by_decile(lsoa_heatwave, 'hei_mean')

    # 缁樺埗璇樊妫掑拰杩炵嚎 - 鍏稿瀷鏃?
    ax_a.errorbar(deciles, stats_typical['mean'], yerr=stats_typical['ci95'],
                  fmt='o-', color=COLOR_TYPICAL, markersize=8, linewidth=2.5,
                  capsize=4, capthick=2, label='Typical Day', zorder=5)

    # 缁樺埗璇樊妫掑拰杩炵嚎 - 鐑氮鏃?
    ax_a.errorbar(deciles, stats_heatwave['mean'], yerr=stats_heatwave['ci95'],
                  fmt='s-', color=COLOR_HEATWAVE, markersize=8, linewidth=2.5,
                  capsize=4, capthick=2, label='Heatwave', zorder=5)

    # 璁＄畻骞舵爣娉?Gap
    gap_typical = stats_typical['mean'].iloc[0] - stats_typical['mean'].iloc[-1]
    gap_heatwave = stats_heatwave['mean'].iloc[0] - stats_heatwave['mean'].iloc[-1]
    amplification = abs(gap_heatwave / gap_typical) if gap_typical != 0 else np.inf

    # 娣诲姞鍙岀澶存爣娉?Gap
    y_d1_hw = stats_heatwave['mean'].iloc[0]
    y_d10_hw = stats_heatwave['mean'].iloc[-1]

    # 鍦ㄥ彸渚ф爣娉ㄧ儹娴棩鐨?Gap
    ax_a.annotate('', xy=(10.3, y_d10_hw), xytext=(10.3, y_d1_hw),
                  arrowprops=dict(arrowstyle='<->', color=COLOR_HEATWAVE, lw=2))
    ax_a.text(10.5, (y_d1_hw + y_d10_hw)/2, f'螖={gap_heatwave:.1f}掳C',
              fontsize=9, color=COLOR_HEATWAVE, fontweight='bold', va='center')

    ax_a.set_xlabel('IMD Decile (1=Most Deprived 鈫?10=Least Deprived)', fontsize=9)
    ax_a.set_ylabel('Heat Exposure Index (掳C)', fontsize=10)
    ax_a.set_title('(a) Heat Exposure Gap Amplified by Heatwave', fontweight='bold', fontsize=11, loc='left')
    ax_a.set_xticks(deciles)
    ax_a.set_xlim(0.5, 11.5)  # 鎵╁ぇX杞磋寖鍥?
    ax_a.set_ylim(34, 46)  # 鎵╁ぇY杞磋寖鍥?
    ax_a.grid(axis='y', linestyle='--', alpha=0.3)
    ax_a.legend(loc='upper center', fontsize=9, framealpha=0.95)  # 绉诲埌涓婃柟涓棿

    # ============ Panel (b): Double Burden 鈥?Low Shade + High Heat Shock ============
    ax_b = fig.add_subplot(gs[1])

    # 璁＄畻"鍙岄噸璐熸媴"姣斾緥锛氫綆閬槾 (<5%) 涓旈珮棰濆鐑毚闇?(>9掳C)
    df_t = lsoa_typical[['lsoa11cd', 'IMD_Decile', 'hei_mean']].copy()
    df_t = df_t.rename(columns={'hei_mean': 'hei_t'})
    df_h = lsoa_heatwave[['lsoa11cd', 'hei_mean', 'shadow_mean']].copy()
    df_h = df_h.rename(columns={'hei_mean': 'hei_h'})
    df_combined = df_t.merge(df_h, on='lsoa11cd')
    df_combined['hei_delta'] = df_combined['hei_h'] - df_combined['hei_t']
    df_combined['low_shade'] = df_combined['shadow_mean'] < 0.05  # <5% 閬槾
    df_combined['high_delta'] = df_combined['hei_delta'] > 9  # >9掳C 棰濆鐑毚闇?

    # 璁＄畻姣忎釜 decile 鐨勫弻閲嶈礋鎷呮瘮渚?
    double_burden_rates = []
    for d in range(1, 11):
        subset = df_combined[df_combined['IMD_Decile'] == d]
        rate = (subset['low_shade'] & subset['high_delta']).sum() / len(subset) * 100
        double_burden_rates.append(rate)

    # 缁樺埗鏌辩姸鍥?
    bars = ax_b.bar(deciles, double_burden_rates, color=COLOR_HEATWAVE, alpha=0.8,
                    edgecolor='darkred', linewidth=1.5)

    # 绐佸嚭鏄剧ず D1 鐨勬煴瀛?
    bars[0].set_color('#8B0000')  # 娣辩孩鑹?
    bars[0].set_edgecolor('black')
    bars[0].set_linewidth(2)

    # 鏍囨敞 D1 鐨勫€?
    d1_rate = double_burden_rates[0]
    ax_b.annotate(f'{d1_rate:.0f}%\nDouble Burden',
                  xy=(1, d1_rate), xytext=(3, d1_rate + 5),
                  fontsize=10, ha='center', color='#8B0000', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='#8B0000', lw=2))

    # 娣诲姞鍙傝€冪嚎
    avg_others = np.mean(double_burden_rates[1:])
    ax_b.axhline(y=avg_others, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_b.text(9, avg_others + 2, f'Avg D2-D10:\n{avg_others:.1f}%', fontsize=8,
              va='bottom', ha='center', color='gray')

    # 娉ㄩ噴妗嗭細瑙ｉ噴鎸囨爣 - 璋冩暣浣嶇疆閬垮厤鍘嬩綇杈规
    # 闃堝€艰鏄? 5% shade 鈮?涓綅鏁? 9掳C 螖T 鈮?涓婂洓鍒嗕綅 (top 17%)
    textstr = 'Double Burden =\nLow shade (<5%) +\nHigh 螖T (>9掳C)'
    props = dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.95, edgecolor='orange')
    ax_b.text(0.97, 0.92, textstr, transform=ax_b.transAxes, fontsize=7,
              verticalalignment='top', horizontalalignment='right', bbox=props)

    ax_b.set_xlabel('IMD Decile (1=Most Deprived 鈫?10=Least Deprived)', fontsize=9)
    ax_b.set_ylabel('% of Neighbourhoods with Double Burden', fontsize=10)
    ax_b.set_title('(b) The "Heat Trap": Deprived Areas Hit Hardest',
                   fontweight='bold', fontsize=11, loc='left')
    ax_b.set_xticks(deciles)
    ax_b.set_xlim(0.5, 10.5)
    ax_b.set_ylim(0, max(double_burden_rates) * 1.25)  # 澧炲姞椤堕儴绌洪棿
    ax_b.grid(axis='y', linestyle='--', alpha=0.3)

    # 淇濆瓨缁熻鏁版嵁渚涘悗缁墦鍗?
    d1_double_burden = d1_rate
    avg_d2_d10 = avg_others

    # ============ Panel (c): Difference from Equality Curve ============
    ax_c = fig.add_subplot(gs[2])

    # 璁＄畻闆嗕腑鏇茬嚎
    ci_t, lorenz_x_t, lorenz_y_t = calculate_concentration_curve(
        lsoa_typical, 'hei_mean', 'IMD_Rank', 'TotPop')
    ci_h, lorenz_x_h, lorenz_y_h = calculate_concentration_curve(
        lsoa_heatwave, 'hei_mean', 'IMD_Rank', 'TotPop')

    # 缁樺埗宸€兼洸绾?(鏀惧ぇ涓嶅钩绛夌殑鍙鍖?
    # 宸€?= L(p) - p (Lorenz - 瀵硅绾?
    # 褰撶儹鏆撮湶闆嗕腑浜庤传鍥扮兢浣撴椂锛孡orenz 鏇茬嚎鍦ㄥ瑙掔嚎涓婃柟
    # L(p) > p锛屽洜姝?L(p) - p > 0 琛ㄧず璐洶缇や綋鎵挎媴鏇村鐑毚闇?

    if lorenz_x_t is not None and lorenz_x_h is not None:
        # 璁＄畻宸€兼洸绾? L(p) - p
        # 姝ｅ€?= 璐洶缇や綋鎵挎媴鏇村鐑毚闇?
        diff_t_plot = lorenz_y_t - lorenz_x_t
        diff_h_plot = lorenz_y_h - lorenz_x_h

        # 濉厖鍏稿瀷鏃ユ洸绾垮尯鍩?
        ax_c.fill_between(lorenz_x_t, 0, diff_t_plot, alpha=0.3, color=COLOR_TYPICAL,
                          label=f'Typical Day (CI={ci_t:.4f})')
        ax_c.plot(lorenz_x_t, diff_t_plot, '-', color=COLOR_TYPICAL, linewidth=2.5)

        # 濉厖鐑氮鏃ユ洸绾垮尯鍩?
        ax_c.fill_between(lorenz_x_h, 0, diff_h_plot, alpha=0.3, color=COLOR_HEATWAVE,
                          label=f'Heatwave (CI={ci_h:.4f})')
        ax_c.plot(lorenz_x_h, diff_h_plot, '-', color=COLOR_HEATWAVE, linewidth=2.5)

        # 闆剁嚎 (瀹屽叏骞崇瓑)
        ax_c.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5,
                     label='Perfect Equality')

        # 璁＄畻宄板€间綅缃拰楂樺害
        max_diff_t = np.max(np.abs(diff_t_plot))
        max_diff_h = np.max(np.abs(diff_h_plot))

        # 鎵惧埌鐑氮鏃ュ嘲鍊间綅缃?
        peak_idx = np.argmax(np.abs(diff_h_plot))
        peak_x_h = lorenz_x_h[peak_idx]
        peak_y_h = diff_h_plot[peak_idx]

        # 鏍囨敞宄板€煎樊寮?- 鏀惧湪鏇茬嚎宄板€奸檮杩?
        if max_diff_t > 0:
            ratio = max_diff_h / max_diff_t
            # 鐩存帴鍦ㄥ嘲鍊兼梺杈规爣娉紝涓嶇敤绠ご
            ax_c.text(peak_x_h + 0.08, peak_y_h, f'{ratio:.1f}脳',
                      fontsize=9, color=COLOR_HEATWAVE, fontweight='bold',
                      va='center', ha='left')

    ax_c.set_xlabel('Cumulative Population Share\n(poorest 鈫?richest)', fontsize=9)
    ax_c.set_ylabel('Deviation from Equality\n[L(p) - p]', fontsize=10)
    ax_c.set_title('(c) Inequality Amplification: Difference Curve',
                   fontweight='bold', fontsize=11, loc='left')
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(-0.0015, 0.006)  # 鎵╁ぇY杞磋寖鍥?
    ax_c.grid(True, linestyle='--', alpha=0.3)
    ax_c.legend(loc='upper right', fontsize=7, framealpha=0.95)

    # 娣诲姞瑙ｉ噴娉ㄩ噴 - 绉诲埌宸︿笂瑙掓洿楂樹綅缃?
    ax_c.text(0.02, 0.97, 'Curve above zero =\nHeat burden on deprived\n(CI < 0 = deprived)',
              transform=ax_c.transAxes, fontsize=7, va='top', ha='left',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95))

    # ============ 娣诲姞鏁翠綋鏍囬 ============
    fig.suptitle('Thermal Inequality: Deprived Neighbourhoods Face Greater Heat Exposure',
                 fontsize=13, fontweight='bold', y=0.98)

    # 淇濆瓨鍥捐〃
    fig.savefig(FIGURES_DIR / 'Fig4_thermal_inequality_v2.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(FIGURES_DIR / 'Fig4_thermal_inequality_v2.pdf',
                bbox_inches='tight', facecolor='white')

    print(f"Figure 4 v2 宸蹭繚瀛樿嚦: {FIGURES_DIR}")
    print("  - Fig4_thermal_inequality_v2.png")
    print("  - Fig4_thermal_inequality_v2.pdf")

    # 鎵撳嵃鍏抽敭缁熻鏁版嵁
    print("\n===== 鍏抽敭缁熻鏁版嵁 =====")
    print(f"\nPanel (a) HEI Gap (D1 - D10):")
    print(f"  Typical Day: {gap_typical:.3f}掳C")
    print(f"  Heatwave:    {gap_heatwave:.3f}掳C")
    print(f"  Amplification: {amplification:.1f}脳")

    print(f"\nPanel (b) Double Burden (Low Shade + High 螖T):")
    print(f"  D1 (Most Deprived): {d1_double_burden:.1f}%")
    print(f"  Avg D2-D10: {avg_d2_d10:.1f}%")
    print(f"  D1 is {d1_double_burden/avg_d2_d10:.1f}脳 higher than average")

    print(f"\nPanel (c) Concentration Index:")
    print(f"  Typical Day: {ci_t:.5f}")
    print(f"  Heatwave:    {ci_h:.5f}")
    print(f"  CI amplification: {abs(ci_h/ci_t):.1f}脳")

    return fig


if __name__ == '__main__':
    print("姝ｅ湪鍔犺浇鏁版嵁...")
    lsoa_typical, lsoa_heatwave = load_data()

    print(f"宸插姞杞?{len(lsoa_typical)} 涓?LSOA 鐨勬暟鎹?)

    print("\n姝ｅ湪缁戝埗 Figure 4 v2...")
    fig = plot_figure4_v2(lsoa_typical, lsoa_heatwave)

    print("\n瀹屾垚!")


