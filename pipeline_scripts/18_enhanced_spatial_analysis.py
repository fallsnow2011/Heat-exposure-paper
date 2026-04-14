import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration / 閰嶇疆
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results" / "inequality_analysis"
OUTPUT_DIR = BASE_DIR / "paper" / "06_supplement"
FIGURES_DIR = BASE_DIR / "paper" / "05_figures"

# Low-saturation colour palette / 浣庨ケ鍜屽害閰嶈壊鏂规
COLORS = {
    'deep_blue': '#1f3a5f',
    'teal': '#3c6e71',
    'gray_teal': '#7ca5a1',
    'warm_gray': '#c2b8ad',
    'light_gray': '#e9e4dd',
    'terracotta': '#c97b63',
    'dark_red': '#7c2e1d',
    'typical': '#7ca5a1',
    'heatwave': '#1f3a5f'
}

def load_lsoa_data():
    """Load LSOA-level summary tables / 鍔犺浇 LSOA 绾у埆鏁版嵁"""
    typical = pd.read_csv(RESULTS_DIR / "lsoa_hei_summary_typical_day.csv")
    heatwave = pd.read_csv(RESULTS_DIR / "lsoa_hei_summary_heatwave.csv")
    return typical, heatwave

def load_lsoa_geometry():
    """Load LSOA geometries when available / 鍔犺浇 LSOA 鍑犱綍鏁版嵁"""
    geojson_path = OUTPUT_DIR / "imd_lsoa.geojson"
    if geojson_path.exists():
        gdf = gpd.read_file(geojson_path)
        return gdf
    return None

def calculate_morans_i_with_stats(gdf, column, w=None):
    """
    Calculate Moran's I and its significance statistics / 璁＄畻 Moran's I 鍙婂叾缁熻鏄捐憲鎬?    """
    try:
        from libpysal.weights import Queen
        from esda.moran import Moran

        # Drop invalid observations before building the weights matrix / 绉婚櫎鏃犳晥鍊?        valid_mask = gdf[column].notna() & np.isfinite(gdf[column])
        gdf_valid = gdf[valid_mask].copy()

        if len(gdf_valid) < 10:
            return None

        # Build a Queen-contiguity spatial weights matrix / 鍒涘缓绌洪棿鏉冮噸鐭╅樀
        if w is None:
            w = Queen.from_dataframe(gdf_valid)

        # Compute global Moran's I / 璁＄畻 Moran's I
        y = gdf_valid[column].values
        moran = Moran(y, w)

        return {
            'I': moran.I,
            'expected_I': moran.EI,
            'variance': moran.VI_norm,
            'z_score': moran.z_norm,
            'p_value': moran.p_norm,
            'n': len(gdf_valid)
        }
    except Exception as e:
        print(f"Moran's I error / Moran's I 璁＄畻閿欒: {e}")
        return None

def calculate_local_morans(gdf, column):
    """
    Calculate Local Moran's I (LISA) / 璁＄畻 Local Moran's I锛圠ISA锛?    """
    try:
        from libpysal.weights import Queen
        from esda.moran import Moran_Local

        valid_mask = gdf[column].notna() & np.isfinite(gdf[column])
        gdf_valid = gdf[valid_mask].copy()

        if len(gdf_valid) < 10:
            return None, None

        w = Queen.from_dataframe(gdf_valid)
        y = gdf_valid[column].values
        lisa = Moran_Local(y, w, seed=0)

        # LISA cluster codes: 1=HH, 2=LH, 3=LL, 4=HL, 0=not significant / LISA 鍒嗙被锛?=HH锛?=LH锛?=LL锛?=HL锛?=涓嶆樉钁?        gdf_valid['lisa_cluster'] = lisa.q
        gdf_valid['lisa_significant'] = lisa.p_sim < 0.05
        gdf_valid['lisa_label'] = 'Not Significant'

        sig_mask = gdf_valid['lisa_significant']
        gdf_valid.loc[sig_mask & (gdf_valid['lisa_cluster'] == 1), 'lisa_label'] = 'High-High'
        gdf_valid.loc[sig_mask & (gdf_valid['lisa_cluster'] == 2), 'lisa_label'] = 'Low-High'
        gdf_valid.loc[sig_mask & (gdf_valid['lisa_cluster'] == 3), 'lisa_label'] = 'Low-Low'
        gdf_valid.loc[sig_mask & (gdf_valid['lisa_cluster'] == 4), 'lisa_label'] = 'High-Low'

        return gdf_valid, lisa
    except Exception as e:
        print(f"Local Moran's I error / Local Moran's I 璁＄畻閿欒: {e}")
        return None, None

def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size / 璁＄畻 Cohen's d 鏁堝簲閲?""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()

    # Pooled standard deviation / 鍚堝苟鏍囧噯宸?    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    if pooled_std == 0:
        return 0

    d = (group1.mean() - group2.mean()) / pooled_std
    return d

def calculate_effect_sizes(typical_df, heatwave_df):
    """Calculate effect sizes for key group comparisons / 璁＄畻鍚勭粍瀵规瘮鐨勬晥搴旈噺"""
    results = []

    for scenario, df in [('typical', typical_df), ('heatwave', heatwave_df)]:
        # IMD Decile 1-2 vs 9-10
        poor = df[df['IMD_Decile'].isin([1, 2])]['hei_mean'].dropna()
        rich = df[df['IMD_Decile'].isin([9, 10])]['hei_mean'].dropna()

        if len(poor) > 0 and len(rich) > 0:
            d = calculate_cohens_d(poor, rich)
            t_stat, p_val = stats.ttest_ind(poor, rich, equal_var=False)

            # Bootstrap 95% CI
            n_boot = 1000
            boot_diffs = []
            for _ in range(n_boot):
                boot_poor = np.random.choice(poor, size=len(poor), replace=True)
                boot_rich = np.random.choice(rich, size=len(rich), replace=True)
                boot_diffs.append(boot_poor.mean() - boot_rich.mean())

            ci_lower = np.percentile(boot_diffs, 2.5)
            ci_upper = np.percentile(boot_diffs, 97.5)

            results.append({
                'scenario': scenario,
                'comparison': 'IMD_Decile_1-2_vs_9-10',
                'mean_diff': poor.mean() - rich.mean(),
                'cohens_d': d,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                't_statistic': t_stat,
                'p_value': p_val,
                'n_poor': len(poor),
                'n_rich': len(rich)
            })

    return pd.DataFrame(results)

def calculate_concentration_index(df, health_var='hei_mean', rank_var='IMD_Rank', weight_var='TotPop'):
    """
    Calculate the concentration index (CI) /
    璁＄畻闆嗕腑鎸囨暟锛圕oncentration Index锛?
    CI = (2 / 渭) * cov(h, r)
    where h is the health-related variable, r is the socioeconomic rank,
    and 渭 is the mean of the health variable /
    鍏朵腑 h 鏄仴搴峰彉閲忥紝r 鏄ぞ浼氱粡娴庢帓鍚嶏紝渭 鏄仴搴峰彉閲忓潎鍊?    """
    valid = df[[health_var, rank_var, weight_var]].dropna()

    if len(valid) < 10:
        return None

    # Sort by socioeconomic rank / 鎸?rank 鎺掑簭
    valid = valid.sort_values(rank_var)

    # Calculate cumulative population shares / 璁＄畻绱Н浜哄彛姣斾緥
    total_pop = valid[weight_var].sum()
    valid['cum_pop'] = valid[weight_var].cumsum() / total_pop
    valid['fractional_rank'] = (valid['cum_pop'] + valid['cum_pop'].shift(1).fillna(0)) / 2

    # Population-weighted mean HEI / 鍔犳潈骞冲潎 HEI
    mean_h = np.average(valid[health_var], weights=valid[weight_var])

    # Calculate the concentration index / 璁＄畻 CI
    ci = 2 * np.cov(valid[health_var], valid['fractional_rank'],
                    aweights=valid[weight_var])[0, 1] / mean_h

    return ci

def plot_lisa_map(gdf_lisa, title, output_path, imd_overlay=True):
    """Plot the LISA cluster map / 缁樺埗 LISA 鑱氱被鍦板浘"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # LISA colour mapping / LISA 棰滆壊鏄犲皠
    lisa_colors = {
        'High-High': '#d7191c',      # red hotspot / 绾㈣壊鐑偣
        'Low-Low': '#2c7bb6',        # blue coldspot / 钃濊壊鍐风偣
        'High-Low': '#fdae61',       # orange spatial outlier / 姗欒壊绌洪棿寮傚父
        'Low-High': '#abd9e9',       # light-blue spatial outlier / 娴呰摑绌洪棿寮傚父
        'Not Significant': '#e0e0e0' # grey background / 鐏拌壊
    }

    gdf_lisa['color'] = gdf_lisa['lisa_label'].map(lisa_colors)
    gdf_lisa.plot(ax=ax, color=gdf_lisa['color'], edgecolor='white', linewidth=0.1)

    # Add the legend patches manually / 娣诲姞鍥句緥
    patches = [mpatches.Patch(color=color, label=label)
               for label, color in lisa_colors.items() if label != 'Not Significant']
    patches.append(mpatches.Patch(color='#e0e0e0', label='Not Significant'))
    ax.legend(handles=patches, loc='lower right', fontsize=9)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved output / 宸蹭繚瀛? {output_path}")

def plot_ci_barplot(ci_results, output_path):
    """Plot a bar chart of concentration indices / 缁樺埗 Concentration Index 鏉″舰鍥?""
    fig, ax = plt.subplots(figsize=(8, 5))

    scenarios = ci_results['scenario'].tolist()
    ci_values = ci_results['CI'].tolist()

    colors = [COLORS['typical'], COLORS['heatwave']]
    bars = ax.bar(scenarios, ci_values, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels above or below bars / 娣诲姞鏁板€兼爣绛?    for bar, val in zip(bars, ci_values):
        height = bar.get_height()
        ax.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=11, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Concentration Index (CI)', fontsize=11)
    ax.set_xlabel('Scenario', fontsize=11)
    ax.set_title('Heat Exposure Concentration Index\n(Negative = burden on deprived areas)', fontsize=12)

    # Expand the y-axis range slightly for labels / 璁剧疆 y 杞磋寖鍥?    y_min = min(ci_values) * 1.3
    y_max = max(0.001, max(ci_values) * 1.3)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved output / 宸蹭繚瀛? {output_path}")

def plot_enhanced_boxplot(typical_df, heatwave_df, output_path):
    """Plot enhanced boxplots with significance annotations / 缁樺埗澧炲己鐗堢绾垮浘锛堝甫鏄捐憲鎬ф爣璁帮級"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (scenario, df, ax) in enumerate([
        ('Typical Day', typical_df, axes[0]),
        ('Heatwave', heatwave_df, axes[1])
    ]):
        # Group observations by IMD decile / 鎸?IMD Decile 鍒嗙粍
        data = [df[df['IMD_Decile'] == d]['hei_mean'].dropna().values
                for d in range(1, 11)]

        color = COLORS['typical'] if idx == 0 else COLORS['heatwave']

        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Compare the two extremes with a t-test / t 妫€楠岋細Decile 1 vs 10
        t_stat, p_val = stats.ttest_ind(data[0], data[9], equal_var=False)

        # Significance labels / 鏄捐憲鎬ф爣璁?
        if p_val < 0.001:
            sig_label = '***'
        elif p_val < 0.01:
            sig_label = '**'
        elif p_val < 0.05:
            sig_label = '*'
        else:
            sig_label = 'ns'

        # Draw the significance bracket above the boxes / 娣诲姞鏄捐憲鎬ф爣璁扮嚎
        y_max = max([max(d) for d in data if len(d) > 0])
        y_line = y_max + 1
        ax.plot([1, 10], [y_line, y_line], 'k-', linewidth=1)
        ax.plot([1, 1], [y_line-0.3, y_line], 'k-', linewidth=1)
        ax.plot([10, 10], [y_line-0.3, y_line], 'k-', linewidth=1)
        ax.text(5.5, y_line + 0.3, sig_label, ha='center', va='bottom', fontsize=14, fontweight='bold')

        ax.set_xlabel('IMD Decile (1 = Most Deprived)', fontsize=11)
        ax.set_ylabel('HEI (掳C)', fontsize=11)
        ax.set_title(f'{scenario}\n(p = {p_val:.2e})', fontsize=12)
        ax.set_xticklabels(range(1, 11))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved output / 宸蹭繚瀛? {output_path}")

def plot_effect_size_forest(effect_df, output_path):
    """Plot a forest-style chart for effect sizes / 缁樺埗鏁堝簲閲忔．鏋楀浘"""
    fig, ax = plt.subplots(figsize=(10, 5))

    y_positions = range(len(effect_df))
    colors = [COLORS['typical'] if s == 'typical' else COLORS['heatwave']
              for s in effect_df['scenario']]

    # Draw point estimates with confidence intervals / 缁樺埗璇樊绾垮拰鐐?    for i, row in effect_df.iterrows():
        ax.errorbar(row['mean_diff'], i,
                    xerr=[[row['mean_diff'] - row['ci_lower']],
                          [row['ci_upper'] - row['mean_diff']]],
                    fmt='o', color=colors[i], markersize=10, capsize=5,
                    capthick=2, elinewidth=2)

        # Annotate each row with Cohen's d / 娣诲姞 Cohen's d 鏍囩
        ax.text(row['ci_upper'] + 0.1, i, f"d = {row['cohens_d']:.2f}",
                va='center', fontsize=10)

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{row['scenario'].title()}" for _, row in effect_df.iterrows()])
    ax.set_xlabel('HEI Difference (Poor - Rich, 掳C)', fontsize=11)
    ax.set_title('Effect Size: Deprived (Decile 1-2) vs Affluent (Decile 9-10)\nwith 95% Bootstrap CI', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved output / 宸蹭繚瀛? {output_path}")

def main():
    print("=" * 60)
    print("Enhanced spatial analysis supplement / 澧炲己绌洪棿鍒嗘瀽琛ュ厖鍒嗘瀽")
    print("=" * 60)

    # 1. Load tabular summaries / 鍔犺浇鏁版嵁
    print("\n[1] Loading tabular summaries / 鍔犺浇鏁版嵁...")
    typical_df, heatwave_df = load_lsoa_data()
    print(f"  Typical day: {len(typical_df)} LSOAs")
    print(f"  Heatwave: {len(heatwave_df)} LSOAs")

    # 2. Load geometry and calculate Moran's I / 鍔犺浇鍑犱綍鏁版嵁骞惰绠?Moran's I
    print("\n[2] Calculating Moran's I / 璁＄畻 Moran's I...")
    gdf = load_lsoa_geometry()

    moran_results = []

    if gdf is not None:
        # Join HEI summaries to the geometry table / 鍚堝苟 HEI 鏁版嵁鍒板嚑浣曟暟鎹?        for scenario, df in [('typical_day', typical_df), ('heatwave', heatwave_df)]:
            gdf_merged = gdf.merge(df[['lsoa11cd', 'hei_mean']],
                                   left_on='lsoa11cd', right_on='lsoa11cd',
                                   how='inner')

            moran = calculate_morans_i_with_stats(gdf_merged, 'hei_mean')
            if moran:
                moran['scenario'] = scenario
                moran_results.append(moran)
                print(f"  {scenario}: I = {moran['I']:.4f}, z = {moran['z_score']:.2f}, p = {moran['p_value']:.2e}")
    else:
        print("  Warning: geometry could not be loaded, skipping Moran's I / 璀﹀憡锛氭棤娉曞姞杞藉嚑浣曟暟鎹紝璺宠繃 Moran's I")

    # Save Moran's I summary / 淇濆瓨 Moran's I 缁撴灉
    if moran_results:
        moran_df = pd.DataFrame(moran_results)
        moran_df.to_csv(OUTPUT_DIR / "morans_i_complete.csv", index=False)
        print(f"  Saved output / 宸蹭繚瀛? {OUTPUT_DIR / 'morans_i_complete.csv'}")

    # 3. Calculate effect sizes / 璁＄畻鏁堝簲閲?    print("\n[3] Calculating Cohen's d effect sizes / 璁＄畻 Cohen's d 鏁堝簲閲?..")
    effect_df = calculate_effect_sizes(typical_df, heatwave_df)
    effect_df.to_csv(OUTPUT_DIR / "effect_sizes_cohens_d.csv", index=False)
    print(f"  Saved output / 宸蹭繚瀛? {OUTPUT_DIR / 'effect_sizes_cohens_d.csv'}")

    for _, row in effect_df.iterrows():
        print(f"  {row['scenario']}: d = {row['cohens_d']:.3f}, "
              f"diff = {row['mean_diff']:.2f}掳C [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]")

    # 4. Calculate the concentration index / 璁＄畻 Concentration Index
    print("\n[4] Calculating the concentration index / 璁＄畻 Concentration Index...")
    ci_results = []
    for scenario, df in [('Typical Day', typical_df), ('Heatwave', heatwave_df)]:
        ci = calculate_concentration_index(df)
        if ci is not None:
            ci_results.append({'scenario': scenario, 'CI': ci})
            print(f"  {scenario}: CI = {ci:.4f}")

    ci_df = pd.DataFrame(ci_results)
    ci_df.to_csv(OUTPUT_DIR / "concentration_index_complete.csv", index=False)

    # 5. Generate enhanced figures / 鐢熸垚澧炲己鍥捐〃
    print("\n[5] Generating enhanced figures / 鐢熸垚澧炲己鍥捐〃...")

    # 5.1 Enhanced boxplots / 澧炲己鐗堢绾垮浘
    plot_enhanced_boxplot(typical_df, heatwave_df,
                          FIGURES_DIR / "fig_box_hei_imd_enhanced.png")

    # 5.2 Effect-size forest plot / 鏁堝簲閲忔．鏋楀浘
    plot_effect_size_forest(effect_df, FIGURES_DIR / "fig_effect_size_forest.png")

    # 5.3 Concentration-index bar plot / CI 鏉″舰鍥?    if ci_results:
        plot_ci_barplot(ci_df, FIGURES_DIR / "fig_concentration_index_bar.png")

    # 6. LISA hotspot analysis / LISA 鐑偣鍒嗘瀽
    print("\n[6] Running LISA hotspot analysis / LISA 鐑偣鍒嗘瀽...")
    if gdf is not None:
        for scenario, df in [('heatwave', heatwave_df)]:
            gdf_merged = gdf.merge(df[['lsoa11cd', 'hei_mean', 'IMD_Decile']],
                                   left_on='lsoa11cd', right_on='lsoa11cd',
                                   how='inner')

            gdf_lisa, lisa = calculate_local_morans(gdf_merged, 'hei_mean')
            if gdf_lisa is not None:
                # Save LISA cluster counts / 淇濆瓨 LISA 缁撴灉
                lisa_summary = gdf_lisa.groupby('lisa_label').size().reset_index(name='count')
                lisa_summary.to_csv(OUTPUT_DIR / f"lisa_clusters_{scenario}.csv", index=False)
                print(f"  {scenario} LISA clusters / LISA 鑱氱被:")
                for _, row in lisa_summary.iterrows():
                    print(f"    {row['lisa_label']}: {row['count']}")

                # Draw the LISA cluster map / 缁樺埗 LISA 鍦板浘
                plot_lisa_map(gdf_lisa,
                              f'LISA Clusters - HEI ({scenario.title()})',
                              FIGURES_DIR / f"fig_lisa_clusters_{scenario}.png")

    # 7. Print a compact summary / 姹囨€绘姤鍛?    print("\n" + "=" * 60)
    print("Analysis summary / 鍒嗘瀽瀹屾垚姹囨€?)
    print("=" * 60)

    print("\nNew files generated / 鏂扮敓鎴愮殑鏂囦欢:")
    print(f"  - {OUTPUT_DIR / 'morans_i_complete.csv'}")
    print(f"  - {OUTPUT_DIR / 'effect_sizes_cohens_d.csv'}")
    print(f"  - {OUTPUT_DIR / 'concentration_index_complete.csv'}")
    print(f"  - {FIGURES_DIR / 'fig_box_hei_imd_enhanced.png'}")
    print(f"  - {FIGURES_DIR / 'fig_effect_size_forest.png'}")
    print(f"  - {FIGURES_DIR / 'fig_concentration_index_bar.png'}")
    print(f"  - {FIGURES_DIR / 'fig_lisa_clusters_heatwave.png'}")

    return {
        'moran': moran_results,
        'effect_sizes': effect_df,
        'ci': ci_results
    }

if __name__ == "__main__":
    results = main()



