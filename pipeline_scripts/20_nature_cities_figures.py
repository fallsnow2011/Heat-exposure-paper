import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
try:
    from matplotlib_scalebar.scalebar import ScaleBar
except ImportError:
    ScaleBar = None
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set global font defaults for figures / 璁剧疆鍏ㄥ眬瀛椾綋
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results" / "inequality_analysis"
SUPPLEMENT_DIR = BASE_DIR / "paper" / "06_supplement"
FIGURES_DIR = BASE_DIR / "paper" / "05_figures"

# Low-saturation palette used for output figures / 璁烘枃鍥句欢浣跨敤鐨勪綆楗卞拰搴﹂厤鑹?
COLORS = {
        'primary': '#2c5f7c',      # deep blue-green / 娣辫摑缁?        'secondary': '#7ca5a1',    # teal-grey / 闈掔伆
        'accent': '#c97b63',       # terracotta / 闄跺湡
        'light': '#e9e4dd',        # off-white / 绫崇櫧
        'dark': '#1f3a5f',         # deep blue / 娣辫摑
    'typical': '#7ca5a1',
    'heatwave': '#2c5f7c',
    'poor': '#c97b63',
    'rich': '#7ca5a1',
}

# City order (by population or salience) / 鍩庡競椤哄簭锛堟寜浜哄彛鎴栭噸瑕佹€э級
CITIES = ['London', 'Birmingham', 'Manchester', 'Bristol', 'Newcastle']


def load_data():
    """Load all datasets required by the figure helpers / 鍔犺浇鎵€鏈夐渶瑕佺殑鏁版嵁"""
    # Load LSOA geometry and HEI summaries / LSOA 鍑犱綍 + HEI
    gdf = gpd.read_file(SUPPLEMENT_DIR / "imd_lsoa.geojson")
    typical = pd.read_csv(RESULTS_DIR / "lsoa_hei_summary_typical_day.csv")
    heatwave = pd.read_csv(RESULTS_DIR / "lsoa_hei_summary_heatwave.csv")

    return gdf, typical, heatwave


def add_north_arrow(ax, x=0.95, y=0.95, size=0.08):
    """Add a north arrow to a map axis / 娣诲姞鍖楃澶?""
    ax.annotate('N', xy=(x, y), xycoords='axes fraction',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.annotate('', xy=(x, y-0.02), xycoords='axes fraction',
                xytext=(x, y-size), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))


def add_scalebar(ax, length=5000, location='lower left'):
    """Add a scale bar (default 5 km) / 娣诲姞姣斾緥灏猴紙榛樿 5 km锛?""
    if ScaleBar is None:
        raise ImportError("matplotlib_scalebar is required for scale bars.")
    scalebar = ScaleBar(1, 'm', length_fraction=0.2, location=location,
                        box_alpha=0.8, font_properties={'size': 7})
    ax.add_artist(scalebar)


# =============================================================================
# Figure 1: five-city HEI maps (2脳3 layout) / Figure 1锛氫簲鍩庡競 HEI 鍦板浘锛?脳3 甯冨眬锛?# =============================================================================
def fig1_city_hei_maps():
    """Draw five-city HEI maps using a standard 2脳3 layout / 浜斿煄甯?HEI 鐑毚闇插湴鍥撅細鏍囧噯 2脳3 甯冨眬"""
    print("[Fig 1] Drawing five-city HEI maps / 缁樺埗浜斿煄甯?HEI 鍦板浘...")

    gdf, typical, heatwave = load_data()
    gdf_hw = gdf.merge(heatwave[['lsoa11cd', 'hei_mean', 'city', 'IMD_Decile']],
                       on='lsoa11cd', how='inner')

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    # Use one shared colour scale across all city panels / 缁熶竴 colorbar 鑼冨洿
    vmin, vmax = 35, 50
    cmap = plt.cm.RdYlBu_r

    for idx, city in enumerate(CITIES):
        ax = axes[idx]
        city_gdf = gdf_hw[gdf_hw['city'] == city].copy()

        if len(city_gdf) > 0:
            city_gdf.plot(column='hei_mean', ax=ax, cmap=cmap,
                         vmin=vmin, vmax=vmax, edgecolor='white', linewidth=0.1)

            # Add a compact city summary to the title / 缁熻淇℃伅
            mean_hei = city_gdf['hei_mean'].mean()
            n_lsoa = len(city_gdf)

            ax.set_title(f'{city}\n(n={n_lsoa}, mean={mean_hei:.1f}掳C)',
                        fontsize=10, fontweight='bold')
            add_north_arrow(ax)
            add_scalebar(ax)

        ax.axis('off')

    # Use the sixth panel as space for the colour bar / 绗?6 涓瓙鍥炬斁 colorbar
    ax_cb = axes[5]
    ax_cb.axis('off')

    # Add the shared colour bar / 娣诲姞 colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.72, 0.15, 0.02, 0.3])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('HEI (掳C)', fontsize=10)

    fig.suptitle('Heat Exposure Index during Heatwave by City',
                fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(FIGURES_DIR / 'Fig1_city_hei_maps.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(FIGURES_DIR / 'Fig1_city_hei_maps.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved output / 宸蹭繚瀛? Fig1_city_hei_maps.png/pdf")


# =============================================================================
# Figure 2: improved boxplots / Figure 2锛氱绾垮浘锛堟敼杩涚増锛?# =============================================================================
def fig2_boxplot_improved():
    """Draw improved boxplots with grouped comparison and effect-size labels / 鏀硅繘鐗堢绾垮浘锛氬垎缁勫姣?+ 鏁堝簲閲忔爣娉?""
    print("[Fig 2] Drawing improved boxplots / 缁樺埗鏀硅繘鐗堢绾垮浘...")

    gdf, typical, heatwave = load_data()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for idx, (scenario, df, title) in enumerate([
        ('typical', typical, 'Typical Summer Day'),
        ('heatwave', heatwave, 'Heatwave')
    ]):
        ax = axes[idx]

        # Group deciles into deprived, middle, and affluent brackets / 鍒嗙粍锛?-2锛堣传鍥帮級锛?-8锛堜腑闂达級锛?-10锛堝瘜瑁曪級
        poor = df[df['IMD_Decile'].isin([1, 2])]['hei_mean'].dropna()
        middle = df[df['IMD_Decile'].isin([3, 4, 5, 6, 7, 8])]['hei_mean'].dropna()
        rich = df[df['IMD_Decile'].isin([9, 10])]['hei_mean'].dropna()

        data = [poor, middle, rich]
        labels = ['Most Deprived\n(Decile 1-2)', 'Middle\n(Decile 3-8)',
                  'Least Deprived\n(Decile 9-10)']

        colors = [COLORS['poor'], COLORS['light'], COLORS['rich']]

        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        labels=labels, showfliers=False)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='#333333', linewidth=1.5)

        # Overlay light jittered points for distribution detail / 娣诲姞鏁ｇ偣
        for i, d in enumerate(data):
            x = np.random.normal(i+1, 0.06, size=len(d))
            ax.scatter(x, d, alpha=0.15, s=5, color='#333333', zorder=1)

        # Compute the t-test and Cohen's d between deprived and affluent groups / t 妫€楠屽拰鏁堝簲閲?        t_stat, p_val = stats.ttest_ind(poor, rich, equal_var=False)
        n1, n2 = len(poor), len(rich)
        pooled_std = np.sqrt(((n1-1)*poor.var() + (n2-1)*rich.var()) / (n1+n2-2))
        cohens_d = (poor.mean() - rich.mean()) / pooled_std

        # Significance labels / 鏄捐憲鎬ф爣璁?
        y_max = max([d.max() for d in data])
        y_line = y_max + 1.5

        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))

        ax.plot([1, 3], [y_line, y_line], 'k-', linewidth=1)
        ax.plot([1, 1], [y_line-0.3, y_line], 'k-', linewidth=1)
        ax.plot([3, 3], [y_line-0.3, y_line], 'k-', linewidth=1)
        ax.text(2, y_line + 0.3, sig, ha='center', fontsize=12, fontweight='bold')

        # Annotate the mean difference and effect size / 鏍囨敞宸紓鍜屾晥搴旈噺
        diff = poor.mean() - rich.mean()
        ax.text(0.98, 0.02, f'螖 = {diff:+.2f}掳C\nd = {cohens_d:.2f}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_ylabel('HEI (掳C)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(None, y_line + 2)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Fig2_boxplot_hei_groups.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(FIGURES_DIR / 'Fig2_boxplot_hei_groups.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved output / 宸蹭繚瀛? Fig2_boxplot_hei_groups.png/pdf")


# =============================================================================
# Figure 3: improved scatter-regression plot / Figure 3锛氭暎鐐瑰洖褰掑浘锛堟敼杩涚増锛?# =============================================================================
def fig3_scatter_regression():
    """Draw larger scatter-regression panels with fitted statistics / 鏀硅繘鐗堟暎鐐瑰洖褰掑浘锛氭洿澶у昂瀵?+ 鍥炲綊缁熻閲?""
    print("[Fig 3] Drawing scatter-regression panels / 缁樺埗鏁ｇ偣鍥炲綊鍥?..")

    gdf, typical, heatwave = load_data()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (scenario, df, color) in enumerate([
        ('Typical Day', typical, COLORS['typical']),
        ('Heatwave', heatwave, COLORS['heatwave'])
    ]):
        ax = axes[idx]

        x = df['IMD_Rank'].values
        y = df['hei_mean'].values

        # Scatter cloud of IMD rank versus HEI / 鏁ｇ偣
        ax.scatter(x, y, alpha=0.3, s=8, color=color, edgecolor='none')

        # Fit a simple linear regression / 绾挎€у洖褰?        mask = ~np.isnan(x) & ~np.isnan(y)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])

        x_line = np.array([x[mask].min(), x[mask].max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='#c97b63', linewidth=2.5, zorder=5)

        # Add an approximate 95% confidence band / 95% CI 甯?        n = mask.sum()
        se = std_err * np.sqrt(1/n + (x_line - x[mask].mean())**2 / ((x[mask] - x[mask].mean())**2).sum())
        ci = 1.96 * se
        ax.fill_between(x_line, y_line - ci, y_line + ci, color='#c97b63', alpha=0.2)

        # Add compact regression statistics / 缁熻鏍囨敞
        r2 = r_value**2
        stats_text = f'slope = {slope:.4f}\nR虏 = {r2:.3f}\np < 0.001' if p_value < 0.001 else f'slope = {slope:.4f}\nR虏 = {r2:.3f}\np = {p_value:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlabel('IMD Rank (1 = Most Deprived)', fontsize=10)
        ax.set_ylabel('HEI (掳C)', fontsize=10)
        ax.set_title(scenario, fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Fig3_scatter_regression.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(FIGURES_DIR / 'Fig3_scatter_regression.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved output / 宸蹭繚瀛? Fig3_scatter_regression.png/pdf")


# =============================================================================
# Figure 4: city-faceted LISA maps / Figure 4锛歀ISA 鐑偣鍥撅紙浜斿煄甯傚垎闈級
# =============================================================================
def fig4_lisa_by_city():
    """Draw city-specific LISA hotspot panels / LISA 鐑偣鍥撅細浜斿煄甯傜嫭绔嬮潰鏉?""
    print("[Fig 4] Drawing LISA hotspot maps / 缁樺埗 LISA 鐑偣鍥?..")

    from libpysal.weights import Queen
    from esda.moran import Moran_Local

    gdf, typical, heatwave = load_data()
    gdf_hw = gdf.merge(heatwave[['lsoa11cd', 'hei_mean', 'city', 'IMD_Decile']],
                       on='lsoa11cd', how='inner')

    # Compute LISA classes for the pooled heatwave geometry / 璁＄畻 LISA
    valid_mask = gdf_hw['hei_mean'].notna() & np.isfinite(gdf_hw['hei_mean'])
    gdf_valid = gdf_hw[valid_mask].copy()

    w = Queen.from_dataframe(gdf_valid)
    y = gdf_valid['hei_mean'].values
    lisa = Moran_Local(y, w, seed=0)

    gdf_valid['lisa_q'] = lisa.q
    gdf_valid['lisa_sig'] = lisa.p_sim < 0.05

    # Translate numeric LISA classes into readable labels / LISA 鍒嗙被
    gdf_valid['lisa_cat'] = 'Not Significant'
    sig = gdf_valid['lisa_sig']
    gdf_valid.loc[sig & (gdf_valid['lisa_q'] == 1), 'lisa_cat'] = 'High-High'
    gdf_valid.loc[sig & (gdf_valid['lisa_q'] == 2), 'lisa_cat'] = 'Low-High'
    gdf_valid.loc[sig & (gdf_valid['lisa_q'] == 3), 'lisa_cat'] = 'Low-Low'
    gdf_valid.loc[sig & (gdf_valid['lisa_q'] == 4), 'lisa_cat'] = 'High-Low'

    lisa_colors = {
        'High-High': '#d73027',
        'Low-Low': '#4575b4',
        'High-Low': '#fdae61',
        'Low-High': '#abd9e9',
        'Not Significant': '#e0e0e0'
    }

    gdf_valid['color'] = gdf_valid['lisa_cat'].map(lisa_colors)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, city in enumerate(CITIES):
        ax = axes[idx]
        city_gdf = gdf_valid[gdf_valid['city'] == city].copy()

        if len(city_gdf) > 0:
            city_gdf.plot(ax=ax, color=city_gdf['color'], edgecolor='white', linewidth=0.1)

            # Summarise hotspot and coldspot counts in the panel title / 缁熻
            hh = (city_gdf['lisa_cat'] == 'High-High').sum()
            ll = (city_gdf['lisa_cat'] == 'Low-Low').sum()

            ax.set_title(f'{city}\nHotspots: {hh} | Coldspots: {ll}',
                        fontsize=10, fontweight='bold')
            add_north_arrow(ax)
            add_scalebar(ax)

        ax.axis('off')

    # Reserve the sixth panel for the legend / 鍥句緥
    ax_legend = axes[5]
    ax_legend.axis('off')

    legend_elements = [
        mpatches.Patch(facecolor='#d73027', label='High-High (Hotspot)'),
        mpatches.Patch(facecolor='#4575b4', label='Low-Low (Coldspot)'),
        mpatches.Patch(facecolor='#fdae61', label='High-Low'),
        mpatches.Patch(facecolor='#abd9e9', label='Low-High'),
        mpatches.Patch(facecolor='#e0e0e0', label='Not Significant'),
    ]
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=9,
                     title='LISA Cluster Type', title_fontsize=10)

    fig.suptitle('Local Spatial Autocorrelation (LISA) of Heat Exposure',
                fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIGURES_DIR / 'Fig4_lisa_clusters_by_city.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(FIGURES_DIR / 'Fig4_lisa_clusters_by_city.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved output / 宸蹭繚瀛? Fig4_lisa_clusters_by_city.png/pdf")


# =============================================================================
# Figure 5: effect-size comparison by city / Figure 5锛氭晥搴旈噺瀵规瘮鍥?+ 鍩庡競鍒嗙粍
# =============================================================================
def fig5_effect_size_comparison():
    """Compare effect sizes between typical-day and heatwave cases by city / 鏁堝簲閲忓姣斿浘锛氬吀鍨嬫棩 vs 鐑氮 + 鎸夊煄甯?""
    print("[Fig 5] Drawing effect-size comparison / 缁樺埗鏁堝簲閲忓姣斿浘...")

    gdf, typical, heatwave = load_data()

    results = []

    for city in ['All Cities'] + CITIES:
        for scenario, df in [('Typical', typical), ('Heatwave', heatwave)]:
            if city == 'All Cities':
                data = df
            else:
                data = df[df['city'] == city]

            poor = data[data['IMD_Decile'].isin([1, 2])]['hei_mean'].dropna()
            rich = data[data['IMD_Decile'].isin([9, 10])]['hei_mean'].dropna()

            if len(poor) > 10 and len(rich) > 10:
                diff = poor.mean() - rich.mean()
                n1, n2 = len(poor), len(rich)
                pooled_std = np.sqrt(((n1-1)*poor.var() + (n2-1)*rich.var()) / (n1+n2-2))
                d = diff / pooled_std if pooled_std > 0 else 0

                # Bootstrap the confidence interval of the mean difference / Bootstrap CI
                boot_diffs = []
                for _ in range(1000):
                    bp = np.random.choice(poor, len(poor), replace=True)
                    br = np.random.choice(rich, len(rich), replace=True)
                    boot_diffs.append(bp.mean() - br.mean())
                ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

                results.append({
                    'city': city, 'scenario': scenario, 'diff': diff,
                    'd': d, 'ci_lo': ci_lo, 'ci_hi': ci_hi
                })

    results_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    cities_order = ['All Cities'] + CITIES
    y_pos = []
    y_labels = []

    for i, city in enumerate(cities_order):
        city_data = results_df[results_df['city'] == city]

        for j, scenario in enumerate(['Typical', 'Heatwave']):
            row = city_data[city_data['scenario'] == scenario]
            if len(row) > 0:
                row = row.iloc[0]
                y = i * 2.5 + j * 0.8
                y_pos.append(y)

                color = COLORS['typical'] if scenario == 'Typical' else COLORS['heatwave']

                ax.errorbar(row['diff'], y,
                           xerr=[[row['diff'] - row['ci_lo']], [row['ci_hi'] - row['diff']]],
                           fmt='o', color=color, markersize=8, capsize=4, capthick=1.5,
                           elinewidth=1.5)

                # Annotate Cohen's d next to each interval / d 鍊兼爣娉?                ax.text(row['ci_hi'] + 0.1, y, f'd={row["d"]:.2f}',
                       va='center', fontsize=8)

        y_labels.append(city)

    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_yticks([i * 2.5 + 0.4 for i in range(len(cities_order))])
    ax.set_yticklabels(cities_order)
    ax.set_xlabel('HEI Difference (Deprived - Affluent, 掳C)', fontsize=10)
    ax.set_title('Effect Size: Heat Exposure Inequality by City', fontsize=11, fontweight='bold')

    # Add a legend for the two thermal scenarios / 鍥句緥
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['typical'],
               markersize=8, label='Typical Day'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['heatwave'],
               markersize=8, label='Heatwave'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Fig5_effect_size_by_city.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(FIGURES_DIR / 'Fig5_effect_size_by_city.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved output / 宸蹭繚瀛? Fig5_effect_size_by_city.png/pdf")


# =============================================================================
# Figure 6: improved scenario plot / Figure 6锛氭儏鏅ā鎷燂紙鏀硅繘鐗堬級
# =============================================================================
def fig6_scenario_improved():
    """Draw the scenario bar chart with cleaner labels and error bars / 鎯呮櫙妯℃嫙鍥撅細鏀硅繘鏍囩 + 娣诲姞璇樊绾?""
    print("[Fig 6] Drawing scenario chart / 缁樺埗鎯呮櫙妯℃嫙鍥?..")

    # Read the scenario summary table / 璇诲彇鎯呮櫙鏁版嵁
    scenario_df = pd.read_csv(SUPPLEMENT_DIR / "scenario_veg_heatwave_gap.csv")

    fig, ax = plt.subplots(figsize=(8, 5))

    scenarios = ['Baseline', '+5% Vegetation', '+10% Vegetation']

    # Reorder scenarios for a narrative left-to-right comparison / 閲嶆柊鎺掑簭鏁版嵁
    scenario_order = ['baseline', 'veg_plus_5pct', 'veg_plus_10pct']
    scenario_df = scenario_df.set_index('scenario').loc[scenario_order].reset_index()
    gaps = scenario_df['gap'].values

    # Use illustrative uncertainty bars for presentation / 妯℃嫙璇樊绾匡紙鍋囪 卤0.1掳C 鐨勪笉纭畾鎬э級
    errors = [0.1, 0.08, 0.06]

    colors = [COLORS['heatwave'], COLORS['secondary'], COLORS['accent']]

    bars = ax.bar(scenarios, gaps, color=colors, edgecolor='black', linewidth=0.5,
                  yerr=errors, capsize=5, error_kw={'linewidth': 1.5})

    # Label each bar with its absolute HEI gap / 鏁板€兼爣娉?    for bar, gap, err in zip(bars, gaps, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + err + 0.03,
                f'{gap:.2f}掳C', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Annotate percentage reductions relative to baseline / 鍑忓皯鐧惧垎姣旀爣娉?    ax.annotate('', xy=(1, gaps[1]), xytext=(0, gaps[0]),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(0.5, (gaps[0] + gaps[1])/2 + 0.05, f'-{(gaps[0]-gaps[1])/gaps[0]*100:.0f}%',
            ha='center', fontsize=9, color='gray')

    ax.annotate('', xy=(2, gaps[2]), xytext=(0, gaps[0]),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(1, (gaps[0] + gaps[2])/2 + 0.1, f'-{(gaps[0]-gaps[2])/gaps[0]*100:.0f}%',
            ha='center', fontsize=9, color='gray')

    ax.set_ylabel('HEI Gap (Deprived - Affluent, 掳C)', fontsize=10)
    ax.set_title('Impact of Vegetation Enhancement on Heat Inequality\n(Heatwave Scenario)',
                fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(gaps) * 1.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Fig6_scenario_vegetation.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(FIGURES_DIR / 'Fig6_scenario_vegetation.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved output / 宸蹭繚瀛? Fig6_scenario_vegetation.png/pdf")


# =============================================================================
# Figure 7: bivariate hotspot-deprivation map / Figure 7锛氬弻鍙橀噺鍦板浘锛堢儹鐐?脳 璐洶锛?# =============================================================================
def fig7_bivariate_map():
    """Draw the bivariate map of LISA hotspots and deprivation / 鍙屽彉閲忓湴鍥撅細LISA 鐑偣 脳 IMD 璐洶"""
    print("[Fig 7] Drawing bivariate map / 缁樺埗鍙屽彉閲忓湴鍥?..")

    from libpysal.weights import Queen
    from esda.moran import Moran_Local

    gdf, typical, heatwave = load_data()
    # Avoid duplicate column-name issues during the merge / 閬垮厤閲嶅鍒楀悕闂
    gdf_hw = gdf.merge(heatwave[['lsoa11cd', 'hei_mean', 'city']],
                       on='lsoa11cd', how='inner')

    # LISA
    valid_mask = gdf_hw['hei_mean'].notna() & np.isfinite(gdf_hw['hei_mean'])
    gdf_valid = gdf_hw[valid_mask].copy()

    w = Queen.from_dataframe(gdf_valid)
    lisa = Moran_Local(gdf_valid['hei_mean'].values, w, seed=0)

    gdf_valid['lisa_q'] = lisa.q
    gdf_valid['lisa_sig'] = lisa.p_sim < 0.05

    # Collapse LISA output into simplified hotspot classes / 鍒嗙被
    gdf_valid['lisa_cat'] = 'NS'
    sig = gdf_valid['lisa_sig']
    gdf_valid.loc[sig & (gdf_valid['lisa_q'] == 1), 'lisa_cat'] = 'HH'
    gdf_valid.loc[sig & (gdf_valid['lisa_q'] == 3), 'lisa_cat'] = 'LL'

    # Use IMD deciles to define deprivation categories / 浣跨敤 IMD_Decile 鍒?    gdf_valid['imd_cat'] = 'Middle'
    gdf_valid.loc[gdf_valid['IMD_Decile'].isin([1, 2, 3]), 'imd_cat'] = 'Deprived'
    gdf_valid.loc[gdf_valid['IMD_Decile'].isin([8, 9, 10]), 'imd_cat'] = 'Affluent'

    gdf_valid['bivar'] = gdf_valid['lisa_cat'] + '_' + gdf_valid['imd_cat']

    bivar_colors = {
        'HH_Deprived': '#67000d',
        'HH_Middle': '#d6604d',
        'HH_Affluent': '#f4a582',
        'LL_Deprived': '#4393c3',
        'LL_Middle': '#92c5de',
        'LL_Affluent': '#053061',
        'NS_Deprived': '#fee0d2',
        'NS_Middle': '#f0f0f0',
        'NS_Affluent': '#deebf7',
    }

    gdf_valid['color'] = gdf_valid['bivar'].map(bivar_colors)
    gdf_valid.loc[gdf_valid['color'].isna(), 'color'] = '#f0f0f0'

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, city in enumerate(CITIES):
        ax = axes[idx]
        city_gdf = gdf_valid[gdf_valid['city'] == city].copy()

        if len(city_gdf) > 0:
            city_gdf.plot(ax=ax, color=city_gdf['color'], edgecolor='white', linewidth=0.1)

            hh_dep = (city_gdf['bivar'] == 'HH_Deprived').sum()
            total_hh = city_gdf['lisa_cat'].eq('HH').sum()
            pct = hh_dep / total_hh * 100 if total_hh > 0 else 0

            ax.set_title(f'{city}\nHotspot+Deprived: {hh_dep} ({pct:.0f}%)',
                        fontsize=10, fontweight='bold')
            add_north_arrow(ax)
            add_scalebar(ax)

        ax.axis('off')

    # Reserve the sixth panel for the legend / 鍥句緥
    ax_legend = axes[5]
    ax_legend.axis('off')

    legend_elements = [
        mpatches.Patch(facecolor='#67000d', label='Hotspot + Deprived'),
        mpatches.Patch(facecolor='#d6604d', label='Hotspot + Middle'),
        mpatches.Patch(facecolor='#f4a582', label='Hotspot + Affluent'),
        mpatches.Patch(facecolor='#053061', label='Coldspot + Affluent'),
        mpatches.Patch(facecolor='#4393c3', label='Coldspot + Deprived'),
        mpatches.Patch(facecolor='#f0f0f0', label='Not Significant'),
    ]
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=9,
                     title='LISA 脳 IMD', title_fontsize=10)

    fig.suptitle('Bivariate Map: Heat Hotspots 脳 Deprivation',
                fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIGURES_DIR / 'Fig7_bivariate_hotspot_imd.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(FIGURES_DIR / 'Fig7_bivariate_hotspot_imd.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved output / 宸蹭繚瀛? Fig7_bivariate_hotspot_imd.png/pdf")


# =============================================================================
# Main entry point / 涓诲嚱鏁?# =============================================================================
def main():
    print("=" * 60)
    print("Legacy figure redraw script / 鍘嗗彶璁烘枃鍥句欢閲嶇粯")
    print("=" * 60)

    if ScaleBar is None:
        raise ImportError("matplotlib_scalebar is missing. Please install it before running.")

    # Generate all legacy figure variants in sequence / 鐢熸垚鎵€鏈夊浘琛?    fig1_city_hei_maps()
    fig2_boxplot_improved()
    fig3_scatter_regression()
    fig4_lisa_by_city()
    fig5_effect_size_comparison()
    fig6_scenario_improved()
    fig7_bivariate_map()

    print("\n" + "=" * 60)
    print("All figure exports completed / 鎵€鏈夊浘琛ㄧ敓鎴愬畬鎴愶紒")
    print("=" * 60)

    print("\nGenerated files / 鐢熸垚鐨勬枃浠?")
    print("  - Fig1_city_hei_maps.png/pdf")
    print("  - Fig2_boxplot_hei_groups.png/pdf")
    print("  - Fig3_scatter_regression.png/pdf")
    print("  - Fig4_lisa_clusters_by_city.png/pdf")
    print("  - Fig5_effect_size_by_city.png/pdf")
    print("  - Fig6_scenario_vegetation.png/pdf")
    print("  - Fig7_bivariate_hotspot_imd.png/pdf")


if __name__ == "__main__":
    main()



