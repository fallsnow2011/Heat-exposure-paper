import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parents[1]


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "paper0128").is_dir():
            return parent
    return BASE_DIR


REPO_ROOT = find_repo_root()


def resolve_large_input(filename: str, *, env_var: str | None = None) -> Path:
    candidates: list[Path] = []

    if env_var:
        env_value = os.environ.get(env_var, "").strip()
        if env_value:
            candidates.append(Path(env_value).expanduser())

    candidates.extend(
        [
            BASE_DIR / filename,
            REPO_ROOT / filename,
            REPO_ROOT / "paper0128" / filename,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    msg = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"Missing required input file: {filename}\nTried:\n{msg}\n"
        "Tip: keep a single copy at repo root (recommended) or under paper0128/."
    )


LCZ_PATH = resolve_large_input("lcz_filter_v3.tif", env_var="LCZ_TIF")
RESULTS_DIR = BASE_DIR / "results" / "inequality_analysis"
SUPPLEMENT_DIR = BASE_DIR / "paper" / "06_supplement"
FIGURES_DIR = BASE_DIR / "paper" / "05_figures"

# LCZ names and colours / LCZ 鍚嶇О鍜岄鑹?LCZ_NAMES = {
    1: 'Compact high-rise',
    2: 'Compact mid-rise',
    3: 'Compact low-rise',
    4: 'Open high-rise',
    5: 'Open mid-rise',
    6: 'Open low-rise',
    7: 'Lightweight low-rise',
    8: 'Large low-rise',
    9: 'Sparsely built',
    10: 'Heavy industry',
    11: 'Dense trees',
    12: 'Scattered trees',
    13: 'Bush/scrub',
    14: 'Low plants',
    15: 'Bare rock/paved',
    16: 'Bare soil/sand',
    17: 'Water'
}

# Official WUDAPT LCZ colors
LCZ_COLORS = {
    1: '#910613',  # Compact high-rise
    2: '#D9081C',  # Compact mid-rise
    3: '#FF0A22',  # Compact low-rise
    4: '#C54F1E',  # Open high-rise
    5: '#FF6628',  # Open mid-rise
    6: '#FF985E',  # Open low-rise
    7: '#FDED3F',  # Lightweight low-rise
    8: '#BBBBBB',  # Large low-rise
    9: '#FFCBAB',  # Sparsely built
    10: '#565656', # Heavy industry
    11: '#006A18', # Dense trees
    12: '#00A926', # Scattered trees
    13: '#628432', # Bush/scrub
    14: '#B5DA7F', # Low plants
    15: '#000000', # Bare rock/paved
    16: '#FCF7B1', # Bare soil/sand
    17: '#656BFA'  # Water
}

# Simplified LCZ groups used in plots and summaries / 绠€鍖栧垎缁?LCZ_GROUPS = {
    'Compact Built': [1, 2, 3],
    'Open Built': [4, 5, 6],
    'Industry/Large': [7, 8, 10],
    'Sparse/Natural': [9, 11, 12, 13, 14],
    'Other': [15, 16, 17]
}

GROUP_COLORS = {
    'Compact Built': '#d10000',
    'Open Built': '#ff9955',
    'Industry/Large': '#bcbcbc',
    'Sparse/Natural': '#00aa00',
    'Other': '#6a6aff'
}


def load_lsoa_boundaries():
    """Load LSOA boundaries from the preferred local sources / 鍔犺浇 LSOA 杈圭晫"""
    lsoa_path = BASE_DIR / "results" / "merged_results" / "all_cities_lsoa_results.gpkg"
    if lsoa_path.exists():
        return gpd.read_file(lsoa_path)

    # Fallback path when the geopackage is unavailable / 澶囬€夎矾寰?    lsoa_path = BASE_DIR / "results" / "inequality_analysis" / "lsoa_hei_summary_typical_day.csv"
    if lsoa_path.exists():
        df = pd.read_csv(lsoa_path)
        # Recover geometries from the IMD lookup file / 闇€瑕佸嚑浣曟暟鎹?        imd_path = resolve_large_input(
            "Index_of_Multiple_Deprivation_(Dec_2019)_Lookup_in_England.geojson",
            env_var="IMD_LOOKUP_GEOJSON",
        )
        gdf = gpd.read_file(imd_path)
        gdf = gdf.merge(df, left_on='lsoa11cd', right_on='LSOA11CD', how='inner')
        return gdf

    return None


def extract_lcz_for_lsoa(lsoa_gdf, lcz_path):
    """Extract LCZ composition statistics for each LSOA / 涓烘瘡涓?LSOA 鎻愬彇 LCZ 缁熻"""
    print("Extracting LCZ distributions for each LSOA / 鎻愬彇 LSOA 鐨?LCZ 鍒嗗竷...")

    results = []

    with rasterio.open(lcz_path) as src:
        # Reproject LSOAs to the raster CRS when needed / 纭繚 CRS 鍖归厤鍒?LCZ 鏍呮牸
        if src.crs is None:
            raise ValueError("LCZ raster CRS is missing.")
        if lsoa_gdf.crs != src.crs:
            lsoa_gdf = lsoa_gdf.to_crs(src.crs)

        for idx, row in lsoa_gdf.iterrows():
            if idx % 500 == 0:
                print(f"  Processing / 澶勭悊 {idx}/{len(lsoa_gdf)}...")

            try:
                # Clip the LCZ raster to one LSOA polygon / 瑁佸壀 LCZ 鍒?LSOA
                geom = [row.geometry.__geo_interface__]
                out_image, out_transform = mask(src, geom, crop=True, nodata=0)
                data = out_image[0]

                # Count LCZ pixels within the polygon / 缁熻鍚?LCZ 绫诲瀷鍍忕礌鏁?                valid_data = data[data > 0]
                if len(valid_data) == 0:
                    continue

                counts = Counter(valid_data)
                total = sum(counts.values())

                result = {
                    'lsoa_code': row.get('lsoa11cd', row.get('LSOA11CD', idx)),
                    'total_pixels': total
                }

                # Percentage share of each LCZ type / 鍚?LCZ 绫诲瀷鍗犳瘮
                for lcz in range(1, 18):
                    result[f'lcz_{lcz}_pct'] = counts.get(lcz, 0) / total * 100

                # Dominant LCZ class / 涓诲 LCZ
                if counts:
                    dominant_lcz = max(counts, key=counts.get)
                    result['dominant_lcz'] = dominant_lcz
                    result['dominant_lcz_name'] = LCZ_NAMES.get(dominant_lcz, 'Unknown')

                # Grouped LCZ shares for simpler interpretation / 鍒嗙粍缁熻
                for group_name, lcz_list in LCZ_GROUPS.items():
                    group_pct = sum(counts.get(lcz, 0) for lcz in lcz_list) / total * 100
                    result[f'group_{group_name.replace("/", "_").replace(" ", "_")}_pct'] = group_pct

                results.append(result)

            except Exception as e:
                continue

    return pd.DataFrame(results)


def analyze_lcz_by_imd(lcz_df, lsoa_df):
    """Analyse the relationship between LCZ patterns and IMD / 鍒嗘瀽 LCZ 涓?IMD 鐨勫叧绯?""
    print("\nAnalysing LCZ versus IMD deprivation / 鍒嗘瀽 LCZ 涓?IMD 璐洶鎸囨暟鐨勫叧绯?..")

    # Merge LCZ summaries with LSOA thermal metrics / 鍚堝苟鏁版嵁
    merged = lcz_df.merge(
        lsoa_df[['lsoa11cd', 'IMD_Decile', 'hei_mean', 'lst_mean', 'TotPop']],
        left_on='lsoa_code',
        right_on='lsoa11cd',
        how='inner'
    )

    print(f"  Merged sample size / 鍚堝苟鍚庢牱鏈暟: {len(merged)}")

    # Summarise LCZ composition by IMD decile / 鎸?IMD Decile 缁熻 LCZ 鍒嗗竷
    lcz_by_imd = []
    for decile in range(1, 11):
        subset = merged[merged['IMD_Decile'] == decile]
        if len(subset) == 0:
            continue

        row = {'IMD_Decile': decile, 'n_lsoa': len(subset)}

        # Group-level LCZ composition / 鍒嗙粍缁熻
        for group_name in LCZ_GROUPS.keys():
            col = f'group_{group_name.replace("/", "_").replace(" ", "_")}_pct'
            if col in subset.columns:
                row[group_name] = subset[col].mean()

        # Average HEI in the decile / 骞冲潎 HEI
        row['mean_hei'] = subset['hei_mean'].mean()

        lcz_by_imd.append(row)

    lcz_imd_df = pd.DataFrame(lcz_by_imd)

    # Print the decile table for quick inspection / 鎵撳嵃缁撴灉
    print("\nLCZ distribution by IMD decile / LCZ 鍒嗗竷 by IMD Decile:")
    print(lcz_imd_df.to_string(index=False))

    return merged, lcz_imd_df


def analyze_lcz_hei_relationship(merged_df, min_sample=50):
    """
    Analyse the relationship between dominant LCZ and HEI /
    鍒嗘瀽 LCZ 涓?HEI 鐨勫叧绯?
    Add a minimum-sample threshold and flag categories below that threshold /
    淇锛氭坊鍔犳牱鏈噺闃堝€硷紝浣庝簬闃堝€肩殑鏍囪涓轰笉鍙潬

    Parameters:
    -----------
    merged_df : DataFrame
        Merged LCZ-HEI table / 鍚堝苟鍚庣殑鏁版嵁
    min_sample : int
        Minimum sample-size threshold, default 50 / 鏈€灏忔牱鏈噺闃堝€硷紝榛樿 50
    """
    print("\nAnalysing LCZ versus HEI / 鍒嗘瀽 LCZ 涓?HEI 鐑毚闇茬殑鍏崇郴...")
    print(f"  Sample threshold / 鏍锋湰閲忛槇鍊? n 鈮?{min_sample}")

    # Summarise HEI by dominant LCZ class / 鎸変富瀵?LCZ 缁熻 HEI
    hei_by_lcz = merged_df.groupby('dominant_lcz').agg({
        'hei_mean': ['mean', 'std', 'count'],
        'IMD_Decile': 'mean'
    }).round(3)

    hei_by_lcz.columns = ['hei_mean', 'hei_std', 'n_lsoa', 'mean_imd_decile']
    hei_by_lcz['lcz_name'] = hei_by_lcz.index.map(LCZ_NAMES)

    # Flag classes with insufficient sample size / 娣诲姞鍙潬鎬ф爣璁?    hei_by_lcz['reliable'] = hei_by_lcz['n_lsoa'] >= min_sample
    hei_by_lcz['note'] = hei_by_lcz.apply(
        lambda x: '' if x['reliable'] else f'Low sample / 鏍锋湰涓嶈冻(n<{min_sample})',
        axis=1
    )

    hei_by_lcz = hei_by_lcz.reset_index()

    print("\nHEI by dominant LCZ / HEI by Dominant LCZ:")
    print("  [*] marks low-sample classes; interpret cautiously / [*] 琛ㄧず鏍锋湰涓嶈冻锛岀粨鏋滀粎渚涘弬鑰?)
    for _, row in hei_by_lcz.sort_values('hei_mean', ascending=False).iterrows():
        marker = '   ' if row['reliable'] else ' * '
        print(f"{marker}LCZ {int(row['dominant_lcz']):2d} ({row['lcz_name']:20s}): "
              f"HEI={row['hei_mean']:.1f}掳C, n={int(row['n_lsoa']):4d}, IMD={row['mean_imd_decile']:.1f}")

    # Count reliable versus low-sample categories / 缁熻鍙潬/涓嶅彲闈犳暟閲?    n_reliable = hei_by_lcz['reliable'].sum()
    n_unreliable = (~hei_by_lcz['reliable']).sum()
    print(f"\n  Reliable classes / 鍙潬绫诲埆: {n_reliable}, low-sample classes / 鏍锋湰涓嶈冻绫诲埆: {n_unreliable}")

    return hei_by_lcz


def plot_lcz_imd_distribution(lcz_imd_df):
    """Plot LCZ composition by IMD decile / 缁樺埗 LCZ 鍒嗗竷 by IMD"""
    print("\nDrawing LCZ-IMD distribution plot / 缁樺埗 LCZ-IMD 鍒嗗竷鍥?..")

    fig, ax = plt.subplots(figsize=(12, 6))

    groups = ['Compact Built', 'Open Built', 'Industry/Large', 'Sparse/Natural']
    x = np.arange(len(lcz_imd_df))
    width = 0.2

    for i, group in enumerate(groups):
        if group in lcz_imd_df.columns:
            bars = ax.bar(x + i*width, lcz_imd_df[group], width,
                         label=group, color=GROUP_COLORS[group], alpha=0.8)

    ax.set_xlabel('IMD Decile (1 = Most Deprived)', fontsize=11)
    ax.set_ylabel('Land Cover (%)', fontsize=11)
    ax.set_title('Local Climate Zone Distribution by Deprivation Level', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(lcz_imd_df['IMD_Decile'].astype(int))
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Fig_lcz_by_imd.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'Fig_lcz_by_imd.pdf', bbox_inches='tight')
    plt.close()
    print("Saved output / 宸蹭繚瀛? Fig_lcz_by_imd.png/pdf")


def plot_lcz_hei_boxplot(merged_df):
    """Plot HEI boxplots for dominant LCZ classes / 缁樺埗鍚?LCZ 绫诲瀷鐨?HEI 绠辩嚎鍥?""
    print("\nDrawing LCZ-HEI boxplots / 缁樺埗 LCZ-HEI 绠辩嚎鍥?..")

    # Restrict the plot to LCZ classes with enough observations / 鍙€夋嫨鏈夎冻澶熸牱鏈殑 LCZ 绫诲瀷
    lcz_counts = merged_df['dominant_lcz'].value_counts()
    valid_lcz = lcz_counts[lcz_counts >= 30].index.tolist()

    fig, ax = plt.subplots(figsize=(14, 6))

    data = [merged_df[merged_df['dominant_lcz'] == lcz]['hei_mean'].dropna().values
            for lcz in sorted(valid_lcz)]
    labels = [LCZ_NAMES.get(lcz, str(lcz)) for lcz in sorted(valid_lcz)]
    colors = [LCZ_COLORS.get(lcz, '#888888') for lcz in sorted(valid_lcz)]

    bp = ax.boxplot(data, patch_artist=True, showfliers=False)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Heat Exposure Index (掳C)', fontsize=11)
    ax.set_title('Heat Exposure by Local Climate Zone', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Fig_lcz_hei_boxplot.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'Fig_lcz_hei_boxplot.pdf', bbox_inches='tight')
    plt.close()
    print("Saved output / 宸蹭繚瀛? Fig_lcz_hei_boxplot.png/pdf")


def plot_compact_vs_open_inequality(merged_df):
    """Compare compact-versus-open urban form in relation to inequality / 缁樺埗绱у噾鍨?vs 寮€鏀惧瀷寤虹瓚鐨勪笉骞崇瓑瀵规瘮"""
    print("\nDrawing compact-versus-open inequality comparison / 缁樺埗绱у噾鍨?vs 寮€鏀惧瀷寤虹瓚涓嶅钩绛夊姣?..")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: compact-built share versus IMD / 宸﹀浘锛欳ompact Built 鍗犳瘮 vs IMD
    ax1 = axes[0]
    compact_col = 'group_Compact_Built_pct'
    if compact_col in merged_df.columns:
        by_imd = merged_df.groupby('IMD_Decile')[compact_col].mean()
        ax1.bar(by_imd.index, by_imd.values, color='#d10000', alpha=0.8)
        ax1.set_xlabel('IMD Decile (1 = Most Deprived)', fontsize=11)
        ax1.set_ylabel('Compact Built Coverage (%)', fontsize=11)
        ax1.set_title('Compact Urban Form by Deprivation', fontsize=12, fontweight='bold')

        # Add a linear trend line / 娣诲姞瓒嬪娍绾?        z = np.polyfit(by_imd.index, by_imd.values, 1)
        p = np.poly1d(z)
        ax1.plot(by_imd.index, p(by_imd.index), 'k--', linewidth=2, alpha=0.7)

        # Annotate the correlation coefficient / 鏍囨敞鐩稿叧鎬?        corr = np.corrcoef(by_imd.index, by_imd.values)[0, 1]
        ax1.text(0.95, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes,
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Right panel: HEI versus compact-built share / 鍙冲浘锛欻EI vs Compact Built
    ax2 = axes[1]
    if compact_col in merged_df.columns:
        # Split LSOAs into high-compactness and low-compactness groups / 鍒嗕负楂?浣?Compact 缁?        median_compact = merged_df[compact_col].median()
        high_compact = merged_df[merged_df[compact_col] >= median_compact]
        low_compact = merged_df[merged_df[compact_col] < median_compact]

        data = [low_compact['hei_mean'].dropna().values,
                high_compact['hei_mean'].dropna().values]

        bp = ax2.boxplot(data, patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('#00aa00')
        bp['boxes'][1].set_facecolor('#d10000')

        ax2.set_xticklabels(['Low Compact\n(Open/Green)', 'High Compact\n(Dense Built)'])
        ax2.set_ylabel('Heat Exposure Index (掳C)', fontsize=11)
        ax2.set_title('HEI by Urban Compactness', fontsize=12, fontweight='bold')

        # Highlight the mean HEI difference / 鏍囨敞宸紓
        diff = high_compact['hei_mean'].mean() - low_compact['hei_mean'].mean()
        ax2.text(0.5, 0.95, f'螖 = {diff:+.2f}掳C', transform=ax2.transAxes,
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'Fig_urban_compactness_inequality.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'Fig_urban_compactness_inequality.pdf', bbox_inches='tight')
    plt.close()
    print("Saved output / 宸蹭繚瀛? Fig_urban_compactness_inequality.png/pdf")


def main():
    print("=" * 60)
    print("LCZ urban-morphology analysis / LCZ 鍩庡競褰㈡€佸垎鏋?)
    print("=" * 60)

    # Load LSOA thermal summaries / 鍔犺浇 LSOA 鏁版嵁
    lsoa_df = pd.read_csv(RESULTS_DIR / "lsoa_hei_summary_typical_day.csv")
    print(f"Loaded LSOA summaries / 鍔犺浇 LSOA 鏁版嵁: {len(lsoa_df)} 鏉¤褰?)

    # Load LSOA geometries from the IMD geopackage / 鍔犺浇 LSOA 鍑犱綍杈圭晫锛堜娇鐢ㄦ湁鍑犱綍鏁版嵁鐨?gpkg锛?    imd_path = BASE_DIR / "city_boundaries" / "Indices_of_Multiple_Deprivation_(IMD)_2019_8404722932957776031.gpkg"
    lsoa_gdf = gpd.read_file(imd_path)
    print(f"Loaded LSOA boundaries / 鍔犺浇 LSOA 杈圭晫: {len(lsoa_gdf)} 涓?LSOA (CRS: {lsoa_gdf.crs})")

    # Keep only LSOAs with HEI summaries / 鍙繚鐣欐湁 HEI 鏁版嵁鐨?LSOA
    lsoa_gdf = lsoa_gdf[lsoa_gdf['lsoa11cd'].isin(lsoa_df['lsoa11cd'])]
    print(f"Matched geometries / 鍖归厤鍑犱綍杈圭晫: {len(lsoa_gdf)} 涓?LSOA")

    # Load LCZ composition from cache when available / 鎻愬彇 LCZ
    lcz_cache = SUPPLEMENT_DIR / "lsoa_lcz_distribution.csv"
    if lcz_cache.exists():
        print("Loading cached LCZ summaries / 浠庣紦瀛樺姞杞?LCZ 鏁版嵁...")
        lcz_df = pd.read_csv(lcz_cache)
    else:
        lcz_df = extract_lcz_for_lsoa(lsoa_gdf, LCZ_PATH)
        lcz_df.to_csv(lcz_cache, index=False)
        print(f"Saved LCZ cache / 宸蹭繚瀛?LCZ 缂撳瓨: {lcz_cache}")

    print(f"LCZ records / LCZ 鏁版嵁: {len(lcz_df)} 涓?LSOA")

    # Analyse LCZ-IMD relationships / 鍒嗘瀽 LCZ-IMD 鍏崇郴
    merged_df, lcz_imd_df = analyze_lcz_by_imd(lcz_df, lsoa_df)

    # Analyse LCZ-HEI relationships / 鍒嗘瀽 LCZ-HEI 鍏崇郴
    hei_by_lcz = analyze_lcz_hei_relationship(merged_df)

    # Save tabular outputs / 淇濆瓨缁撴灉
    lcz_imd_df.to_csv(SUPPLEMENT_DIR / "lcz_distribution_by_imd.csv", index=False)
    hei_by_lcz.to_csv(SUPPLEMENT_DIR / "hei_by_dominant_lcz.csv", index=False)
    merged_df.to_csv(SUPPLEMENT_DIR / "lsoa_lcz_hei_merged.csv", index=False)

    # Generate summary plots / 缁樺浘
    plot_lcz_imd_distribution(lcz_imd_df)
    plot_lcz_hei_boxplot(merged_df)
    plot_compact_vs_open_inequality(merged_df)

    print("\n" + "=" * 60)
    print("Analysis complete / 鍒嗘瀽瀹屾垚锛?)
    print("=" * 60)

    print("\nGenerated files / 鐢熸垚鐨勬枃浠?")
    print("  Data / 鏁版嵁:")
    print("    - lsoa_lcz_distribution.csv")
    print("    - lcz_distribution_by_imd.csv")
    print("    - hei_by_dominant_lcz.csv")
    print("    - lsoa_lcz_hei_merged.csv")
    print("  Figures / 鍥捐〃:")
    print("    - Fig_lcz_by_imd.png/pdf")
    print("    - Fig_lcz_hei_boxplot.png/pdf")
    print("    - Fig_urban_compactness_inequality.png/pdf")


if __name__ == "__main__":
    main()



