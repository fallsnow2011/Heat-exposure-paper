#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ============ 璺緞璁剧疆 ============
# Repo root is 3 levels up from this file.
BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / 'results'
SUPPLEMENT_DIR = BASE_DIR / 'paper' / '06_supplement'

# ============ 鍩庡競鍏冩暟鎹?============
CITY_META = {
    'London': {'lat': 51.51, 'type': 'Capital', 'region': 'South East'},
    'Bristol': {'lat': 51.45, 'type': 'Regional hub', 'region': 'South West'},
    'Birmingham': {'lat': 52.48, 'type': 'Industrial', 'region': 'West Midlands'},
    'Manchester': {'lat': 53.48, 'type': 'Industrial', 'region': 'North West'},
    'Newcastle': {'lat': 54.98, 'type': 'Regional hub', 'region': 'North East'},
}


def load_data():
    """鍔犺浇鎵€鏈夐渶瑕佺殑鏁版嵁"""
    lsoa_heatwave = pd.read_csv(RESULTS_DIR / 'inequality_analysis' / 'lsoa_hei_summary_heatwave.csv')
    lsoa_typical = pd.read_csv(RESULTS_DIR / 'inequality_analysis' / 'lsoa_hei_summary_typical_day.csv')
    hei_summary = pd.read_csv(RESULTS_DIR / 'heat_exposure' / 'hei_cni_tcni_summary_improved.csv')
    ndvi_cni = pd.read_csv(RESULTS_DIR / 'ndvi_analysis' / 'lsoa_ndvi_cni_merged_heatwave.csv')
    return lsoa_heatwave, lsoa_typical, hei_summary, ndvi_cni


def bootstrap_ci(data1, data2, weights1, weights2, n_boot=1000, ci=95):
    """Bootstrap confidence interval for weighted mean difference"""
    gaps = []
    for _ in range(n_boot):
        idx1 = np.random.choice(len(data1), size=len(data1), replace=True)
        idx2 = np.random.choice(len(data2), size=len(data2), replace=True)
        gap = np.average(data1.iloc[idx1], weights=weights1.iloc[idx1]) - \
              np.average(data2.iloc[idx2], weights=weights2.iloc[idx2])
        gaps.append(gap)
    alpha = (100 - ci) / 2
    return np.percentile(gaps, [alpha, 100-alpha])


def calculate_city_stats_v2(lsoa_heatwave, lsoa_typical, hei_summary, ndvi_cni):
    """璁＄畻姣忎釜鍩庡競鐨勮缁嗙粺璁℃暟鎹?""

    # 鎸夌含搴︽帓搴?(鍗椻啋鍖?
    cities = sorted(CITY_META.keys(), key=lambda x: CITY_META[x]['lat'])

    results = []

    for city in cities:
        hw = lsoa_heatwave[lsoa_heatwave['city'] == city]
        typ = lsoa_typical[lsoa_typical['city'] == city]
        ndvi = ndvi_cni[ndvi_cni['city'] == city]
        ndvi = ndvi[ndvi['n_roads'] >= 10].dropna(subset=['ndvi_mean', 'hei_mean'])

        meta = CITY_META[city]

        # Basic info
        population = hw['TotPop'].sum()
        n_lsoa = len(hw)
        n_roads = hei_summary[(hei_summary['city'] == city) &
                              (hei_summary['scenario'] == 'heatwave')].iloc[0]['n_roads']

        # Deprived/Affluent breakdown
        hw_dep = hw[hw['IMD_Decile'].isin([1, 2, 3])]
        hw_aff = hw[hw['IMD_Decile'].isin([8, 9, 10])]
        typ_dep = typ[typ['IMD_Decile'].isin([1, 2, 3])]
        typ_aff = typ[typ['IMD_Decile'].isin([8, 9, 10])]

        dep_pop_pct = hw_dep['TotPop'].sum() / population * 100
        aff_pop_pct = hw_aff['TotPop'].sum() / population * 100

        # HEI statistics
        hei_typ_mean = np.average(typ['hei_mean'], weights=typ['TotPop'])
        hei_hw_mean = np.average(hw['hei_mean'], weights=hw['TotPop'])

        # TCNI
        tcni_typ = hei_summary[(hei_summary['city'] == city) &
                               (hei_summary['scenario'] == 'typical_day')].iloc[0]['tcni']
        tcni_hw = hei_summary[(hei_summary['city'] == city) &
                              (hei_summary['scenario'] == 'heatwave')].iloc[0]['tcni']
        tcni_collapse = (tcni_typ - tcni_hw) / tcni_typ * 100  # % reduction

        # HEI Gap with 95% CI
        if len(hw_aff) >= 5:
            gap_typ = np.average(typ_dep['hei_mean'], weights=typ_dep['TotPop']) - \
                     np.average(typ_aff['hei_mean'], weights=typ_aff['TotPop'])
            gap_hw = np.average(hw_dep['hei_mean'], weights=hw_dep['TotPop']) - \
                    np.average(hw_aff['hei_mean'], weights=hw_aff['TotPop'])

            ci_typ = bootstrap_ci(typ_dep['hei_mean'], typ_aff['hei_mean'],
                                  typ_dep['TotPop'], typ_aff['TotPop'])
            ci_hw = bootstrap_ci(hw_dep['hei_mean'], hw_aff['hei_mean'],
                                hw_dep['TotPop'], hw_aff['TotPop'])
        else:
            gap_typ, gap_hw = np.nan, np.nan
            ci_typ, ci_hw = [np.nan, np.nan], [np.nan, np.nan]

        # Cohen's d effect size
        if len(hw_aff) >= 5:
            pooled_std = np.sqrt((hw_dep['hei_mean'].std()**2 + hw_aff['hei_mean'].std()**2) / 2)
            cohen_d = (hw_dep['hei_mean'].mean() - hw_aff['hei_mean'].mean()) / pooled_std
        else:
            cohen_d = np.nan

        # NDVI vs HEI correlation (more meaningful than NDVI vs CNI)
        if len(ndvi) > 10:
            r_ndvi_hei, p_ndvi_hei = stats.pearsonr(ndvi['ndvi_mean'], ndvi['hei_mean'])
        else:
            r_ndvi_hei, p_ndvi_hei = np.nan, np.nan

        # Shadow coverage
        shadow_mean = np.average(hw['shadow_mean'], weights=hw['TotPop']) * 100

        results.append({
            'City': city,
            'Latitude': meta['lat'],
            'Type': meta['type'],
            'Region': meta['region'],
            'Population': population,
            'n_LSOA': n_lsoa,
            'n_roads': n_roads,
            'Deprived_pct': dep_pop_pct,
            'Affluent_pct': aff_pop_pct,
            'HEI_typical': hei_typ_mean,
            'HEI_heatwave': hei_hw_mean,
            'HEI_increase': hei_hw_mean - hei_typ_mean,
            'TCNI_typical': tcni_typ,
            'TCNI_heatwave': tcni_hw,
            'TCNI_collapse_pct': tcni_collapse,
            'Gap_typical': gap_typ,
            'Gap_typical_CI_low': ci_typ[0],
            'Gap_typical_CI_high': ci_typ[1],
            'Gap_heatwave': gap_hw,
            'Gap_heatwave_CI_low': ci_hw[0],
            'Gap_heatwave_CI_high': ci_hw[1],
            'Cohen_d': cohen_d,
            'r_NDVI_HEI': r_ndvi_hei,
            'p_NDVI_HEI': p_ndvi_hei,
            'Shadow_pct': shadow_mean,
        })

    # Add Total row
    all_hw = lsoa_heatwave
    all_typ = lsoa_typical
    all_ndvi = ndvi_cni[ndvi_cni['n_roads'] >= 10].dropna(subset=['ndvi_mean', 'hei_mean'])

    all_dep_hw = all_hw[all_hw['IMD_Decile'].isin([1, 2, 3])]
    all_aff_hw = all_hw[all_hw['IMD_Decile'].isin([8, 9, 10])]
    all_dep_typ = all_typ[all_typ['IMD_Decile'].isin([1, 2, 3])]
    all_aff_typ = all_typ[all_typ['IMD_Decile'].isin([8, 9, 10])]

    total_pop = all_hw['TotPop'].sum()

    gap_typ_all = np.average(all_dep_typ['hei_mean'], weights=all_dep_typ['TotPop']) - \
                  np.average(all_aff_typ['hei_mean'], weights=all_aff_typ['TotPop'])
    gap_hw_all = np.average(all_dep_hw['hei_mean'], weights=all_dep_hw['TotPop']) - \
                 np.average(all_aff_hw['hei_mean'], weights=all_aff_hw['TotPop'])

    ci_typ_all = bootstrap_ci(all_dep_typ['hei_mean'], all_aff_typ['hei_mean'],
                              all_dep_typ['TotPop'], all_aff_typ['TotPop'])
    ci_hw_all = bootstrap_ci(all_dep_hw['hei_mean'], all_aff_hw['hei_mean'],
                             all_dep_hw['TotPop'], all_aff_hw['TotPop'])

    r_all, p_all = stats.pearsonr(all_ndvi['ndvi_mean'], all_ndvi['hei_mean'])

    results.append({
        'City': 'All cities',
        'Latitude': np.nan,
        'Type': '鈥?,
        'Region': '鈥?,
        'Population': total_pop,
        'n_LSOA': len(all_hw),
        'n_roads': hei_summary[hei_summary['scenario']=='heatwave']['n_roads'].sum(),  # 缁熶竴浣跨敤 hei_summary 鏉ユ簮
        'Deprived_pct': all_dep_hw['TotPop'].sum() / total_pop * 100,
        'Affluent_pct': all_aff_hw['TotPop'].sum() / total_pop * 100,
        'HEI_typical': np.average(all_typ['hei_mean'], weights=all_typ['TotPop']),
        'HEI_heatwave': np.average(all_hw['hei_mean'], weights=all_hw['TotPop']),
        'HEI_increase': np.average(all_hw['hei_mean'], weights=all_hw['TotPop']) - \
                        np.average(all_typ['hei_mean'], weights=all_typ['TotPop']),
        'TCNI_typical': np.nan,  # Not meaningful to average
        'TCNI_heatwave': np.nan,
        'TCNI_collapse_pct': np.nan,
        'Gap_typical': gap_typ_all,
        'Gap_typical_CI_low': ci_typ_all[0],
        'Gap_typical_CI_high': ci_typ_all[1],
        'Gap_heatwave': gap_hw_all,
        'Gap_heatwave_CI_low': ci_hw_all[0],
        'Gap_heatwave_CI_high': ci_hw_all[1],
        'Cohen_d': np.nan,
        'r_NDVI_HEI': r_all,
        'p_NDVI_HEI': p_all,
        'Shadow_pct': np.average(all_hw['shadow_mean'], weights=all_hw['TotPop']) * 100,
    })

    return pd.DataFrame(results)


def format_table_panel_a(df):
    """Panel A: City characteristics and sample"""
    table_data = []

    for _, row in df.iterrows():
        if row['City'] == 'All cities':
            lat_str = '鈥?
        else:
            lat_str = f"{row['Latitude']:.1f}掳N"

        pop_str = f"{row['Population']/1e6:.1f}" if row['Population'] >= 1e6 else f"{row['Population']/1e3:.0f}k"

        table_data.append({
            'City': row['City'],
            'Lat.': lat_str,
            'Type': row['Type'],
            'Pop.': pop_str,
            'LSOAs': f"{int(row['n_LSOA']):,}",
            'Roads': f"{int(row['n_roads']):,}",
            'Deprived (%)': f"{row['Deprived_pct']:.0f}",
            'Affluent (%)': f"{row['Affluent_pct']:.0f}",
        })

    return pd.DataFrame(table_data)


def format_table_panel_b(df):
    """Panel B: Thermal metrics (Typical Day vs Heatwave)"""
    table_data = []

    for _, row in df.iterrows():
        # Format Gap with 95% CI
        if pd.notna(row['Gap_typical']):
            gap_typ_str = f"+{row['Gap_typical']:.2f} [{row['Gap_typical_CI_low']:.2f}, {row['Gap_typical_CI_high']:.2f}]"
        else:
            gap_typ_str = "鈥?

        if pd.notna(row['Gap_heatwave']):
            gap_hw_str = f"+{row['Gap_heatwave']:.2f} [{row['Gap_heatwave_CI_low']:.2f}, {row['Gap_heatwave_CI_high']:.2f}]"
        else:
            gap_hw_str = "鈥?

        # TCNI with collapse %
        if pd.notna(row['TCNI_typical']):
            tcni_str = f"{row['TCNI_typical']:.1f} 鈫?{row['TCNI_heatwave']:.1f} (鈭抺row['TCNI_collapse_pct']:.0f}%)"
        else:
            tcni_str = "鈥?

        # r(NDVI~HEI) with significance
        if pd.notna(row['r_NDVI_HEI']):
            if row['p_NDVI_HEI'] < 0.001:
                r_str = f"{row['r_NDVI_HEI']:.2f}***"
            elif row['p_NDVI_HEI'] < 0.01:
                r_str = f"{row['r_NDVI_HEI']:.2f}**"
            elif row['p_NDVI_HEI'] < 0.05:
                r_str = f"{row['r_NDVI_HEI']:.2f}*"
            else:
                r_str = f"{row['r_NDVI_HEI']:.2f}"
        else:
            r_str = "鈥?

        # Cohen's d interpretation
        if pd.notna(row['Cohen_d']):
            d = row['Cohen_d']
            if abs(d) < 0.2:
                d_interp = 'negligible'
            elif abs(d) < 0.5:
                d_interp = 'small'
            elif abs(d) < 0.8:
                d_interp = 'medium'
            else:
                d_interp = 'large'
            d_str = f"{d:.2f} ({d_interp})"
        else:
            d_str = "鈥?

        table_data.append({
            'City': row['City'],
            'HEI Typical (掳C)': f"{row['HEI_typical']:.1f}",
            'HEI Heatwave (掳C)': f"{row['HEI_heatwave']:.1f}",
            '螖HEI (掳C)': f"+{row['HEI_increase']:.1f}",
            'TCNI': tcni_str,
            'Gap Typical [95% CI]': gap_typ_str,
            'Gap Heatwave [95% CI]': gap_hw_str,
            "Cohen's d": d_str,
            'r(NDVI~HEI)': r_str,
        })

    return pd.DataFrame(table_data)


def create_latex_table_v2(df_a, df_b):
    """鐢熸垚浼樺寲鍚庣殑 LaTeX 琛ㄦ牸"""

    def texify_cell(s: str) -> str:
        """Convert unicode formatting and escape LaTeX specials for table cells."""
        if s is None:
            return "鈥?
        s = str(s)

        # Normalize dashes/arrows/minus to LaTeX-safe forms.
        s = s.replace("鈥?, "---")
        s = s.replace("鈫?, r"$\to$")
        s = s.replace("鈭?, r"$-$")

        # Degree symbol in latitude (e.g., 51.5掳N).
        s = s.replace("掳N", r"\textdegree N")

        # Escape LaTeX special chars that can appear in cells.
        # NOTE: We intentionally do NOT blanket-escape backslashes here.
        s = s.replace("%", r"\%")

        return s

    latex = r"""
\begin{table}[htbp]
\centering
\caption{\textbf{Summary statistics for five English cities.}
(a) City characteristics and sample composition. Deprived = IMD Decile 1--3; Affluent = IMD Decile 8--10.
(b) Thermal metrics under typical summer day and heatwave conditions. HEI = Heat Exposure Index; TCNI = Thermal Connectivity Network Index; Gap = population-weighted mean HEI difference (Deprived $-$ Affluent); Cohen's d = effect size of inequality.
Values in brackets are 95\% bootstrap confidence intervals.
Significance: ***$p<0.001$, **$p<0.01$, *$p<0.05$.}
\label{tab:city_summary}

\small
\textbf{(a) City characteristics}
\begin{tabular}{llllrrrr}
\toprule
City & Lat. & Type & Pop. & LSOAs & Roads & Deprived (\%) & Affluent (\%) \\
\midrule
"""

    for _, row in df_a.iterrows():
        if row['City'] == 'All cities':
            latex += r"\midrule" + "\n"
        latex += (
            f"{texify_cell(row['City'])} & {texify_cell(row['Lat.'])} & {texify_cell(row['Type'])} & "
            f"{texify_cell(row['Pop.'])} & {texify_cell(row['LSOAs'])} & {texify_cell(row['Roads'])} & "
            f"{texify_cell(row['Deprived (%)'])} & {texify_cell(row['Affluent (%)'])} \\\\\n"
        )

    latex += r"""\bottomrule
\end{tabular}

\vspace{0.5cm}

\textbf{(b) Thermal metrics}
\begin{tabular}{lcccccc}
\toprule
City & HEI$_{\text{typ}}$ & HEI$_{\text{hw}}$ & TCNI$_{\text{typ}} \to_{\text{hw}}$ & Gap$_{\text{typ}}$ [95\% CI] & Gap$_{\text{hw}}$ [95\% CI] & Cohen's $d$ \\
\midrule
"""

    for _, row in df_b.iterrows():
        if row['City'] == 'All cities':
            latex += r"\midrule" + "\n"
        cohen_d = row["Cohen's d"]
        latex += (
            f"{texify_cell(row['City'])} & {texify_cell(row['HEI Typical (掳C)'])} & {texify_cell(row['HEI Heatwave (掳C)'])} & "
            f"{texify_cell(row['TCNI'])} & {texify_cell(row['Gap Typical [95% CI]'])} & {texify_cell(row['Gap Heatwave [95% CI]'])} & "
            f"{texify_cell(cohen_d)} \\\\\n"
        )

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def create_markdown_table_v2(df_a, df_b):
    """鐢熸垚浼樺寲鍚庣殑 Markdown 琛ㄦ牸"""

    md = "# Table 1: City-Level Summary Statistics\n\n"
    md += "## (a) City characteristics and sample composition\n\n"
    md += "| City | Lat. | Type | Pop. | LSOAs | Roads | Deprived (%) | Affluent (%) |\n"
    md += "|------|------|------|------|-------|-------|--------------|-------------|\n"

    for _, row in df_a.iterrows():
        md += f"| {row['City']} | {row['Lat.']} | {row['Type']} | {row['Pop.']} | {row['LSOAs']} | {row['Roads']} | {row['Deprived (%)']} | {row['Affluent (%)']} |\n"

    md += "\n## (b) Thermal metrics\n\n"
    md += "| City | HEI_typ (掳C) | HEI_hw (掳C) | TCNI | Gap_typ [95% CI] | Gap_hw [95% CI] | Cohen's d |\n"
    md += "|------|--------------|-------------|------|------------------|-----------------|----------|\n"

    for _, row in df_b.iterrows():
        cohen_d = row["Cohen's d"]
        md += f"| {row['City']} | {row['HEI Typical (掳C)']} | {row['HEI Heatwave (掳C)']} | {row['TCNI']} | {row['Gap Typical [95% CI]']} | {row['Gap Heatwave [95% CI]']} | {cohen_d} |\n"

    md += "\n\n**Notes:**\n"
    md += "- Lat. = Latitude (掳N)\n"
    md += "- Pop. = Population (M = million, k = thousand)\n"
    md += "- Deprived = IMD Decile 1鈥?; Affluent = IMD Decile 8鈥?0\n"
    md += "- HEI = Heat Exposure Index (掳C); typ = typical summer day; hw = heatwave\n"
    md += "- TCNI = Thermal Connectivity Network Index (arrows show collapse from typical 鈫?heatwave)\n"
    md += "- Gap = population-weighted mean HEI difference (Deprived 鈭?Affluent)\n"
    md += "- Cohen's d = effect size: negligible (<0.2), small (0.2鈥?.5), medium (0.5鈥?.8), large (>0.8)\n"
    md += "- Significance: ***p<0.001, **p<0.01, *p<0.05\n"

    return md


def main():
    print("Loading data...")
    lsoa_heatwave, lsoa_typical, hei_summary, ndvi_cni = load_data()

    print("Calculating city statistics (v2)...")
    df_stats = calculate_city_stats_v2(lsoa_heatwave, lsoa_typical, hei_summary, ndvi_cni)

    print("\n" + "="*100)
    print("RAW STATISTICS (v2)")
    print("="*100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_stats.to_string(index=False))

    # Format panels
    df_panel_a = format_table_panel_a(df_stats)
    df_panel_b = format_table_panel_b(df_stats)

    print("\n" + "="*100)
    print("PANEL A: City Characteristics")
    print("="*100)
    print(df_panel_a.to_string(index=False))

    print("\n" + "="*100)
    print("PANEL B: Thermal Metrics")
    print("="*100)
    # Windows terminals may use a legacy encoding (e.g. GBK) that cannot
    # print certain Unicode glyphs (e.g. '鈭?, '鈫?). Keep stdout robust.
    try:
        print(df_panel_b.to_string(index=False))
    except UnicodeEncodeError:
        safe = (
            df_panel_b.to_string(index=False)
            .replace("鈭?, "-")
            .replace("鈫?, "->")
        )
        print(safe)

    # Save files
    df_stats.to_csv(SUPPLEMENT_DIR / 'Table1_city_summary_v2_raw.csv', index=False)
    df_panel_a.to_csv(SUPPLEMENT_DIR / 'Table1_city_summary_v2_panel_a.csv', index=False)
    df_panel_b.to_csv(SUPPLEMENT_DIR / 'Table1_city_summary_v2_panel_b.csv', index=False)

    # LaTeX
    latex_code = create_latex_table_v2(df_panel_a, df_panel_b)
    print("\n" + "="*100)
    print("LATEX TABLE (v2)")
    print("="*100)
    try:
        print(latex_code)
    except UnicodeEncodeError:
        print(latex_code.replace("鈭?, "-").replace("鈫?, "->"))

    with open(SUPPLEMENT_DIR / 'Table1_city_summary_v2.tex', 'w', encoding='utf-8') as f:
        f.write(latex_code)

    # Markdown
    md_table = create_markdown_table_v2(df_panel_a, df_panel_b)
    print("\n" + "="*100)
    print("MARKDOWN TABLE (v2)")
    print("="*100)
    try:
        print(md_table)
    except UnicodeEncodeError:
        print(md_table.replace("鈭?, "-").replace("鈫?, "->"))

    with open(SUPPLEMENT_DIR / 'Table1_city_summary_v2.md', 'w', encoding='utf-8') as f:
        f.write(md_table)

    print("\n" + "="*100)
    print("FILES SAVED:")
    print("="*100)
    print(f"  - {SUPPLEMENT_DIR / 'Table1_city_summary_v2_raw.csv'}")
    print(f"  - {SUPPLEMENT_DIR / 'Table1_city_summary_v2_panel_a.csv'}")
    print(f"  - {SUPPLEMENT_DIR / 'Table1_city_summary_v2_panel_b.csv'}")
    print(f"  - {SUPPLEMENT_DIR / 'Table1_city_summary_v2.tex'}")
    print(f"  - {SUPPLEMENT_DIR / 'Table1_city_summary_v2.md'}")

    # Key findings summary
    print("\n" + "="*100)
    print("KEY FINDINGS FOR MANUSCRIPT")
    print("="*100)

    cities_only = df_stats[df_stats['City'] != 'All cities']

    print(f"\n1. SAMPLE: {int(df_stats[df_stats['City']=='All cities']['n_LSOA'].values[0]):,} LSOAs across 5 cities")
    print(f"   Population: {df_stats[df_stats['City']=='All cities']['Population'].values[0]/1e6:.1f} million")

    print(f"\n2. HEATWAVE AMPLIFICATION:")
    print(f"   Mean HEI increase: +{cities_only['HEI_increase'].mean():.1f}掳C (range: {cities_only['HEI_increase'].min():.1f}鈥搟cities_only['HEI_increase'].max():.1f}掳C)")
    print(f"   TCNI collapse: {cities_only['TCNI_collapse_pct'].mean():.0f}% average reduction")

    print(f"\n3. INEQUALITY:")
    print(f"   Largest Gap (heatwave): Manchester +{cities_only.loc[cities_only['Gap_heatwave'].idxmax(), 'Gap_heatwave']:.2f}掳C")
    print(f"   Strongest effect size: Manchester d={cities_only.loc[cities_only['Cohen_d'].idxmax(), 'Cohen_d']:.2f}")

    print(f"\n4. SAMPLE REPRESENTATIVENESS WARNING:")
    for _, row in cities_only.iterrows():
        if row['Affluent_pct'] < 10:
            # Keep stdout ASCII-clean on Windows consoles.
            print(
                f"   [WARNING] {row['City']}: Only {row['Affluent_pct']:.1f}% affluent population (limited comparison)"
            )


if __name__ == '__main__':
    main()


