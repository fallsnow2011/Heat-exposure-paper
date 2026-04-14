import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results" / "inequality_analysis"
SUPPLEMENT_DIR = BASE_DIR / "paper" / "06_supplement"
FIGURES_DIR = BASE_DIR / "paper" / "05_figures"
IMD_GPKG = BASE_DIR / "city_boundaries" / "Indices_of_Multiple_Deprivation_(IMD)_2019_8404722932957776031.gpkg"

# Colour mapping for policy scenarios / 閰嶈壊
COLORS = {
    'scenario1': '#c97b63',  # Equity-first (terracotta)
    'scenario2': '#7ca5a1',  # Main-corridor (teal)
    'scenario3': '#2c5f7c',  # City-wide (deep blue)
    'baseline': '#1f3a5f',   # Baseline (dark)
}

# HEI parameters / HEI 鍙傛暟
ALPHA_B = 0.6  # building-shade cooling coefficient / 寤虹瓚闃村奖鍐峰嵈绯绘暟
ALPHA_V = 0.8  # vegetation-shade cooling coefficient / 妞嶈闃村奖鍐峰嵈绯绘暟
DELTA_T_VEG = 2.0  # additional vegetation cooling term (掳C) proportional to shadow_vegetation / 妞嶈棰濆闄嶆俯椤癸紙掳C锛夛紝涓?shadow_vegetation 鎴愭姣?

def load_lsoa_data():
    """Load LSOA summaries and attach polygon areas / 鍔犺浇 LSOA 鏁版嵁锛屽苟娣诲姞闈㈢Н淇℃伅"""
    typical = pd.read_csv(RESULTS_DIR / "lsoa_hei_summary_typical_day.csv")
    heatwave = pd.read_csv(RESULTS_DIR / "lsoa_hei_summary_heatwave.csv")

    # Load LSOA geometries to recover area / 鍔犺浇 LSOA 鍑犱綍鏁版嵁鑾峰彇闈㈢Н
    gdf = gpd.read_file(IMD_GPKG)
    # `st_areasha` is in square metres; convert it to km虏 / `st_areasha` 鏄钩鏂圭背锛岃浆鎹负骞虫柟鍏噷
    area_df = gdf[['lsoa11cd', 'st_areasha']].copy()
    area_df['area_km2'] = area_df['st_areasha'] / 1e6

    # Attach area fields to both scenarios / 鍚堝苟闈㈢Н鏁版嵁
    typical = typical.merge(area_df[['lsoa11cd', 'area_km2']], on='lsoa11cd', how='left')
    heatwave = heatwave.merge(area_df[['lsoa11cd', 'area_km2']], on='lsoa11cd', how='left')

    return typical, heatwave


# =============================================================================
# Fix 1: population-weighted Gini coefficient / 淇 1锛氫汉鍙ｅ姞鏉?Gini 绯绘暟
# =============================================================================
def calculate_gini(values, weights=None):
    """
    Calculate the Gini coefficient / 璁＄畻 Gini 绯绘暟

    Parameters:
    -----------
    values : array-like
        Value array / 鏁板€兼暟缁?    weights : array-like, optional
        Population-weight array / 浜哄彛鏉冮噸鏁扮粍

    Returns:
    --------
    float : Gini coefficient / Gini 绯绘暟
    """
    values = np.array(values)
    mask = ~np.isnan(values)
    values = values[mask]

    if len(values) == 0:
        return np.nan

    if weights is None:
        # Unweighted version / 鏈姞鏉冪増鏈?        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumulative = np.cumsum(sorted_values)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
    else:
        # Population-weighted version / 浜哄彛鍔犳潈鐗堟湰
        weights = np.array(weights)[mask]
        # Sort by the target value / 鎸夊€兼帓搴?        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]

        # Cumulative weights / 璁＄畻绱Н鏉冮噸
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights[-1]

        # Weighted Gini coefficient /
        # 鍔犳潈 Gini
        # G = 1 - 2 * 鈭?L(p))dp, where L(p) is the Lorenz curve /
        # G = 1 - 2 * 鈭?L(p))dp锛屽叾涓?L(p) 鏄?Lorenz 鏇茬嚎
        cum_values = np.cumsum(sorted_values * sorted_weights)
        total_value = cum_values[-1]

        # Approximate the Lorenz area by the trapezoid rule / 姊舰娉曞垯杩戜技绉垎
        p = cum_weights / total_weight
        L = cum_values / total_value

        # Gini = 1 - 2 * Area under Lorenz curve
        gini = 1 - 2 * np.trapz(L, p)

    return gini


def calculate_concentration_index(df, health_var, rank_var='IMD_Rank'):
    """
    Calculate the population-weighted concentration index / 璁＄畻闆嗕腑鎸囨暟锛堜汉鍙ｅ姞鏉冿級

    CI = (2/渭) * cov_w(h, r)
    where r is the fractional rank ordered by deprivation /
    鍏朵腑 r 鏄寜璐洶鎺掑悕鐨勫垎鏁扮З
    """
    valid = df[[health_var, rank_var, 'TotPop']].dropna()
    if len(valid) < 10:
        return np.nan

    valid = valid.sort_values(rank_var)
    total_pop = valid['TotPop'].sum()

    # Fractional rank = midpoint of cumulative population / 鍒嗘暟绉?= 锛堢疮绉汉鍙ｇ殑涓偣锛? 鎬讳汉鍙?    valid['cum_pop'] = valid['TotPop'].cumsum()
    valid['frac_rank'] = (valid['cum_pop'] - valid['TotPop']/2) / total_pop

    # Population-weighted mean / 浜哄彛鍔犳潈骞冲潎
    mean_h = np.average(valid[health_var], weights=valid['TotPop'])

    # Population-weighted covariance / 浜哄彛鍔犳潈鍗忔柟宸?    ci = 2 * np.cov(valid[health_var], valid['frac_rank'],
                    aweights=valid['TotPop'])[0, 1] / mean_h

    return ci


# =============================================================================
# Fix 2: dual-metric affluent-versus-deprived gap / 淇 2锛氳传瀵屽樊璺濊绠?- 鍚屾椂鎶ュ憡涓ょ鍙ｅ緞
# =============================================================================
def calculate_gap(df, var, pop_var='TotPop'):
    """
    Calculate the deprived-versus-affluent gap with two reporting conventions /
    璁＄畻璐瘜宸窛锛岃繑鍥炰袱绉嶅彛寰?
    Use the D1-D3 versus D8-D10 definition (30% vs 30%) to stay consistent
    with Table 1 and Figure 4 /
    浣跨敤 D1-D3 vs D8-D10锛?0% vs 30%锛夊彛寰勶紝涓?Table 1 鍜?Fig 4 淇濇寔涓€鑷?
    Returns:
    --------
    dict: {
        'gap_lsoa_avg': LSOA-average gap / LSOA 骞冲潎宸窛,
        'gap_pop_weighted': population-weighted gap / 浜哄彛鍔犳潈宸窛,
        'poor_lsoa_avg': deprived-group LSOA average / 璐洶缁?LSOA 骞冲潎,
        'poor_pop_weighted': deprived-group population-weighted average / 璐洶缁勪汉鍙ｅ姞鏉冨钩鍧?
        'rich_lsoa_avg': affluent-group LSOA average / 瀵岃缁?LSOA 骞冲潎,
        'rich_pop_weighted': affluent-group population-weighted average / 瀵岃缁勪汉鍙ｅ姞鏉冨钩鍧?    }
    """
    # Use D1-D3 versus D8-D10 to match the manuscript tables and figures / 浣跨敤 D1-D3 vs D8-D10锛屼笌 Table 1 鍜?Fig 4 淇濇寔涓€鑷?    poor = df[df['IMD_Decile'].isin([1, 2, 3])]
    rich = df[df['IMD_Decile'].isin([8, 9, 10])]

    # LSOA-average gap / LSOA 骞冲潎锛堝尯鍩熷钩鍧囷級
    poor_lsoa = poor[var].mean()
    rich_lsoa = rich[var].mean()
    gap_lsoa = poor_lsoa - rich_lsoa

    # Population-weighted mean / 浜哄彛鍔犳潈骞冲潎
    poor_pop = np.average(poor[var], weights=poor[pop_var]) if len(poor) > 0 else np.nan
    rich_pop = np.average(rich[var], weights=rich[pop_var]) if len(rich) > 0 else np.nan
    gap_pop = poor_pop - rich_pop

    return {
        'gap_lsoa_avg': gap_lsoa,
        'gap_pop_weighted': gap_pop,
        'poor_lsoa_avg': poor_lsoa,
        'poor_pop_weighted': poor_pop,
        'rich_lsoa_avg': rich_lsoa,
        'rich_pop_weighted': rich_pop
    }


# =============================================================================
# Part 4: CNI(LST) versus CNI(HEI) comparison / 绗洓鐐癸細CNI(LST) vs CNI(HEI) 瀵规瘮鍒嗘瀽锛堜慨澶嶇増锛?# =============================================================================
def analyze_lst_vs_hei_inequality():
    """Analyse LST-versus-HEI inequality with both reporting conventions / 鍒嗘瀽 LST vs HEI 鐨勪笉骞崇瓑宸紓 - 鍚屾椂鎶ュ憡涓ょ鍙ｅ緞"""
    print("=" * 60)
    print("Part 4. LST vs HEI inequality comparison / 鍥涖€丆NI(LST) vs CNI(HEI) 涓嶅钩绛夊姣斿垎鏋?)
    print("=" * 60)

    typical, heatwave = load_lsoa_data()

    results = []

    for scenario, df in [('typical_day', typical), ('heatwave', heatwave)]:
        # LST inequality metrics / LST 鐨?Gini 鍜?CI
        gini_lst_unweighted = calculate_gini(df['lst_mean'].dropna())
        gini_lst_weighted = calculate_gini(df['lst_mean'].values, df['TotPop'].values)
        ci_lst = calculate_concentration_index(df, 'lst_mean')

        # HEI inequality metrics / HEI 鐨?Gini 鍜?CI
        gini_hei_unweighted = calculate_gini(df['hei_mean'].dropna())
        gini_hei_weighted = calculate_gini(df['hei_mean'].values, df['TotPop'].values)
        ci_hei = calculate_concentration_index(df, 'hei_mean')

        # Deprived-versus-affluent gaps with two conventions / 璐瘜宸窛锛堜袱绉嶅彛寰勶級
        gap_lst = calculate_gap(df, 'lst_mean')
        gap_hei = calculate_gap(df, 'hei_mean')

        results.append({
            'scenario': scenario,
            # Gini
            'gini_lst_unweighted': gini_lst_unweighted,
            'gini_lst_pop_weighted': gini_lst_weighted,
            'gini_hei_unweighted': gini_hei_unweighted,
            'gini_hei_pop_weighted': gini_hei_weighted,
            # CI
            'ci_lst': ci_lst,
            'ci_hei': ci_hei,
            # Gap - LSOA average / LSOA 骞冲潎
            'gap_lst_lsoa_avg': gap_lst['gap_lsoa_avg'],
            'gap_hei_lsoa_avg': gap_hei['gap_lsoa_avg'],
            # Gap - population weighted / 浜哄彛鍔犳潈
            'gap_lst_pop_weighted': gap_lst['gap_pop_weighted'],
            'gap_hei_pop_weighted': gap_hei['gap_pop_weighted'],
            # Cooling benefit attributable to shade adjustment / 闃村奖缂撹В鏁堟灉
            'shadow_effect_lsoa': gap_lst['gap_lsoa_avg'] - gap_hei['gap_lsoa_avg'],
            'shadow_effect_pop': gap_lst['gap_pop_weighted'] - gap_hei['gap_pop_weighted']
        })

        print(f"\n{scenario}:")
        print("  [LSOA-average metric / LSOA 骞冲潎鍙ｅ緞]")
        print(f"    LST - Gini: {gini_lst_unweighted:.4f}, Gap: {gap_lst['gap_lsoa_avg']:+.2f}掳C")
        print(f"    HEI - Gini: {gini_hei_unweighted:.4f}, Gap: {gap_hei['gap_lsoa_avg']:+.2f}掳C")
        print(f"    Shade mitigation / 闃村奖缂撹В: {gap_lst['gap_lsoa_avg'] - gap_hei['gap_lsoa_avg']:+.2f}掳C")
        print("  [Population-weighted metric / 浜哄彛鍔犳潈鍙ｅ緞]")
        print(f"    LST - Gini: {gini_lst_weighted:.4f}, Gap: {gap_lst['gap_pop_weighted']:+.2f}掳C")
        print(f"    HEI - Gini: {gini_hei_weighted:.4f}, Gap: {gap_hei['gap_pop_weighted']:+.2f}掳C")
        print(f"    Shade mitigation / 闃村奖缂撹В: {gap_lst['gap_pop_weighted'] - gap_hei['gap_pop_weighted']:+.2f}掳C")
        print("  [Concentration index CI / 闆嗕腑鎸囨暟 CI]锛堜汉鍙ｅ姞鏉冿級")
        print(f"    LST CI: {ci_lst:.4f}, HEI CI: {ci_hei:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(SUPPLEMENT_DIR / "lst_vs_hei_inequality_comparison_v2.csv", index=False)
    print("\nSaved output / 宸蹭繚瀛? lst_vs_hei_inequality_comparison_v2.csv")

    return results_df


# =============================================================================
# Fix 3: shadow clipping helpers / 淇 3锛氶槾褰辫鍓嚱鏁?# =============================================================================
def clip_shadow(shadow_values):
    """
    Ensure shadow fractions remain within [0, 1] / 纭繚闃村奖鍊煎湪 [0, 1] 鑼冨洿鍐?
    Fix note /
    淇璇存槑锛?    Directly applying `shadow + 0.10` can push values above 1 /
    鍘熶唬鐮佺洿鎺?`shadow + 0.10` 鍙兘瀵艰嚧鍊艰秴杩?1
    """
    return np.clip(shadow_values, 0, 1)


def calculate_hei(lst, shadow_building, shadow_vegetation):
    """
    Recalculate HEI while enforcing valid shade fractions / 璁＄畻 HEI锛岀‘淇濋槾褰卞€煎悎娉?
    HEI = LST 脳 (1 - 伪_b 脳 S_b - 伪_v 脳 S_v) - 螖T_v 脳 S_v
    """
    lst = np.asarray(lst, dtype=float)
    shadow_building = clip_shadow(np.asarray(shadow_building, dtype=float))
    shadow_vegetation = clip_shadow(np.asarray(shadow_vegetation, dtype=float))

    # Robustness: ensure S_b + S_v <= 1 by proportional scaling (normally already satisfied)
    shadow_total = shadow_building + shadow_vegetation
    scale = np.where(shadow_total > 1, 1 / shadow_total, 1)
    shadow_building = shadow_building * scale
    shadow_vegetation = shadow_vegetation * scale

    shadow_cooling = ALPHA_B * shadow_building + ALPHA_V * shadow_vegetation
    hei_base = lst * (1 - shadow_cooling)
    vegetation_cooling = DELTA_T_VEG * shadow_vegetation
    return hei_base - vegetation_cooling


# =============================================================================
# Part 5: three policy scenarios / 绗簲鐐癸細涓夌鏀跨瓥鎯呮櫙妯℃嫙锛堜慨澶嶇増锛?# =============================================================================
def simulate_s3_equity_first(df, shadow_increase=0.10):
    """
    S3: Equity First
    Add vegetation shade in deprived, hot, low-vegetation-shade LSOAs /
    鍦ㄨ传鍥伴珮 + HEI 楂?+ 妞嶈闃村奖浣庣殑 LSOA 涓鍔犳琚槾褰?
    Fix: clip shadow values after intervention / 淇锛氭坊鍔犻槾褰辫鍓?    """
    df = df.copy()

    # Select target LSOAs: IMD 1-3 AND HEI > median AND vegetation shade < median / 璇嗗埆鐩爣 LSOA锛欼MD Decile 1-3 AND HEI > median AND shadow_veg < median
    hei_median = df['hei_mean'].median()
    shadow_median = df['shadow_vegetation_mean'].median()

    target_mask = (
        (df['IMD_Decile'].isin([1, 2, 3])) &
        (df['hei_mean'] > hei_median) &
        (df['shadow_vegetation_mean'] < shadow_median)
    )

    n_target = target_mask.sum()
    pop_target = df.loc[target_mask, 'TotPop'].sum()
    pop_total = df['TotPop'].sum()

    print("  S3 (Equity First) target area / 鐩爣鍖哄煙:")
    print(f"    LSOA: {n_target} ({n_target/len(df)*100:.1f}%)")
    print(f"    Population / 浜哄彛: {pop_target:,.0f} ({pop_target/pop_total*100:.1f}%)")

    # Increase vegetation shade in target zones and clip it to [0, 1] / 鍦ㄧ洰鏍囧尯鍩熷鍔犳琚槾褰憋紙瑁佸壀鍒?[0, 1]锛?    df['shadow_vegetation_new'] = df['shadow_vegetation_mean'].copy()
    df.loc[target_mask, 'shadow_vegetation_new'] = clip_shadow(
        df.loc[target_mask, 'shadow_vegetation_mean'] + shadow_increase
    )

    # Recalculate HEI after the intervention / 閲嶆柊璁＄畻 HEI
    df['hei_new'] = calculate_hei(
        df['lst_mean'],
        df['shadow_building_mean'],
        df['shadow_vegetation_new']
    )

    return df, target_mask


def simulate_s2_corridors(df, shadow_increase=0.10):
    """
    S2: Corridors
    Add vegetation shade in high road-density areas /
    鍦ㄩ珮璺綉瀵嗗害鍖哄煙澧炲姞闃村奖

    Fix: compute road density as total_length / area_km2 /
    淇锛氳矾缃戝瘑搴﹁绠楁敼涓?total_length / area_km2
    """
    df = df.copy()

    # Use the corrected road-density definition / 浣跨敤姝ｇ‘鐨勮矾缃戝瘑搴﹁绠?    # road_length_density = total_length (m) / area (km虏) / road_length_density = total_length锛堢背锛? area锛堝钩鏂瑰叕閲岋級
    if 'total_length' in df.columns and 'area_km2' in df.columns:
        # Road density = total road length (m) / area (km虏) = m/km虏 / 璺綉瀵嗗害 = 閬撹矾鎬婚暱搴︼紙绫筹級/ 闈㈢Н锛堝钩鏂瑰叕閲岋級= 绫?骞虫柟鍏噷
        df['road_density'] = df['total_length'] / (df['area_km2'] + 0.001)  # 閬垮厤闄ら浂
        density_75 = df['road_density'].quantile(0.75)
        target_mask = df['road_density'] >= density_75
        print("  Road-density formula / 璺綉瀵嗗害璁＄畻: total_length / area_km2")
        print(f"  P75 threshold / P75 闃堝€? {density_75:.0f} 绫?骞虫柟鍏噷")
    elif 'total_length' in df.columns:
        # Fallback to a population-based proxy when area is missing / 閫€鍥炲埌浣跨敤浜哄彛瀵嗗害浠ｇ悊锛堢己灏戦潰绉暟鎹級
        print("  Warning: area data missing, using a population proxy / 璀﹀憡锛氱己灏戦潰绉暟鎹紝浣跨敤浜哄彛瀵嗗害浠ｇ悊")
        target_mask = df['TotPop'] >= df['TotPop'].quantile(0.75)
    else:
        # If road-network inputs are absent, also fall back to a population proxy / 濡傛灉娌℃湁璺綉鏁版嵁锛屼娇鐢ㄤ汉鍙ｅ瘑搴︿唬鐞?        print("  Warning: road-network data missing, using a population proxy / 璀﹀憡锛氱己灏戣矾缃戞暟鎹紝浣跨敤浜哄彛瀵嗗害浠ｇ悊")
        target_mask = df['TotPop'] >= df['TotPop'].quantile(0.75)

    n_target = target_mask.sum()
    pop_target = df.loc[target_mask, 'TotPop'].sum()
    pop_total = df['TotPop'].sum()

    print("  S2 (Corridors) target area / 鐩爣鍖哄煙:")
    print(f"    LSOA: {n_target} ({n_target/len(df)*100:.1f}%)")
    print(f"    Population / 浜哄彛: {pop_target:,.0f} ({pop_target/pop_total*100:.1f}%)")

    # Increase vegetation shade and clip it to [0, 1] / 澧炲姞妞嶈闃村奖锛堣鍓埌 [0, 1]锛?    df['shadow_vegetation_new'] = df['shadow_vegetation_mean'].copy()
    df.loc[target_mask, 'shadow_vegetation_new'] = clip_shadow(
        df.loc[target_mask, 'shadow_vegetation_mean'] + shadow_increase
    )

    # Recalculate HEI after the corridor intervention / 閲嶆柊璁＄畻 HEI
    df['hei_new'] = calculate_hei(
        df['lst_mean'],
        df['shadow_building_mean'],
        df['shadow_vegetation_new']
    )

    return df, target_mask


def simulate_s1_citywide(df, shadow_increase=0.10):
    """
    S1: Citywide
    Add 10% vegetation shade everywhere in the city /
    鍏ㄥ煄甯傛墍鏈夊尯鍩熷潎鍖€澧炲姞 10% 妞嶈闃村奖

    Fix: clip shade values after the intervention / 淇锛氭坊鍔犻槾褰辫鍓?    """
    df = df.copy()

    target_mask = pd.Series([True] * len(df), index=df.index)
    n_target = target_mask.sum()
    pop_target = df['TotPop'].sum()

    print("  S1 (Citywide) target area / 鐩爣鍖哄煙:")
    print(f"    LSOA: {n_target} ({n_target/len(df)*100:.1f}%)")
    print(f"    Population / 浜哄彛: {pop_target:,.0f} (100.0%)")

    # Apply the intervention everywhere and clip it to [0, 1] / 鍏ㄩ儴鍖哄煙澧炲姞闃村奖锛堣鍓埌 [0, 1]锛?    df['shadow_vegetation_new'] = clip_shadow(
        df['shadow_vegetation_mean'] + shadow_increase
    )

    # Recalculate HEI after the citywide intervention / 閲嶆柊璁＄畻 HEI
    df['hei_new'] = calculate_hei(
        df['lst_mean'],
        df['shadow_building_mean'],
        df['shadow_vegetation_new']
    )

    return df, target_mask


def run_all_scenarios():
    """Run all three policy scenarios and report both metrics / 杩愯鎵€鏈変笁绉嶆儏鏅ā鎷?- 鎶ュ憡鍙屽彛寰勭粨鏋?""
    print("\n" + "=" * 60)
    print("Part 5. Policy scenarios / 浜斻€佹斂绛栨儏鏅ā鎷燂紙淇鐗堬級")
    print("=" * 60)

    typical, heatwave = load_lsoa_data()

    all_results = []
    scenario_details = []

    for scenario_name, df in [('typical_day', typical), ('heatwave', heatwave)]:
        print(f"\n{'='*40}")
        print(f"--- {scenario_name} ---")
        print(f"{'='*40}")

        # Baseline metrics under the two reporting conventions / Baseline - 涓ょ鍙ｅ緞
        gap_baseline = calculate_gap(df, 'hei_mean')

        all_results.append({
            'time_scenario': scenario_name,
            'policy_scenario': 'baseline',
            'scenario_order': 0,
            'scenario_label': 'Baseline',
            # LSOA-average metric / LSOA 骞冲潎鍙ｅ緞
            'poor_hei_lsoa': gap_baseline['poor_lsoa_avg'],
            'rich_hei_lsoa': gap_baseline['rich_lsoa_avg'],
            'gap_lsoa': gap_baseline['gap_lsoa_avg'],
            # Population-weighted metric / 浜哄彛鍔犳潈鍙ｅ緞
            'poor_hei_pop': gap_baseline['poor_pop_weighted'],
            'rich_hei_pop': gap_baseline['rich_pop_weighted'],
            'gap_pop': gap_baseline['gap_pop_weighted'],
            # Coverage metrics / 瑕嗙洊鐜?            'target_lsoa_pct': 0,
            'target_pop_pct': 0,
            'gap_reduction_lsoa_pct': 0,
            'gap_reduction_pop_pct': 0
        })

        print(f"\nBaseline:")
        print(f"  LSOA-average gap / LSOA 骞冲潎 Gap: {gap_baseline['gap_lsoa_avg']:+.2f}掳C")
        print(f"  Population-weighted gap / 浜哄彛鍔犳潈 Gap: {gap_baseline['gap_pop_weighted']:+.2f}掳C")

        scenarios = [
            ('S1_citywide', 1, 'S1: Citywide', simulate_s1_citywide, 'S1: Citywide (+10% vegetation shade)'),
            ('S2_corridors', 2, 'S2: Corridors', simulate_s2_corridors, 'S2: Corridors (high road-density)'),
            ('S3_equity_first', 3, 'S3: Equity First', simulate_s3_equity_first, 'S3: Equity First (IMD 1鈥? 鈭?high HEI 鈭?low veg shade)'),
        ]

        for scenario_id, scenario_order, scenario_label, scenario_func, scenario_title in scenarios:
            print(f"\n{scenario_title}")
            df_sim, mask = scenario_func(df.copy())

            # Compute the new gap under both reporting conventions / 璁＄畻鏂扮殑宸窛锛堜袱绉嶅彛寰勶級
            gap_new = calculate_gap(df_sim, 'hei_new')

            # Coverage metrics / 瑕嗙洊鐜?            n_target = mask.sum()
            pop_target = df_sim.loc[mask, 'TotPop'].sum()
            pop_total = df_sim['TotPop'].sum()

            # Percentage reduction relative to baseline / 鍑忓皯鐧惧垎姣?            reduction_lsoa = (gap_baseline['gap_lsoa_avg'] - gap_new['gap_lsoa_avg']) / abs(gap_baseline['gap_lsoa_avg']) * 100
            reduction_pop = (gap_baseline['gap_pop_weighted'] - gap_new['gap_pop_weighted']) / abs(gap_baseline['gap_pop_weighted']) * 100

            all_results.append({
                'time_scenario': scenario_name,
                'policy_scenario': scenario_id,
                'scenario_order': scenario_order,
                'scenario_label': scenario_label,
                # LSOA-average metric / LSOA 骞冲潎鍙ｅ緞
                'poor_hei_lsoa': gap_new['poor_lsoa_avg'],
                'rich_hei_lsoa': gap_new['rich_lsoa_avg'],
                'gap_lsoa': gap_new['gap_lsoa_avg'],
                # Population-weighted metric / 浜哄彛鍔犳潈鍙ｅ緞
                'poor_hei_pop': gap_new['poor_pop_weighted'],
                'rich_hei_pop': gap_new['rich_pop_weighted'],
                'gap_pop': gap_new['gap_pop_weighted'],
                # Coverage metrics / 瑕嗙洊鐜?                'target_lsoa_pct': n_target / len(df) * 100,
                'target_pop_pct': pop_target / pop_total * 100,
                'gap_reduction_lsoa_pct': reduction_lsoa,
                'gap_reduction_pop_pct': reduction_pop
            })

            print("  Results / 缁撴灉:")
            print(f"    LSOA-average / LSOA 骞冲潎: Gap {gap_baseline['gap_lsoa_avg']:.2f}掳C 鈫?{gap_new['gap_lsoa_avg']:.2f}掳C (鈫搟reduction_lsoa:.1f}%)")
            print(f"    Population-weighted / 浜哄彛鍔犳潈: Gap {gap_baseline['gap_pop_weighted']:.2f}掳C 鈫?{gap_new['gap_pop_weighted']:.2f}掳C (鈫搟reduction_pop:.1f}%)")

            # Save decile-level outputs for later plotting / 淇濆瓨鍚?Decile 鐨勮缁嗙粨鏋?            for d in range(1, 11):
                decile_data = df_sim[df_sim['IMD_Decile'] == d]
                scenario_details.append({
                    'time_scenario': scenario_name,
                    'policy_scenario': scenario_id,
                    'scenario_order': scenario_order,
                    'scenario_label': scenario_label,
                    'IMD_Decile': d,
                    'hei_mean_lsoa': decile_data['hei_new'].mean(),
                    'hei_mean_pop': np.average(decile_data['hei_new'], weights=decile_data['TotPop']) if len(decile_data) > 0 else np.nan,
                    'n_lsoa': len(decile_data),
                    'total_pop': decile_data['TotPop'].sum()
                })

    # Save scenario-level and decile-level outputs / 淇濆瓨缁撴灉
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(SUPPLEMENT_DIR / "policy_scenarios_summary_v2.csv", index=False)
    print("\nSaved output / 宸蹭繚瀛? policy_scenarios_summary_v2.csv")

    details_df = pd.DataFrame(scenario_details)
    details_df.to_csv(SUPPLEMENT_DIR / "policy_scenarios_by_decile_v2.csv", index=False)
    print("Saved output / 宸蹭繚瀛? policy_scenarios_by_decile_v2.csv")

    return results_df, details_df


def plot_scenario_comparison():
    """Plot the policy-scenario comparison under both metrics / 缁樺埗涓夌鎯呮櫙瀵规瘮鍥?- 鏄剧ず涓ょ鍙ｅ緞"""
    print("\nDrawing policy-scenario comparison / 缁樺埗鏀跨瓥鎯呮櫙瀵规瘮鍥?..")

    results_df = pd.read_csv(SUPPLEMENT_DIR / "policy_scenarios_summary_v2.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for row, gap_type in enumerate(['lsoa', 'pop']):
        gap_col = f'gap_{gap_type}'
        reduction_col = f'gap_reduction_{gap_type}_pct'
        title_suffix = 'LSOA Average' if gap_type == 'lsoa' else 'Population-Weighted'

        for col, time_scenario in enumerate(['typical_day', 'heatwave']):
            ax = axes[row, col]
            data = results_df[results_df['time_scenario'] == time_scenario]

            scenarios = ['baseline', 'S1_citywide', 'S2_corridors', 'S3_equity_first']
            labels = ['Baseline', 'S1:\nCitywide', 'S2:\nCorridors', 'S3:\nEquity First']
            # Keep colours semantically aligned with scenario meaning, not ordering / 閰嶈壊鎸夋儏鏅惈涔夎€屼笉鏄『搴忓搴?            colors = [COLORS['baseline'], COLORS['scenario3'], COLORS['scenario2'], COLORS['scenario1']]

            gaps = [data[data['policy_scenario'] == s][gap_col].values[0] for s in scenarios]
            reductions = [data[data['policy_scenario'] == s][reduction_col].values[0] for s in scenarios]

            bars = ax.bar(labels, gaps, color=colors, edgecolor='black', linewidth=0.5)

            for bar, gap, red in zip(bars, gaps, reductions):
                height = bar.get_height()
                y_pos = height + 0.02 if height >= 0 else height - 0.1
                ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                       f'{gap:.2f}掳C', ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=9, fontweight='bold')
                if red > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, max(height/2, 0.1),
                           f'鈫搟red:.0f}%', ha='center', va='center', fontsize=8, color='white')

            ax.set_ylabel('HEI Gap (掳C)', fontsize=10)
            time_title = 'Typical Day' if col == 0 else 'Heatwave'
            ax.set_title(f'{time_title} - {title_suffix}', fontsize=11, fontweight='bold')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_ylim(min(gaps) - 0.3, max(gaps) * 1.3)

    plt.suptitle('Policy Scenario Comparison: Impact on Heat Exposure Inequality\n(Two Metrics: LSOA Average vs Population-Weighted)',
                fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIGURES_DIR / 'Fig_policy_scenarios_comparison_v2.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'Fig_policy_scenarios_comparison_v2.pdf', bbox_inches='tight')
    plt.close()
    print("Saved output / 宸蹭繚瀛? Fig_policy_scenarios_comparison_v2.png/pdf")


def main():
    print("=" * 60)
    print("Full policy-scenario simulation and urban-form analysis (revised version) / 瀹屾暣鏀跨瓥鎯呮櫙妯℃嫙涓庡煄甯傚舰鎬佸垎鏋愶紙淇鐗堬級")
    print("=" * 60)
    print("\nFix summary / 淇鍐呭:")
    print("  1. S2 road density / S2 璺綉瀵嗗害: total_length / area_km2")
    print("  2. Shadow clipping / 闃村奖瑁佸壀: ensure shadow 鈭?[0, 1]")
    print("  3. Dual reporting / 鍙屽彛寰勬姤鍛? LSOA 骞冲潎 + 浜哄彛鍔犳潈")
    print("  4. Coverage metrics / 瑕嗙洊鐜囧弻鍙ｅ緞: LSOA% + 浜哄彛%")
    # NOTE: Use ASCII only to avoid Windows GBK console encoding errors.
    print("  5. HEI formula consistency / HEI 鍏紡涓€鑷? add the vegetation term -DeltaT_v*S_v")

    # Part 4: LST versus HEI inequality comparison / 绗洓閮ㄥ垎锛欳NI(LST) vs CNI(HEI) 瀵规瘮
    analyze_lst_vs_hei_inequality()

    # Part 5: run the three policy scenarios / 绗簲閮ㄥ垎锛氫笁绉嶆斂绛栨儏鏅ā鎷?    run_all_scenarios()
    plot_scenario_comparison()

    print("\n" + "=" * 60)
    print("Analysis complete / 鍒嗘瀽瀹屾垚锛?)
    print("=" * 60)

    print("\nGenerated files (v2) / 鐢熸垚鐨勬枃浠讹紙v2 鐗堟湰锛?")
    print("  Data / 鏁版嵁:")
    print("    - lst_vs_hei_inequality_comparison_v2.csv")
    print("    - policy_scenarios_summary_v2.csv")
    print("    - policy_scenarios_by_decile_v2.csv")
    print("  Figures / 鍥捐〃:")
    print("    - Fig_policy_scenarios_comparison_v2.png/pdf")


if __name__ == "__main__":
    main()



