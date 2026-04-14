import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
import warnings
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from shapely.ops import linemerge

warnings.filterwarnings('ignore')

# Logging setup / 璁剧疆鏃ュ織
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / 'results' / 'inequality_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'figures').mkdir(exist_ok=True)

log_file = OUTPUT_DIR / f'imd_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration / 閰嶇疆
CITIES = ['London', 'Birmingham', 'Bristol', 'Manchester', 'Newcastle']
HEI_DIR = BASE_DIR / 'results' / 'heat_exposure'
IMD_PATH = BASE_DIR / 'city_boundaries' / 'Indices_of_Multiple_Deprivation_(IMD)_2019_8404722932957776031.gpkg'

# Temperature thresholds used in the inequality summaries / 娓╁害闃堝€?THRESHOLDS = [26, 28, 30, 35]
GCC_SNAP_METERS = 1.0  # endpoint snap tolerance in metres for road-graph connectivity / 绔偣鍚搁檮绮惧害锛堢背锛夛紝鐢ㄤ簬鏋勫缓閬撹矾杩為€氬浘


def fit_city_fixed_effects(df: pd.DataFrame, y_col: str, x_col: str, city_col: str) -> dict:
    """
    OLS with city fixed effects:
        y = 尾0 + 尾1*x + 危 纬_c * I(city=c) + 蔚

    Returns slope 尾1, p-value, R虏, and per-city intercepts (尾0 + 纬_c).

    Notes:
    - Uses classical OLS standard errors (no robust SE).
    - Avoids statsmodels dependency (repo env may not include it).
    """
    if df.empty:
        return {'slope': np.nan, 'p_value': np.nan, 'r2': np.nan, 'intercepts': {}, 'baseline_city': None}

    data = df[[y_col, x_col, city_col]].dropna().copy()
    if data.empty:
        return {'slope': np.nan, 'p_value': np.nan, 'r2': np.nan, 'intercepts': {}, 'baseline_city': None}

    y = data[y_col].to_numpy(dtype=float)
    x = data[x_col].to_numpy(dtype=float)
    cities = sorted(data[city_col].astype(str).unique().tolist())

    # Build design matrix: intercept + x + city dummies (drop_first)
    if len(cities) > 1:
        dummies = pd.get_dummies(data[city_col].astype(str), drop_first=True)
        dummy_cols = dummies.columns.tolist()
        X = np.column_stack([np.ones(len(data)), x, dummies.to_numpy(dtype=float)])
    else:
        dummy_cols = []
        X = np.column_stack([np.ones(len(data)), x])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    df_resid = len(y) - X.shape[1]
    if df_resid <= 0:
        p_value = np.nan
    else:
        s2 = ss_res / df_resid
        xtx_inv = np.linalg.pinv(X.T @ X)
        cov = s2 * xtx_inv
        se = np.sqrt(np.diag(cov))
        se_slope = float(se[1]) if len(se) > 1 else np.nan
        slope = float(beta[1])
        if np.isfinite(se_slope) and se_slope > 0:
            t_stat = slope / se_slope
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_resid))
        else:
            p_value = np.nan

    baseline_city = cities[0] if cities else None
    intercepts: dict[str, float] = {}
    if baseline_city is not None:
        intercept_base = float(beta[0])
        intercepts[baseline_city] = intercept_base
        for i, c in enumerate(dummy_cols):
            intercepts[str(c)] = intercept_base + float(beta[2 + i])

    return {
        'slope': float(beta[1]) if len(beta) > 1 else np.nan,
        'p_value': float(p_value) if p_value is not None else np.nan,
        'r2': float(r2) if r2 is not None else np.nan,
        'intercepts': intercepts,
        'baseline_city': baseline_city,
    }


def load_imd_data():
    """Load IMD polygons and deprivation attributes / 鍔犺浇 IMD 鏁版嵁"""
    logger.info("Loading IMD data / 鍔犺浇 IMD 鏁版嵁...")
    imd = gpd.read_file(IMD_PATH)

    # Reproject to British National Grid / 杞崲鍒?British National Grid
    if imd.crs.to_epsg() != 27700:
        imd = imd.to_crs(epsg=27700)

    logger.info(f"  Total LSOAs / LSOA 鎬绘暟: {len(imd)}")
    logger.info(f"  IMD decile range / IMD Decile 鑼冨洿: {imd['IMD_Decile'].min()} - {imd['IMD_Decile'].max()}")

    return imd


def load_roads_data(city, scenario='typical_day'):
    """Load road-level HEI data for one city and one scenario / 鍔犺浇閬撹矾 HEI 鏁版嵁"""
    file_path = HEI_DIR / f'{city}_roads_hei_improved_{scenario}.gpkg'
    if not file_path.exists():
        logger.warning(f"File not found / 鏂囦欢涓嶅瓨鍦? {file_path}")
        return None

    roads = gpd.read_file(file_path)

    # Ensure CRS consistency / 纭繚 CRS 涓€鑷?
    if roads.crs.to_epsg() != 27700:
        roads = roads.to_crs(epsg=27700)

    # Compute road-segment length in projected units / 璁＄畻閬撹矾闀垮害
    roads['length'] = roads.geometry.length
    roads['city'] = city

    return roads


def _snap_xy(xy: tuple[float, float], snap_m: float = 1.0) -> tuple[int, int]:
    """Snap coordinates to a metre grid to avoid disconnection from floating-point noise / 灏嗗潗鏍囧惛闄勫埌鎸囧畾绫崇骇缃戞牸锛岄伩鍏嶆诞鐐硅宸鑷寸殑鏂繛銆?""
    x, y = xy
    return (int(round(x / snap_m)), int(round(y / snap_m)))


def _representative_linestring(geom):
    """Extract a representative line segment for endpoint-based connectivity / 浠?`LineString|MultiLineString` 涓彁鍙栧彲鐢ㄤ簬绔偣杩為€氱殑浠ｈ〃绾挎銆?""
    if geom is None or geom.is_empty:
        return None

    gtype = geom.geom_type
    if gtype == 'LineString':
        return geom
    if gtype == 'MultiLineString':
        if len(geom.geoms) == 1:
            return geom.geoms[0]
        try:
            merged = linemerge(geom)
            if merged.geom_type == 'LineString':
                return merged
        except Exception:
            pass
        # Fallback to the longest component line / 鍏滃簳锛氬彇鏈€闀垮瓙绾挎
        return max(list(geom.geoms), key=lambda g: g.length)

    return None


def add_gcc_membership_by_hei(roads: gpd.GeoDataFrame, thresholds=THRESHOLDS,
                              hei_col: str = 'hei_improved', snap_m: float = 1.0) -> gpd.GeoDataFrame:
    """
    Step 2.3 from the design document: compute citywide GCC membership under HEI thresholds / Step 2.3锛坉esign doc锛夛細鍩轰簬 HEI 闃堝€肩殑鍏ㄥ煄杩為€氭€э紙GCC锛?
    - Treat each road as an undirected edge connecting its two endpoints / 灏嗘瘡鏉￠亾璺涓鸿繛鎺ュ叾绔偣鐨勬棤鍚戣竟
    - For each threshold `t`, build the cool-edge subgraph where `HEI <= t` and compute its giant connected component / 瀵规瘡涓槇鍊?`t`锛屽彇婊¤冻 `HEI <= t` 鐨勨€滃喎杈光€濆瓙鍥撅紝璁＄畻鏈€澶ц繛閫氬垎閲忥紙GCC锛?    - Output column `in_gcc_{t}` is `True` when a road belongs to the GCC at threshold `t` / 杈撳嚭鍒?`in_gcc_{t}` 涓紝`True` 琛ㄧず璇ラ亾璺湪闃堝€?`t` 涓嬪睘浜?GCC 鐨勫喎缃戠粶
    """
    roads = roads.copy()

    # Initialize GCC membership columns as False / 鍒濆鍖栧垪锛堥粯璁?False锛?    for t in thresholds:
        roads[f'in_gcc_{t}'] = False

    if hei_col not in roads.columns:
        logger.warning(f"Missing column {hei_col}, skipping GCC calculation / 缂哄皯鍒?{hei_col}锛岃烦杩?GCC 璁＄畻")
        return roads

    hei = roads[hei_col].to_numpy()
    valid_edge = np.isfinite(hei)
    if valid_edge.sum() == 0:
        return roads

    # Extract road endpoints and snap them to a common grid / 鎻愬彇绔偣骞跺惛闄?    u_nodes = []
    v_nodes = []
    for geom in roads.geometry:
        line = _representative_linestring(geom)
        if line is None or line.is_empty:
            u_nodes.append(None)
            v_nodes.append(None)
            continue
        coords = list(line.coords)
        if len(coords) < 2:
            u_nodes.append(None)
            v_nodes.append(None)
            continue
        u_nodes.append(_snap_xy(coords[0], snap_m=snap_m))
        v_nodes.append(_snap_xy(coords[-1], snap_m=snap_m))

    # Valid graph edges must have both endpoints and a finite HEI value / 鏈夋晥鐨勫浘杈癸紙鏃㈡湁绔偣涔熸湁鏈夋晥 HEI锛?    has_uv = np.array([(u is not None and v is not None) for u, v in zip(u_nodes, v_nodes)])
    edge_mask = valid_edge & has_uv

    if edge_mask.sum() == 0:
        return roads

    edge_indices = np.flatnonzero(edge_mask)
    u_list = [u_nodes[i] for i in edge_indices]
    v_list = [v_nodes[i] for i in edge_indices]
    hei_arr = hei[edge_mask]

    # Union-find structure for GCC tracking / 骞舵煡闆?    parents: dict[object, object] = {}
    sizes: dict[object, int] = {}

    def find(x):
        parents.setdefault(x, x)
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        sa, sb = sizes.get(ra, 1), sizes.get(rb, 1)
        if sa < sb:
            ra, rb = rb, ra
            sa, sb = sb, sa
        parents[rb] = ra
        sizes[ra] = sa + sb
        sizes[rb] = 0

    # Initialize all snapped nodes in the union-find structure / 鍒濆鍖栬妭鐐归泦鍚?    nodes = set(u_list).union(set(v_list))
    for n in nodes:
        parents[n] = n
        sizes[n] = 1

    # Add edges incrementally as thresholds increase / 鎸?HEI 鍗囧簭锛岄槇鍊奸€掑鏃跺閲忓姞鍏ヨ竟
    order = np.argsort(hei_arr)
    hei_sorted = hei_arr[order]
    u_sorted = [u_list[i] for i in order]
    v_sorted = [v_list[i] for i in order]

    thresholds_sorted = sorted(thresholds)
    ptr = 0

    for t in thresholds_sorted:
        while ptr < len(hei_sorted) and hei_sorted[ptr] <= t:
            union(u_sorted[ptr], v_sorted[ptr])
            ptr += 1

        # Identify the current GCC root with the largest component size / 褰撳墠 GCC 鏍癸紙鏈€澶?sizes锛?        gcc_root, gcc_size = max(sizes.items(), key=lambda kv: kv[1])
        if gcc_size <= 1:
            continue

        # Mark roads that are cool and whose two endpoints both belong to the GCC / 鏍囪閬撹矾鏄惁鍦?GCC锛氬喎杈逛笖涓や釜绔偣閮藉睘浜?GCC
        roots_u = [find(u) for u in u_list]
        roots_v = [find(v) for v in v_list]

        in_gcc = np.fromiter(
            (h <= t and ru == gcc_root and rv == gcc_root for h, ru, rv in zip(hei_arr, roots_u, roots_v)),
            dtype=bool,
            count=len(hei_arr),
        )
        col = f'in_gcc_{t}'
        roads.iloc[edge_indices[in_gcc], roads.columns.get_loc(col)] = True

    return roads


def spatial_join_roads_to_lsoa(roads, imd):
    """
    Step 1: spatially link roads to LSOAs using road centroids / Step 1锛氶亾璺?LSOA 绌洪棿鍏宠仈锛屼娇鐢ㄩ亾璺腑蹇冪偣锛坈entroid锛夎繘琛岀┖闂磋繛鎺?    """
    # Replace road geometries with centroids for the spatial join / 鍒涘缓閬撹矾涓績鐐?    roads_centroid = roads.copy()
    roads_centroid['geometry'] = roads_centroid.geometry.centroid

    # Perform the spatial join / 绌洪棿杩炴帴
    roads_with_lsoa = gpd.sjoin(
        roads_centroid,
        imd[['lsoa11cd', 'lsoa11nm', 'IMD_Decile', 'IMD_Rank', 'IMDScore',
             'IncDec', 'EmpDec', 'EnvDec', 'TotPop', 'geometry']],
        how='left',
        predicate='within'
    )

    # Restore the original road geometries after the join / 鎭㈠鍘熷閬撹矾鍑犱綍
    roads_with_lsoa['geometry'] = roads.geometry.values

    # Report join coverage / 缁熻鍖归厤鎯呭喌
    matched = roads_with_lsoa['lsoa11cd'].notna().sum()
    total = len(roads_with_lsoa)
    logger.info(f"  Spatial matches / 绌洪棿鍖归厤: {matched}/{total} ({matched/total*100:.1f}%)")

    return roads_with_lsoa


def aggregate_lsoa_metrics(roads_with_lsoa, thresholds=THRESHOLDS):
    """Step 2: aggregate road metrics to the LSOA scale / Step 2锛歀SOA 绾у埆鎸囨爣鑱氬悎"""
    # Drop roads without an LSOA match / 杩囨护鎺夋湭鍖归厤鐨勯亾璺?    roads_matched = roads_with_lsoa[roads_with_lsoa['lsoa11cd'].notna()].copy()

    # Keep only roads with valid thermal metrics to avoid raster-edge artefacts / 浠呬繚鐣欐湁鏈夋晥鐑毚闇叉暟鎹殑閬撹矾锛堥伩鍏嶆爡鏍艰寖鍥村閲囨牱瀵艰嚧鐨?0/NaN 骞叉壈锛?    roads_matched = roads_matched[np.isfinite(roads_matched['lst']) & np.isfinite(roads_matched['hei_improved'])].copy()

    # Aggregate by LSOA / 鎸?LSOA 鍒嗙粍鑱氬悎
    lsoa_stats = roads_matched.groupby('lsoa11cd').agg({
        'city': 'first',
        'lsoa11nm': 'first',
        'IMD_Decile': 'first',
        'IMD_Rank': 'first',
        'IMDScore': 'first',
        'IncDec': 'first',
        'EmpDec': 'first',
        'EnvDec': 'first',
        'TotPop': 'first',
        'lst': ['mean', 'std', 'min', 'max'],
        'hei_improved': ['mean', 'std', 'min', 'max'],
        'shadow_daily_avg': 'mean',
        'shadow_building_avg': 'mean',
        'shadow_vegetation_avg': 'mean',
        'length': ['sum', 'count']
    }).reset_index()

    # Flatten the grouped column names / 灞曞钩鍒楀悕
    lsoa_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                          for col in lsoa_stats.columns]

    # Rename grouped columns to cleaner output names / 閲嶅懡鍚嶏紝鍘绘帀 `_first` 鍚庣紑骞舵爣鍑嗗寲
    rename_dict = {
        'city_first': 'city',
        'lsoa11nm_first': 'lsoa11nm',
        'IMD_Decile_first': 'IMD_Decile',
        'IMD_Rank_first': 'IMD_Rank',
        'IMDScore_first': 'IMDScore',
        'IncDec_first': 'IncDec',
        'EmpDec_first': 'EmpDec',
        'EnvDec_first': 'EnvDec',
        'TotPop_first': 'TotPop',
        'lst_mean': 'lst_mean',
        'lst_std': 'lst_std',
        'hei_improved_mean': 'hei_mean',
        'hei_improved_std': 'hei_std',
        'shadow_daily_avg_mean': 'shadow_mean',
        'shadow_building_avg_mean': 'shadow_building_mean',
        'shadow_vegetation_avg_mean': 'shadow_vegetation_mean',
        'length_sum': 'total_length',
        'length_count': 'n_roads'
    }
    lsoa_stats = lsoa_stats.rename(columns=rename_dict)

    # Compute mean cooling benefit as LST minus HEI / 璁＄畻娓╁害闄嶄綆閲?    lsoa_stats['temp_reduction'] = lsoa_stats['lst_mean'] - lsoa_stats['hei_mean']

    # Denominator: number of valid road segments inside each LSOA / 鍒嗘瘝锛歀SOA 鍐?valid 鎬婚亾璺鏁?    total_counts = roads_matched.groupby('lsoa11cd').size()

    # Compute CoolShare and CNI for each threshold / 璁＄畻鍚勯槇鍊肩殑 CoolShare
    for threshold in thresholds:
        # CoolShare: HEI <= threshold 鐨勯亾璺鏁版瘮渚嬶紙鍒嗘瘝=LSOA鍐?valid 鎬婚亾璺鏁帮級
        cool_counts = roads_matched[roads_matched['hei_improved'] <= threshold].groupby('lsoa11cd').size()
        cool_share = (cool_counts / total_counts).fillna(0)
        lsoa_stats[f'cool_share_{threshold}'] = lsoa_stats['lsoa11cd'].map(cool_share)

        # CNI: 鍐蜂笖灞炰簬鍏ㄥ煄 GCC 鐨勯亾璺鏁版瘮渚嬶紙鍒嗘瘝=LSOA鍐?valid 鎬婚亾璺鏁帮紱闇€棰勫厛璁＄畻 in_gcc_{threshold}锛?        gcc_col = f'in_gcc_{threshold}'
        if gcc_col in roads_matched.columns:
            cni_counts = roads_matched[roads_matched[gcc_col]].groupby('lsoa11cd').size()
            cni_share = (cni_counts / total_counts).fillna(0)
            lsoa_stats[f'cni_{threshold}'] = lsoa_stats['lsoa11cd'].map(cni_share)

    return lsoa_stats


def calculate_imd_group_stats(lsoa_stats):
    """
    Step 3: summarise metrics by IMD decile / Step 3锛氭寜 IMD 鍗佸垎浣嶆眹鎬绘寚鏍?    """
    agg_dict = {
        'hei_mean': ['mean', 'std', 'count'],
        'lst_mean': ['mean', 'std'],
        'temp_reduction': ['mean', 'std'],
        'shadow_mean': ['mean', 'std'],
        'cool_share_28': ['mean', 'std'],
        'cool_share_35': ['mean', 'std'],
        'n_roads': 'sum',
        'TotPop': 'sum'
    }

    # Include CNI metrics when GCC membership is available / 濡傛灉宸茶绠?CNI锛堝喎涓斿睘浜?GCC 鐨勯亾璺鏁板崰姣旓級锛屼篃绾冲叆鍒嗙粍缁熻
    for t in THRESHOLDS:
        col = f'cni_{t}'
        if col in lsoa_stats.columns:
            agg_dict[col] = ['mean', 'std']

    # Aggregate metrics by IMD decile / 鎸?IMD 鍗佸垎浣嶅垎缁勭粺璁?    decile_stats = lsoa_stats.groupby('IMD_Decile').agg(agg_dict).reset_index()

    # Flatten column names / 灞曞钩鍒楀悕
    decile_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                            for col in decile_stats.columns]

    return decile_stats


def statistical_tests(lsoa_stats):
    """
    Compare the most deprived and most affluent groups statistically / 瀵规渶璐洶缁勪笌鏈€瀵岃缁勮繘琛岀粺璁℃楠?    """
    results = {}

    # Compare the most deprived 20% with the least deprived 20% / 鏈€璐洶 20%锛圖ecile 1-2锛塿s 鏈€瀵岃 20%锛圖ecile 9-10锛?    poor = lsoa_stats[lsoa_stats['IMD_Decile'].isin([1, 2])]
    rich = lsoa_stats[lsoa_stats['IMD_Decile'].isin([9, 10])]

    metrics = ['hei_mean', 'lst_mean', 'temp_reduction', 'shadow_mean', 'cool_share_28', 'cool_share_35']
    for t in THRESHOLDS:
        col = f'cni_{t}'
        if col in lsoa_stats.columns:
            metrics.append(col)

    for metric in metrics:
        poor_vals = poor[metric].dropna()
        rich_vals = rich[metric].dropna()

        # Welch's t-test / Welch t 妫€楠?        t_stat, p_value = stats.ttest_ind(poor_vals, rich_vals, equal_var=False)

        # Mann-Whitney U test / Mann-Whitney U 妫€楠?        u_stat, u_pvalue = stats.mannwhitneyu(poor_vals, rich_vals, alternative='two-sided')

        results[metric] = {
            'poor_mean': poor_vals.mean(),
            'poor_std': poor_vals.std(),
            'rich_mean': rich_vals.mean(),
            'rich_std': rich_vals.std(),
            'difference': poor_vals.mean() - rich_vals.mean(),
            't_statistic': t_stat,
            't_pvalue': p_value,
            'u_statistic': u_stat,
            'u_pvalue': u_pvalue
        }

    return results


def calculate_concentration_index(lsoa_stats, value_col='hei_mean', rank_col='IMD_Rank', weight_col='TotPop'):
    """
    Calculate the deprivation-ranked concentration curve and concentration index /
    璁＄畻鎸夎传鍥版帓搴忕殑 Concentration curve 鍜?Concentration index锛圕I锛?
    Consistent with the study design /
    涓庤璁℃枃妗ｄ竴鑷达細
    - X axis: cumulative population share from most deprived to least deprived /
      X 杞达細浜哄彛浠庘€滄渶璐洶 鈫?鏈€瀵岃鈥濈殑绱Н姣斾緥锛堟寜 IMD_Rank 鍗囧簭锛?    - Y axis: cumulative population-weighted heat exposure /
      Y 杞达細鐑毚闇茬殑绱Н姣斾緥锛堜汉鍙ｅ姞鏉冿級

    CI = 1 - 2 * area_under_curve
    - CI < 0 means heat exposure is concentrated among deprived groups /
      CI < 0 琛ㄧず鐑毚闇叉洿闆嗕腑鍦ㄨ传鍥扮兢浣擄紙鏇茬嚎鍦ㄥ瑙掔嚎涓婃柟锛?    - CI > 0 means heat exposure is concentrated among affluent groups /
      CI > 0 琛ㄧず鐑毚闇叉洿闆嗕腑鍦ㄥ瘜瑁曠兢浣擄紙鏇茬嚎鍦ㄥ瑙掔嚎涓嬫柟锛?    """
    df = lsoa_stats[[value_col, rank_col, weight_col]].dropna().copy()

    # Sort from most deprived to least deprived; smaller IMD rank means more deprived / 鎸夎传鍥扮▼搴︽帓搴忥細IMD_Rank 瓒婂皬瓒婅传鍥?    df = df.sort_values(rank_col, ascending=True)

    # Cumulative population share / 绱Н浜哄彛姣斾緥
    df['cum_pop'] = df[weight_col].cumsum() / df[weight_col].sum()

    # Cumulative population-weighted heat exposure / 绱Н鐑毚闇诧紙浜哄彛鍔犳潈锛?    df['weighted_value'] = df[value_col] * df[weight_col]
    df['cum_value'] = df['weighted_value'].cumsum() / df['weighted_value'].sum()

    # Area under the curve and final concentration index / 璁＄畻鏇茬嚎闈㈢Н涓庢渶缁?CI
    curve_area = np.trapz(df['cum_value'], df['cum_pop'])
    ci = 1 - 2 * curve_area

    return df[['cum_pop', 'cum_value']], ci


def plot_imd_boxplots(lsoa_stats, scenario, output_dir):
    """
    Plot boxplots by IMD decile / 缁樺埗 IMD Decile 绠辩嚎鍥?    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    cni_col = 'cni_28' if 'cni_28' in lsoa_stats.columns else 'cool_share_28'
    cni_title = 'Connected Cool Share (CNI, HEI 鈮?28掳C)' if cni_col == 'cni_28' else 'Cool Road Share (HEI 鈮?28掳C)'

    metrics = [
        ('hei_mean', 'Heat Exposure Index (掳C)', 'HEI Mean'),
        (cni_col, 'Proportion', cni_title),
        ('temp_reduction', 'Cooling Benefit (掳C)', '螖HEI = LST - HEI'),
        ('shadow_mean', 'Proportion', 'Shadow Coverage'),
    ]

    for ax, (metric, ylabel, title) in zip(axes.flatten(), metrics):
        data = [lsoa_stats[lsoa_stats['IMD_Decile'] == d][metric].dropna()
                for d in range(1, 11)]

        bp = ax.boxplot(data, labels=range(1, 11), patch_artist=True)

        # Colour ramp from deprived (red) to affluent (blue) / 棰滆壊锛氳传鍥帮紙绾級鈫?瀵岃锛堣摑锛?        colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, 10))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel('IMD Decile (1=Most Deprived, 10=Least Deprived)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Heat Exposure by IMD Decile - {scenario.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'imd_boxplots_{scenario}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"淇濆瓨: {output_path}")
    plt.close()


def plot_concentration_curve(curve_lst, ci_lst, curve_hei, ci_hei, scenario, output_dir):
    """
    Plot concentration curves ranked by IMD / 缁樺埗鎸?IMD 鎺掑簭鐨?Concentration curve
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect equality line / 瀹屽叏骞崇瓑绾?    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Equality', linewidth=1)

    # LST-only curve
    ax.plot(curve_lst['cum_pop'], curve_lst['cum_value'],
            'r-', linewidth=2, label=f'LST-only (CI={ci_lst:+.3f})')

    # HEI-based curve
    ax.plot(curve_hei['cum_pop'], curve_hei['cum_value'],
            'b-', linewidth=2, label=f'HEI-based (CI={ci_hei:+.3f})')

    ax.set_xlabel('Cumulative Population (Most Deprived 鈫?Least Deprived)')
    ax.set_ylabel('Cumulative Heat Exposure')
    ax.set_title(f'Concentration Curve (Ranked by IMD)\n{scenario.replace("_", " ").title()}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    output_path = output_dir / f'lorenz_curve_{scenario}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved figure / 淇濆瓨鍥句欢: {output_path}")
    plt.close()


def plot_scatter_regression(lsoa_stats, scenario, output_dir):
    """
    Plot scatterplots and fixed-effect regressions for IMD versus HEI or CNI /
    缁樺埗 IMD 涓?HEI 鎴?CNI 鐨勬暎鐐瑰浘鍙婂浐瀹氭晥搴斿洖褰掔嚎
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # IMD Rank vs HEI
    ax1 = axes[0]
    cni_col = 'cni_28' if 'cni_28' in lsoa_stats.columns else 'cool_share_28'
    df = lsoa_stats[['hei_mean', cni_col, 'IMD_Rank', 'city']].dropna().copy()
    df['imd_rank_norm'] = df['IMD_Rank'] / df['IMD_Rank'].max()

    fit_hei = fit_city_fixed_effects(df, 'hei_mean', 'imd_rank_norm', 'city')

    # Draw city-coloured scatter points and city fixed-effect lines with a shared slope / 鎸夊煄甯傜潃鑹叉暎鐐瑰苟缁樺埗鍥哄畾鏁堝簲鍥炲綊绾匡紙鍚屼竴鏂滅巼锛屼笉鍚屾埅璺濓級
    cities = sorted(df['city'].astype(str).unique().tolist())
    colors = plt.cm.tab10(np.linspace(0, 1, len(cities)))

    for c, col in zip(cities, colors):
        sub = df[df['city'].astype(str) == c]
        ax1.scatter(sub['imd_rank_norm'], sub['hei_mean'], alpha=0.25, s=10, color=col, label=c)

        intercept = fit_hei['intercepts'].get(c, np.nan)
        if np.isfinite(intercept) and np.isfinite(fit_hei['slope']):
            ax1.plot([0, 1], [intercept, intercept + fit_hei['slope']], color=col, linewidth=1.5, alpha=0.9)

    ax1.set_xlabel('IMD Rank (Normalized, 0=Most Deprived)')
    ax1.set_ylabel('Heat Exposure Index (掳C)')
    ax1.set_title(
        'IMD Rank vs Heat Exposure\n'
        f"FE slope={fit_hei['slope']:.2f}, p={fit_hei['p_value']:.2e}, R虏={fit_hei['r2']:.3f}"
    )
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # IMD Rank vs CNI (or CoolShare fallback)
    ax2 = axes[1]
    fit_cni = fit_city_fixed_effects(df, cni_col, 'imd_rank_norm', 'city')

    for c, col in zip(cities, colors):
        sub = df[df['city'].astype(str) == c]
        ax2.scatter(sub['imd_rank_norm'], sub[cni_col], alpha=0.25, s=10, color=col, label=c)

        intercept = fit_cni['intercepts'].get(c, np.nan)
        if np.isfinite(intercept) and np.isfinite(fit_cni['slope']):
            ax2.plot([0, 1], [intercept, intercept + fit_cni['slope']], color=col, linewidth=1.5, alpha=0.9)

    ax2.set_xlabel('IMD Rank (Normalized, 0=Most Deprived)')
    ax2.set_ylabel('CNI (HEI 鈮?28掳C)' if cni_col == 'cni_28' else 'Cool Road Share (HEI 鈮?28掳C)')
    ax2.set_title(
        ('IMD Rank vs CNI\n' if cni_col == 'cni_28' else 'IMD Rank vs Cool Road Share\n') +
        f"FE slope={fit_cni['slope']:.3f}, p={fit_cni['p_value']:.2e}, R虏={fit_cni['r2']:.3f}"
    )
    ax2.legend(loc='lower right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Socioeconomic Deprivation vs Heat Exposure - {scenario.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'imd_scatter_regression_{scenario}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved figure / 淇濆瓨鍥句欢: {output_path}")
    plt.close()

    return {
        'hei_slope_fe': fit_hei['slope'], 'hei_p_fe': fit_hei['p_value'], 'hei_r2_fe': fit_hei['r2'],
        'cni_slope_fe': fit_cni['slope'], 'cni_p_fe': fit_cni['p_value'], 'cni_r2_fe': fit_cni['r2']
    }


def process_scenario(scenario, imd):
    """Process one scenario end-to-end / 澶勭悊鍗曚釜鍦烘櫙鐨勫畬鏁存祦绋?""
    logger.info(f"\n{'='*60}")
    logger.info(f"Scenario / 鍦烘櫙: {scenario}")
    logger.info(f"{'='*60}")

    all_roads = []

    # Load road-level data for all cities / 鍔犺浇鎵€鏈夊煄甯傜殑閬撹矾鏁版嵁
    for city in CITIES:
        logger.info(f"\nProcessing city / 澶勭悊鍩庡競: {city}...")
        roads = load_roads_data(city, scenario)
        if roads is None:
            continue

        logger.info(f"  Road segments / 閬撹矾鏁伴噺: {len(roads)}")

        # Compute citywide GCC membership under HEI thresholds / 璁＄畻鍏ㄥ煄 GCC 鎴愬憳锛堝熀浜?HEI 闃堝€硷級
        roads = add_gcc_membership_by_hei(roads, thresholds=THRESHOLDS, hei_col='hei_improved', snap_m=GCC_SNAP_METERS)

        # Spatially attach roads to LSOAs / 绌洪棿鍏宠仈
        roads_with_lsoa = spatial_join_roads_to_lsoa(roads, imd)
        roads_with_lsoa['city'] = city
        all_roads.append(roads_with_lsoa)

    # Merge all city-level road tables / 鍚堝苟鎵€鏈夊煄甯傜粨鏋?    if not all_roads:
        raise RuntimeError("No road data loaded for any city; check input files and scenarios.")
    all_roads_df = pd.concat(all_roads, ignore_index=True)
    logger.info(f"\nTotal road segments / 鎬婚亾璺暟: {len(all_roads_df)}")

    # Aggregate road metrics to the LSOA level / LSOA 绾у埆鑱氬悎
    logger.info("\nAggregating LSOA-level metrics / 鑱氬悎 LSOA 绾у埆鎸囨爣...")
    lsoa_stats = aggregate_lsoa_metrics(all_roads_df)
    logger.info(f"  Valid LSOAs / 鏈夋晥 LSOA 鏁? {len(lsoa_stats)}")

    # Save LSOA summary table / 淇濆瓨 LSOA 姹囨€?    lsoa_path = OUTPUT_DIR / f'lsoa_hei_summary_{scenario}.csv'
    lsoa_stats.to_csv(lsoa_path, index=False)
    logger.info(f"Saved output / 淇濆瓨缁撴灉: {lsoa_path}")

    # Summarise by IMD decile / IMD 鍒嗙粍缁熻
    logger.info("\nSummarising by IMD decile / IMD 鍒嗙粍缁熻...")
    decile_stats = calculate_imd_group_stats(lsoa_stats)
    decile_path = OUTPUT_DIR / f'imd_decile_stats_{scenario}.csv'
    decile_stats.to_csv(decile_path, index=False)
    logger.info(f"Saved output / 淇濆瓨缁撴灉: {decile_path}")

    # Statistical tests comparing deprived and affluent groups / 缁熻妫€楠?    logger.info("\nRunning statistical tests (deprived vs affluent) / 缁熻妫€楠岋紙璐洶缁?vs 瀵岃缁勶級...")
    test_results = statistical_tests(lsoa_stats)

    for metric, result in test_results.items():
        logger.info(f"\n  {metric}:")
        logger.info(f"    Deprived group (D1-2) / 璐洶缁勶紙D1-2锛? {result['poor_mean']:.3f} 卤 {result['poor_std']:.3f}")
        logger.info(f"    Affluent group (D9-10) / 瀵岃缁勶紙D9-10锛? {result['rich_mean']:.3f} 卤 {result['rich_std']:.3f}")
        logger.info(f"    Difference / 宸紓: {result['difference']:+.3f}")
        logger.info(f"    t-test p-value / t 妫€楠?p 鍊? {result['t_pvalue']:.2e}")

    # Concentration curve/index ranked by IMD / 鎸?IMD 鎺掑簭鐨?Concentration curve 鍜?CI
    logger.info("\nCalculating concentration curves and concentration indices / 璁＄畻 Concentration curve 鍜?Concentration index...")
    curve_lst, ci_lst = calculate_concentration_index(lsoa_stats, 'lst_mean', 'IMD_Rank', 'TotPop')
    curve_hei, ci_hei = calculate_concentration_index(lsoa_stats, 'hei_mean', 'IMD_Rank', 'TotPop')
    logger.info(f"  LST-only CI: {ci_lst:+.4f}")
    logger.info(f"  HEI-based CI: {ci_hei:+.4f}")
    logger.info(f"  CI change (HEI-LST) / CI 鍙樺寲锛圚EI-LST锛? {ci_hei - ci_lst:+.4f}")

    # Generate output figures / 鐢熸垚鍙鍖?    logger.info("\nGenerating figures / 鐢熸垚鍙鍖?..")
    plot_imd_boxplots(lsoa_stats, scenario, OUTPUT_DIR / 'figures')
    plot_concentration_curve(curve_lst, ci_lst, curve_hei, ci_hei, scenario, OUTPUT_DIR / 'figures')
    reg_results = plot_scatter_regression(lsoa_stats, scenario, OUTPUT_DIR / 'figures')

    return {
        'scenario': scenario,
        'n_lsoa': len(lsoa_stats),
        'ci_lst': ci_lst,
        'ci_hei': ci_hei,
        'test_results': test_results,
        'regression': reg_results
    }


def sensitivity_analysis_income_decile(lsoa_stats, scenario):
    """
    Sensitivity analysis 1: replace overall IMD with Income Decile /
    鏁忔劅鎬у垎鏋?1锛氫娇鐢?Income Decile 鏇夸唬缁煎悎 IMD
    """
    logger.info("\nSensitivity analysis: Income Decile / 鏁忔劅鎬у垎鏋愶細Income Decile 鍒嗙粍")

    # Aggregate metrics by income decile / 鎸?IncDec 鍒嗙粍缁熻
    income_stats = lsoa_stats.groupby('IncDec').agg({
        'hei_mean': ['mean', 'std', 'count'],
        'lst_mean': ['mean', 'std'],
        'temp_reduction': ['mean', 'std'],
        'shadow_mean': ['mean', 'std'],
        'cool_share_28': ['mean', 'std'],
        'cool_share_35': ['mean', 'std'],
        'TotPop': 'sum'
    }).reset_index()

    income_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                            for col in income_stats.columns]

    # Statistical tests: lowest-income 20% vs highest-income 20% / 缁熻妫€楠岋細鏀跺叆鏈€浣?20% vs 鏀跺叆鏈€楂?20%
    poor = lsoa_stats[lsoa_stats['IncDec'].isin([1, 2])]
    rich = lsoa_stats[lsoa_stats['IncDec'].isin([9, 10])]

    results = {}
    for metric in ['hei_mean', 'lst_mean', 'cool_share_28']:
        poor_vals = poor[metric].dropna()
        rich_vals = rich[metric].dropna()

        t_stat, p_value = stats.ttest_ind(poor_vals, rich_vals, equal_var=False)

        results[metric] = {
            'poor_mean': poor_vals.mean(),
            'rich_mean': rich_vals.mean(),
            'difference': poor_vals.mean() - rich_vals.mean(),
            't_pvalue': p_value
        }

        logger.info(
            f"  {metric}: low income / 浣庢敹鍏?{poor_vals.mean():.3f} vs "
            f"high income / 楂樻敹鍏?{rich_vals.mean():.3f}, "
            f"difference / 宸紓 {poor_vals.mean() - rich_vals.mean():+.3f}, p={p_value:.2e}"
        )

    # Save summary output / 淇濆瓨缁撴灉
    income_path = OUTPUT_DIR / 'sensitivity' / f'income_decile_stats_{scenario}.csv'
    income_stats.to_csv(income_path, index=False)
    logger.info(f"Saved output / 淇濆瓨缁撴灉: {income_path}")

    return results


def sensitivity_analysis_single_city(city, scenario, imd):
    """
    Sensitivity analysis 2: single-city analysis (London) /
    鏁忔劅鎬у垎鏋?2锛氬崟鍩庡競鍒嗘瀽锛圠ondon锛?    """
    logger.info(f"\nSensitivity analysis: single-city case / 鏁忔劅鎬у垎鏋愶細{city} 鍗曞煄甯傚垎鏋?)

    roads = load_roads_data(city, scenario)
    if roads is None:
        return None

    roads_with_lsoa = spatial_join_roads_to_lsoa(roads, imd)
    lsoa_stats = aggregate_lsoa_metrics(roads_with_lsoa)

    logger.info(f"  {city} LSOA count / LSOA 鏁? {len(lsoa_stats)}")

    # Statistical tests for the selected city / 鍗曞煄甯傜粺璁℃楠?    test_results = statistical_tests(lsoa_stats)

    hei_test = test_results['hei_mean']
    logger.info(f"  HEI gap (deprived-affluent) / HEI 宸紓锛堣传鍥?瀵岃锛? {hei_test['difference']:+.3f}掳C (p={hei_test['t_pvalue']:.2e})")

    cool_test = test_results['cool_share_28']
    logger.info(f"  CoolShare gap / CoolShare 宸紓: {cool_test['difference']:+.3f} (p={cool_test['t_pvalue']:.2e})")

    # Concentration index for a single city, still ranked by IMD / 鍗曞煄甯傛儏褰笅鎸?IMD 鎺掑簭璁＄畻闆嗕腑鎸囨暟
    _, ci_hei = calculate_concentration_index(lsoa_stats, 'hei_mean', 'IMD_Rank', 'TotPop')
    logger.info(f"  CI(HEI): {ci_hei:+.4f}")

    # Save single-city summary / 淇濆瓨鍗曞煄甯傛眹鎬?    city_path = OUTPUT_DIR / 'sensitivity' / f'{city.lower()}_lsoa_stats_{scenario}.csv'
    lsoa_stats.to_csv(city_path, index=False)
    logger.info(f"Saved output / 淇濆瓨缁撴灉: {city_path}")

    # Reuse the boxplot function for the single-city case / 澶嶇敤绠辩嚎鍥惧嚱鏁扮粯鍒跺崟鍩庡競缁撴灉
    plot_imd_boxplots(lsoa_stats, f'{city.lower()}_{scenario}', OUTPUT_DIR / 'sensitivity')

    return {
        'city': city,
        'n_lsoa': len(lsoa_stats),
        'ci_hei': ci_hei,
        'hei_difference': hei_test['difference'],
        'hei_pvalue': hei_test['t_pvalue'],
        'cool_difference': cool_test['difference'],
        'cool_pvalue': cool_test['t_pvalue']
    }


def sensitivity_analysis_employment_decile(lsoa_stats, scenario):
    """
    Sensitivity analysis 3: replace IMD with Employment Decile /
    鏁忔劅鎬у垎鏋?3锛氫娇鐢?Employment Decile
    """
    logger.info("\nSensitivity analysis: Employment Decile / 鏁忔劅鎬у垎鏋愶細Employment Decile 鍒嗙粍")

    # Statistical tests: lowest-employment-decile 20% vs highest 20% / 缁熻妫€楠岋細灏变笟璐洶鏈€浣?20% vs 鏈€楂?20%
    poor = lsoa_stats[lsoa_stats['EmpDec'].isin([1, 2])]
    rich = lsoa_stats[lsoa_stats['EmpDec'].isin([9, 10])]

    results = {}
    for metric in ['hei_mean', 'lst_mean', 'cool_share_28']:
        poor_vals = poor[metric].dropna()
        rich_vals = rich[metric].dropna()

        t_stat, p_value = stats.ttest_ind(poor_vals, rich_vals, equal_var=False)

        results[metric] = {
            'poor_mean': poor_vals.mean(),
            'rich_mean': rich_vals.mean(),
            'difference': poor_vals.mean() - rich_vals.mean(),
            't_pvalue': p_value
        }

        logger.info(
            f"  {metric}: low employment / 浣庡氨涓?{poor_vals.mean():.3f} vs "
            f"high employment / 楂樺氨涓?{rich_vals.mean():.3f}, "
            f"difference / 宸紓 {poor_vals.mean() - rich_vals.mean():+.3f}, p={p_value:.2e}"
        )

    return results


def run_sensitivity_analyses(imd):
    """
    Run all sensitivity checks in sequence / 杩愯鍏ㄩ儴鏁忔劅鎬у垎鏋?    """
    logger.info("\n" + "#" * 60)
    logger.info("Step 5: sensitivity analyses / Step 5锛氭晱鎰熸€у垎鏋?)
    logger.info("#" * 60)

    # Create the output directory for sensitivity results / 鍒涘缓鏁忔劅鎬у垎鏋愯緭鍑虹洰褰?    sensitivity_dir = OUTPUT_DIR / 'sensitivity'
    sensitivity_dir.mkdir(exist_ok=True)

    sensitivity_results = []

    for scenario in ['typical_day', 'heatwave']:
        logger.info(f"\n{'='*50}")
        logger.info(f"Scenario / 鍦烘櫙: {scenario}")
        logger.info('='*50)

        # Read the previously saved LSOA summary / 璇诲彇宸蹭繚瀛樼殑 LSOA 鏁版嵁
        lsoa_path = OUTPUT_DIR / f'lsoa_hei_summary_{scenario}.csv'
        lsoa_stats = pd.read_csv(lsoa_path)

        # Sensitivity analysis 1: Income Decile / 鏁忔劅鎬у垎鏋?1锛欼ncome Decile
        income_results = sensitivity_analysis_income_decile(lsoa_stats, scenario)

        # Sensitivity analysis 2: Employment Decile / 鏁忔劅鎬у垎鏋?2锛欵mployment Decile
        emp_results = sensitivity_analysis_employment_decile(lsoa_stats, scenario)

        # Sensitivity analysis 3: single-city case (London) / 鏁忔劅鎬у垎鏋?3锛氬崟鍩庡競锛圠ondon锛?        london_results = sensitivity_analysis_single_city('London', scenario, imd)

        sensitivity_results.append({
            'scenario': scenario,
            'income_hei_diff': income_results['hei_mean']['difference'],
            'income_hei_pvalue': income_results['hei_mean']['t_pvalue'],
            'emp_hei_diff': emp_results['hei_mean']['difference'],
            'emp_hei_pvalue': emp_results['hei_mean']['t_pvalue'],
            'london_hei_diff': london_results['hei_difference'] if london_results else None,
            'london_hei_pvalue': london_results['hei_pvalue'] if london_results else None,
            'london_ci': london_results['ci_hei'] if london_results else None
        })

    # Save the combined sensitivity summary / 姹囨€绘晱鎰熸€у垎鏋愮粨鏋?    sensitivity_df = pd.DataFrame(sensitivity_results)
    sensitivity_path = OUTPUT_DIR / 'sensitivity' / 'sensitivity_summary.csv'
    sensitivity_df.to_csv(sensitivity_path, index=False)
    logger.info(f"\nSaved sensitivity summary / 鏁忔劅鎬у垎鏋愭眹鎬诲凡淇濆瓨: {sensitivity_path}")

    return sensitivity_results


def main():
    logger.info("=" * 60)
    logger.info("IMD inequality analysis / IMD 涓嶅钩绛夊垎鏋?)
    logger.info("=" * 60)

    # Load IMD polygons and attributes / 鍔犺浇 IMD 鏁版嵁
    imd = load_imd_data()

    all_results = []

    # Process both thermal scenarios / 澶勭悊涓や釜鍦烘櫙
    for scenario in ['typical_day', 'heatwave']:
        result = process_scenario(scenario, imd)
        all_results.append(result)

    # Print a concise summary for both scenarios / 姹囨€荤粨鏋?    logger.info("\n" + "=" * 60)
    logger.info("Analysis summary / 鍒嗘瀽姹囨€?)
    logger.info("=" * 60)

    for result in all_results:
        logger.info(f"\n{result['scenario']}:")
        logger.info(f"  Analysed LSOAs / 鍒嗘瀽 LSOA 鏁? {result['n_lsoa']}")
        logger.info(f"  CI(LST): {result['ci_lst']:+.4f}")
        logger.info(f"  CI(HEI): {result['ci_hei']:+.4f}")
        logger.info(
            "  HEI regression (city fixed effects) / HEI 鍥炲綊锛堝煄甯傚浐瀹氭晥搴旓級: "
            f"slope={result['regression']['hei_slope_fe']:.3f} "
            f"(p={result['regression']['hei_p_fe']:.2e}, R虏={result['regression']['hei_r2_fe']:.3f})"
        )

        hei_test = result['test_results']['hei_mean']
        logger.info(f"  HEI gap (deprived-affluent) / HEI 宸紓锛堣传鍥?瀵岃锛? {hei_test['difference']:+.2f}掳C (p={hei_test['t_pvalue']:.2e})")

    # Run sensitivity analyses after the main summaries / 杩愯鏁忔劅鎬у垎鏋?    sensitivity_results = run_sensitivity_analyses(imd)

    # Print the sensitivity summary / 鎵撳嵃鏁忔劅鎬у垎鏋愭眹鎬?    logger.info("\n" + "=" * 60)
    logger.info("Sensitivity summary / 鏁忔劅鎬у垎鏋愭眹鎬?)
    logger.info("=" * 60)

    for result in sensitivity_results:
        logger.info(f"\n{result['scenario']}:")
        logger.info(f"  Income Decile: HEI gap / HEI 宸紓 {result['income_hei_diff']:+.3f}掳C (p={result['income_hei_pvalue']:.2e})")
        logger.info(f"  Employment Decile: HEI gap / HEI 宸紓 {result['emp_hei_diff']:+.3f}掳C (p={result['emp_hei_pvalue']:.2e})")
        if result['london_hei_diff'] is not None:
            logger.info(f"  London single-city / London 鍗曞煄甯? HEI gap / HEI 宸紓 {result['london_hei_diff']:+.3f}掳C (p={result['london_hei_pvalue']:.2e})")
            logger.info(f"  London CI: {result['london_ci']:+.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete / 鍒嗘瀽瀹屾垚锛?)
    logger.info(f"Output directory / 杈撳嚭鐩綍: {OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()



