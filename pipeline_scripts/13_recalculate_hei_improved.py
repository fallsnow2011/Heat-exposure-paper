import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from pathlib import Path
import warnings
import logging
from datetime import datetime
from tqdm import tqdm
from shapely.ops import linemerge

warnings.filterwarnings('ignore')

# Logging setup / 璁剧疆鏃ュ織
LOG_DIR = Path('results/heat_exposure')
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f'hei_improved_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

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
LST_DIR = Path('GEE_LST_Baseline/lst_scenarios')
SHADOW_DIR = Path('results/shadow_attribution')
OUTPUT_DIR = Path('results/heat_exposure')

# Scenario configuration / 鍦烘櫙閰嶇疆
SCENARIOS = {
    'typical_day': {
        'lst_pattern': 'LST_typical_summer_{city}_30m.tif',
        'shadow_pattern': '{city}_roads_shadow_attribution_typical_day.gpkg'
    },
    'heatwave': {
        'lst_pattern': 'LST_heatwave_2022_{city}_30m.tif',
        'shadow_pattern': '{city}_roads_shadow_attribution_v2.gpkg'
    }
}

# Baseline parameters from the sensitivity analysis / 鍩哄噯鍙傛暟锛堟潵鑷晱鎰熸€у垎鏋愶級
BASELINE_PARAMS = {
    'alpha_building': 0.6,
    'alpha_vegetation': 0.8,
    'delta_t_vegetation': 2.0
}

# CNI/TCNI parameters / CNI/TCNI 鍙傛暟
TEMPERATURE_THRESHOLDS = np.linspace(20, 45, 51)
SAMPLE_STEP = 5.0
GCC_SNAP_METERS = 1.0  # endpoint snap tolerance in metres for road-graph connectivity / 绔偣鍚搁檮绮惧害锛堢背锛夛紝鐢ㄤ簬鏋勫缓閬撹矾杩為€氬浘


def calculate_hei_improved(lst_values, shadow_building, shadow_vegetation,
                           alpha_b=0.6, alpha_v=0.8, delta_t_veg=2.0):
    """
    Calculate the improved HEI formulation / 璁＄畻鏀硅繘鐨?HEI 鍏紡

    HEI = LST 脳 (1 - (伪_b 脳 S_building + 伪_v 脳 S_vegetation)) - 螖T_veg 脳 S_vegetation
    """
    # Shade-induced cooling effect / 闃村奖闄嶆俯鏁堝簲
    shadow_cooling = alpha_b * shadow_building + alpha_v * shadow_vegetation

    # Base HEI (with shade adjustment) / 鍩虹 HEI锛堣€冭檻闃村奖锛?
    hei_base = lst_values * (1 - shadow_cooling)

    # Additional vegetation cooling scales with vegetation shade fraction / 妞嶈棰濆闄嶆俯鏁堝簲锛堜笌妞嶈闃村奖姣斾緥鎴愭姣旓級
    vegetation_cooling = delta_t_veg * shadow_vegetation

    # Final HEI / 鏈€缁?HEI
    hei = hei_base - vegetation_cooling

    return hei


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


def calculate_coolshare_cni_curve(
    roads_gdf: gpd.GeoDataFrame,
    values: np.ndarray,
    thresholds: np.ndarray = TEMPERATURE_THRESHOLDS,
    snap_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Calculate CoolShare(胃) and CNI(胃), then integrate CNI to obtain TCNI / 璁＄畻 CoolShare(胃) 涓?CNI(胃) 鏇茬嚎锛屽苟杩斿洖 `TCNI = 鈭?CNI(胃) d胃`

    Definitions / 鍙ｅ緞锛堢敤鎴风‘璁わ級锛?    - `CoolShare(胃) = #(valid roads with value <= 胃) / #(valid roads)`
    - `CNI(胃) = #(valid roads in the GCC of the cool subgraph at 胃) / #(valid roads)`
    - The denominator is the total count of valid road segments, without length weighting or node weighting / 鍒嗘瘝鏄?valid 鎬婚亾璺鏁帮紙涓嶅仛闀垮害鍔犳潈銆佷笉鐢ㄨ妭鐐瑰崰姣旓級
    """
    values = np.asarray(values, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    # Snap endpoints to a common metre grid / 绔偣鍚搁檮
    u_nodes: list[tuple[int, int] | None] = []
    v_nodes: list[tuple[int, int] | None] = []
    for geom in roads_gdf.geometry:
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

    has_uv = np.fromiter((u is not None and v is not None for u, v in zip(u_nodes, v_nodes)), dtype=bool, count=len(values))
    valid_edge = np.isfinite(values) & has_uv
    edge_indices = np.flatnonzero(valid_edge)
    if edge_indices.size == 0:
        coolshare_curve = np.full_like(thresholds, np.nan, dtype=float)
        cni_curve = np.full_like(thresholds, np.nan, dtype=float)
        return coolshare_curve, cni_curve, np.nan, 0

    vals = values[edge_indices]
    u_list = [u_nodes[i] for i in edge_indices]
    v_list = [v_nodes[i] for i in edge_indices]

    # Compress snapped node coordinates to integer identifiers / 鑺傜偣鍘嬬缉鍒版暣鏁?ID
    node_set = set(u_list).union(set(v_list))
    node_id = {n: i for i, n in enumerate(node_set)}
    u_id = np.fromiter((node_id[n] for n in u_list), dtype=np.int64, count=len(u_list))
    v_id = np.fromiter((node_id[n] for n in v_list), dtype=np.int64, count=len(v_list))

    # Process edges in ascending value order so edges are added incrementally with 胃 / 鎸夊€煎崌搴忓鐞嗭紙闃堝€奸€掑鏃跺閲忓姞鍏ヨ竟锛?    order = np.argsort(vals)
    vals_sorted = vals[order]
    u_sorted = u_id[order]
    v_sorted = v_id[order]

    thresholds_sorted = np.sort(thresholds)
    total_edges = int(len(vals_sorted))
    n_nodes = int(len(node_id))

    parent = np.arange(n_nodes, dtype=np.int64)
    node_size = np.ones(n_nodes, dtype=np.int64)
    edge_size = np.zeros(n_nodes, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a: int, b: int) -> int:
        ra, rb = find(a), find(b)
        if ra == rb:
            edge_size[ra] += 1
            return ra
        if node_size[ra] < node_size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        node_size[ra] += node_size[rb]
        edge_size[ra] += edge_size[rb] + 1
        node_size[rb] = 0
        edge_size[rb] = 0
        return ra

    coolshare_curve = np.zeros(len(thresholds_sorted), dtype=float)
    cni_curve = np.zeros(len(thresholds_sorted), dtype=float)

    ptr = 0
    gcc_root = 0 if n_nodes > 0 else None
    gcc_nodes = int(node_size[gcc_root]) if gcc_root is not None else 0

    for i, t in enumerate(thresholds_sorted):
        while ptr < total_edges and vals_sorted[ptr] <= t:
            root = union(int(u_sorted[ptr]), int(v_sorted[ptr]))
            root = find(root)
            if node_size[root] > gcc_nodes:
                gcc_root = root
                gcc_nodes = int(node_size[root])
            ptr += 1

        coolshare_curve[i] = ptr / total_edges
        gcc_root_now = find(int(gcc_root)) if gcc_root is not None else None
        cni_curve[i] = (edge_size[gcc_root_now] / total_edges) if gcc_root_now is not None else 0.0

    tcni = float(np.trapz(cni_curve, thresholds_sorted))
    return coolshare_curve, cni_curve, tcni, total_edges


def _value_at_threshold(thresholds: np.ndarray, curve: np.ndarray, target: float) -> float:
    idx = np.where(np.isclose(thresholds, target))[0]
    if len(idx) == 0:
        # Fallback to the nearest available threshold / 鍏滃簳锛氬彇鏈€杩戠偣
        i = int(np.argmin(np.abs(thresholds - target)))
        return float(curve[i])
    return float(curve[int(idx[0])])


def sample_lst_to_roads(roads_gdf, lst_path, sample_step=5.0):
    """Sample LST values along each road geometry / 娌块亾璺噰鏍?LST 鍊?""
    with rasterio.open(lst_path) as src:
        lst_crs = src.crs
        nodata = src.nodata

        if roads_gdf.crs != lst_crs:
            roads_gdf = roads_gdf.to_crs(lst_crs)

        lst_values = []
        for idx, row in tqdm(roads_gdf.iterrows(), total=len(roads_gdf), desc="Sampling LST / 閲囨牱 LST", leave=False):
            geom = row.geometry
            if geom is None or geom.is_empty:
                lst_values.append(np.nan)
                continue

            length = geom.length
            if length < sample_step:
                point = geom.interpolate(0.5, normalized=True)
                coords = [(point.x, point.y)]
            else:
                n_samples = max(2, int(length / sample_step))
                coords = []
                for j in range(n_samples):
                    frac = j / (n_samples - 1) if n_samples > 1 else 0.5
                    point = geom.interpolate(frac, normalized=True)
                    coords.append((point.x, point.y))

            sampled = list(src.sample(coords, masked=True))

            valid_values = []
            for v in sampled:
                val = v[0]
                if np.ma.is_masked(val):
                    continue
                val = float(val)
                if nodata is not None and val == nodata:
                    continue
                if not np.isfinite(val):
                    continue
                if not (0.0 < val < 100.0):
                    continue
                valid_values.append(val)

            if len(valid_values) > 0:
                lst_values.append(np.mean(valid_values))
            else:
                lst_values.append(np.nan)

    return np.array(lst_values)


def process_city_scenario(city, scenario):
    """Process one city under one scenario / 澶勭悊鍗曚釜鍩庡競鍦烘櫙"""
    scenario_config = SCENARIOS[scenario]

    lst_path = LST_DIR / scenario_config['lst_pattern'].format(city=city)
    shadow_path = SHADOW_DIR / scenario_config['shadow_pattern'].format(city=city)

    if not lst_path.exists():
        logger.warning(f"LST file not found / LST 鏂囦欢涓嶅瓨鍦? {lst_path}")
        return None, None
    if not shadow_path.exists():
        logger.warning(f"Shade-attribution file not found / 闃村奖褰掑洜鏂囦欢涓嶅瓨鍦? {shadow_path}")
        return None, None

    logger.info(f"Processing {city} - {scenario} / 澶勭悊 {city} - {scenario}")

    # Read the road-level dataset / 璇诲彇閬撹矾鏁版嵁
    roads = gpd.read_file(shadow_path)
    logger.info(f"  Road count / 閬撹矾鏁伴噺: {len(roads)}")

    # Sample LST along roads / 閲囨牱 LST
    lst_values = sample_lst_to_roads(roads, lst_path, sample_step=SAMPLE_STEP)

    # Retrieve daily-average shade attribution fields / 鑾峰彇闃村奖鏁版嵁锛堜娇鐢ㄦ棩鍧囧€硷級
    shadow_building = roads['shadow_building_avg'].values
    shadow_vegetation = roads['shadow_vegetation_avg'].values
    total_shadow = roads['shadow_daily_avg'].values

    # Compute improved HEI values / 璁＄畻鏀硅繘鐨?HEI
    hei = calculate_hei_improved(
        lst_values, shadow_building, shadow_vegetation,
        BASELINE_PARAMS['alpha_building'],
        BASELINE_PARAMS['alpha_vegetation'],
        BASELINE_PARAMS['delta_t_vegetation']
    )

    # Compute CoolShare, CNI, and TCNI using the GCC-based definition / 璁＄畻 CoolShare/CNI/TCNI锛圕NI = GCC-based锛汿CNI = 鈭獵NI d胃锛?    coolshare_curve, cni_curve, tcni, n_valid_roads = calculate_coolshare_cni_curve(
        roads, hei, thresholds=TEMPERATURE_THRESHOLDS, snap_m=GCC_SNAP_METERS
    )
    thresholds_sorted = np.sort(TEMPERATURE_THRESHOLDS)

    cool_share_26 = _value_at_threshold(thresholds_sorted, coolshare_curve, 26.0)
    cool_share_28 = _value_at_threshold(thresholds_sorted, coolshare_curve, 28.0)
    cool_share_30 = _value_at_threshold(thresholds_sorted, coolshare_curve, 30.0)
    cool_share_35 = _value_at_threshold(thresholds_sorted, coolshare_curve, 35.0)

    cni_26 = _value_at_threshold(thresholds_sorted, cni_curve, 26.0)
    cni_28 = _value_at_threshold(thresholds_sorted, cni_curve, 28.0)
    cni_30 = _value_at_threshold(thresholds_sorted, cni_curve, 30.0)
    cni_35 = _value_at_threshold(thresholds_sorted, cni_curve, 35.0)

    # Collect summary statistics / 姹囨€荤粺璁?    summary = {
        'city': city,
        'scenario': scenario,
        'n_roads': len(roads),
        'lst_mean': np.nanmean(lst_values),
        'lst_min': np.nanmin(lst_values),
        'lst_max': np.nanmax(lst_values),
        'hei_mean': np.nanmean(hei),
        'hei_min': np.nanmin(hei),
        'hei_max': np.nanmax(hei),
        'shadow_mean': np.nanmean(total_shadow),
        'shadow_building_mean': np.nanmean(shadow_building),
        'shadow_vegetation_mean': np.nanmean(shadow_vegetation),
        'n_valid_roads': n_valid_roads,
        'cool_share_26': cool_share_26,
        'cool_share_28': cool_share_28,
        'cool_share_30': cool_share_30,
        'cool_share_35': cool_share_35,
        'cni_26': cni_26,
        'cni_28': cni_28,
        'cni_30': cni_30,
        'cni_35': cni_35,
        'tcni': tcni
    }

    logger.info(f"  LST: {summary['lst_mean']:.2f}掳C (min={summary['lst_min']:.2f}, max={summary['lst_max']:.2f})")
    logger.info(f"  HEI: {summary['hei_mean']:.2f}掳C (min={summary['hei_min']:.2f}, max={summary['hei_max']:.2f})")
    logger.info(f"  Shade / 闃村奖: total / 鎬讳綋={summary['shadow_mean']:.1%}, building / 寤虹瓚={summary['shadow_building_mean']:.1%}, vegetation / 妞嶈={summary['shadow_vegetation_mean']:.1%}")
    logger.info(
        f"  CoolShare@28掳C: {cool_share_28:.1%}, CNI@28掳C: {cni_28:.1%}, "
        f"CoolShare@35掳C: {cool_share_35:.1%}, CNI@35掳C: {cni_35:.1%}, TCNI: {tcni:.2f}"
    )

    # Save output / 淇濆瓨閬撹矾绾у埆鏁版嵁
    roads_output = roads.copy()
    roads_output['lst'] = lst_values
    roads_output['hei_improved'] = hei

    # Compute road-level building and vegetation cooling contributions / 璁＄畻姣忔潯閬撹矾鐨勫缓绛?妞嶈闃村奖璐＄尞
    roads_output['cooling_building'] = BASELINE_PARAMS['alpha_building'] * shadow_building * lst_values
    roads_output['cooling_vegetation'] = (BASELINE_PARAMS['alpha_vegetation'] * shadow_vegetation * lst_values +
                                           BASELINE_PARAMS['delta_t_vegetation'] * shadow_vegetation)
    roads_output['cooling_total'] = roads_output['cooling_building'] + roads_output['cooling_vegetation']

    return summary, roads_output


def main():
    logger.info("=" * 60)
    logger.info("Recalculate HEI, CNI, and TCNI (improved formula) / 閲嶆柊璁＄畻 HEI/CNI/TCNI锛堟敼杩涘叕寮忥級")
    logger.info("=" * 60)
    logger.info(f"Baseline parameters / 鍩哄噯鍙傛暟锛?)
    logger.info(f"  伪_building / 寤虹瓚闄嶆俯绯绘暟 = {BASELINE_PARAMS['alpha_building']}")
    logger.info(f"  伪_vegetation / 妞嶈闄嶆俯绯绘暟 = {BASELINE_PARAMS['alpha_vegetation']}")
    logger.info(f"  螖T_vegetation / 妞嶈棰濆闄嶆俯 = {BASELINE_PARAMS['delta_t_vegetation']}掳C")

    all_summaries = []

    for scenario in ['typical_day', 'heatwave']:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Scenario / 鍦烘櫙: {scenario}")
        logger.info(f"{'#' * 60}")

        for city in CITIES:
            summary, roads_data = process_city_scenario(city, scenario)

            if summary:
                all_summaries.append(summary)

                # Save the road-level output / 淇濆瓨閬撹矾绾у埆鏁版嵁
                output_path = OUTPUT_DIR / f'{city}_roads_hei_improved_{scenario}.gpkg'
                roads_data.to_file(output_path, driver='GPKG')
                logger.info(f"  Saved output / 淇濆瓨: {output_path}")

    # Summary outputs / 姹囨€荤粨鏋?
    summary_df = pd.DataFrame(all_summaries)
    summary_path = OUTPUT_DIR / 'hei_cni_tcni_summary_improved.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nSaved summary results / 姹囨€荤粨鏋滀繚瀛? {summary_path}")

    # Print the summary table / 鎵撳嵃姹囨€昏〃
    logger.info("\n" + "=" * 60)
    logger.info("Summary results / 姹囨€荤粨鏋?)
    logger.info("=" * 60)
    logger.info("\n" + summary_df.to_string())

    # Compare against the original simplified formula / 涓庡師濮嬬粨鏋滃姣?    original_path = OUTPUT_DIR / 'hei_cni_tcni_summary.csv'
    if original_path.exists():
        original_df = pd.read_csv(original_path)
        logger.info("\n" + "-" * 60)
        logger.info("Comparison with the original simplified formula (difference in mean HEI) / 涓庡師濮嬬畝鍖栧叕寮忓姣旓紙HEI 鍧囧€煎樊寮傦級")
        logger.info("-" * 60)
        for scenario in ['typical_day', 'heatwave']:
            logger.info(f"\n{scenario}:")
            for city in CITIES:
                orig = original_df[(original_df['city'] == city) & (original_df['scenario'] == scenario)]
                new = summary_df[(summary_df['city'] == city) & (summary_df['scenario'] == scenario)]
                if len(orig) > 0 and len(new) > 0:
                    diff = new['hei_mean'].values[0] - orig['hei_mean'].values[0]
                    logger.info(f"  {city}: original / 鍘熷={orig['hei_mean'].values[0]:.2f}掳C, "
                               f"improved / 鏀硅繘={new['hei_mean'].values[0]:.2f}掳C, difference / 宸紓={diff:+.2f}掳C")

    logger.info("\n" + "=" * 60)
    logger.info("Calculation complete / 璁＄畻瀹屾垚锛?)
    logger.info("=" * 60)

    return summary_df


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()



