import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from pathlib import Path
import warnings
import logging
from datetime import datetime
from tqdm import tqdm
import time
import itertools
from shapely.ops import linemerge

warnings.filterwarnings('ignore')

# Logging setup / 璁剧疆鏃ュ織
LOG_DIR = Path('results/sensitivity_analysis')
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f'hei_sensitivity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

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
OUTPUT_DIR = Path('results/sensitivity_analysis')
HEI_DIR = Path('results/heat_exposure')  # Reuse precomputed road-level LST/shade summaries when available / 鑻ュ凡瀛樺湪閬撹矾绾?LST/闃村奖姹囨€荤粨鏋滐紝鍙洿鎺ュ鐢ㄤ互鍔犻€?
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

# Sensitivity-analysis parameter ranges / 鏁忔劅鎬у垎鏋愬弬鏁拌寖鍥?SENSITIVITY_PARAMS = {
    'alpha_building': [0.5, 0.6, 0.7],      # Building cooling coefficient / 寤虹瓚闄嶆俯绯绘暟
    'alpha_vegetation': [0.7, 0.8, 0.9],    # Vegetation cooling coefficient / 鏍戞湪闄嶆俯绯绘暟
    'delta_t_vegetation': [1.0, 2.0, 3.0]   # Additional vegetation cooling (掳C) / 鏍戞湪骞冲潎闄嶆俯鏁堝簲锛埪癈锛?}

# Baseline parameter set / 鍩哄噯鍙傛暟
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
                           alpha_b, alpha_v, delta_t_veg):
    """
    Calculate the improved HEI formulation / 璁＄畻鏀硅繘鐨?HEI 鍏紡

    HEI = LST 脳 (1 - (伪_b 脳 S_building + 伪_v 脳 S_vegetation)) - 螖T_veg 脳 S_vegetation

    Parameters / 鍙傛暟锛?    - `lst_values`: LST values / LST 娓╁害鍊?    - `shadow_building`: building shade fraction / 寤虹瓚闃村奖瑕嗙洊鐜?    - `shadow_vegetation`: vegetation shade fraction / 妞嶈闃村奖瑕嗙洊鐜?    - `alpha_b`: building-shade cooling coefficient (0.5-0.7) / 寤虹瓚闃村奖闄嶆俯绯绘暟锛?.5-0.7锛?    - `alpha_v`: vegetation-shade cooling coefficient (0.7-0.9) / 妞嶈闃村奖闄嶆俯绯绘暟锛?.7-0.9锛?    - `delta_t_veg`: extra vegetation cooling effect (1-3掳C) / 妞嶈棰濆闄嶆俯鏁堝簲锛?-3掳C锛?
    Returns / 杩斿洖锛?    - `hei`: Heat Exposure Index / 鐑毚闇叉寚鏁?    """
    # Shade-induced cooling effect / 闃村奖闄嶆俯鏁堝簲
    shadow_cooling = alpha_b * shadow_building + alpha_v * shadow_vegetation

    # Base HEI (with shade adjustment) / 鍩虹 HEI锛堣€冭檻闃村奖锛?
    hei_base = lst_values * (1 - shadow_cooling)

    # Additional vegetation cooling scales with vegetation shade fraction / 妞嶈棰濆闄嶆俯鏁堝簲锛堜笌妞嶈闃村奖姣斾緥鎴愭姣旓級
    # Only locations with vegetation shade receive this extra cooling / 鍙湁鏈夋琚槾褰辩殑鍦版柟鎵嶆湁棰濆闄嶆俯
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

    Definitions / 鍙ｅ緞锛?    - `CoolShare(胃) = #(valid roads with value <= 胃) / #(valid roads)`
    - `CNI(胃) = #(valid roads in the GCC of the cool subgraph at 胃) / #(valid roads)`
    - The denominator is the total count of valid road segments, without length weighting or node weighting / 鍒嗘瘝涓?valid 鎬婚亾璺鏁帮紙涓嶅仛闀垮害鍔犳潈銆佷笉鐢ㄨ妭鐐瑰崰姣旓級
    """
    values = np.asarray(values, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

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

    node_set = set(u_list).union(set(v_list))
    node_id = {n: i for i, n in enumerate(node_set)}
    u_id = np.fromiter((node_id[n] for n in u_list), dtype=np.int64, count=len(u_list))
    v_id = np.fromiter((node_id[n] for n in v_list), dtype=np.int64, count=len(v_list))

    order = np.argsort(vals)
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


def prepare_connectivity_inputs(roads_gdf: gpd.GeoDataFrame, snap_m: float = 1.0) -> dict:
    """Precompute endpoint-based connectivity inputs to speed up sensitivity analysis / 棰勮绠楅亾璺鐐硅繛閫氫俊鎭紙鐢ㄤ簬鏁忔劅鎬у垎鏋愬姞閫燂級銆?""
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

    has_uv = np.fromiter((u is not None and v is not None for u, v in zip(u_nodes, v_nodes)), dtype=bool, count=len(u_nodes))
    uv_indices = np.flatnonzero(has_uv)
    if uv_indices.size == 0:
        return {'uv_indices': uv_indices, 'u_id': np.array([], dtype=np.int64), 'v_id': np.array([], dtype=np.int64), 'n_nodes': 0}

    u_list = [u_nodes[i] for i in uv_indices]
    v_list = [v_nodes[i] for i in uv_indices]

    node_set = set(u_list).union(set(v_list))
    node_id = {n: i for i, n in enumerate(node_set)}

    u_id = np.fromiter((node_id[n] for n in u_list), dtype=np.int64, count=len(u_list))
    v_id = np.fromiter((node_id[n] for n in v_list), dtype=np.int64, count=len(v_list))

    return {'uv_indices': uv_indices, 'u_id': u_id, 'v_id': v_id, 'n_nodes': int(len(node_id))}


def calculate_coolshare_cni_curve_prepared(
    values: np.ndarray,
    prepared: dict,
    thresholds: np.ndarray = TEMPERATURE_THRESHOLDS,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Reuse precomputed endpoint data to calculate CoolShare, CNI, and TCNI / 浣跨敤棰勮绠楃鐐逛俊鎭绠?CoolShare/CNI/TCNI銆?""
    values = np.asarray(values, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    uv_indices = prepared['uv_indices']
    if uv_indices.size == 0:
        coolshare_curve = np.full_like(thresholds, np.nan, dtype=float)
        cni_curve = np.full_like(thresholds, np.nan, dtype=float)
        return coolshare_curve, cni_curve, np.nan, 0

    vals_all = values[uv_indices]
    finite_mask = np.isfinite(vals_all)
    if finite_mask.sum() == 0:
        coolshare_curve = np.full_like(thresholds, np.nan, dtype=float)
        cni_curve = np.full_like(thresholds, np.nan, dtype=float)
        return coolshare_curve, cni_curve, np.nan, 0

    vals = vals_all[finite_mask]
    u_id = prepared['u_id'][finite_mask]
    v_id = prepared['v_id'][finite_mask]

    order = np.argsort(vals)
    vals_sorted = vals[order]
    u_sorted = u_id[order]
    v_sorted = v_id[order]

    thresholds_sorted = np.sort(thresholds)
    total_edges = int(len(vals_sorted))
    n_nodes = int(prepared.get('n_nodes', 0))

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


def process_city_scenario_sensitivity(city, scenario, params):
    """Process the sensitivity analysis for one city and one scenario / 澶勭悊鍗曚釜鍩庡競鍦烘櫙鐨勬晱鎰熸€у垎鏋愩€?""
    cache_key = f"{city}_{scenario}"

    # Cache road-level data to avoid repeatedly reading the same city-scenario inputs / 缂撳瓨锛氶伩鍏嶅鍚屼竴 city脳scenario 鍙嶅璇绘枃浠?绠楃鐐?    if not hasattr(process_city_scenario_sensitivity, 'data_cache'):
        process_city_scenario_sensitivity.data_cache = {}

    if cache_key not in process_city_scenario_sensitivity.data_cache:
        # Reuse precomputed road-level HEI inputs when available to avoid resampling LST / 浼樺厛澶嶇敤宸茬敓鎴愮殑閬撹矾绾?HEI 鏁版嵁锛堝寘鍚?lst + shadow attribution锛夛紝閬垮厤閲嶅閲囨牱 LST
        hei_roads_path = HEI_DIR / f'{city}_roads_hei_improved_{scenario}.gpkg'

        if hei_roads_path.exists():
            roads = gpd.read_file(hei_roads_path)
            lst_values = roads['lst'].to_numpy()
        else:
            scenario_config = SCENARIOS[scenario]
            lst_path = LST_DIR / scenario_config['lst_pattern'].format(city=city)
            shadow_path = SHADOW_DIR / scenario_config['shadow_pattern'].format(city=city)

            if not lst_path.exists() or not shadow_path.exists():
                return None

            roads = gpd.read_file(shadow_path)
            lst_values = sample_lst_to_roads(roads, lst_path, sample_step=SAMPLE_STEP)

        shadow_building = roads['shadow_building_avg'].to_numpy()
        shadow_vegetation = roads['shadow_vegetation_avg'].to_numpy()

        prepared = prepare_connectivity_inputs(roads, snap_m=GCC_SNAP_METERS)

        process_city_scenario_sensitivity.data_cache[cache_key] = {
            'n_roads': int(len(roads)),
            'lst': lst_values,
            'shadow_building': shadow_building,
            'shadow_vegetation': shadow_vegetation,
            'prepared': prepared,
        }

    cached = process_city_scenario_sensitivity.data_cache[cache_key]
    lst_values = cached['lst']
    shadow_building = cached['shadow_building']
    shadow_vegetation = cached['shadow_vegetation']
    prepared = cached['prepared']

    # Compute HEI for the current parameter combination / 璁＄畻 HEI
    hei = calculate_hei_improved(
        lst_values, shadow_building, shadow_vegetation,
        params['alpha_building'], params['alpha_vegetation'], params['delta_t_vegetation']
    )

    # Compute CoolShare, CNI, and TCNI under the GCC-based definition / 璁＄畻 CoolShare/CNI/TCNI锛圕NI = GCC-based锛汿CNI = 鈭獵NI d胃锛?    thresholds_sorted = np.sort(TEMPERATURE_THRESHOLDS)
    coolshare_curve, cni_curve, tcni, n_valid_roads = calculate_coolshare_cni_curve_prepared(
        hei, prepared, thresholds=TEMPERATURE_THRESHOLDS
    )

    cool_share_28 = _value_at_threshold(thresholds_sorted, coolshare_curve, 28.0)
    cool_share_30 = _value_at_threshold(thresholds_sorted, coolshare_curve, 30.0)
    cool_share_35 = _value_at_threshold(thresholds_sorted, coolshare_curve, 35.0)

    cni_28 = _value_at_threshold(thresholds_sorted, cni_curve, 28.0)
    cni_30 = _value_at_threshold(thresholds_sorted, cni_curve, 30.0)
    cni_35 = _value_at_threshold(thresholds_sorted, cni_curve, 35.0)

    return {
        'city': city,
        'scenario': scenario,
        'alpha_building': params['alpha_building'],
        'alpha_vegetation': params['alpha_vegetation'],
        'delta_t_vegetation': params['delta_t_vegetation'],
        'n_roads': int(cached['n_roads']),
        'n_valid_roads': int(n_valid_roads),
        'hei_mean': np.nanmean(hei),
        'hei_std': np.nanstd(hei),
        'cool_share_28': cool_share_28,
        'cool_share_30': cool_share_30,
        'cool_share_35': cool_share_35,
        'cni_28': cni_28,
        'cni_30': cni_30,
        'cni_35': cni_35,
        'tcni': tcni
    }


def run_sensitivity_analysis():
    """Run the full sensitivity-analysis workflow / 杩愯瀹屾暣鐨勬晱鎰熸€у垎鏋?""
    logger.info("="*60)
    logger.info("HEI sensitivity analysis / HEI 鏁忔劅鎬у垎鏋?)
    logger.info("="*60)
    logger.info(f"Parameter ranges / 鍙傛暟鑼冨洿锛?)
    logger.info(f"  Building cooling coefficient / 寤虹瓚闄嶆俯绯绘暟: {SENSITIVITY_PARAMS['alpha_building']}")
    logger.info(f"  Vegetation cooling coefficient / 妞嶈闄嶆俯绯绘暟: {SENSITIVITY_PARAMS['alpha_vegetation']}")
    logger.info(f"  Additional vegetation cooling / 妞嶈棰濆闄嶆俯: {SENSITIVITY_PARAMS['delta_t_vegetation']}掳C")

    all_results = []

    # Enumerate all parameter combinations / 鐢熸垚鎵€鏈夊弬鏁扮粍鍚?    param_combinations = list(itertools.product(
        SENSITIVITY_PARAMS['alpha_building'],
        SENSITIVITY_PARAMS['alpha_vegetation'],
        SENSITIVITY_PARAMS['delta_t_vegetation']
    ))

    logger.info(f"Number of parameter combinations / 鎬诲弬鏁扮粍鍚堟暟: {len(param_combinations)}")
    logger.info(f"Total workload / 鎬昏绠楅噺: {len(param_combinations)} 脳 {len(CITIES)} 脳 2 鍦烘櫙 = {len(param_combinations) * len(CITIES) * 2}")

    total_start = time.time()

    for scenario in ['typical_day', 'heatwave']:
        logger.info(f"\n{'#'*60}")
        logger.info(f"Scenario / 鍦烘櫙: {scenario}")
        logger.info(f"{'#'*60}")

        for city in CITIES:
            logger.info(f"\nProcessing {city} / 澶勭悊 {city}...")

            for alpha_b, alpha_v, delta_t in tqdm(param_combinations, desc=f"{city}"):
                params = {
                    'alpha_building': alpha_b,
                    'alpha_vegetation': alpha_v,
                    'delta_t_vegetation': delta_t
                }

                try:
                    result = process_city_scenario_sensitivity(city, scenario, params)
                    if result:
                        all_results.append(result)
                except Exception as e:
                    logger.error(f"Error for {city} {scenario} {params} / 閿欒 {city} {scenario} {params}: {e}")

    # Summary outputs / 姹囨€荤粨鏋?
    results_df = pd.DataFrame(all_results)

    # Save the full result table / 淇濆瓨瀹屾暣缁撴灉
    full_results_path = OUTPUT_DIR / 'sensitivity_full_results.csv'
    results_df.to_csv(full_results_path, index=False)
    logger.info(f"\nSaved full results / 瀹屾暣缁撴灉淇濆瓨: {full_results_path}")

    # Generate the summary table / 鐢熸垚鏁忔劅鎬у垎鏋愭憳瑕?    generate_sensitivity_summary(results_df)

    total_elapsed = time.time() - total_start
    logger.info(f"\nTotal runtime / 鎬昏€楁椂: {total_elapsed/60:.1f} 鍒嗛挓")

    return results_df


def generate_sensitivity_summary(results_df):
    """Generate the summary products for the sensitivity analysis / 鐢熸垚鏁忔劅鎬у垎鏋愭憳瑕?""
    logger.info("\n" + "="*60)
    logger.info("Sensitivity-analysis summary / 鏁忔劅鎬у垎鏋愭憳瑕?)
    logger.info("="*60)

    # 1. Baseline-scenario results / 鍩哄噯鎯呮櫙缁撴灉
    baseline = results_df[
        (results_df['alpha_building'] == BASELINE_PARAMS['alpha_building']) &
        (results_df['alpha_vegetation'] == BASELINE_PARAMS['alpha_vegetation']) &
        (results_df['delta_t_vegetation'] == BASELINE_PARAMS['delta_t_vegetation'])
    ]

    logger.info(f"\nBaseline parameters / 鍩哄噯鍙傛暟锛?伪_b={BASELINE_PARAMS['alpha_building']}, "
                f"伪_v={BASELINE_PARAMS['alpha_vegetation']}, "
                f"螖T={BASELINE_PARAMS['delta_t_vegetation']}掳C")
    logger.info("\nBaseline-scenario results / 鍩哄噯鎯呮櫙缁撴灉:")
    logger.info(
        baseline[['city', 'scenario', 'n_valid_roads', 'hei_mean',
                  'cool_share_28', 'cool_share_35', 'cni_28', 'cni_35', 'tcni']].to_string()
    )

    # 2. Parameter-by-parameter sensitivity summary / 鍙傛暟鏁忔劅鎬у垎鏋?    summaries = []

    for scenario in ['typical_day', 'heatwave']:
        scenario_data = results_df[results_df['scenario'] == scenario]

        # Vary one parameter while keeping the others at the baseline values / 鍥哄畾鍏朵粬鍙傛暟锛屽彉鍖栧崟涓弬鏁?        for param_name, param_values in SENSITIVITY_PARAMS.items():
            other_params = {k: v for k, v in BASELINE_PARAMS.items() if k != param_name}

            mask = True
            for k, v in other_params.items():
                mask = mask & (scenario_data[k] == v)

            param_data = scenario_data[mask]

            for val in param_values:
                val_data = param_data[param_data[param_name] == val]
                if len(val_data) > 0:
                    summaries.append({
                        'scenario': scenario,
                        'varied_param': param_name,
                        'param_value': val,
                        'hei_mean_avg': val_data['hei_mean'].mean(),
                        'cool_share_28_avg': val_data['cool_share_28'].mean(),
                        'cool_share_35_avg': val_data['cool_share_35'].mean(),
                        'cni_28_avg': val_data['cni_28'].mean(),
                        'cni_35_avg': val_data['cni_35'].mean(),
                        'tcni_avg': val_data['tcni'].mean()
                    })

    summary_df = pd.DataFrame(summaries)
    summary_path = OUTPUT_DIR / 'sensitivity_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nSaved sensitivity summary / 鏁忔劅鎬ф憳瑕佷繚瀛? {summary_path}")

    # 3. Print the main findings / 鎵撳嵃鍏抽敭鍙戠幇
    logger.info("\n" + "-"*40)
    logger.info("Key findings / 鍏抽敭鍙戠幇:")
    logger.info("-"*40)

    for scenario in ['typical_day', 'heatwave']:
        logger.info(f"\n{scenario}:")
        scenario_summary = summary_df[summary_df['scenario'] == scenario]

        for param in SENSITIVITY_PARAMS.keys():
            param_summary = scenario_summary[scenario_summary['varied_param'] == param]
            if len(param_summary) > 0:
                tcni_range = param_summary['tcni_avg'].max() - param_summary['tcni_avg'].min()
                cni28_range = param_summary['cni_28_avg'].max() - param_summary['cni_28_avg'].min()
                logger.info(f"  {param}: TCNI range / TCNI 鍙樺寲鑼冨洿 = {tcni_range:.2f}, CNI@28掳C change / CNI@28掳C 鍙樺寲 = {cni28_range:.1%}")

    # 4. Save the baseline-detail table for follow-up analysis / 淇濆瓨鍩哄噯鎯呮櫙璇︾粏缁撴灉鐢ㄤ簬鍚庣画鍒嗘瀽
    baseline_path = OUTPUT_DIR / 'baseline_scenario_results.csv'
    baseline.to_csv(baseline_path, index=False)
    logger.info(f"\nSaved baseline results / 鍩哄噯鎯呮櫙缁撴灉淇濆瓨: {baseline_path}")

    return summary_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_sensitivity_analysis()

    logger.info("\n" + "="*60)
    logger.info("Sensitivity analysis finished / 鏁忔劅鎬у垎鏋愬畬鎴愶紒")
    logger.info("="*60)
    logger.info(f"Output directory / 杈撳嚭鐩綍: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()



