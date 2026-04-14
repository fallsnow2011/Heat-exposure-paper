import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path
import warnings
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from shapely.ops import linemerge

warnings.filterwarnings('ignore')

# Logging setup / 璁剧疆鏃ュ織
LOG_DIR = Path('results/heat_exposure')
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f'lst_vs_hei_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

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
OUTPUT_DIR = Path('results/heat_exposure')

# Scenario configuration using precomputed HEI road data / 鍦烘櫙閰嶇疆锛屼娇鐢ㄥ凡璁＄畻濂界殑 HEI 鏁版嵁
SCENARIOS = {
    'typical_day': '{city}_roads_hei_improved_typical_day.gpkg',
    'heatwave': '{city}_roads_hei_improved_heatwave.gpkg'
}

# CNI/TCNI parameters / CNI/TCNI 鍙傛暟
TEMPERATURE_THRESHOLDS = np.linspace(20, 45, 51)  # 20-45掳C with 0.5掳C spacing / 20-45掳C锛?.5掳C 闂撮殧
GCC_SNAP_METERS = 1.0  # endpoint snap tolerance in metres for road-graph connectivity / 绔偣鍚搁檮绮惧害锛堢背锛夛紝鐢ㄤ簬鏋勫缓閬撹矾杩為€氬浘


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
    - The denominator is the total count of valid road segments, without length weighting or node weighting / 鍒嗘瘝 = valid 鎬婚亾璺鏁帮紙涓嶅仛闀垮害鍔犳潈銆佷笉鐢ㄨ妭鐐瑰崰姣旓級
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


def _value_at_threshold(thresholds: np.ndarray, curve: np.ndarray, target: float) -> float:
    idx = np.where(np.isclose(thresholds, target))[0]
    if len(idx) == 0:
        i = int(np.argmin(np.abs(thresholds - target)))
        return float(curve[i])
    return float(curve[int(idx[0])])


def process_city_scenario(city, scenario):
    """Process one city-scenario pair and compare LST-based and HEI-based metrics / 澶勭悊鍗曚釜鍩庡競鍦烘櫙锛岃绠?LST 鍜?HEI 鐨?CNI/TCNI"""

    data_path = OUTPUT_DIR / SCENARIOS[scenario].format(city=city)

    if not data_path.exists():
        logger.warning(f"Data file not found / 鏁版嵁鏂囦欢涓嶅瓨鍦? {data_path}")
        return None

    logger.info(f"Processing {city} - {scenario} / 澶勭悊 {city} - {scenario}")

    # Read the precomputed road-level dataset / 璇诲彇鏁版嵁
    roads = gpd.read_file(data_path)
    logger.info(f"  Road count / 閬撹矾鏁伴噺: {len(roads)}")

    # Read LST and HEI values / 鑾峰彇 LST 鍜?HEI 鍊?    lst_values = roads['lst'].values
    hei_values = roads['hei_improved'].values

    thresholds_sorted = np.sort(TEMPERATURE_THRESHOLDS)

    # Compute LST-only metrics under the GCC-based definition / 璁＄畻 LST-only CNI/TCNI锛圕NI = GCC-based锛汿CNI = 鈭獵NI d胃锛?    _, cni_curve_lst, tcni_lst, n_valid_roads = calculate_coolshare_cni_curve(
        roads, lst_values, thresholds=TEMPERATURE_THRESHOLDS, snap_m=GCC_SNAP_METERS
    )
    cni_lst_28 = _value_at_threshold(thresholds_sorted, cni_curve_lst, 28.0)
    cni_lst_30 = _value_at_threshold(thresholds_sorted, cni_curve_lst, 30.0)
    cni_lst_35 = _value_at_threshold(thresholds_sorted, cni_curve_lst, 35.0)

    # Compute HEI-based metrics under the same definition / 璁＄畻 HEI-based CNI/TCNI锛堝悓鍙ｅ緞锛?    _, cni_curve_hei, tcni_hei, n_valid_roads_hei = calculate_coolshare_cni_curve(
        roads, hei_values, thresholds=TEMPERATURE_THRESHOLDS, snap_m=GCC_SNAP_METERS
    )
    # The valid-road counts should match because missing LST also implies missing HEI / 涓よ€呭簲鐩稿悓锛圠ST 缂哄け浼氬鑷?HEI 缂哄け锛夛紝淇濈暀涓€涓嵆鍙?    n_valid_roads = min(n_valid_roads, n_valid_roads_hei) if n_valid_roads and n_valid_roads_hei else max(n_valid_roads, n_valid_roads_hei)

    cni_hei_28 = _value_at_threshold(thresholds_sorted, cni_curve_hei, 28.0)
    cni_hei_30 = _value_at_threshold(thresholds_sorted, cni_curve_hei, 30.0)
    cni_hei_35 = _value_at_threshold(thresholds_sorted, cni_curve_hei, 35.0)

    # Compute 螖TCNI_c as the network-scale cooling benefit of shade / 璁＄畻宸€?螖TCNI_c锛堥槾褰卞甫鏉ョ殑鍐峰嵈鏁堢泭锛?    # Positive values mean that using HEI makes more roads effectively cool / 姝ｅ€艰〃绀?HEI 鑰冭檻闃村奖鍚庯紝鏇村閬撹矾鍙樷€滃噳鐖解€?    delta_tcni = tcni_hei - tcni_lst
    delta_cni_28 = cni_hei_28 - cni_lst_28
    delta_cni_35 = cni_hei_35 - cni_lst_35

    # Summary outputs / 姹囨€荤粨鏋?
    summary = {
        'city': city,
        'scenario': scenario,
        'n_roads': len(roads),
        'n_valid_roads': n_valid_roads,
        # LST summary / LST 缁熻
        'lst_mean': np.nanmean(lst_values),
        'lst_std': np.nanstd(lst_values),
        # HEI summary / HEI 缁熻
        'hei_mean': np.nanmean(hei_values),
        'hei_std': np.nanstd(hei_values),
        # Mean temperature reduction / 娓╁害闄嶄綆
        'temp_reduction': np.nanmean(lst_values) - np.nanmean(hei_values),
        # LST-only CNI/TCNI
        'cni_lst_28': cni_lst_28,
        'cni_lst_30': cni_lst_30,
        'cni_lst_35': cni_lst_35,
        'tcni_lst': tcni_lst,
        # HEI-based CNI/TCNI
        'cni_hei_28': cni_hei_28,
        'cni_hei_30': cni_hei_30,
        'cni_hei_35': cni_hei_35,
        'tcni_hei': tcni_hei,
        # Metric differences (shade cooling benefit) / 宸€硷紙闃村奖鍐峰嵈鏁堢泭锛?        'delta_cni_28': delta_cni_28,
        'delta_cni_35': delta_cni_35,
        'delta_tcni': delta_tcni,
        # Relative improvement percentage / 鐩稿鏀瑰杽鐧惧垎姣?        'tcni_improvement_pct': (delta_tcni / tcni_lst * 100) if tcni_lst > 0 else np.nan
    }

    logger.info(f"  LST: {summary['lst_mean']:.2f}卤{summary['lst_std']:.2f}掳C")
    logger.info(f"  HEI: {summary['hei_mean']:.2f}卤{summary['hei_std']:.2f}掳C")
    logger.info(f"  Temperature reduction / 娓╁害闄嶄綆: {summary['temp_reduction']:.2f}掳C")
    logger.info(f"  TCNI(LST): {tcni_lst:.2f}, TCNI(HEI): {tcni_hei:.2f}")
    logger.info(f"  螖TCNI: {delta_tcni:+.2f} ({summary['tcni_improvement_pct']:+.1f}%)")

    # Save the CNI curves for later plotting / 淇濆瓨 CNI 鏇茬嚎
    cni_curves = pd.DataFrame({
        'threshold': thresholds_sorted,
        'cni_lst': cni_curve_lst,
        'cni_hei': cni_curve_hei,
        'delta_cni': cni_curve_hei - cni_curve_lst
    })

    return summary, cni_curves


def create_comparison_plots(all_summaries, all_curves):
    """Create the comparison figure for all cities and scenarios / 鍒涘缓鍙鍖栧姣斿浘"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    scenarios = ['typical_day', 'heatwave']
    colors = plt.cm.tab10(np.linspace(0, 1, len(CITIES)))

    for idx, scenario in enumerate(scenarios):
        # Compare CNI curves / CNI 鏇茬嚎瀵规瘮
        ax1 = axes[idx, 0]
        ax2 = axes[idx, 1]
        ax3 = axes[idx, 2]

        for i, city in enumerate(CITIES):
            key = f"{city}_{scenario}"
            if key in all_curves:
                curves = all_curves[key]

                # LST CNI curve shown with a dashed line / LST CNI 鏇茬嚎锛堣櫄绾匡級
                ax1.plot(curves['threshold'], curves['cni_lst'],
                        linestyle='--', color=colors[i], alpha=0.7, label=f'{city} (LST)')
                # HEI CNI curve shown with a solid line / HEI CNI 鏇茬嚎锛堝疄绾匡級
                ax1.plot(curves['threshold'], curves['cni_hei'],
                        linestyle='-', color=colors[i], label=f'{city} (HEI)')

        ax1.set_xlabel('Temperature Threshold (掳C)')
        ax1.set_ylabel('CNI')
        ax1.set_title(f'{scenario.replace("_", " ").title()}: CNI Curves')
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(20, 45)
        ax1.set_ylim(0, 1)

        # Plot the 螖CNI curve / 螖CNI 鏇茬嚎
        for i, city in enumerate(CITIES):
            key = f"{city}_{scenario}"
            if key in all_curves:
                curves = all_curves[key]
                ax2.plot(curves['threshold'], curves['delta_cni'],
                        color=colors[i], label=city)

        ax2.set_xlabel('Temperature Threshold (掳C)')
        ax2.set_ylabel('螖CNI (HEI - LST)')
        ax2.set_title(f'{scenario.replace("_", " ").title()}: Shadow Cooling Benefit')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlim(20, 45)

        # Plot the TCNI comparison bars / TCNI 瀵规瘮鏌辩姸鍥?        scenario_data = [s for s in all_summaries if s['scenario'] == scenario]
        if scenario_data:
            cities = [s['city'] for s in scenario_data]
            tcni_lst = [s['tcni_lst'] for s in scenario_data]
            tcni_hei = [s['tcni_hei'] for s in scenario_data]

            x = np.arange(len(cities))
            width = 0.35

            bars1 = ax3.bar(x - width/2, tcni_lst, width, label='TCNI (LST-only)', color='coral')
            bars2 = ax3.bar(x + width/2, tcni_hei, width, label='TCNI (HEI-based)', color='steelblue')

            ax3.set_xlabel('City')
            ax3.set_ylabel('TCNI')
            ax3.set_title(f'{scenario.replace("_", " ").title()}: TCNI Comparison')
            ax3.set_xticks(x)
            ax3.set_xticklabels(cities, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'lst_vs_hei_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved comparison figure / 鍙鍖栦繚瀛? {output_path}")
    plt.close()


def main():
    logger.info("=" * 60)
    logger.info("Comparison of LST-only and HEI-based CNI/TCNI metrics / LST-only 涓?HEI-based CNI/TCNI 瀵规瘮鍒嗘瀽")
    logger.info("=" * 60)

    all_summaries = []
    all_curves = {}

    for scenario in ['typical_day', 'heatwave']:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Scenario / 鍦烘櫙: {scenario}")
        logger.info(f"{'#' * 60}")

        for city in CITIES:
            result = process_city_scenario(city, scenario)

            if result:
                summary, curves = result
                all_summaries.append(summary)
                all_curves[f"{city}_{scenario}"] = curves

                # Save the per-city CNI curves / 淇濆瓨 CNI 鏇茬嚎
                curve_path = OUTPUT_DIR / f'{city}_cni_curves_{scenario}.csv'
                curves.to_csv(curve_path, index=False)

    # Summary outputs / 姹囨€荤粨鏋?
    summary_df = pd.DataFrame(all_summaries)
    summary_path = OUTPUT_DIR / 'lst_vs_hei_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nSaved summary results / 姹囨€荤粨鏋滀繚瀛? {summary_path}")

    # Print the summary table / 鎵撳嵃姹囨€昏〃
    logger.info("\n" + "=" * 60)
    logger.info("Summary results / 姹囨€荤粨鏋?)
    logger.info("=" * 60)

    # Format the main summary fields for logging / 鏍煎紡鍖栬緭鍑?    print_cols = ['city', 'scenario', 'lst_mean', 'hei_mean', 'temp_reduction',
                  'tcni_lst', 'tcni_hei', 'delta_tcni', 'tcni_improvement_pct']
    logger.info("\n" + summary_df[print_cols].to_string())

    # Create the comparison figure / 鍒涘缓鍙鍖?    create_comparison_plots(all_summaries, all_curves)

    # Report overall statistics by scenario / 璁＄畻鏁翠綋缁熻
    logger.info("\n" + "-" * 60)
    logger.info("Overall statistics / 鏁翠綋缁熻")
    logger.info("-" * 60)

    for scenario in ['typical_day', 'heatwave']:
        scenario_data = summary_df[summary_df['scenario'] == scenario]
        logger.info(f"\n{scenario}:")
        logger.info(f"  Mean temperature reduction / 骞冲潎娓╁害闄嶄綆: {scenario_data['temp_reduction'].mean():.2f}掳C")
        logger.info(f"  Mean 螖TCNI / 骞冲潎 螖TCNI: {scenario_data['delta_tcni'].mean():.2f}")
        logger.info(f"  Mean TCNI improvement / 骞冲潎 TCNI 鏀瑰杽: {scenario_data['tcni_improvement_pct'].mean():.1f}%")

    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete / 鍒嗘瀽瀹屾垚锛?)
    logger.info("=" * 60)

    return summary_df


if __name__ == '__main__':
    main()



