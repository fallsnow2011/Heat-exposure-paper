import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results" / "inequality_analysis"
OUTPUT_DIR = BASE_DIR / "paper" / "06_supplement"
FIGURES_DIR = BASE_DIR / "paper" / "05_figures"

def load_data():
    """Load LISA inputs and deprivation summaries / 鍔犺浇鏁版嵁"""
    gdf = gpd.read_file(OUTPUT_DIR / "imd_lsoa.geojson")
    heatwave_df = pd.read_csv(RESULTS_DIR / "lsoa_hei_summary_heatwave.csv")

    # Merge attributes and keep the city identifier / 鍚堝苟骞朵繚鐣?city 鍒?    gdf = gdf.merge(heatwave_df[['lsoa11cd', 'hei_mean', 'IMD_Decile', 'city']],
                    on='lsoa11cd', how='inner', suffixes=('', '_hw'))

    return gdf

def calculate_lisa(gdf):
    """Calculate Local Moran's I clusters / 璁＄畻 LISA"""
    from libpysal.weights import Queen
    from esda.moran import Moran_Local

    valid_mask = gdf['hei_mean'].notna() & np.isfinite(gdf['hei_mean'])
    gdf_valid = gdf[valid_mask].copy()

    w = Queen.from_dataframe(gdf_valid)
    y = gdf_valid['hei_mean'].values
    lisa = Moran_Local(y, w, seed=0)

    gdf_valid['lisa_q'] = lisa.q
    gdf_valid['lisa_p'] = lisa.p_sim
    gdf_valid['lisa_sig'] = lisa.p_sim < 0.05

    return gdf_valid

def create_bivariate_classification(gdf):
    """Create the bivariate hotspot-deprivation classes / 鍒涘缓鍙屽彉閲忓垎绫?""
    gdf = gdf.copy()

    gdf['lisa_cat'] = 'Other'
    sig_mask = gdf['lisa_sig']
    gdf.loc[sig_mask & (gdf['lisa_q'] == 1), 'lisa_cat'] = 'HH'
    gdf.loc[sig_mask & (gdf['lisa_q'] == 3), 'lisa_cat'] = 'LL'

    gdf['imd_cat'] = 'Middle'
    gdf.loc[gdf['IMD_Decile'].isin([1, 2, 3]), 'imd_cat'] = 'Deprived'
    gdf.loc[gdf['IMD_Decile'].isin([8, 9, 10]), 'imd_cat'] = 'Affluent'

    gdf['bivar'] = gdf['lisa_cat'] + '_' + gdf['imd_cat']

    return gdf

def plot_bivariate_map(gdf, output_path):
    """Plot the bivariate LISA-deprivation map / 缁樺埗鍙屽彉閲忓湴鍥?""

    bivar_colors = {
        'HH_Deprived': '#8B0000',
        'HH_Middle': '#CD5C5C',
        'HH_Affluent': '#F08080',
        'LL_Deprived': '#4682B4',
        'LL_Middle': '#87CEEB',
        'LL_Affluent': '#1E90FF',
        'Other_Deprived': '#FFE4B5',
        'Other_Middle': '#F5F5F5',
        'Other_Affluent': '#E0FFFF',
    }

    gdf['color'] = gdf['bivar'].map(bivar_colors)
    gdf.loc[gdf['color'].isna(), 'color'] = '#F5F5F5'

    fig, axes = plt.subplots(2, 3, figsize=(18, 14))
    cities = ['London', 'Birmingham', 'Manchester', 'Bristol', 'Newcastle']

    for idx, city in enumerate(cities):
        ax = axes[idx // 3, idx % 3]

        # Filter each facet by city / 浣跨敤 city 鍒楃瓫閫?        city_gdf = gdf[gdf['city'] == city].copy()

        if len(city_gdf) > 0:
            city_gdf.plot(ax=ax, color=city_gdf['color'], edgecolor='white', linewidth=0.1)

            hh_dep = (city_gdf['bivar'] == 'HH_Deprived').sum()
            ll_aff = (city_gdf['bivar'] == 'LL_Affluent').sum()
            total_hh = (city_gdf['lisa_cat'] == 'HH').sum()

            ax.set_title(f'{city}\nHotspots: {total_hh} | Hotspot+Deprived: {hh_dep}',
                        fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'{city}\n(No data)', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)

        ax.axis('off')

    # Legend / 鍥句緥
    ax_legend = axes[1, 2]
    ax_legend.axis('off')

    legend_elements = [
        mpatches.Patch(facecolor='#8B0000', edgecolor='black', label='Heat Hotspot + Deprived'),
        mpatches.Patch(facecolor='#CD5C5C', edgecolor='black', label='Heat Hotspot + Middle'),
        mpatches.Patch(facecolor='#F08080', edgecolor='black', label='Heat Hotspot + Affluent'),
        mpatches.Patch(facecolor='#4682B4', edgecolor='black', label='Heat Coldspot + Deprived'),
        mpatches.Patch(facecolor='#87CEEB', edgecolor='black', label='Heat Coldspot + Middle'),
        mpatches.Patch(facecolor='#1E90FF', edgecolor='black', label='Heat Coldspot + Affluent'),
        mpatches.Patch(facecolor='#FFE4B5', edgecolor='black', label='Not Sig. + Deprived'),
        mpatches.Patch(facecolor='#F5F5F5', edgecolor='black', label='Not Significant'),
    ]

    ax_legend.legend(handles=legend_elements, loc='center', fontsize=10,
                     title='HEI LISA 脳 IMD Deprivation', title_fontsize=12)

    plt.suptitle('Bivariate Map: Heat Exposure Hotspots 脳 Deprivation Level (Heatwave)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved output / 宸蹭繚瀛? {output_path}")

def main():
    print("=" * 60)
    print("Bivariate map: LISA hotspots 脳 IMD deprivation / 鍙屽彉閲忓湴鍥撅細LISA 鐑偣 脳 IMD 璐洶锛堜慨璁㈢増锛?)
    print("=" * 60)

    gdf = load_data()
    print(f"  Total LSOAs / 鍏?{len(gdf)} 涓?LSOA")
    print(f"  Cities / 鍩庡競: {gdf['city'].unique()}")

    gdf = calculate_lisa(gdf)
    gdf = create_bivariate_classification(gdf)

    # Summarise hotspot counts by city / 鎸夊煄甯傜粺璁?    print("\nCity-level summary / 鎸夊煄甯傜粺璁?")
    for city in gdf['city'].unique():
        city_gdf = gdf[gdf['city'] == city]
        hh = (city_gdf['lisa_cat'] == 'HH').sum()
        hh_dep = (city_gdf['bivar'] == 'HH_Deprived').sum()
        print(f"  {city}: hotspots / 鐑偣={hh}, hotspot+deprived / 鐑偣+璐洶={hh_dep}")

    plot_bivariate_map(gdf, FIGURES_DIR / "fig_bivariate_lisa_imd.png")

    # Print headline totals / 杈撳嚭姹囨€?    stats = {
        'total_hotspots': (gdf['lisa_cat'] == 'HH').sum(),
        'hotspot_deprived': (gdf['bivar'] == 'HH_Deprived').sum(),
        'hotspot_deprived_pct': (gdf['bivar'] == 'HH_Deprived').sum() / max(1, (gdf['lisa_cat'] == 'HH').sum()) * 100
    }
    print(
        f"\nOverall / 鎬昏: hotspots / 鐑偣={stats['total_hotspots']}, "
        f"hotspot+deprived / 鐑偣+璐洶={stats['hotspot_deprived']} "
        f"({stats['hotspot_deprived_pct']:.1f}%)"
    )

if __name__ == "__main__":
    main()



