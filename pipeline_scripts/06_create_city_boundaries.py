import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths / 璺緞
gpkg_path = Path('city_boundaries/Indices_of_Multiple_Deprivation_(IMD)_2019_8404722932957776031.gpkg')
buildings_dir = Path('city_boundaries/GEE_Buildings')
output_dir = Path('city_boundaries')
output_dir.mkdir(exist_ok=True)

# Read the LSOA layer / 璇诲彇 LSOA 鏁版嵁
print("Read IMD LSOA data / 璇诲彇 IMD LSOA 鏁版嵁...")
gdf = gpd.read_file(gpkg_path)
print(f"  Total LSOA count / 鎬?LSOA 鏁? {len(gdf)}")
print(f"  CRS: {gdf.crs}")

# Define the LAD names associated with each city / 瀹氫箟姣忎釜鍩庡競瀵瑰簲鐨?LAD 鍚嶇О
# The 33 London boroughs (excluding surrounding non-London areas) / 浼︽暒 33 涓?borough锛堜笉鍚鸡鏁︿互澶栧尯鍩燂級
LONDON_BOROUGHS = [
    'City of London',
    'Barking and Dagenham',
    'Barnet',
    'Bexley',
    'Brent',
    'Bromley',
    'Camden',
    'Croydon',
    'Ealing',
    'Enfield',
    'Greenwich',
    'Hackney',
    'Hammersmith and Fulham',
    'Haringey',
    'Harrow',
    'Havering',
    'Hillingdon',
    'Hounslow',
    'Islington',
    'Kensington and Chelsea',
    'Kingston upon Thames',
    'Lambeth',
    'Lewisham',
    'Merton',
    'Newham',
    'Redbridge',
    'Richmond upon Thames',
    'Southwark',
    'Sutton',
    'Tower Hamlets',
    'Waltham Forest',
    'Wandsworth',
    'Westminster'
]

CITY_LADS = {
    'London': LONDON_BOROUGHS,
    'Birmingham': ['Birmingham'],
    'Bristol': ['Bristol, City of'],
    'Manchester': ['Manchester'],
    'Newcastle': ['Newcastle upon Tyne']
}

# Process each city / 澶勭悊姣忎釜鍩庡競
boundaries = {}
for city_name, lad_names in CITY_LADS.items():
    print(f"\nProcessing {city_name} / 澶勭悊 {city_name}...")

    # Filter LSOAs / 绛涢€?LSOA
    city_lsoas = gdf[gdf['LADnm'].isin(lad_names)].copy()
    print(f"  LSOA count / LSOA 鏁伴噺: {len(city_lsoas)}")

    # Dissolve into a single boundary / 鍚堝苟涓哄崟涓€杈圭晫
    boundary = city_lsoas.dissolve()
    boundary = boundary[['geometry']].copy()
    boundary['city'] = city_name

    # Reproject to WGS84 (EPSG:4326) / 杞崲鍒?WGS84锛圗PSG:4326锛?
    boundary_wgs84 = boundary.to_crs(epsg=4326)

    # Save output / 淇濆瓨杈圭晫
    boundary_file = output_dir / f'{city_name}_boundary.geojson'
    boundary_wgs84.to_file(boundary_file, driver='GeoJSON')
    print(f"  Saved boundary / 淇濆瓨杈圭晫: {boundary_file}")

    boundaries[city_name] = boundary_wgs84

    # Calculate the area in square kilometres / 璁＄畻闈㈢Н
    boundary_utm = boundary.to_crs(epsg=32630)  # UTM zone 30N for UK
    area_km2 = boundary_utm.geometry.area.values[0] / 1e6
    print(f"  Area / 闈㈢Н: {area_km2:.2f} km虏")

print("\n" + "="*60)
print("City boundary creation complete / 鍩庡競杈圭晫鍒涘缓瀹屾垚锛?)
print("="*60)

# Clip the building data to each city boundary / 瑁佸壀寤虹瓚鏁版嵁
print("\nStart clipping building datasets / 寮€濮嬭鍓缓绛戞暟鎹?..")

for city_name, boundary in boundaries.items():
    buildings_file = buildings_dir / f'MS_Buildings_{city_name}.geojson'

    if not buildings_file.exists():
        print(f"\n{city_name}: building file not found / 寤虹瓚鏂囦欢涓嶅瓨鍦?{buildings_file}")
        continue

    print(f"\nProcessing building footprints for {city_name} / 澶勭悊 {city_name} 寤虹瓚...")
    print(f"  Reading file / 璇诲彇: {buildings_file}")

    # Read building footprints / 璇诲彇寤虹瓚鏁版嵁
    buildings = gpd.read_file(buildings_file)
    print(f"  Original building count / 鍘熷寤虹瓚鏁? {len(buildings)}")

    # Clip buildings to the city boundary / 瑁佸壀鍒板煄甯傝竟鐣?    buildings_clipped = gpd.clip(buildings, boundary)
    print(f"  Clipped building count / 瑁佸壀鍚庡缓绛戞暟: {len(buildings_clipped)}")

    # Save output / 淇濆瓨
    output_file = output_dir / f'{city_name}_buildings_clipped.gpkg'
    buildings_clipped.to_file(output_file, driver='GPKG')
    print(f"  Saved output / 淇濆瓨: {output_file}")

    # Statistics / 缁熻
    if 'height' in buildings_clipped.columns:
        mean_height = buildings_clipped['height'].mean()
        print(f"  Mean building height / 骞冲潎寤虹瓚楂樺害: {mean_height:.2f} m")

print("\n" + "="*60)
print("All processing complete / 鎵€鏈夊鐞嗗畬鎴愶紒")
print("="*60)



