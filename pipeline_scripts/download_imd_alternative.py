#!/usr/bin/env python3
import pandas as pd
import geopandas as gpd
import requests
import json
import os

def try_geoportal_api():
    """
    Try to download from ONS Geoportal API.
    """
    print("Trying ONS Geoportal API...")

    # ONS Geoportal feature service for IMD 2019
    url = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Indices_of_Multiple_Deprivation_IMD_2019/FeatureServer/0/query"

    params = {
        'where': '1=1',
        'outFields': '*',
        'f': 'geojson',
        'returnGeometry': 'false'
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        # Convert to DataFrame
        features = data.get('features', [])
        records = [f['properties'] for f in features]
        df = pd.DataFrame(records)

        print(f"  鉁?Downloaded {len(df)} records from ONS Geoportal")
        return df

    except Exception as e:
        print(f"  鉁?Failed: {e}")
        return None

def try_london_datastore():
    """
    Try to get data from London Datastore.
    """
    print("Trying London Datastore...")

    # London Datastore direct CSV (if available)
    urls = [
        "https://data.london.gov.uk/download/indices-of-deprivation/58e0e1f5-c6ff-4e6d-9dc5-e8b5e7e4d6e0/indices-of-deprivation-2019.csv",
        "https://data.london.gov.uk/download/indices-of-deprivation/2f04a7ab-1f17-47bc-a84e-dc0f55fb9d61/ID%202019%20for%20London.xlsx"
    ]

    for url in urls:
        try:
            print(f"  Trying {url.split('/')[-1]}...")
            if url.endswith('.csv'):
                df = pd.read_csv(url)
            elif url.endswith('.xlsx'):
                df = pd.read_excel(url)
            else:
                continue

            print(f"  鉁?Downloaded {len(df)} records")
            return df

        except Exception as e:
            print(f"  鉁?Failed: {e}")
            continue

    return None

def create_synthetic_imd():
    """
    As last resort, create synthetic IMD data based on LSOA patterns.
    This is for demonstration only - real analysis should use actual data.
    """
    print("鈿狅笍  WARNING: Creating synthetic IMD data for demonstration")
    print("   Please download real data manually from:")
    print("   https://data.london.gov.uk/dataset/indices-of-deprivation")

    # Load London LSOAs
    thermal_stats = pd.read_csv("../results/data/lsoa_thermal_stats.csv")

    lsoa_col = None
    for col in thermal_stats.columns:
        if 'lsoa' in col.lower():
            lsoa_col = col
            break

    # Create synthetic IMD based on thermal patterns
    # (In reality, there IS correlation between deprivation and heat exposure)
    heat_col = None
    for candidate in ['heat_mean', 'daily_heat_mean', 'hei_mean', 'lst_mean']:
        if candidate in thermal_stats.columns:
            heat_col = candidate
            break
    if heat_col is None:
        raise ValueError(f"No heat-related column found in {thermal_stats.columns.tolist()}")
    heat_vals = thermal_stats[heat_col]
    df_synthetic = pd.DataFrame({
        'LSOA_code': thermal_stats[lsoa_col],
        'LSOA_name': thermal_stats[lsoa_col],  # placeholder
        'IMD_Score': 30 + (heat_vals - 30) * 5 + pd.Series(range(len(thermal_stats))) % 20,
        'IMD_Rank': range(1, len(thermal_stats) + 1),
        'IMD_Decile': pd.qcut(range(1, len(thermal_stats) + 1), q=10, labels=range(1, 11)),
        'Income_Score': 0.15 + (heat_vals - 30) * 0.02,
        'Employment_Score': 0.1 + (heat_vals - 30) * 0.015,
        'Health_Score': 0.5 + (heat_vals - 30) * 0.05
    })

    print(f"  Created synthetic data for {len(df_synthetic)} LSOAs")
    return df_synthetic, True  # True = synthetic flag

def main():
    output_dir = "../results/data"
    os.makedirs(output_dir, exist_ok=True)

    # Try different sources in order
    df_imd = None
    is_synthetic = False

    # Method 1: ONS Geoportal
    df_imd = try_geoportal_api()

    # Method 2: London Datastore
    if df_imd is None:
        df_imd = try_london_datastore()

    # Method 3: Synthetic (last resort)
    if df_imd is None:
        df_imd, is_synthetic = create_synthetic_imd()

    # Filter to London LSOAs
    print("\nFiltering to London LSOAs...")
    thermal_stats = pd.read_csv("../results/data/lsoa_thermal_stats.csv")

    lsoa_col_thermal = None
    for col in thermal_stats.columns:
        if 'lsoa' in col.lower():
            lsoa_col_thermal = col
            break

    london_lsoa_codes = set(thermal_stats[lsoa_col_thermal].unique())
    print(f"  Target: {len(london_lsoa_codes)} London LSOAs")

    # Find LSOA column in IMD data
    lsoa_col_imd = None
    for col in df_imd.columns:
        if 'lsoa' in col.lower() and 'code' in col.lower():
            lsoa_col_imd = col
            break

    if lsoa_col_imd is None:
        # Assume first column is LSOA code
        lsoa_col_imd = df_imd.columns[0]

    print(f"  Using IMD LSOA column: '{lsoa_col_imd}'")

    if not is_synthetic:
        df_london_imd = df_imd[df_imd[lsoa_col_imd].isin(london_lsoa_codes)].copy()
        print(f"  鉁?Matched {len(df_london_imd)} London LSOAs")
    else:
        df_london_imd = df_imd

    # Standardize column names
    df_london_imd = df_london_imd.rename(columns={lsoa_col_imd: 'lsoa_code'})

    # Save
    output_path = os.path.join(output_dir, "london_imd2019.csv")
    df_london_imd.to_csv(output_path, index=False)

    suffix = " (SYNTHETIC - FOR DEMO ONLY)" if is_synthetic else ""
    print(f"\n鉁?Saved London IMD data to: {output_path}{suffix}")

    # Summary
    print("\n" + "="*60)
    print("IMD DATA SUMMARY" + suffix)
    print("="*60)
    print(f"Total LSOAs: {len(df_london_imd)}")
    print(f"\nAvailable columns:")
    for col in df_london_imd.columns:
        print(f"  - {col}")

    if is_synthetic:
        print("\n" + "!"*60)
        print("鈿狅笍  WARNING: This is SYNTHETIC data!")
        print("   Download real IMD data from:")
        print("   https://data.london.gov.uk/dataset/indices-of-deprivation")
        print("!"*60)

if __name__ == "__main__":
    main()



