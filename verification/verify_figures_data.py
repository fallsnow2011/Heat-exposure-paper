
import pandas as pd
import numpy as np
from pathlib import Path

def verify_stats():
    base_dir = Path("results/inequality_analysis")
    heatwave_path = base_dir / "lsoa_hei_summary_heatwave.csv"
    typical_path = base_dir / "lsoa_hei_summary_typical_day.csv"
    
    if not heatwave_path.exists() or not typical_path.exists():
        print("Error: Summary files not found.")
        return

    df_hw = pd.read_csv(heatwave_path)
    df_typ = pd.read_csv(typical_path)

    # Merge on LSOA
    df = df_hw.merge(df_typ[['lsoa11cd', 'hei_mean']], on='lsoa11cd', suffixes=('_hw', '_typ'))
    
    # Calculate Amplification
    df['amplification'] = df['hei_mean_hw'] - df['hei_mean_typ']
    
    # 1. Check Median Shadow
    # Assuming shadow_mean is fraction (0-1) based on previous peek (0.50 etc in London)
    # The manuscript says "median 5.4%". So we expect median to be ~0.054.
    median_shadow = df['shadow_mean'].median()
    print(f"Median Shadow (fraction): {median_shadow:.4f} ({median_shadow*100:.2f}%)")
    
    # 2. Check Heat Trap % in Decile 1
    # Definition: Shade < 5% (0.05) AND Amplification > 9
    # Manuscript says: "low shade coverage (<5%; below the overall median 5.4%)"
    # So the threshold is likely 5% (0.05).
    
    heat_trap_mask = (df['shadow_mean'] < 0.05) & (df['amplification'] > 9.0)
    df['is_heat_trap'] = heat_trap_mask
    
    decile1 = df[df['IMD_Decile'] == 1]
    heat_trap_pct_d1 = decile1['is_heat_trap'].mean() * 100
    
    print(f"Heat Trap % in Decile 1: {heat_trap_pct_d1:.2f}%")
    print(f"  (Count: {decile1['is_heat_trap'].sum()}/{len(decile1)})")
    
    # 3. Check Inequality Gap (Pop weighted)
    # We need population data
    def weighted_mean(series, weights):
        return np.average(series, weights=weights)
    
    pop = df['TotPop']
    
    # Typical Gap
    poor_typ = weighted_mean(df[df['IMD_Decile'].isin([1,2,3])]['hei_mean_typ'], df[df['IMD_Decile'].isin([1,2,3])]['TotPop'])
    rich_typ = weighted_mean(df[df['IMD_Decile'].isin([8,9,10])]['hei_mean_typ'], df[df['IMD_Decile'].isin([8,9,10])]['TotPop'])
    gap_typ = poor_typ - rich_typ
    
    # Heatwave Gap
    poor_hw = weighted_mean(df[df['IMD_Decile'].isin([1,2,3])]['hei_mean_hw'], df[df['IMD_Decile'].isin([1,2,3])]['TotPop'])
    rich_hw = weighted_mean(df[df['IMD_Decile'].isin([8,9,10])]['hei_mean_hw'], df[df['IMD_Decile'].isin([8,9,10])]['TotPop'])
    gap_hw = poor_hw - rich_hw
    
    print(f"Typical Gap (Pop Weighted): {gap_typ:.2f}°C")
    print(f"Heatwave Gap (Pop Weighted): {gap_hw:.2f}°C")

if __name__ == "__main__":
    verify_stats()
