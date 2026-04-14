#!/usr/bin/env python3
import os
import glob
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import json

class ShadowResultAnalyzer:
    def __init__(self, shadow_dir):
        self.shadow_dir = shadow_dir
        self.shadow_files = sorted(glob.glob(os.path.join(shadow_dir, "shadow_*.tif")))
        self.stats_data = []

    def analyze_single_raster(self, tif_path):
        """Extract statistics from a single shadow raster"""
        filename = os.path.basename(tif_path)
        # Parse filename: shadow_YYYY-MM-DD_HH00.tif
        parts = filename.replace('.tif', '').split('_')
        date_str = parts[1]
        time_str = parts[2]
        hour = int(time_str[:2])

        with rasterio.open(tif_path) as src:
            data = src.read(1)
            nodata = src.nodata if src.nodata is not None else 255

            # Calculate statistics
            total_pixels = data.size
            shadow_pixels = np.sum(data == 0)  # Shadow
            lit_pixels = np.sum(data == 1)      # Lit
            nodata_pixels = np.sum(data == nodata) # NoData
            valid_pixels = shadow_pixels + lit_pixels

            if valid_pixels > 0:
                shadow_ratio = shadow_pixels / valid_pixels
                lit_ratio = lit_pixels / valid_pixels
            else:
                shadow_ratio = np.nan
                lit_ratio = np.nan

            nodata_ratio = nodata_pixels / total_pixels

            # File size
            file_size_mb = os.path.getsize(tif_path) / (1024 * 1024)

            return {
                'filename': filename,
                'date': date_str,
                'hour': hour,
                'total_pixels': total_pixels,
                'valid_pixels': valid_pixels,
                'shadow_pixels': shadow_pixels,
                'lit_pixels': lit_pixels,
                'nodata_pixels': nodata_pixels,
                'shadow_ratio': shadow_ratio,
                'lit_ratio': lit_ratio,
                'nodata_ratio': nodata_ratio,
                'file_size_mb': file_size_mb,
                'crs': str(src.crs),
                'width': src.width,
                'height': src.height,
                'bounds': src.bounds
            }

    def run_analysis(self):
        """Analyze all shadow rasters"""
        print(f"Found {len(self.shadow_files)} shadow rasters")

        for tif_path in tqdm(self.shadow_files, desc="Analyzing shadows"):
            try:
                stats = self.analyze_single_raster(tif_path)
                self.stats_data.append(stats)
            except Exception as e:
                print(f"Error processing {tif_path}: {e}")

        self.df = pd.DataFrame(self.stats_data)
        return self.df

    def generate_report(self, output_dir):
        """Generate comprehensive analysis report"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Summary Statistics
        print("\n" + "="*60)
        print("SHADOW ANALYSIS SUMMARY REPORT")
        print("="*60)

        print(f"\nDataset Overview:")
        print(f"  Total rasters: {len(self.df)}")
        print(f"  Date: {self.df['date'].iloc[0]}")
        print(f"  Time range: {self.df['hour'].min():02d}:00 - {self.df['hour'].max():02d}:00")
        print(f"  Raster size: {self.df['width'].iloc[0]} x {self.df['height'].iloc[0]} pixels")
        print(f"  CRS: {self.df['crs'].iloc[0]}")

        print(f"\nShadow Coverage Statistics:")
        print(f"  Mean shadow ratio: {self.df['shadow_ratio'].mean():.2%}")
        print(f"  Std shadow ratio: {self.df['shadow_ratio'].std():.2%}")
        print(f"  Min shadow ratio: {self.df['shadow_ratio'].min():.2%} (Hour {self.df.loc[self.df['shadow_ratio'].idxmin(), 'hour']:02d}:00)")
        print(f"  Max shadow ratio: {self.df['shadow_ratio'].max():.2%} (Hour {self.df.loc[self.df['shadow_ratio'].idxmax(), 'hour']:02d}:00)")

        print(f"\nData Quality:")
        print(f"  Mean NoData ratio: {self.df['nodata_ratio'].mean():.2%}")
        print(f"  Mean valid pixels: {self.df['valid_pixels'].mean()/1e6:.1f}M")

        print(f"\nStorage:")
        print(f"  Total size: {self.df['file_size_mb'].sum():.1f} MB")
        print(f"  Average file size: {self.df['file_size_mb'].mean():.1f} MB")

        # Save to file
        with open(os.path.join(output_dir, 'shadow_summary.txt'), 'w') as f:
            f.write("SHADOW ANALYSIS SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {self.df['date'].iloc[0]}\n")
            f.write(f"Time range: {self.df['hour'].min():02d}:00 - {self.df['hour'].max():02d}:00\n")
            f.write(f"Mean shadow ratio: {self.df['shadow_ratio'].mean():.2%}\n")
            f.write(f"Max shadow at: {self.df.loc[self.df['shadow_ratio'].idxmax(), 'hour']:02d}:00\n")
            f.write(f"Min shadow at: {self.df.loc[self.df['shadow_ratio'].idxmin(), 'hour']:02d}:00\n")

        # 2. Hourly Statistics Table
        hourly_stats = self.df[['hour', 'shadow_ratio', 'lit_ratio', 'nodata_ratio']].copy()
        hourly_stats.to_csv(os.path.join(output_dir, 'hourly_statistics.csv'), index=False)
        print(f"\nHourly Statistics:")
        print(hourly_stats.to_string(index=False))

        # 3. Visualizations
        self._plot_shadow_time_series(output_dir)
        self._plot_shadow_distribution(output_dir)
        self._plot_quality_metrics(output_dir)

        # 4. JSON metadata
        metadata = {
            'analysis_date': datetime.now().isoformat(),
            'shadow_date': self.df['date'].iloc[0],
            'num_rasters': len(self.df),
            'mean_shadow_ratio': float(self.df['shadow_ratio'].mean()),
            'peak_shadow_hour': int(self.df.loc[self.df['shadow_ratio'].idxmax(), 'hour']),
            'peak_sun_hour': int(self.df.loc[self.df['shadow_ratio'].idxmin(), 'hour'])
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nReport saved to: {output_dir}")

    def _plot_shadow_time_series(self, output_dir):
        """Plot shadow ratio over time"""
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(self.df['hour'], self.df['shadow_ratio'] * 100,
                marker='o', linewidth=2, markersize=8, color='#2E86AB')
        ax.fill_between(self.df['hour'], self.df['shadow_ratio'] * 100,
                         alpha=0.3, color='#2E86AB')

        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Shadow Coverage (%)', fontsize=12)
        ax.set_title('Temporal Pattern of Urban Shadow Coverage\n2024-06-21 (Summer Solstice)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(self.df['hour'])
        ax.set_xticklabels([f'{h:02d}:00' for h in self.df['hour']], rotation=45)

        # Mark peak shadow
        max_idx = self.df['shadow_ratio'].idxmax()
        ax.scatter(self.df.loc[max_idx, 'hour'],
                   self.df.loc[max_idx, 'shadow_ratio'] * 100,
                   color='red', s=200, zorder=5, marker='*',
                   label=f"Peak Shadow: {self.df.loc[max_idx, 'shadow_ratio']:.1%}")
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shadow_time_series.png'), dpi=300)
        plt.close()

    def _plot_shadow_distribution(self, output_dir):
        """Plot distribution comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Stacked bar chart
        ax1 = axes[0]
        x = self.df['hour']
        shadow_pct = self.df['shadow_ratio'] * 100
        lit_pct = self.df['lit_ratio'] * 100

        ax1.bar(x, shadow_pct, label='Shadow', color='#264653')
        ax1.bar(x, lit_pct, bottom=shadow_pct, label='Sunlit', color='#E9C46A')

        ax1.set_xlabel('Hour of Day', fontsize=11)
        ax1.set_ylabel('Percentage of Valid Pixels (%)', fontsize=11)
        ax1.set_title('Shadow vs. Sunlit Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{h:02d}' for h in x])
        ax1.grid(axis='y', alpha=0.3)

        # Box plot (if we had spatial variance data, this would be useful)
        ax2 = axes[1]
        shadow_values = [self.df['shadow_ratio'].values]
        bp = ax2.boxplot(shadow_values, vert=True, patch_artist=True,
                         labels=['All Hours'])
        bp['boxes'][0].set_facecolor('#2E86AB')
        bp['boxes'][0].set_alpha(0.7)

        ax2.set_ylabel('Shadow Ratio', fontsize=11)
        ax2.set_title('Shadow Ratio Distribution Across Day', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shadow_distribution.png'), dpi=300)
        plt.close()

    def _plot_quality_metrics(self, output_dir):
        """Plot data quality metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # NoData ratio
        ax1 = axes[0]
        ax1.bar(self.df['hour'], self.df['nodata_ratio'] * 100,
                color='#E76F51', alpha=0.7)
        ax1.set_xlabel('Hour of Day', fontsize=11)
        ax1.set_ylabel('NoData Ratio (%)', fontsize=11)
        ax1.set_title('Data Coverage Quality', fontsize=12, fontweight='bold')
        ax1.set_xticks(self.df['hour'])
        ax1.set_xticklabels([f'{h:02d}' for h in self.df['hour']])
        ax1.grid(axis='y', alpha=0.3)

        # File size
        ax2 = axes[1]
        ax2.bar(self.df['hour'], self.df['file_size_mb'],
                color='#2A9D8F', alpha=0.7)
        ax2.set_xlabel('Hour of Day', fontsize=11)
        ax2.set_ylabel('File Size (MB)', fontsize=11)
        ax2.set_title('Compression Efficiency (LZW)', fontsize=12, fontweight='bold')
        ax2.set_xticks(self.df['hour'])
        ax2.set_xticklabels([f'{h:02d}' for h in self.df['hour']])
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_metrics.png'), dpi=300)
        plt.close()

    def compare_morning_afternoon(self):
        """Compare morning vs afternoon shadow patterns"""
        morning = self.df[self.df['hour'] < 12]['shadow_ratio'].mean()
        afternoon = self.df[self.df['hour'] >= 12]['shadow_ratio'].mean()

        print(f"\nMorning (9-11): {morning:.2%} shadow")
        print(f"Afternoon (12-17): {afternoon:.2%} shadow")
        print(f"Difference: {abs(morning - afternoon):.2%}")

        return morning, afternoon

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze computed shadow results")
    parser.add_argument("--shadow-dir",
                        default="./shadow_maps",
                        help="Directory containing shadow_*.tif files")
    parser.add_argument("--output-dir",
                        default="./shadow_analysis_report",
                        help="Output directory for analysis report")
    args = parser.parse_args()

    if not os.path.exists(args.shadow_dir):
        print(f"Error: Shadow directory not found: {args.shadow_dir}")
        return

    analyzer = ShadowResultAnalyzer(args.shadow_dir)
    df = analyzer.run_analysis()

    if len(df) == 0:
        print("No shadow files found!")
        return

    analyzer.generate_report(args.output_dir)
    analyzer.compare_morning_afternoon()

if __name__ == "__main__":
    main()



