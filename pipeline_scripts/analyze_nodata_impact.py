#!/usr/bin/env python3
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
import geopandas as gpd
import os

class NoDataAnalyzer:
    def __init__(self, shadow_tif_path):
        self.shadow_path = shadow_tif_path

        with rasterio.open(shadow_tif_path) as src:
            self.data = src.read(1)
            self.nodata_value = src.nodata if src.nodata is not None else 255
            self.transform = src.transform
            self.crs = src.crs
            self.bounds = src.bounds
            self.width = src.width
            self.height = src.height

        self.nodata_mask = (self.data == self.nodata_value)
        self.valid_mask = ~self.nodata_mask

    def generate_spatial_distribution_map(self, output_path="nodata_spatial_dist.png"):
        """
        Create a visual map showing where NoData pixels are located.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left: NoData binary map
        ax1 = axes[0]
        im1 = ax1.imshow(self.nodata_mask, cmap='RdYlGn_r', interpolation='nearest')
        ax1.set_title('NoData Spatial Distribution\n(Red = NoData, Green = Valid)', fontsize=14)
        ax1.set_xlabel('Pixel X')
        ax1.set_ylabel('Pixel Y')
        plt.colorbar(im1, ax=ax1, label='NoData (1) vs Valid (0)')

        # Right: Shadow data with NoData masked
        ax2 = axes[1]
        shadow_masked = np.ma.masked_where(self.nodata_mask, self.data)
        im2 = ax2.imshow(shadow_masked, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax2.set_title('Shadow Map\n(White areas = NoData)', fontsize=14)
        ax2.set_xlabel('Pixel X')
        ax2.set_ylabel('Pixel Y')
        cbar2 = plt.colorbar(im2, ax=ax2, label='Shadow (0) vs Lit (1)')
        cbar2.set_ticks([0, 1])
        cbar2.set_ticklabels(['Shadow', 'Lit'])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"鉁?Saved spatial distribution map: {output_path}")

    def analyze_clustering(self):
        """
        Analyze if NoData pixels are clustered (indicating systematic regions)
        or randomly distributed (indicating data quality issues).
        """
        # Label connected components of NoData
        labeled_nodata, num_clusters = ndimage.label(self.nodata_mask)

        cluster_sizes = []
        for label in range(1, num_clusters + 1):
            size = np.sum(labeled_nodata == label)
            cluster_sizes.append(size)

        cluster_sizes = np.array(cluster_sizes)

        print("\n" + "="*60)
        print("NODATA CLUSTERING ANALYSIS")
        print("="*60)
        print(f"Total NoData clusters: {num_clusters}")
        print(f"Largest cluster: {cluster_sizes.max():,} pixels ({cluster_sizes.max() / self.nodata_mask.sum() * 100:.1f}% of all NoData)")
        print(f"Mean cluster size: {cluster_sizes.mean():.1f} pixels")
        print(f"Median cluster size: {np.median(cluster_sizes):.1f} pixels")

        # Top 5 largest clusters
        top5_indices = np.argsort(-cluster_sizes)[:5]
        print("\nTop 5 largest NoData regions:")
        for i, idx in enumerate(top5_indices, 1):
            size = cluster_sizes[idx]
            percent = size / self.nodata_mask.sum() * 100
            print(f"  {i}. Cluster #{idx+1}: {size:,} pixels ({percent:.1f}%)")

        return labeled_nodata, num_clusters, cluster_sizes

    def plot_cluster_histogram(self, cluster_sizes, output_path="nodata_cluster_hist.png"):
        """
        Histogram of NoData cluster sizes.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Log-scale histogram
        bins = np.logspace(0, np.log10(cluster_sizes.max()), 50)
        ax.hist(cluster_sizes, bins=bins, alpha=0.7, color='#E76F51', edgecolor='black')
        ax.set_xscale('log')
        ax.set_xlabel('Cluster Size (pixels)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of NoData Cluster Sizes', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Annotate largest cluster
        ax.axvline(cluster_sizes.max(), color='red', linestyle='--', linewidth=2,
                   label=f'Largest: {cluster_sizes.max():,} px')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"鉁?Saved cluster histogram: {output_path}")

    def analyze_edge_proximity(self):
        """
        Check if NoData is concentrated at raster edges (boundary issue)
        or distributed throughout (water/parks).
        """
        margin = 100  # pixels

        # Define edge regions
        top_edge = self.nodata_mask[:margin, :]
        bottom_edge = self.nodata_mask[-margin:, :]
        left_edge = self.nodata_mask[:, :margin]
        right_edge = self.nodata_mask[:, -margin:]
        interior = self.nodata_mask[margin:-margin, margin:-margin]

        edge_nodata = (top_edge.sum() + bottom_edge.sum() +
                       left_edge.sum() + right_edge.sum())
        interior_nodata = interior.sum()
        total_nodata = self.nodata_mask.sum()

        print("\n" + "="*60)
        print("NODATA EDGE vs INTERIOR ANALYSIS")
        print("="*60)
        print(f"Edge region NoData: {edge_nodata:,} ({edge_nodata/total_nodata*100:.1f}%)")
        print(f"Interior NoData: {interior_nodata:,} ({interior_nodata/total_nodata*100:.1f}%)")

        if edge_nodata / total_nodata > 0.5:
            print("鈿狅笍  WARNING: >50% NoData is on edges 鈫?DSM boundary issue!")
        else:
            print("鉁?NoData is distributed in interior 鈫?likely water/parks (acceptable)")

    def quantify_network_impact(self, network_gpkg=None):
        """
        Estimate how many network edges are affected by NoData.
        """
        if network_gpkg is None or not os.path.exists(network_gpkg):
            print("\n鈿狅笍  Network GPKG not provided, skipping network impact analysis")
            return

        print("\n" + "="*60)
        print("NETWORK IMPACT ANALYSIS")
        print("="*60)

        gdf = gpd.read_file(network_gpkg, layer='edges')
        print(f"Total network edges: {len(gdf)}")

        # Reproject if needed
        if gdf.crs != self.crs:
            print(f"Reprojecting network from {gdf.crs} to {self.crs}...")
            gdf = gdf.to_crs(self.crs)

        # Sample edges at their midpoints
        midpoints = gdf.geometry.interpolate(0.5, normalized=True)

        # Convert to pixel coordinates
        with rasterio.open(self.shadow_path) as src:
            coords = [(p.x, p.y) for p in midpoints]
            sampled = list(src.sample(coords))

        sampled_values = np.array([v[0] for v in sampled])
        nodata_edges = np.sum(sampled_values == 255)
        nodata_pct = nodata_edges / len(gdf) * 100

        print(f"Edges with NoData at midpoint: {nodata_edges}/{len(gdf)} ({nodata_pct:.1f}%)")

        if nodata_pct > 30:
            print("馃敶 HIGH IMPACT: >30% of edges affected!")
            print("   Recommendation: Investigate and potentially expand DSM coverage")
        elif nodata_pct > 10:
            print("馃煛 MODERATE IMPACT: 10-30% of edges affected")
            print("   Recommendation: Use spatial interpolation for missing values")
        else:
            print("鉁?LOW IMPACT: <10% of edges affected (acceptable)")

        return nodata_edges, len(gdf)

    def generate_report(self, output_dir="nodata_analysis"):
        """
        Generate comprehensive NoData analysis report.
        """
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*60)
        print("COMPREHENSIVE NODATA ANALYSIS REPORT")
        print("="*60)

        # Basic statistics
        total_pixels = self.data.size
        nodata_count = self.nodata_mask.sum()
        valid_count = self.valid_mask.sum()
        nodata_pct = nodata_count / total_pixels * 100

        print(f"\nRaster: {os.path.basename(self.shadow_path)}")
        print(f"Size: {self.width} x {self.height} = {total_pixels:,} pixels")
        print(f"NoData: {nodata_count:,} pixels ({nodata_pct:.2f}%)")
        print(f"Valid: {valid_count:,} pixels ({100-nodata_pct:.2f}%)")

        # Generate visualizations
        self.generate_spatial_distribution_map(
            os.path.join(output_dir, "nodata_spatial_distribution.png")
        )

        labeled, num_clusters, sizes = self.analyze_clustering()

        self.plot_cluster_histogram(sizes,
            os.path.join(output_dir, "nodata_cluster_histogram.png")
        )

        self.analyze_edge_proximity()

        # Save labeled cluster map
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(labeled, cmap='tab20', interpolation='nearest')
        ax.set_title(f'NoData Clusters (n={num_clusters})', fontsize=14)
        plt.colorbar(im, ax=ax, label='Cluster ID')
        plt.savefig(os.path.join(output_dir, "nodata_clusters_labeled.png"), dpi=300)
        plt.close()

        # Save summary
        with open(os.path.join(output_dir, "nodata_summary.txt"), 'w') as f:
            f.write("NODATA ANALYSIS SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Raster: {os.path.basename(self.shadow_path)}\n")
            f.write(f"NoData Percentage: {nodata_pct:.2f}%\n")
            f.write(f"Number of Clusters: {num_clusters}\n")
            f.write(f"Largest Cluster: {sizes.max():,} pixels\n")

        print(f"\n鉁?Full report saved to: {output_dir}/")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze NoData distribution in shadow rasters")
    parser.add_argument("--shadow",
                        default="./shadow_maps/shadow_2024-06-21_1200.tif",
                        help="Path to shadow raster")
    parser.add_argument("--network",
                        default="./london_daily_shadow_network.gpkg",
                        help="Optional: Network GPKG to assess impact")
    parser.add_argument("--output-dir",
                        default="./nodata_analysis",
                        help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.shadow):
        print(f"Error: Shadow raster not found: {args.shadow}")
        return

    analyzer = NoDataAnalyzer(args.shadow)
    analyzer.generate_report(args.output_dir)

    # Network impact (optional)
    if os.path.exists(args.network):
        analyzer.quantify_network_impact(args.network)

if __name__ == "__main__":
    main()



