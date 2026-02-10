"""
Visualization and Export Module for Qanat Detection Results
Handles exporting to multiple formats and creating interactive visualizations.
Updated for the current pipeline with bbox and centroid support.
"""

import os
import logging
from typing import Dict, Optional

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)


class ResultsExporter:
    """Export detection results to multiple formats."""
    
    def __init__(self, output_folder: str = "exports"):
        """Initialize exporter.
        
        Args:
            output_folder: Base folder for all exports
        """
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Initialized exporter: {output_folder}")
    
    def to_shapefile(
        self,
        geopackage_path: str,
        filename: str = "qanats_detected.shp"
    ) -> str:
        """Export results to Shapefile format.
        
        Args:
            geopackage_path: Path to input GeoPackage
            filename: Output filename (default: qanats_detected.shp)
            
        Returns:
            Path to exported Shapefile
            
        Example:
            >>> exporter = ResultsExporter("exports")
            >>> shp_path = exporter.to_shapefile("results.gpkg", "qanats.shp")
        """
        try:
            logger.info(f"Loading GeoPackage: {geopackage_path}")
            gdf = gpd.read_file(geopackage_path)
            
            output_path = os.path.join(self.output_folder, filename)
            logger.info(f"Exporting {len(gdf)} features to Shapefile...")
            
            # Drop WKT columns if present (not needed for shapefile)
            cols_to_drop = [col for col in gdf.columns if col.endswith('_wkt')]
            if cols_to_drop:
                gdf = gdf.drop(columns=cols_to_drop)
            
            gdf.to_file(output_path, driver='ESRI Shapefile')
            
            logger.info(f"✓ Shapefile exported: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to export Shapefile: {e}", exc_info=True)
            return None
    
    def to_geojson(
        self,
        geopackage_path: str,
        filename: str = "qanats_detected.geojson"
    ) -> str:
        """Export results to GeoJSON format.
        
        Args:
            geopackage_path: Path to input GeoPackage
            filename: Output filename (default: qanats_detected.geojson)
            
        Returns:
            Path to exported GeoJSON
            
        Example:
            >>> exporter = ResultsExporter("exports")
            >>> geojson_path = exporter.to_geojson("results.gpkg")
        """
        try:
            logger.info(f"Loading GeoPackage: {geopackage_path}")
            gdf = gpd.read_file(geopackage_path)
            
            output_path = os.path.join(self.output_folder, filename)
            logger.info(f"Exporting {len(gdf)} features to GeoJSON...")
            
            # Drop WKT columns if present
            cols_to_drop = [col for col in gdf.columns if col.endswith('_wkt')]
            if cols_to_drop:
                gdf = gdf.drop(columns=cols_to_drop)
            
            gdf.to_file(output_path, driver='GeoJSON')
            
            logger.info(f"✓ GeoJSON exported: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to export GeoJSON: {e}", exc_info=True)
            return None
    
    def to_csv(
        self,
        geopackage_path: str,
        filename: str = "qanats_detected.csv",
        include_geometry: bool = True,
        use_centroids: bool = True
    ) -> str:
        """Export results to CSV format.
        
        Args:
            geopackage_path: Path to input GeoPackage
            filename: Output filename (default: qanats_detected.csv)
            include_geometry: Include longitude/latitude columns
            use_centroids: If True, use centroid coordinates; if False, use bbox center
            
        Returns:
            Path to exported CSV
            
        Example:
            >>> exporter = ResultsExporter("exports")
            >>> csv_path = exporter.to_csv("results.gpkg", include_geometry=True)
        """
        try:
            logger.info(f"Loading GeoPackage: {geopackage_path}")
            gdf = gpd.read_file(geopackage_path)
            
            output_path = os.path.join(self.output_folder, filename)
            logger.info(f"Exporting {len(gdf)} features to CSV...")
            
            # Create copy for export
            df = gdf.copy()
            
            # Add coordinates if requested
            if include_geometry:
                # Check if centroid columns already exist
                if use_centroids and 'centroid_x' in df.columns and 'centroid_y' in df.columns:
                    logger.info("Using existing centroid coordinates")
                    df['longitude'] = df['centroid_x']
                    df['latitude'] = df['centroid_y']
                else:
                    # Calculate from geometry
                    logger.info("Calculating coordinates from geometry")
                    centroids = df.geometry.centroid
                    df['longitude'] = centroids.x
                    df['latitude'] = centroids.y
            
            # Select relevant columns
            columns_to_export = []
            
            if include_geometry:
                columns_to_export.extend(['longitude', 'latitude'])
            
            # Add attribute columns (exclude geometry and WKT columns)
            for col in df.columns:
                if col not in ['geometry', 'longitude', 'latitude', 'centroid_wkt', 'bbox_wkt']:
                    columns_to_export.append(col)
            
            df[columns_to_export].to_csv(output_path, index=False)
            
            logger.info(f"✓ CSV exported: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}", exc_info=True)
            return None
    
    def to_all_formats(
        self,
        geopackage_path: str,
        formats: list = ['shapefile', 'geojson', 'csv']
    ) -> Dict[str, str]:
        """Export to multiple formats at once.
        
        Args:
            geopackage_path: Path to input GeoPackage
            formats: List of formats to export
                     Options: 'shapefile', 'geojson', 'csv'
                     
        Returns:
            Dictionary with format -> filepath mappings
            
        Example:
            >>> exporter = ResultsExporter("exports")
            >>> exports = exporter.to_all_formats(
            ...     "results.gpkg",
            ...     formats=['shapefile', 'geojson', 'csv']
            ... )
            >>> print(f"Shapefile: {exports['shapefile']}")
        """
        logger.info(f"\n[EXPORT] Starting multi-format export...")
        exported = {}
        
        if 'shapefile' in formats:
            shp_path = self.to_shapefile(geopackage_path)
            if shp_path:
                exported['shapefile'] = shp_path
        
        if 'geojson' in formats:
            geojson_path = self.to_geojson(geopackage_path)
            if geojson_path:
                exported['geojson'] = geojson_path
        
        if 'csv' in formats:
            csv_path = self.to_csv(geopackage_path)
            if csv_path:
                exported['csv'] = csv_path
        
        logger.info(f"✓ Export complete! {len(exported)} formats exported\n")
        return exported


class ResultsVisualizer:
    """Create interactive visualizations of detection results."""
    
    def __init__(self):
        """Initialize visualizer."""
        logger.info("Initialized visualizer")
    
    def create_interactive_map(
        self,
        geopackage_path: str,
        zoom: int = 12,
        center: tuple = None,
        basemap: str = "Esri.WorldImagery",
        use_centroids: bool = True
    ):
        """Create interactive map using leafmap.
        
        Args:
            geopackage_path: Path to GeoPackage with detection results
            zoom: Map zoom level (1-20, default 12)
            center: Map center as (latitude, longitude), auto-detected if None
            basemap: Leafmap basemap name (default: Esri.WorldImagery)
                     Options: 'Esri.WorldImagery', 'OpenStreetMap.Mapnik', etc.
            use_centroids: If True and centroids file exists, use it for display
            
        Returns:
            leafmap.Map object or None if leafmap not available
            
        Example:
            >>> visualizer = ResultsVisualizer()
            >>> m = visualizer.create_interactive_map("results.gpkg", zoom=13)
            >>> display(m)  # In Jupyter
        """
        try:
            try:
                import leafmap
            except ImportError:
                logger.error("leafmap not installed. Install with: pip install leafmap")
                return None
            
            logger.info("\n[VISUALIZATION] Creating interactive map...")
            
            # Load data
            if not os.path.exists(geopackage_path):
                logger.error(f"GeoPackage not found: {geopackage_path}")
                return None
            
            # Try to use centroids file if available and requested
            if use_centroids:
                centroids_path = geopackage_path.replace('.gpkg', '_centroids.gpkg')
                if os.path.exists(centroids_path):
                    logger.info(f"Using centroids file: {centroids_path}")
                    geopackage_path = centroids_path
                else:
                    logger.info("Centroids file not found, using main file")
            
            logger.info(f"Loading results from {geopackage_path}...")
            gdf = gpd.read_file(geopackage_path)
            logger.info(f"Loaded {len(gdf)} features")
            
            # Ensure WGS84
            if gdf.crs != "EPSG:4326":
                logger.info("Reprojecting to EPSG:4326...")
                gdf = gdf.to_crs("EPSG:4326")
            
            # Auto-detect center if not provided
            if center is None:
                bounds = gdf.total_bounds
                center = (
                    (bounds[1] + bounds[3]) / 2,  # latitude
                    (bounds[0] + bounds[2]) / 2   # longitude
                )
                logger.info(f"Auto-detected center: {center}")
            
            # Create map
            logger.info(f"Creating map centered at {center}, zoom={zoom}...")
            m = leafmap.Map(center=center, zoom=zoom, height="800px", basemap=basemap)
            
            # Add detected features as layer
            logger.info(f"Adding {len(gdf)} detected features to map...")
            try:
                m.add_gdf(
                    gdf,
                    layer_name="Detected Qanats",
                    zoom=False,
                    color="red",
                    fill_color="red",
                    opacity=0.7,
                    fill_opacity=0.5
                )
                logger.info(f"✓ Added features as layer")
            except Exception as e:
                logger.warning(f"Could not add as GeoDataFrame layer: {e}")
                # Fallback: add features as markers
                logger.info("Fallback: Adding features as markers...")
                gdf_display = gdf.copy()
                
                # Get centroids
                centroids = gdf_display.geometry.centroid
                gdf_display["longitude"] = centroids.x
                gdf_display["latitude"] = centroids.y
                
                n_sample = min(500, len(gdf_display))
                gdf_sample = gdf_display.sample(n=n_sample, random_state=42) if len(gdf_display) > n_sample else gdf_display
                
                logger.info(f"Adding {len(gdf_sample)} markers...")
                for idx, row in gdf_sample.iterrows():
                    try:
                        lat = row['latitude']
                        lon = row['longitude']
                        cluster_id = row.get('cluster', 'N/A')
                        confidence = row.get('confidence', 'N/A')
                        popup_text = f"Cluster: {cluster_id}<br>Confidence: {confidence}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}"
                        
                        m.add_circle_marker(
                            location=[lat, lon],
                            radius=4,
                            color="red",
                            fill_color="red",
                            popup=popup_text,
                            opacity=0.8
                        )
                    except Exception as e:
                        logger.debug(f"Error adding marker: {e}")
            
            # Add legend
            legend_dict = {
                "Detected Qanats": "red",
            }
            m.add_legend(title="Qanat Detection Results", legend_dict=legend_dict)
            
            logger.info("✓ Map created successfully!")
            return m
        
        except Exception as e:
            logger.error(f"Visualization failed: {e}", exc_info=True)
            return None
    
    def get_summary_statistics(self, geopackage_path: str) -> Dict:
        """Calculate summary statistics from detection results.
        
        Args:
            geopackage_path: Path to GeoPackage
            
        Returns:
            Dictionary with summary statistics
            
        Example:
            >>> visualizer = ResultsVisualizer()
            >>> stats = visualizer.get_summary_statistics("results.gpkg")
            >>> print(f"Total features: {stats['n_features']}")
        """
        try:
            logger.info("Calculating summary statistics...")
            gdf = gpd.read_file(geopackage_path)
            
            stats = {
                'n_features': len(gdf),
                'crs': str(gdf.crs),
                'bounds': gdf.total_bounds.tolist(),  # [minx, miny, maxx, maxy]
                'area_km2': None,
            }
            
            # Calculate area if possible
            try:
                if gdf.crs != "EPSG:3857":
                    gdf_projected = gdf.to_crs("EPSG:3857")
                else:
                    gdf_projected = gdf
                
                stats['area_km2'] = gdf_projected.geometry.area.sum() / 1e6
            except Exception as e:
                logger.warning(f"Could not calculate area: {e}")
            
            # Get cluster statistics if available
            if 'cluster' in gdf.columns:
                stats['n_clusters'] = gdf['cluster'].nunique()
                stats['clusters'] = gdf['cluster'].value_counts().to_dict()
            
            # Get confidence statistics if available
            if 'confidence' in gdf.columns:
                stats['mean_confidence'] = gdf['confidence'].mean()
                stats['median_confidence'] = gdf['confidence'].median()
                stats['min_confidence'] = gdf['confidence'].min()
                stats['max_confidence'] = gdf['confidence'].max()
            
            logger.info(f"✓ Statistics calculated")
            return stats
        
        except Exception as e:
            logger.error(f"Failed to calculate statistics: {e}")
            return {}
    
    def print_summary(self, geopackage_path: str) -> None:
        """Print formatted summary of results.
        
        Args:
            geopackage_path: Path to GeoPackage
            
        Example:
            >>> visualizer = ResultsVisualizer()
            >>> visualizer.print_summary("results.gpkg")
        """
        stats = self.get_summary_statistics(geopackage_path)
        
        print("\n" + "="*80)
        print("QANAT DETECTION RESULTS SUMMARY")
        print("="*80)
        
        print(f"\n📊 Basic Statistics:")
        print(f"   • Total detected features: {stats.get('n_features', 'N/A')}")
        print(f"   • Coordinate system: {stats.get('crs', 'N/A')}")
        
        if stats.get('n_clusters'):
            print(f"   • Number of clusters: {stats['n_clusters']}")
        
        if stats.get('area_km2'):
            print(f"   • Total area covered: {stats['area_km2']:.2f} km²")
        
        if stats.get('mean_confidence'):
            print(f"\n🎯 Confidence Statistics:")
            print(f"   • Mean: {stats['mean_confidence']:.1f}")
            print(f"   • Median: {stats['median_confidence']:.1f}")
            print(f"   • Min: {stats['min_confidence']:.1f}")
            print(f"   • Max: {stats['max_confidence']:.1f}")
        
        bounds = stats.get('bounds', [])
        if bounds:
            print(f"\n🗺️  Spatial Extent:")
            print(f"   • Min X (Longitude): {bounds[0]:.6f}")
            print(f"   • Min Y (Latitude):  {bounds[1]:.6f}")
            print(f"   • Max X (Longitude): {bounds[2]:.6f}")
            print(f"   • Max Y (Latitude):  {bounds[3]:.6f}")
        
        if stats.get('clusters'):
            print(f"\n🔢 Clusters (top 10):")
            clusters_sorted = sorted(stats['clusters'].items(), key=lambda x: x[1], reverse=True)[:10]
            for cluster_id, count in clusters_sorted:
                print(f"   • Cluster {cluster_id}: {count} features")
        
        print("\n" + "="*80)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
