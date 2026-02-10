"""
Visualization utilities for displaying detection results on maps.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import geopandas as gpd
import leafmap

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """Visualize detection results on interactive maps."""
    
    def __init__(self, basemap: str = "Google Satellite"):
        """
        Initialize ResultVisualizer.
        
        Args:
            basemap: Base map to use for visualization
        """
        self.basemap = basemap
    
    def create_map(
        self,
        center: Optional[List[float]] = None,
        zoom: int = 12,
        height: str = "800px"
    ) -> leafmap.Map:
        """
        Create a new map instance.
        
        Args:
            center: Map center as [lat, lon]
            zoom: Initial zoom level
            height: Map height
            
        Returns:
            Leafmap Map object
        """
        m = leafmap.Map(
            center=center,
            zoom=zoom,
            height=height,
            #basemap=self.basemap
            basemap = 'Esri.WorldImagery',
        )
        
        return m
    
    def visualize_results(
        self,
        raster_path: str,
        gdf: gpd.GeoDataFrame,
        output_html: Optional[str] = None,
        bbox_polygon: Optional[object] = None,
        layer_name: str = "Detected Features",
        marker_color: str = "#FF0DFF",
        marker_radius: int = 3
    ) -> leafmap.Map:
        """
        Create an interactive map with raster and detection results.
        
        Args:
            raster_path: Path to the raster image
            gdf: GeoDataFrame with detection results
            output_html: Optional path to save HTML map
            bbox_polygon: Optional bounding box polygon
            layer_name: Name for the detection layer
            marker_color: Color for detection markers
            marker_radius: Radius of detection markers
            
        Returns:
            Leafmap Map object
        """
        # Ensure GeoDataFrame is in EPSG:4326
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        
        # Calculate map center from GeoDataFrame
        bounds = gdf.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        
        # Create map
        m = self.create_map(center=center)
        
        # Add raster layer
        try:
            m.add_raster(
                raster_path,
                layer_name=Path(raster_path).stem,
                crs="EPSG:4326"
            )
            logger.info(f"Added raster layer: {raster_path}")
        except Exception as e:
            logger.warning(f"Failed to add raster layer: {e}")
        
        # Extract coordinates for markers
        gdf["longitude"] = gdf.geometry.x
        gdf["latitude"] = gdf.geometry.y
        
        # Add detection markers
        m.add_circle_markers_from_xy(
            gdf,
            x="longitude",
            y="latitude",
            radius=marker_radius,
            color=marker_color,
            fill_color=marker_color,
            layer_name=layer_name,
            stroke=False,
            fill_opacity=0.5
        )
        
        logger.info(f"Added {len(gdf)} detection markers")
        
        # Add bounding box if provided
        if bbox_polygon is not None:
            bbox_gdf = gpd.GeoDataFrame(
                {"geometry": [bbox_polygon]},
                crs="EPSG:4326"
            )
            
            m.add_gdf(
                bbox_gdf,
                layer_name="Bounding Box",
                style={"fillOpacity": 0.1, "color": "#FF0000"}
            )
            
            logger.info("Added bounding box polygon")
        
        # Add legend
        legend_dict = {
            layer_name: marker_color
        }
        
        if bbox_polygon is not None:
            legend_dict["Bounding Box"] = "#FF0000"
        
        m.add_legend(title="Detection Results", legend_dict=legend_dict)
        
        # Add layer control
        m.add_layer_control()
        
        # Save to HTML if requested
        if output_html:
            output_path = Path(output_html)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            m.to_html(str(output_path))
            logger.info(f"Saved map to {output_path}")
        
        return m
    
    def visualize_clusters(
        self,
        raster_path: str,
        gdf: gpd.GeoDataFrame,
        cluster_column: str = "cluster",
        output_html: Optional[str] = None
    ) -> leafmap.Map:
        """
        Visualize clustered detections with different colors per cluster.
        
        Args:
            raster_path: Path to the raster image
            gdf: GeoDataFrame with cluster labels
            cluster_column: Name of the cluster column
            output_html: Optional path to save HTML map
            
        Returns:
            Leafmap Map object
        """
        # Ensure GeoDataFrame is in EPSG:4326
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        
        # Calculate map center
        bounds = gdf.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
        
        # Create map
        m = self.create_map(center=center)
        
        # Add raster layer
        try:
            m.add_raster(
                raster_path,
                layer_name=Path(raster_path).stem,
                crs="EPSG:4326"
            )
        except Exception as e:
            logger.warning(f"Failed to add raster layer: {e}")
        
        # Get unique clusters (excluding noise)
        clusters = sorted(gdf[gdf[cluster_column] != -1][cluster_column].unique())
        
        # Generate colors for each cluster
        colors = self._generate_colors(len(clusters))
        
        # Add markers for each cluster
        for cluster_id, color in zip(clusters, colors):
            cluster_gdf = gdf[gdf[cluster_column] == cluster_id].copy()
            cluster_gdf["longitude"] = cluster_gdf.geometry.x
            cluster_gdf["latitude"] = cluster_gdf.geometry.y
            
            m.add_circle_markers_from_xy(
                cluster_gdf,
                x="longitude",
                y="latitude",
                radius=3,
                color=color,
                fill_color=color,
                layer_name=f"Qanat {cluster_id}",
                stroke=False,
                fill_opacity=0.6
            )
        
        # Add noise points if they exist
        noise_gdf = gdf[gdf[cluster_column] == -1]
        if len(noise_gdf) > 0:
            noise_gdf = noise_gdf.copy()
            noise_gdf["longitude"] = noise_gdf.geometry.x
            noise_gdf["latitude"] = noise_gdf.geometry.y
            
            m.add_circle_markers_from_xy(
                noise_gdf,
                x="longitude",
                y="latitude",
                radius=2,
                color="#888888",
                fill_color="#888888",
                layer_name="Noise",
                stroke=False,
                fill_opacity=0.3
            )
        
        # Add legend
        legend_dict = {f"Qanat {i}": color for i, color in zip(clusters, colors)}
        if len(noise_gdf) > 0:
            legend_dict["Noise"] = "#888888"
        
        m.add_legend(title="Qanat Clusters", legend_dict=legend_dict)
        
        # Add layer control
        m.add_layer_control()
        
        # Save to HTML if requested
        if output_html:
            output_path = Path(output_html)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            m.to_html(str(output_path))
            logger.info(f"Saved clustered map to {output_path}")
        
        logger.info(f"Visualized {len(clusters)} qanat clusters with {len(gdf)} total shafts")
        
        return m
    
    def _generate_colors(self, n: int) -> List[str]:
        """
        Generate n distinct colors for visualization.
        
        Args:
            n: Number of colors to generate
            
        Returns:
            List of color hex strings
        """
        import colorsys
        
        colors = []
        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )
            colors.append(hex_color)
        
        return colors
    
    def create_comparison_map(
        self,
        raster_path: str,
        gdf_list: List[gpd.GeoDataFrame],
        labels: List[str],
        output_html: Optional[str] = None
    ) -> leafmap.Map:
        """
        Create a map comparing multiple detection results.
        
        Args:
            raster_path: Path to the raster image
            gdf_list: List of GeoDataFrames to compare
            labels: Labels for each GeoDataFrame
            output_html: Optional path to save HTML map
            
        Returns:
            Leafmap Map object
        """
        if len(gdf_list) != len(labels):
            raise ValueError("Number of GeoDataFrames must match number of labels")
        
        # Calculate combined bounds
        all_bounds = [gdf.to_crs("EPSG:4326").total_bounds for gdf in gdf_list]
        min_lon = min(b[0] for b in all_bounds)
        min_lat = min(b[1] for b in all_bounds)
        max_lon = max(b[2] for b in all_bounds)
        max_lat = max(b[3] for b in all_bounds)
        
        center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
        
        # Create map
        m = self.create_map(center=center)
        
        # Add raster layer
        try:
            m.add_raster(
                raster_path,
                layer_name=Path(raster_path).stem,
                crs="EPSG:4326"
            )
        except Exception as e:
            logger.warning(f"Failed to add raster layer: {e}")
        
        # Generate colors
        colors = self._generate_colors(len(gdf_list))
        
        # Add each GeoDataFrame
        legend_dict = {}
        
        for gdf, label, color in zip(gdf_list, labels, colors):
            gdf = gdf.to_crs("EPSG:4326").copy()
            gdf["longitude"] = gdf.geometry.x
            gdf["latitude"] = gdf.geometry.y
            
            m.add_circle_markers_from_xy(
                gdf,
                x="longitude",
                y="latitude",
                radius=3,
                color=color,
                fill_color=color,
                layer_name=label,
                stroke=False,
                fill_opacity=0.5
            )
            
            legend_dict[label] = color
        
        # Add legend and layer control
        m.add_legend(title="Comparison", legend_dict=legend_dict)
        m.add_layer_control()
        
        # Save to HTML if requested
        if output_html:
            output_path = Path(output_html)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            m.to_html(str(output_path))
            logger.info(f"Saved comparison map to {output_path}")
        
        return m
