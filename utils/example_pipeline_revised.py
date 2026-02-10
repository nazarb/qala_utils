"""
REVISED Example pipeline for qanat detection from satellite imagery.

This script demonstrates the CORRECT workflow matching the original notebook:
1. Load or download RGB imagery
2. Convert to grayscale 8-bit
3. Reproject to EPSG:4326
4. Clip to regular size (8192×8192)
5. Generate tiles with georeference preservation
6. Run YOLO detection
7. Merge and postprocess results using clipped image geotransform
8. Cluster into qanat systems
9. Visualize results
"""

import os
import logging
from pathlib import Path
from osgeo import gdal

# Import all utilities
from qala_processor import (
    ImageDownloader,
    ImageLoader,
    ImagePreprocessor,
    TileGenerator,
    YOLODetector,
    DetectionPostprocessor,
    QalaPipeline,
    GeospatialUtils,
    ResultVisualizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_qanat_detection_pipeline(
    image_source: str,
    model_path: str,
    output_dir: str = "results",
    is_url: bool = False,
    bbox: tuple = None,
    tile_size: int = 1024,
    overlap: int = 0,  # Original notebook doesn't use overlap
    conf_threshold: float = 0.1,  # Match notebook default
    regular_size: int = 8192  # Must be multiple of tile_size
):
    """
    Run the complete qanat detection pipeline with proper georeferencing.
    
    Args:
        image_source: Path to local TIF file or URL/bbox for download
        model_path: Path to YOLO model weights
        output_dir: Directory for outputs
        is_url: Whether image_source is a URL or bbox coordinates
        bbox: Bounding box for download (minx, miny, maxx, maxy) if applicable
        tile_size: Size of tiles for processing (1024 in original)
        overlap: Overlap between tiles in pixels (0 in original, can be >0)
        conf_threshold: Confidence threshold for detections
        regular_size: Size to clip image to (must be multiple of tile_size)
    """
    
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = output_dir / "temp"
    tiles_dir = output_dir / "tiles"
    detections_dir = output_dir / "detections"
    
    temp_dir.mkdir(exist_ok=True)
    tiles_dir.mkdir(exist_ok=True)
    detections_dir.mkdir(exist_ok=True)
    
    # ==========================================
    # STEP 1: Load or Download Image (RGB)
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading/Downloading RGB Image")
    logger.info("=" * 60)
    
    if is_url or bbox:
        downloader = ImageDownloader(output_dir=str(temp_dir))
        
        if bbox:
            image_path_rgb = downloader.download_from_bbox(
                bbox=bbox,
                filename="satellite_image",
                zoom=18,
                source="Satellite"
            )
        else:
            image_path_rgb = downloader.download_from_url(
                url=image_source,
                filename="satellite_image.tif"
            )
    else:
        image_path_rgb = image_source
    
    # Validate image
    loader = ImageLoader()
    if not loader.validate_image(image_path_rgb):
        raise ValueError(f"Invalid image file: {image_path_rgb}")
    
    logger.info(f"RGB image loaded: {image_path_rgb}")
    
    # ==========================================
    # STEP 2: Convert to Grayscale 8-bit
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 2: Converting to Grayscale 8-bit")
    logger.info("=" * 60)
    
    preprocessor = ImagePreprocessor(
        target_width=regular_size,
        target_height=regular_size,
        convert_to_grayscale=True
    )
    
    grayscale_path = temp_dir / "image_grayscale.tif"
    preprocessor.convert_to_grayscale_8bit(image_path_rgb, str(grayscale_path))
    
    # ==========================================
    # STEP 3: Reproject to EPSG:4326
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 3: Reprojecting to EPSG:4326")
    logger.info("=" * 60)
    
    reprojected_path = temp_dir / "image_4326.tif"
    preprocessor.reproject_to_4326(str(grayscale_path), str(reprojected_path))
    
    # ==========================================
    # STEP 4: Clip to Regular Size
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 4: Clipping to Regular Size")
    logger.info("=" * 60)
    
    clipped_path = temp_dir / "image_4326_clipped.tif"
    preprocessor.clip_to_regular_size(
        str(reprojected_path),
        str(clipped_path),
        regular_width=regular_size,
        regular_height=regular_size
    )
    
    # THIS IS THE REFERENCE IMAGE - its geotransform is used for all coordinate mapping
    reference_path = clipped_path
    
    # Get reference geotransform
    ref_ds = gdal.Open(str(reference_path))
    reference_geotransform = ref_ds.GetGeoTransform()
    reference_projection = ref_ds.GetProjection()
    ref_ds = None
    
    logger.info(f"Reference image (for coordinates): {reference_path}")
    logger.info(f"Reference geotransform: {reference_geotransform}")
    
    # ==========================================
    # STEP 5: Generate Tiles (Georeferenced)
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 5: Generating Georeferenced Tiles")
    logger.info("=" * 60)
    
    tile_generator = TileGenerator(
        tile_size=tile_size,
        overlap=overlap,
        min_tile_coverage=0.5
    )
    
    # Generate tiles from GeoTIFF (preserves georeference)
    tile_info = tile_generator.generate_tiles_from_geotiff(
        input_path=str(clipped_path),
        output_dir=str(tiles_dir),
        basename="tile"
    )
    
    logger.info(f"Generated {len(tile_info)} tiles")
    logger.info(f"Tile size: {tile_size}x{tile_size}, Overlap: {overlap}px")
    
    # ==========================================
    # STEP 6: Run YOLO Detection on Tiles
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 6: Running YOLO Detection")
    logger.info("=" * 60)
    
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=0.45
    )
    
    logger.info(f"Model info: {detector.get_model_info()}")
    
    # Run detection on all tiles
    tile_detections = detector.detect_tiles(
        tile_info=tile_info,
        batch_size=16,
        save_results=True,
        output_dir=str(detections_dir)
    )
    
    # Count total detections
    total_detections = sum(d['num_detections'] for d in tile_detections)
    logger.info(f"Total detections across all tiles: {total_detections}")
    
    # ==========================================
    # STEP 7: Merge Tile Detections
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 7: Merging Tile Detections")
    logger.info("=" * 60)
    
    postprocessor = DetectionPostprocessor(iou_threshold=0.5)
    
    # Merge detections - coordinates will be in clipped image pixel space
    merged_detections = postprocessor.merge_tile_detections(
        tile_detections,
        image_shape=(regular_size, regular_size)  # Shape of clipped image
    )
    
    logger.info(f"Merged detections: {merged_detections['num_detections']}")
    
    if merged_detections['num_detections'] == 0:
        logger.warning("No detections found!")
        return None
    
    # Convert to centroids (still in clipped image pixel space)
    centroids = postprocessor.boxes_to_centroids(merged_detections['boxes'])
    
    # ==========================================
    # STEP 8: Cluster into Qanat Systems
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 8: Clustering into Qanat Systems")
    logger.info("=" * 60)
    
    clusterer = QalaPipeline(
        eps=100.0,
        min_samples=3,
        min_shaft_distance=10.0,
        max_shaft_distance=300.0
    )
    
    # Cluster shafts
    labels = clusterer.cluster_shafts(centroids)
    
    # Calculate cluster statistics
    cluster_stats = clusterer.calculate_shaft_distances(centroids, labels)
    logger.info(f"Cluster statistics: {cluster_stats}")
    
    # Filter by spacing
    filtered_centroids, filtered_labels, filtered_confidences = clusterer.filter_by_shaft_spacing(
        centroids,
        labels,
        merged_detections['confidences']
    )
    
    # Create GeoDataFrame using REFERENCE (clipped) image geotransform
    gdf = clusterer.create_geodataframe(
        filtered_centroids,
        filtered_labels,
        filtered_confidences,
        reference_geotransform=reference_geotransform,  # From clipped EPSG:4326 image!
        crs="EPSG:4326"
    )
    
    # Export results
    output_gpkg = output_dir / "qanat_detections.gpkg"
    clusterer.export_results(gdf, str(output_gpkg), driver="GPKG")
    
    logger.info(f"Exported {len(gdf)} shafts to {output_gpkg}")
    
    # ==========================================
    # STEP 9: Visualize Results
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 9: Visualizing Results")
    logger.info("=" * 60)
    
    visualizer = ResultVisualizer(basemap="Google Satellite")
    
    # Get bounding box from clipped reference image
    bbox_polygon = GeospatialUtils.get_image_bbox(str(reference_path), target_crs="EPSG:4326")
    
    # Create clustered visualization using clipped image
    map_obj = visualizer.visualize_clusters(
        raster_path=str(reference_path),
        gdf=gdf,
        cluster_column="cluster",
        output_html=str(output_dir / "results_map.html")
    )
    
    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"- Reference (clipped) image: {reference_path}")
    logger.info(f"- Tiles: {tiles_dir}")
    logger.info(f"- Detections: {detections_dir}")
    logger.info(f"- GeoPackage: {output_gpkg}")
    logger.info(f"- Map: {output_dir / 'results_map.html'}")
    logger.info("=" * 60)
    logger.info("IMPORTANT: Coordinates are based on the clipped EPSG:4326 image")
    logger.info("=" * 60)
    
    return gdf


if __name__ == "__main__":
    # Example 1: Using a local TIF file
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Processing Local TIF File")
    print("=" * 80)
    
    gdf = run_qanat_detection_pipeline(
        image_source="/path/to/your/satellite_image.tif",
        model_path="/path/to/your/yolo_model.pt",
        output_dir="results/local_image",
        is_url=False,
        tile_size=1024,
        overlap=0,  # Original notebook uses no overlap
        conf_threshold=0.1,  # Match notebook
        regular_size=8192
    )
    
    # Example 2: Downloading from bounding box
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Downloading from Bounding Box")
    print("=" * 80)
    
    gdf = run_qanat_detection_pipeline(
        image_source=None,
        model_path="/path/to/your/yolo_model.pt",
        output_dir="results/downloaded_bbox",
        is_url=False,
        bbox=(89.30, 42.70, 89.40, 42.80),  # (minx, miny, maxx, maxy) in EPSG:4326
        tile_size=1024,
        overlap=0,
        conf_threshold=0.1,
        regular_size=8192
    )
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
