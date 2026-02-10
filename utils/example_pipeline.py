"""
Example pipeline for qanat detection from satellite imagery.

This script demonstrates the complete workflow:
1. Load or download imagery
2. Preprocess and tile with overlap
3. Run YOLO detection
4. Merge and postprocess results
5. Cluster into qanat systems
6. Visualize results
"""

import os
import logging
from pathlib import Path

# Import all utilities
from utils import (
    ImageDownloader,
    ImageLoader,
    ImagePreprocessor,
    TileGenerator,
    YOLODetector,
    DetectionPostprocessor,
    QanatClusterer,
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
    overlap: int = 128,
    conf_threshold: float = 0.25
):
    """
    Run the complete qanat detection pipeline.
    
    Args:
        image_source: Path to local TIF file or URL/bbox for download
        model_path: Path to YOLO model weights
        output_dir: Directory for outputs
        is_url: Whether image_source is a URL or bbox coordinates
        bbox: Bounding box for download (minx, miny, maxx, maxy) if applicable
        tile_size: Size of tiles for processing
        overlap: Overlap between tiles in pixels
        conf_threshold: Confidence threshold for detections
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
    # STEP 1: Load or Download Image
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading/Downloading Image")
    logger.info("=" * 60)
    
    if is_url or bbox:
        # Download image
        downloader = ImageDownloader(output_dir=str(temp_dir))
        
        if bbox:
            image_path = downloader.download_from_bbox(
                bbox=bbox,
                filename="satellite_image",
                zoom=18,
                source="Satellite"
            )
        else:
            image_path = downloader.download_from_url(
                url=image_source,
                filename="satellite_image.tif"
            )
    else:
        # Use local file
        image_path = image_source
    
    # Validate image
    loader = ImageLoader()
    if not loader.validate_image(image_path):
        raise ValueError(f"Invalid image file: {image_path}")
    
    # Load image and metadata
    image, metadata = loader.load_image(image_path)
    logger.info(f"Loaded image: {image.shape}")
    logger.info(f"Image metadata: {metadata}")
    
    # ==========================================
    # STEP 2: Preprocess Image
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing Image")
    logger.info("=" * 60)
    
    preprocessor = ImagePreprocessor(
        target_width=8192,
        target_height=8192
    )
    
    # Resize to standard dimensions
    resized_image = preprocessor.resize_image(image)
    
    # Prepare for YOLO
    prepared_image = preprocessor.prepare_for_yolo(resized_image)
    
    # Save preprocessed image
    preprocessed_path = temp_dir / "preprocessed.tif"
    preprocessor.save_image(
        prepared_image,
        str(preprocessed_path),
        metadata=metadata
    )
    
    logger.info(f"Preprocessed image shape: {prepared_image.shape}")
    
    # ==========================================
    # STEP 3: Generate Tiles with Overlap
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 3: Generating Tiles with Overlap")
    logger.info("=" * 60)
    
    tile_generator = TileGenerator(
        tile_size=tile_size,
        overlap=overlap,
        min_tile_coverage=0.5
    )
    
    # Generate tiles
    tile_info = tile_generator.generate_tiles(
        prepared_image,
        output_dir=str(tiles_dir),
        basename="tile",
        save_metadata=True
    )
    
    logger.info(f"Generated {len(tile_info)} tiles")
    logger.info(f"Tile size: {tile_size}x{tile_size}, Overlap: {overlap}px, Stride: {tile_generator.stride}px")
    
    # ==========================================
    # STEP 4: Run YOLO Detection on Tiles
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 4: Running YOLO Detection")
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
    # STEP 5: Merge Tile Detections
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 5: Merging Tile Detections")
    logger.info("=" * 60)
    
    postprocessor = DetectionPostprocessor(iou_threshold=0.5)
    
    # Merge detections from overlapping tiles
    merged_detections = postprocessor.merge_tile_detections(
        tile_detections,
        image_shape=prepared_image.shape[:2]
    )
    
    logger.info(f"Merged detections: {merged_detections['num_detections']}")
    
    if merged_detections['num_detections'] == 0:
        logger.warning("No detections found!")
        return
    
    # Convert to centroids
    centroids = postprocessor.boxes_to_centroids(merged_detections['boxes'])
    
    # ==========================================
    # STEP 6: Cluster into Qanat Systems
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 6: Clustering into Qanat Systems")
    logger.info("=" * 60)
    
    clusterer = QanatClusterer(
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
    
    # Create GeoDataFrame
    gdf = clusterer.create_geodataframe(
        filtered_centroids,
        filtered_labels,
        filtered_confidences,
        geotransform=metadata['geotransform'],
        crs="EPSG:4326"
    )
    
    # Export results
    output_gpkg = output_dir / "qanat_detections.gpkg"
    clusterer.export_results(gdf, str(output_gpkg), driver="GPKG")
    
    logger.info(f"Exported {len(gdf)} shafts to {output_gpkg}")
    
    # ==========================================
    # STEP 7: Visualize Results
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 7: Visualizing Results")
    logger.info("=" * 60)
    
    visualizer = ResultVisualizer(basemap="Google Satellite")
    
    # Get bounding box
    bbox_polygon = GeospatialUtils.get_image_bbox(image_path, target_crs="EPSG:4326")
    
    # Create clustered visualization
    map_obj = visualizer.visualize_clusters(
        raster_path=image_path,
        gdf=gdf,
        cluster_column="cluster",
        output_html=str(output_dir / "results_map.html")
    )
    
    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"- Tiles: {tiles_dir}")
    logger.info(f"- Detections: {detections_dir}")
    logger.info(f"- GeoPackage: {output_gpkg}")
    logger.info(f"- Map: {output_dir / 'results_map.html'}")
    
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
        overlap=128,
        conf_threshold=0.25
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
        overlap=128,
        conf_threshold=0.25
    )
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
