# Qanat Detection Pipeline

A modular Python toolkit for automated qanat detection from satellite imagery using YOLO object detection with tiled processing and overlap handling.

## Features

- **Flexible Image Input**: Support for both local TIF files and downloadable satellite imagery
- **Tiled Processing with Overlap**: Process large images efficiently with configurable tile size and overlap
- **YOLO Integration**: State-of-the-art object detection for qanat shaft identification
- **Smart Postprocessing**: Merge overlapping detections with NMS and cluster into qanat systems
- **Geospatial Support**: Full georeferencing support with coordinate transformations
- **Interactive Visualization**: Create interactive maps with Leafmap for result exploration

## Installation

```bash
# Clone or download the repository
cd qanat_detection

# Install dependencies
pip install -r requirements.txt
```

**Note**: GDAL installation can be tricky. On Ubuntu/Debian:
```bash
sudo apt-get install gdal-bin libgdal-dev
pip install gdal==$(gdal-config --version)
```

## Project Structure

```
qanat_detection/
├── qala_processor/
│   ├── __init__.py              # Package initialization
│   ├── image_io.py              # Image loading and downloading
│   ├── preprocessing.py         # Tiling and preprocessing
│   ├── yolo_inference.py        # YOLO detection
│   ├── postprocessing.py        # Detection merging and clustering
│   ├── geospatial.py            # Coordinate transformations
│   └── visualization.py         # Interactive mapping
├── example_pipeline.py          # Complete example workflow
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Quick Start

### Example 1: Process a Local TIF File

```python
from qala_processor import (
    ImageLoader, ImagePreprocessor, TileGenerator,
    YOLODetector, DetectionPostprocessor, QalaPipeline,
    GeospatialUtils, ResultVisualizer
)

# 1. Load image
loader = ImageLoader()
image, metadata = loader.load_image("path/to/image.tif")

# 2. Preprocess
preprocessor = ImagePreprocessor(target_width=8192, target_height=8192)
resized = preprocessor.resize_image(image)
prepared = preprocessor.prepare_for_yolo(resized)

# 3. Generate tiles with overlap
tile_gen = TileGenerator(tile_size=1024, overlap=128)
tile_info = tile_gen.generate_tiles(prepared, "tiles/", basename="tile")

# 4. Run detection
detector = YOLODetector("path/to/model.pt", conf_threshold=0.25)
detections = detector.detect_tiles(tile_info, batch_size=16)

# 5. Merge and cluster
postprocessor = DetectionPostprocessor(iou_threshold=0.5)
merged = postprocessor.merge_tile_detections(detections, prepared.shape[:2])

clusterer = QalaPipeline(eps=100.0, min_samples=3)
centroids = postprocessor.boxes_to_centroids(merged['boxes'])
labels = clusterer.cluster_shafts(centroids)

# 6. Create GeoDataFrame and export
gdf = clusterer.create_geodataframe(
    centroids, labels, merged['confidences'],
    metadata['geotransform'], crs="EPSG:4326"
)
clusterer.export_results(gdf, "results.gpkg")

# 7. Visualize
visualizer = ResultVisualizer()
map_obj = visualizer.visualize_clusters(
    "path/to/image.tif", gdf, output_html="map.html"
)
```

### Example 2: Download and Process Satellite Imagery

```python
from qala_processor import ImageDownloader

# Download imagery for a bounding box
downloader = ImageDownloader(output_dir="downloads/")
image_path = downloader.download_from_bbox(
    bbox=(89.30, 42.70, 89.40, 42.80),  # (minx, miny, maxx, maxy)
    filename="satellite_image",
    zoom=18,
    source="Satellite"
)

# Then process as in Example 1
# ... (same as above)
```

### Complete Pipeline Script

The `example_pipeline.py` script demonstrates the complete workflow:

```python
from example_pipeline import run_qanat_detection_pipeline

# Process local file
gdf = run_qanat_detection_pipeline(
    image_source="/path/to/image.tif",
    model_path="/path/to/model.pt",
    output_dir="results/",
    tile_size=1024,
    overlap=128,
    conf_threshold=0.25
)

# Or download from bbox
gdf = run_qanat_detection_pipeline(
    image_source=None,
    model_path="/path/to/model.pt",
    output_dir="results/",
    bbox=(89.30, 42.70, 89.40, 42.80),
    tile_size=1024,
    overlap=128,
    conf_threshold=0.25
)
```

## Module Documentation

### `image_io.py` - Image I/O

**ImageDownloader**: Download satellite imagery
- `download_from_bbox()`: Download from bounding box coordinates
- `download_from_url()`: Download from URL

**ImageLoader**: Load and validate GeoTIFF files
- `load_image()`: Load image with metadata
- `validate_image()`: Check if file is valid
- `get_image_info()`: Get detailed image information

### `preprocessing.py` - Preprocessing

**ImagePreprocessor**: Prepare images for detection
- `resize_image()`: Resize to target dimensions
- `prepare_for_yolo()`: Ensure correct format
- `save_image()`: Save with georeferencing

**TileGenerator**: Generate overlapping tiles
- `tile_size`: Size of each tile (default: 1024)
- `overlap`: Overlap between tiles in pixels (default: 128)
- `stride`: Calculated as `tile_size - overlap`
- `calculate_tile_positions()`: Calculate tile grid
- `generate_tiles()`: Create and save tiles
- `map_detection_to_original()`: Map coordinates back to source image

**Key Feature**: The tile generator automatically handles edge cases and ensures proper coverage with adjustable overlap.

### `yolo_inference.py` - Detection

**YOLODetector**: Run YOLO inference
- `detect_single()`: Process one image
- `detect_batch()`: Process multiple images
- `detect_tiles()`: Process tiles with metadata
- `filter_by_confidence()`: Filter low-confidence detections
- `filter_by_class()`: Filter by object class

### `postprocessing.py` - Postprocessing

**DetectionPostprocessor**: Merge tile detections
- `merge_tile_detections()`: Merge from overlapping tiles using NMS
- `boxes_to_centroids()`: Convert boxes to points
- Non-Maximum Suppression handles duplicate detections in overlap regions

**QalaPipeline**: Cluster shafts into qanat systems
- `cluster_shafts()`: DBSCAN clustering
- `calculate_shaft_distances()`: Distance statistics per cluster
- `filter_by_shaft_spacing()`: Quality filtering
- `create_geodataframe()`: Create spatial data
- `export_results()`: Save to GeoPackage/Shapefile/GeoJSON

### `geospatial.py` - Geospatial Operations

**GeospatialUtils**: Coordinate transformations and clipping
- `get_image_bbox()`: Extract image bounding box
- `clip_raster_to_bbox()`: Clip raster to region
- `reproject_raster()`: Change CRS
- `pixel_to_geo_coords()`: Pixel → Geographic coordinates
- `geo_to_pixel_coords()`: Geographic → Pixel coordinates
- `calculate_ground_sample_distance()`: Get GSD from geotransform

### `visualization.py` - Visualization

**ResultVisualizer**: Interactive mapping
- `create_map()`: Initialize map
- `visualize_results()`: Show detections on map
- `visualize_clusters()`: Color-coded cluster visualization
- `create_comparison_map()`: Compare multiple results

## Tiling with Overlap: How It Works

The tiling system is designed to handle large satellite images efficiently while ensuring no features are missed at tile boundaries.

### Parameters

- **tile_size** (default: 1024): Size of each tile in pixels
- **overlap** (default: 128): Number of pixels overlapping between adjacent tiles
- **stride**: Calculated as `tile_size - overlap` (e.g., 896 for defaults)

### Example

For an 8192×8192 image with tile_size=1024 and overlap=128:
- Stride = 896 pixels between tile starts
- Tiles needed: ⌈(8192 - 128) / 896⌉ = 10 tiles per dimension
- Total tiles: 10 × 10 = 100 tiles
- Each tile overlaps 128 pixels with its neighbors

### Benefits

1. **No missed detections**: Features near tile edges are captured in multiple tiles
2. **Quality assurance**: Detections appearing in multiple tiles boost confidence
3. **NMS deduplication**: Post-processing removes duplicates from overlap regions
4. **Adjustable**: Increase overlap for small features, decrease for speed

### Coordinate Mapping

The system automatically handles coordinate transformations:
```
Tile coordinates → Original image coordinates → Geographic coordinates
```

This ensures all detections are correctly positioned regardless of tiling.

## Configuration Tips

### Tile Size and Overlap

- **Large features** (>200px): Use larger tiles (1024-2048), smaller overlap (64-128)
- **Small features** (<100px): Use smaller tiles (512-1024), larger overlap (128-256)
- **Memory constraints**: Reduce tile_size
- **Speed optimization**: Reduce overlap (but ensure >0 to catch edge features)

### Detection Thresholds

- **conf_threshold** (0.1-0.5): Higher = fewer false positives, more missed detections
- **iou_threshold** (0.3-0.7): Higher = more detections kept, more duplicates
- Start with conf=0.25, iou=0.45 and adjust based on results

### Clustering Parameters

- **eps** (50-200): Maximum distance between shafts in same qanat
- **min_samples** (2-5): Minimum shafts to form a qanat
- **min_shaft_distance** (5-20): Minimum spacing between adjacent shafts
- **max_shaft_distance** (100-500): Maximum spacing for qanat continuity

## Performance Notes

- **GPU highly recommended** for YOLO inference
- Batch processing tiles improves speed significantly
- Tile metadata is saved for reproducibility
- Progress logging helps monitor long-running processes

## Output Files

The pipeline generates:
- **tiles/**: Individual tile images
- **tiles/tile_metadata.json**: Tile positioning information
- **detections/**: Visualizations of detections per tile
- **qanat_detections.gpkg**: GeoPackage with all detections and clusters
- **results_map.html**: Interactive HTML map

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@article{your_article,
  title={Automated Qanat Detection from Satellite Imagery},
  author={Your Name},
  journal={Journal of Archaeological Science},
  year={2024}
}
```

## License

[Specify your license here]

## Contact

[Your contact information]

## Acknowledgments

This toolkit uses:
- **Ultralytics YOLOv8/v9** for object detection
- **GDAL** for geospatial processing
- **Leafmap** for interactive visualization
- **samgeo** for satellite imagery download
