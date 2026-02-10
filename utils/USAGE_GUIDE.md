# Qanat Detection - Usage Guide

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/qanat-detection.git
cd qanat-detection
```

### 2. Install Dependencies

#### Option A: Using pip
```bash
pip install -r requirements.txt
```

#### Option B: Install as package
```bash
pip install -e .
```

#### GDAL Installation (if needed)
GDAL can be tricky. Platform-specific instructions:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev
export CFLAGS=$(gdal-config --cflags)
export LDFLAGS=$(gdal-config --libs)
pip install gdal==$(gdal-config --version)
```

**macOS (with Homebrew):**
```bash
brew install gdal
pip install gdal==$(gdal-config --version)
```

**Windows:**
Use pre-built wheels from https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal

### 3. Verify Installation
```bash
python test_installation.py
```

## Quick Start

### Command Line Usage

#### Process a Local TIF File
```bash
python run_detection.py \
    --image /path/to/satellite_image.tif \
    --model /path/to/yolo_model.pt \
    --output results/
```

#### Download from Bounding Box
```bash
python run_detection.py \
    --bbox 89.30 42.70 89.40 42.80 \
    --model /path/to/yolo_model.pt \
    --output results/
```

#### With Custom Parameters
```bash
python run_detection.py \
    --image satellite.tif \
    --model model.pt \
    --tile-size 1024 \
    --overlap 128 \
    --conf-threshold 0.3 \
    --eps 80 \
    --min-samples 4 \
    --output results/
```

### Python API Usage

#### Basic Example
```python
from utils import (
    ImageLoader, TileGenerator, YOLODetector,
    DetectionPostprocessor, QanatClusterer
)

# 1. Load image
loader = ImageLoader()
image, metadata = loader.load_image("image.tif")

# 2. Generate tiles
tile_gen = TileGenerator(tile_size=1024, overlap=128)
tile_info = tile_gen.generate_tiles(image, "tiles/")

# 3. Run detection
detector = YOLODetector("model.pt", conf_threshold=0.25)
detections = detector.detect_tiles(tile_info)

# 4. Merge and cluster
postprocessor = DetectionPostprocessor()
merged = postprocessor.merge_tile_detections(detections, image.shape[:2])

clusterer = QanatClusterer()
centroids = postprocessor.boxes_to_centroids(merged['boxes'])
labels = clusterer.cluster_shafts(centroids)

# 5. Export results
gdf = clusterer.create_geodataframe(
    centroids, labels, merged['confidences'],
    metadata['geotransform']
)
gdf.to_file("results.gpkg", driver="GPKG")
```

## Configuration Presets

The package includes several configuration presets optimized for different scenarios:

### Default Configuration
Balanced speed and accuracy for general use.
```bash
python run_detection.py --image image.tif --model model.pt --config default
```

### High Resolution
For detailed mapping with high precision.
```bash
python run_detection.py --image image.tif --model model.pt --config high_res
```
- Tile size: 1024px
- Overlap: 256px (25%)
- Higher confidence threshold: 0.30
- Stricter clustering: eps=80, min_samples=5

### Fast Processing
For quick processing of large areas.
```bash
python run_detection.py --image image.tif --model model.pt --config fast
```
- Larger tiles: 2048px
- Smaller overlap: 128px (~6%)
- Lower confidence: 0.20
- Larger batches: 32

### Maximum Quality
Highest accuracy, slowest processing.
```bash
python run_detection.py --image image.tif --model model.pt --config max_quality
```
- Small tiles: 512px
- Large overlap: 256px (50%)
- High confidence: 0.35
- Fine clustering: eps=50, min_samples=4

## Advanced Workflows

### 1. Processing Multiple Images

```python
import glob
from pathlib import Path

image_paths = glob.glob("data/*.tif")

for image_path in image_paths:
    output_dir = Path("results") / Path(image_path).stem
    
    run_qanat_detection_pipeline(
        image_source=image_path,
        model_path="model.pt",
        output_dir=str(output_dir),
        tile_size=1024,
        overlap=128
    )
```

### 2. Custom Clustering Parameters

```python
from utils import QanatClusterer

# For closely-spaced qanats
clusterer = QanatClusterer(
    eps=50.0,              # Smaller search radius
    min_samples=5,         # More shafts required
    min_shaft_distance=15.0,
    max_shaft_distance=150.0
)

# For widely-spaced qanats
clusterer = QanatClusterer(
    eps=200.0,             # Larger search radius
    min_samples=3,         # Fewer shafts required
    min_shaft_distance=20.0,
    max_shaft_distance=500.0
)
```

### 3. Multi-Scale Processing

Process the same image at different resolutions:

```python
from utils import ImagePreprocessor

preprocessor = ImagePreprocessor()

scales = [4096, 8192, 16384]

for scale in scales:
    resized = preprocessor.resize_image(
        image,
        width=scale,
        height=scale
    )
    
    # Process at this scale
    # ...
```

### 4. Region of Interest Processing

```python
from utils import GeospatialUtils

# Define ROI
roi_bbox = (89.30, 42.70, 89.35, 42.75)  # (minx, miny, maxx, maxy)

# Clip raster to ROI
clipped_path = GeospatialUtils.clip_raster_to_bbox(
    input_path="large_image.tif",
    output_path="roi.tif",
    bbox=roi_bbox,
    target_crs="EPSG:4326"
)

# Process clipped image
# ...
```

### 5. Batch Download and Process

```python
from utils import ImageDownloader

downloader = ImageDownloader(output_dir="downloads/")

# Define multiple regions
regions = [
    ("region1", (89.30, 42.70, 89.40, 42.80)),
    ("region2", (89.40, 42.80, 89.50, 42.90)),
    ("region3", (89.50, 42.90, 89.60, 43.00)),
]

for name, bbox in regions:
    # Download
    image_path = downloader.download_from_bbox(
        bbox=bbox,
        filename=name,
        zoom=18
    )
    
    # Process
    run_qanat_detection_pipeline(
        image_source=image_path,
        model_path="model.pt",
        output_dir=f"results/{name}"
    )
```

## Optimizing Tile Parameters

### Choosing Tile Size

The tile size should balance:
1. **Memory constraints**: Larger tiles use more GPU memory
2. **Feature size**: Tiles should be larger than your features
3. **Context**: Larger tiles provide more context for detection

**Recommendations:**
- Small features (<50px): tile_size=512-1024
- Medium features (50-200px): tile_size=1024-2048
- Large features (>200px): tile_size=2048-4096

### Choosing Overlap

Overlap prevents missing features at tile boundaries:

**Formula:** `overlap ≥ max_feature_size + safety_margin`

**Examples:**
- Features up to 100px: overlap=128 (100px + 28px margin)
- Features up to 200px: overlap=256
- Features up to 400px: overlap=512

**Trade-off:** More overlap = more computation but better coverage.

### Memory Requirements

Approximate GPU memory usage per tile:

| Tile Size | YOLO Model | Approximate VRAM |
|-----------|------------|------------------|
| 512px     | YOLOv8n    | ~500 MB         |
| 1024px    | YOLOv8n    | ~1 GB           |
| 1024px    | YOLOv8x    | ~2 GB           |
| 2048px    | YOLOv8n    | ~3 GB           |
| 2048px    | YOLOv8x    | ~6 GB           |

Reduce tile size or batch size if you encounter OOM errors.

## Output Files

The pipeline generates:

```
results/
├── temp/
│   ├── preprocessed.tif         # Preprocessed image
│   └── satellite_image.tif      # Downloaded image (if applicable)
├── tiles/
│   ├── tile_0000.jpg
│   ├── tile_0001.jpg
│   ├── ...
│   └── tile_metadata.json       # Tile positioning info
├── detections/
│   ├── tile_0000_detected.jpg   # Detection visualizations
│   ├── tile_0001_detected.jpg
│   └── ...
├── qanat_detections.gpkg        # Final results (GeoPackage)
├── results_map.html             # Interactive map
└── pipeline.log                 # Processing log
```

## Visualization

### Opening the Interactive Map

The pipeline generates an HTML map file:

```bash
# Open in default browser
open results/results_map.html

# Or use Python
python -m http.server 8000
# Navigate to http://localhost:8000/results/results_map.html
```

### Creating Custom Visualizations

```python
from utils import ResultVisualizer

visualizer = ResultVisualizer(basemap="Google Satellite")

# Simple visualization
map_obj = visualizer.visualize_results(
    raster_path="image.tif",
    gdf=gdf,
    output_html="map.html"
)

# Clustered visualization
map_obj = visualizer.visualize_clusters(
    raster_path="image.tif",
    gdf=gdf,
    cluster_column="cluster",
    output_html="clusters.html"
)

# Comparison visualization
map_obj = visualizer.create_comparison_map(
    raster_path="image.tif",
    gdf_list=[gdf1, gdf2],
    labels=["Method 1", "Method 2"],
    output_html="comparison.html"
)
```

## Troubleshooting

### Common Issues

#### 1. GDAL Import Error
```
ImportError: No module named 'osgeo'
```
**Solution:** Reinstall GDAL with system bindings:
```bash
# Ubuntu/Debian
sudo apt-get install python3-gdal
# Or compile from source with pip
```

#### 2. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce `tile_size`: `--tile-size 512`
- Reduce `batch_size`: `--batch-size 4`
- Use CPU: Add `device='cpu'` to YOLODetector

#### 3. No Detections Found
```
WARNING: No detections found!
```
**Solutions:**
- Lower `conf_threshold`: `--conf-threshold 0.15`
- Check if image has correct bands (RGB)
- Verify model is trained for your imagery type
- Inspect tile visualizations in `detections/`

#### 4. Too Many False Positives
**Solutions:**
- Increase `conf_threshold`: `--conf-threshold 0.35`
- Adjust clustering parameters: stricter `min_samples`
- Use `filter_by_confidence()` in postprocessing

#### 5. Clusters Not Forming
**Solutions:**
- Increase `eps`: Try doubling the current value
- Decrease `min_samples`: Try 2-3 for sparse qanats
- Check shaft spacing with `calculate_shaft_distances()`

## Performance Tips

1. **Use GPU**: ~10-50x faster than CPU for inference
2. **Batch processing**: Larger batches are more efficient
3. **Optimize tile size**: Larger tiles = fewer tiles = less overhead
4. **Reduce overlap**: Less overlap = fewer tiles (but less coverage)
5. **Skip visualization**: Use `--no-viz` for faster processing
6. **Parallel processing**: Process multiple images in parallel

## Best Practices

1. **Always validate results**: Inspect output maps and adjust parameters
2. **Start with default config**: Then fine-tune based on results
3. **Save intermediate outputs**: Keep tiles and detections for debugging
4. **Document parameters**: Record which parameters worked for each region
5. **Use consistent CRS**: Keep all data in the same coordinate system
6. **Backup model weights**: Version your model files
7. **Test on subset first**: Try on small area before full processing

## Getting Help

- Check the README for feature documentation
- Run `test_installation.py` to verify setup
- Examine example scripts for usage patterns
- Enable debug logging: `--log-level DEBUG`
- Review pipeline.log for detailed processing info

## Next Steps

1. Train a custom YOLO model on your own annotated data
2. Experiment with different tile configurations
3. Develop domain-specific postprocessing rules
4. Integrate with GIS workflows
5. Create automated batch processing scripts
