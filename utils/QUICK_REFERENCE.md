# Qanat Detection - Quick Reference

## Installation
```bash
pip install -r requirements.txt
python test_installation.py
```

## Basic Usage

### Command Line
```bash
# Local file
python run_detection.py -i image.tif -m model.pt -o results/

# Download from bbox
python run_detection.py -b 89.3 42.7 89.4 42.8 -m model.pt

# Custom parameters
python run_detection.py -i image.tif -m model.pt \
    --tile-size 1024 --overlap 128 --conf-threshold 0.25
```

### Python API
```python
from utils import *

# Load
image, meta = ImageLoader().load_image("image.tif")

# Tile (with overlap!)
tiles = TileGenerator(1024, overlap=128).generate_tiles(image, "tiles/")

# Detect
detections = YOLODetector("model.pt").detect_tiles(tiles)

# Merge
merged = DetectionPostprocessor().merge_tile_detections(detections, image.shape[:2])

# Cluster
labels = QanatClusterer(eps=100).cluster_shafts(centroids)

# Export
gdf = clusterer.create_geodataframe(centroids, labels, confidences, meta['geotransform'])
gdf.to_file("results.gpkg")
```

## Key Parameters

### Tiling
| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| tile_size | 1024 | Tile dimensions (px) |
| overlap | 128 | Overlap between tiles (px) |
| stride | auto | tile_size - overlap |

### Detection
| Parameter | Range | Description |
|-----------|-------|-------------|
| conf_threshold | 0.15-0.40 | Min confidence (lower=more detections) |
| iou_threshold | 0.30-0.70 | NMS threshold (lower=fewer duplicates) |
| batch_size | 8-32 | Images per batch |

### Clustering
| Parameter | Range | Description |
|-----------|-------|-------------|
| eps | 50-200 | Max distance between shafts (px) |
| min_samples | 2-5 | Min shafts per qanat |
| min_shaft_distance | 5-20 | Min spacing (px) |
| max_shaft_distance | 100-500 | Max spacing (px) |

## Configuration Presets

```bash
--config default      # Balanced
--config high_res     # Detailed, slower
--config fast         # Quick, less accurate
--config max_quality  # Best quality, slowest
```

## Module Quick Reference

```python
# Image I/O
from utils import ImageDownloader, ImageLoader
downloader = ImageDownloader("output/")
image, meta = ImageLoader().load_image("image.tif")

# Preprocessing
from utils import ImagePreprocessor, TileGenerator
preprocessor = ImagePreprocessor(8192, 8192)
tile_gen = TileGenerator(1024, overlap=128)

# Detection
from utils import YOLODetector
detector = YOLODetector("model.pt", conf_threshold=0.25)
detections = detector.detect_tiles(tiles, batch_size=16)

# Postprocessing
from utils import DetectionPostprocessor, QanatClusterer
postprocessor = DetectionPostprocessor(iou_threshold=0.5)
clusterer = QanatClusterer(eps=100, min_samples=3)

# Geospatial
from utils import GeospatialUtils
bbox = GeospatialUtils.get_image_bbox("image.tif")
coords = GeospatialUtils.pixel_to_geo_coords(pixels, geotransform)

# Visualization
from utils import ResultVisualizer
visualizer = ResultVisualizer()
map_obj = visualizer.visualize_clusters("image.tif", gdf)
```

## Tiling Cheat Sheet

### Choosing Parameters
```
Small features (<50px):   tile=512,  overlap=128-256
Medium features (50-200): tile=1024, overlap=128-256
Large features (>200px):  tile=2048, overlap=256-512
```

### Overlap Rules
```
Minimum:   overlap >= max_feature_size
Safe:      overlap = max_feature_size × 1.5
Maximum:   overlap = tile_size × 0.5
```

### Memory Estimation
```
Tile 512:  ~500 MB VRAM
Tile 1024: ~1 GB VRAM
Tile 2048: ~3 GB VRAM
Tile 4096: ~8 GB VRAM
```

## Common Workflows

### Workflow 1: Single Image
```bash
python run_detection.py -i image.tif -m model.pt
```

### Workflow 2: Multiple Images
```python
import glob
for path in glob.glob("data/*.tif"):
    run_qanat_detection_pipeline(path, "model.pt", f"results/{Path(path).stem}")
```

### Workflow 3: Download & Process
```bash
python run_detection.py -b 89.3 42.7 89.4 42.8 -m model.pt --zoom 18
```

### Workflow 4: Custom Processing
```python
# Your custom parameters
tile_gen = TileGenerator(tile_size=2048, overlap=256)
detector = YOLODetector("model.pt", conf_threshold=0.30)
clusterer = QanatClusterer(eps=80, min_samples=5)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce tile_size or batch_size |
| No detections | Lower conf_threshold |
| Too many false positives | Raise conf_threshold |
| No clusters | Increase eps or decrease min_samples |
| GDAL error | `sudo apt-get install python3-gdal` |

## Output Files

```
results/
├── tiles/                    # Tile images + metadata
├── detections/              # Detection visualizations
├── qanat_detections.gpkg    # Final results
├── results_map.html         # Interactive map
└── pipeline.log             # Processing log
```

## Quick Tips

1. **Start with defaults**, adjust based on results
2. **Use GPU** for 10-50× speedup
3. **Larger overlap** = better coverage, slower processing
4. **Test on small area** before full processing
5. **Check visualizations** in detections/ folder
6. **Inspect tile_metadata.json** to understand tiling
7. **Use --log-level DEBUG** for detailed output

## Performance Optimization

```python
# Fast processing
run_detection(image, model, tile_size=2048, overlap=64, batch_size=32)

# High quality
run_detection(image, model, tile_size=512, overlap=256, batch_size=8)

# Balanced
run_detection(image, model, tile_size=1024, overlap=128, batch_size=16)
```

## Getting Help

- README.md - Feature documentation
- USAGE_GUIDE.md - Detailed usage examples
- PROJECT_SUMMARY.md - Architecture overview
- test_installation.py - Verify setup
- --help - Command line help

---

**Created by:** Archaeological ML Pipeline  
**Version:** 1.0.0  
**Last Updated:** 2025
