# Qanat Detection Pipeline - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    QANAT DETECTION PIPELINE                      │
│                         (Version 1.0.0)                          │
└─────────────────────────────────────────────────────────────────┘

INPUT                 PROCESSING                      OUTPUT
─────                 ──────────                      ──────

┌─────────────┐
│ Local TIF   │─────┐
└─────────────┘     │
                    │
┌─────────────┐     │    ┌──────────────────────────────────────┐
│ URL/Bbox    │─────┼───→│      IMAGE ACQUISITION              │
└─────────────┘     │    │  • ImageDownloader                  │
                    │    │  • ImageLoader                      │
┌─────────────┐     │    │  • Validation                       │
│ User Upload │─────┘    └────────────┬─────────────────────────┘
└─────────────┘                       │
                                      ↓
                            ┌──────────────────────────────────────┐
                            │      PREPROCESSING                   │
                            │  • Resize to standard dimensions     │
                            │  • RGB conversion                    │
                            │  • Format validation                 │
                            └────────────┬─────────────────────────┘
                                         ↓
                            ┌──────────────────────────────────────┐
                            │    TILING WITH OVERLAP ⭐            │
                            │  • Calculate tile positions          │
                            │  • Generate overlapping tiles        │
                            │  • Save metadata                     │
                            │                                      │
                            │  Config: tile_size, overlap          │
                            │  Output: N tiles with metadata       │
                            └────────────┬─────────────────────────┘
                                         ↓
                            ┌──────────────────────────────────────┐
                            │      YOLO DETECTION                  │
                            │  • Batch processing                  │
                            │  • GPU acceleration                  │
                            │  • Per-tile detection                │
                            │                                      │
                            │  Input: N tiles                      │
                            │  Output: N detection sets            │
                            └────────────┬─────────────────────────┘
                                         ↓
                            ┌──────────────────────────────────────┐
                            │    POSTPROCESSING                    │
                            │  • Coordinate mapping                │
                            │  • Merge overlapping detections      │
                            │  • Non-Maximum Suppression (NMS)     │
                            │  • Duplicate removal                 │
                            └────────────┬─────────────────────────┘
                                         ↓
                            ┌──────────────────────────────────────┐
                            │      CLUSTERING                      │
                            │  • DBSCAN algorithm                  │
                            │  • Distance filtering                │
                            │  • Qanat identification              │
                            │  • Statistical analysis              │
                            └────────────┬─────────────────────────┘
                                         ↓
                            ┌──────────────────────────────────────┐
                            │    GEOSPATIAL PROCESSING             │
                            │  • Pixel → Geographic coords         │
                            │  • GeoDataFrame creation             │
                            │  • CRS handling                      │
                            └────────────┬─────────────────────────┘
                                         ↓
                     ┌───────────────────┴────────────────────┐
                     ↓                                        ↓
        ┌────────────────────────┐               ┌────────────────────────┐
        │    FILE EXPORT         │               │   VISUALIZATION        │
        │  • GeoPackage          │               │  • Interactive map     │
        │  • GeoJSON             │               │  • Cluster colors      │
        │  • Shapefile           │               │  • HTML export         │
        └────────────────────────┘               └────────────────────────┘
```

## Module Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         qala_processor/ PACKAGE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐    ┌────────────────┐    ┌──────────────┐ │
│  │  image_io.py   │    │preprocessing.py│    │yolo_inference│ │
│  │                │    │                │    │    .py       │ │
│  │ • Download     │───→│ • Preprocess   │───→│ • Detection  │ │
│  │ • Load         │    │ • Tile+Overlap │    │ • Batch      │ │
│  │ • Validate     │    │ • Coordinate   │    │ • Filter     │ │
│  └────────────────┘    └────────────────┘    └──────┬───────┘ │
│                                                      │          │
│                                                      ↓          │
│  ┌────────────────┐    ┌────────────────┐    ┌──────────────┐ │
│  │visualization   │←───│  geospatial    │←───│postprocessing│ │
│  │    .py         │    │     .py        │    │    .py       │ │
│  │ • Maps         │    │ • Transforms   │    │ • Merge      │ │
│  │ • Clusters     │    │ • CRS          │    │ • NMS        │ │
│  │ • Export       │    │ • Clip/Reproject│    │ • Cluster    │ │
│  └────────────────┘    └────────────────┘    └──────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow with Tiling

```
ORIGINAL IMAGE (8192 × 8192)
         │
         │ [PREPROCESSING]
         ↓
RESIZED IMAGE (8192 × 8192 × 3 RGB)
         │
         │ [TILING] tile_size=1024, overlap=128
         ↓
┌────────────────────────────────────────────────────────────┐
│  TILES WITH OVERLAP (10×10 grid = 100 tiles)               │
│                                                             │
│  [T0 ]─128px─[T1 ]─128px─[T2 ]                            │
│   │            │            │                               │
│  128px       128px        128px                            │
│   │            │            │                               │
│  [T10]────────[T11]────────[T12]                           │
│   │            │            │                               │
│  ...          ...          ...                             │
│                                                             │
│  Each tile: 1024×1024 px                                   │
│  Overlap: 128 px on each side                              │
│  Stride: 896 px (1024-128)                                 │
└────────────────────────────────────────────────────────────┘
         │
         │ [YOLO DETECTION] batch_size=16
         ↓
┌────────────────────────────────────────────────────────────┐
│  DETECTIONS PER TILE                                        │
│                                                             │
│  Tile 0:  15 detections  →  boxes, confidences, classes   │
│  Tile 1:  23 detections  →  boxes, confidences, classes   │
│  ...                                                        │
│  Tile 99: 8  detections  →  boxes, confidences, classes   │
│                                                             │
│  Total: ~1200 detections (before merging)                  │
└────────────────────────────────────────────────────────────┘
         │
         │ [COORDINATE MAPPING]
         │ Tile coords → Original image coords
         ↓
┌────────────────────────────────────────────────────────────┐
│  MAPPED DETECTIONS IN ORIGINAL IMAGE SPACE                 │
│                                                             │
│  Detection from Tile 5 at (100, 200):                      │
│    → Original: (100 + tile_x_offset, 200 + tile_y_offset) │
│                                                             │
│  Detections in overlap regions appear MULTIPLE times       │
└────────────────────────────────────────────────────────────┘
         │
         │ [NON-MAXIMUM SUPPRESSION]
         ↓
┌────────────────────────────────────────────────────────────┐
│  MERGED DETECTIONS (duplicates removed)                    │
│                                                             │
│  Before NMS: ~1200 detections                              │
│  After NMS:  ~800 unique detections                        │
│                                                             │
│  Overlap detections validated and merged                   │
└────────────────────────────────────────────────────────────┘
         │
         │ [CLUSTERING] DBSCAN(eps=100, min_samples=3)
         ↓
┌────────────────────────────────────────────────────────────┐
│  QANAT SYSTEMS                                              │
│                                                             │
│  Cluster 0: 45 shafts  (Qanat system 1)                   │
│  Cluster 1: 38 shafts  (Qanat system 2)                   │
│  Cluster 2: 52 shafts  (Qanat system 3)                   │
│  ...                                                        │
│  Noise:     15 shafts  (isolated detections)               │
└────────────────────────────────────────────────────────────┘
         │
         │ [GEOSPATIAL TRANSFORM]
         ↓
┌────────────────────────────────────────────────────────────┐
│  GEOGRAPHIC COORDINATES (EPSG:4326)                        │
│                                                             │
│  Pixel (4096, 4096) → Lat/Lon (42.75, 89.35)             │
│  + Cluster labels                                          │
│  + Confidence scores                                       │
│  + Metadata                                                │
└────────────────────────────────────────────────────────────┘
         │
         ├──→ GeoPackage (.gpkg)
         ├──→ Interactive Map (.html)
         └──→ Statistics (.json)
```

## Tiling Algorithm Detail

```
STEP 1: Calculate Grid
─────────────────────────
Input: image_width=8192, image_height=8192
       tile_size=1024, overlap=128
       
stride = tile_size - overlap = 896

n_cols = ceil((image_width - overlap) / stride) = 10
n_rows = ceil((image_height - overlap) / stride) = 10

Total tiles = n_cols × n_rows = 100


STEP 2: Generate Positions
───────────────────────────
for row in range(10):
    y_start = row × stride = row × 896
    y_end = y_start + tile_size = y_start + 1024
    
    for col in range(10):
        x_start = col × stride = col × 896
        x_end = x_start + tile_size = x_start + 1024
        
        create_tile(x_start, y_start, x_end, y_end)


STEP 3: Handle Boundaries
──────────────────────────
If last tile extends beyond image:
    - Adjust start position to fit
    - Pad with zeros if needed
    
If tile too small (< min_coverage):
    - Extend to meet minimum size
    - Remove duplicates


STEP 4: Save Metadata
─────────────────────
{
  "tile_id": 0,
  "x_start": 0,
  "y_start": 0,
  "x_end": 1024,
  "y_end": 1024,
  "width": 1024,
  "height": 1024
}
```

## NMS (Non-Maximum Suppression) Process

```
INPUT: All detections from all tiles
───────────────────────────────────

Box 1: [100, 100, 200, 200] conf=0.9  (from Tile 5)
Box 2: [105, 105, 205, 205] conf=0.8  (from Tile 6, overlap region)
Box 3: [500, 500, 600, 600] conf=0.95 (from Tile 8)


STEP 1: Calculate IOU
─────────────────────
IOU(Box 1, Box 2) = 0.75  (high overlap - likely same feature)
IOU(Box 1, Box 3) = 0.0   (no overlap - different features)
IOU(Box 2, Box 3) = 0.0


STEP 2: Sort by Confidence
───────────────────────────
Box 3: conf=0.95  ← Keep (highest confidence)
Box 1: conf=0.9   ← Keep (no overlap with Box 3)
Box 2: conf=0.8   ← Remove (overlaps Box 1 with IOU > threshold)


OUTPUT: Merged detections
─────────────────────────
Box 1: [100, 100, 200, 200] conf=0.9
Box 3: [500, 500, 600, 600] conf=0.95

Duplicate removed! 2 detections instead of 3.
```

## Configuration System

```
┌─────────────────────────────────────────────────────────┐
│                   config.py                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  DEFAULT_CONFIG          HIGH_RES_CONFIG                │
│  ├─ tile_size: 1024     ├─ tile_size: 1024             │
│  ├─ overlap: 128        ├─ overlap: 256 (25%)          │
│  ├─ conf: 0.25          ├─ conf: 0.30                   │
│  └─ eps: 100            └─ eps: 80                      │
│                                                          │
│  FAST_CONFIG            MAX_QUALITY_CONFIG              │
│  ├─ tile_size: 2048     ├─ tile_size: 512              │
│  ├─ overlap: 128 (6%)   ├─ overlap: 256 (50%)          │
│  ├─ conf: 0.20          ├─ conf: 0.35                   │
│  └─ eps: 150            └─ eps: 50                      │
│                                                          │
└─────────────────────────────────────────────────────────┘
         │
         ├──→ CLI (run_detection.py)
         ├──→ Python API
         └──→ Custom configs
```

## Performance Characteristics

```
┌──────────────────────────────────────────────────────────┐
│              PROCESSING PIPELINE TIMING                   │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  8192×8192 image, 1024px tiles, 128px overlap           │
│  YOLO v8n model, NVIDIA A40 GPU, batch_size=16          │
│                                                           │
│  Step                    Time        Notes               │
│  ─────────────────────  ──────────  ─────────────────    │
│  Image loading           1-2 sec    I/O bound           │
│  Preprocessing           2-3 sec    CPU                 │
│  Tile generation         5-10 sec   100 tiles           │
│  YOLO detection          30-60 sec  GPU accelerated     │
│  NMS merging             2-5 sec    CPU                 │
│  Clustering              1-2 sec    CPU                 │
│  Geospatial transform    1-2 sec    CPU                 │
│  Visualization           5-10 sec   Map generation      │
│  ─────────────────────  ──────────  ─────────────────    │
│  TOTAL                   ~60-90 sec                      │
│                                                           │
│  Without tiling: 3-5 minutes (memory issues likely)     │
│  Speedup: ~3-4× with proper tiling                      │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

## Class Hierarchy

```
ImageDownloader
├── download_from_bbox()
├── download_from_url()
└── output_dir

ImageLoader
├── load_image()
├── validate_image()
└── get_image_info()

ImagePreprocessor
├── resize_image()
├── prepare_for_yolo()
└── save_image()

TileGenerator ⭐
├── tile_size
├── overlap
├── stride (calculated)
├── calculate_tile_positions()
├── generate_tiles()
└── map_detection_to_original()

YOLODetector
├── model
├── conf_threshold
├── iou_threshold
├── detect_single()
├── detect_batch()
├── detect_tiles()
└── filter_by_confidence()

DetectionPostprocessor
├── iou_threshold
├── merge_tile_detections()
├── _non_maximum_suppression()
└── boxes_to_centroids()

QalaPipeline
├── eps
├── min_samples
├── cluster_shafts()
├── calculate_shaft_distances()
├── filter_by_shaft_spacing()
├── create_geodataframe()
└── export_results()

GeospatialUtils (static methods)
├── get_image_bbox()
├── clip_raster_to_bbox()
├── reproject_raster()
├── pixel_to_geo_coords()
└── geo_to_pixel_coords()

ResultVisualizer
├── basemap
├── create_map()
├── visualize_results()
├── visualize_clusters()
└── create_comparison_map()
```

## Design Principles

1. **Modularity**: Each module has a single responsibility
2. **Reusability**: Functions work independently
3. **Composability**: Modules combine easily
4. **Testability**: Each component can be tested separately
5. **Documentation**: Comprehensive docstrings
6. **Error Handling**: Robust validation and logging
7. **Performance**: GPU acceleration, batching, efficient algorithms
8. **Reproducibility**: Metadata tracking, configuration management

## Key Innovations

1. **Tiled Processing with Overlap** ⭐
   - Handles images of any size
   - Prevents edge detection failures
   - Automatic duplicate removal

2. **Coordinate Mapping**
   - Tile space → Image space → Geographic space
   - Maintains georeferencing throughout

3. **Smart NMS**
   - Merges detections from overlap regions
   - Preserves best detection per feature

4. **Configuration Presets**
   - Ready-to-use configs for different scenarios
   - Easy parameter tuning

5. **Comprehensive Pipeline**
   - End-to-end workflow
   - CLI and Python API
   - Production-ready code
