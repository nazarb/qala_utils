"""
Configuration file for qanat detection pipeline.

Modify these parameters to customize the detection workflow.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Model path
YOLO_MODEL_PATH = MODELS_DIR / "yolo_qanat_best.pt"

# =============================================================================
# IMAGE ACQUISITION
# =============================================================================

# For downloading imagery
DOWNLOAD_CONFIG = {
    'zoom': 18,                    # Zoom level for tile download (higher = better resolution)
    'source': 'Satellite',         # Tile source ('Satellite', 'OpenStreetMap', etc.)
    'overwrite': False             # Whether to overwrite existing downloads
}

# =============================================================================
# PREPROCESSING
# =============================================================================

PREPROCESSING_CONFIG = {
    # Target dimensions for resizing
    'target_width': 8192,
    'target_height': 8192,
    
    # Interpolation method
    'interpolation': 'linear',     # 'linear', 'cubic', 'nearest'
}

# =============================================================================
# TILING
# =============================================================================

TILING_CONFIG = {
    # Tile dimensions
    'tile_size': 1024,             # Size of each tile (width and height in pixels)
    
    # Overlap between adjacent tiles
    'overlap': 128,                # Overlap in pixels (recommended: 10-25% of tile_size)
    
    # Minimum coverage for edge tiles
    'min_tile_coverage': 0.5,      # Minimum fraction of tile that must contain data
    
    # Save tile metadata
    'save_metadata': True
}

# =============================================================================
# YOLO DETECTION
# =============================================================================

YOLO_CONFIG = {
    # Model parameters
    'conf_threshold': 0.25,        # Confidence threshold (0.1-0.5)
    'iou_threshold': 0.45,         # IOU threshold for NMS (0.3-0.7)
    
    # Device
    'device': None,                # 'cuda', 'cpu', or None for auto-detection
    
    # Batch processing
    'batch_size': 16,              # Number of images to process at once
    
    # Save detection visualizations
    'save_visualizations': True
}

# =============================================================================
# POSTPROCESSING
# =============================================================================

POSTPROCESSING_CONFIG = {
    # NMS for merging overlapping detections
    'iou_threshold': 0.5,          # IOU threshold for merging (0.3-0.7)
    
    # Minimum confidence for final detections
    'min_confidence': 0.20,        # Filter out low-confidence detections
}

# =============================================================================
# CLUSTERING (QANAT-SPECIFIC)
# =============================================================================

CLUSTERING_CONFIG = {
    # DBSCAN parameters
    'eps': 100.0,                  # Maximum distance between shafts (in pixels or meters)
    'min_samples': 3,              # Minimum shafts to form a qanat
    
    # Shaft spacing constraints
    'min_shaft_distance': 10.0,    # Minimum distance between consecutive shafts
    'max_shaft_distance': 300.0,   # Maximum distance between consecutive shafts
    
    # Filtering
    'filter_by_spacing': True,     # Apply spacing filters
    'filter_noise': True,          # Remove noise points (cluster label = -1)
}

# =============================================================================
# GEOSPATIAL
# =============================================================================

GEOSPATIAL_CONFIG = {
    # Coordinate reference system
    'target_crs': 'EPSG:4326',     # Output CRS (WGS84)
    
    # Clipping
    'clip_to_bbox': False,         # Whether to clip results to a bounding box
    
    # Output format
    'output_driver': 'GPKG',       # 'GPKG', 'GeoJSON', 'ESRI Shapefile'
}

# =============================================================================
# VISUALIZATION
# =============================================================================

VISUALIZATION_CONFIG = {
    # Base map
    'basemap': 'Google Satellite',  # Base map style
    
    # Map settings
    'default_zoom': 12,
    'map_height': '800px',
    
    # Marker styling
    'marker_radius': 3,
    'marker_color': '#FF0DFF',
    'marker_opacity': 0.5,
    
    # Cluster colors (automatically generated if not specified)
    'cluster_colors': None,
    
    # Save outputs
    'save_html': True,
    'save_image': False
}

# =============================================================================
# LOGGING
# =============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',               # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_to_file': True,
    'log_file': RESULTS_DIR / 'pipeline.log'
}

# =============================================================================
# EXAMPLE CONFIGURATIONS FOR DIFFERENT SCENARIOS
# =============================================================================

# High-resolution detailed mapping
HIGH_RES_CONFIG = {
    'tile_size': 1024,
    'overlap': 256,                # 25% overlap
    'conf_threshold': 0.30,
    'eps': 80.0,
    'min_samples': 5
}

# Fast processing for large areas
FAST_CONFIG = {
    'tile_size': 2048,
    'overlap': 128,                # ~6% overlap
    'conf_threshold': 0.20,
    'batch_size': 32,
    'eps': 150.0,
    'min_samples': 3
}

# Maximum quality (slow)
MAX_QUALITY_CONFIG = {
    'tile_size': 512,
    'overlap': 256,                # 50% overlap
    'conf_threshold': 0.35,
    'batch_size': 8,
    'eps': 50.0,
    'min_samples': 4,
    'min_shaft_distance': 15.0
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_config_for_scenario(scenario: str = 'default'):
    """
    Get configuration for a specific scenario.
    
    Args:
        scenario: 'default', 'high_res', 'fast', or 'max_quality'
    
    Returns:
        Dictionary with configuration parameters
    """
    configs = {
        'default': {
            **TILING_CONFIG,
            **YOLO_CONFIG,
            **CLUSTERING_CONFIG
        },
        'high_res': HIGH_RES_CONFIG,
        'fast': FAST_CONFIG,
        'max_quality': MAX_QUALITY_CONFIG
    }
    
    return configs.get(scenario, configs['default'])


def create_directories():
    """Create all necessary directories."""
    directories = [
        DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        RESULTS_DIR / 'temp',
        RESULTS_DIR / 'tiles',
        RESULTS_DIR / 'detections'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Create directories when config is run directly
    create_directories()
    print("Configuration loaded and directories created.")
    print(f"Base directory: {BASE_DIR}")
    print(f"Results will be saved to: {RESULTS_DIR}")
