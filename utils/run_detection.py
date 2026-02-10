#!/usr/bin/env python3
"""
Command-line interface for qanat detection pipeline.

Usage examples:
    # Process local file
    python run_detection.py --image path/to/image.tif --model path/to/model.pt
    
    # Download and process
    python run_detection.py --bbox 89.3 42.7 89.4 42.8 --model path/to/model.pt
    
    # Use custom configuration
    python run_detection.py --image image.tif --model model.pt --config high_res
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from example_pipeline import run_qanat_detection_pipeline
from config import (
    get_config_for_scenario,
    LOGGING_CONFIG,
    RESULTS_DIR
)


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Configure logging."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format=LOGGING_CONFIG['format'],
        handlers=handlers
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Qanat Detection Pipeline - Automated detection from satellite imagery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a local TIF file
  %(prog)s --image satellite.tif --model weights.pt --output results/
  
  # Download from bounding box and process
  %(prog)s --bbox 89.3 42.7 89.4 42.8 --model weights.pt
  
  # Use high-resolution configuration
  %(prog)s --image satellite.tif --model weights.pt --config high_res
  
  # Custom tile settings
  %(prog)s --image satellite.tif --model weights.pt --tile-size 2048 --overlap 256
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image', '-i',
        type=str,
        help='Path to input GeoTIFF image'
    )
    input_group.add_argument(
        '--bbox', '-b',
        type=float,
        nargs=4,
        metavar=('MINX', 'MINY', 'MAXX', 'MAXY'),
        help='Bounding box for downloading imagery (minx miny maxx maxy in EPSG:4326)'
    )
    input_group.add_argument(
        '--url', '-u',
        type=str,
        help='URL to download image from'
    )
    
    # Required arguments
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to YOLO model weights (.pt file)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=str(RESULTS_DIR),
        help=f'Output directory (default: {RESULTS_DIR})'
    )
    
    # Configuration presets
    parser.add_argument(
        '--config', '-c',
        type=str,
        choices=['default', 'high_res', 'fast', 'max_quality'],
        default='default',
        help='Configuration preset (default: default)'
    )
    
    # Tiling parameters
    parser.add_argument(
        '--tile-size',
        type=int,
        help='Size of tiles in pixels (overrides config)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        help='Overlap between tiles in pixels (overrides config)'
    )
    
    # Detection parameters
    parser.add_argument(
        '--conf-threshold',
        type=float,
        help='Confidence threshold for detections (0.0-1.0, overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for inference (overrides config)'
    )
    
    # Clustering parameters
    parser.add_argument(
        '--eps',
        type=float,
        help='DBSCAN epsilon parameter (overrides config)'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        help='DBSCAN minimum samples (overrides config)'
    )
    
    # Download options
    parser.add_argument(
        '--zoom',
        type=int,
        default=18,
        help='Zoom level for tile download (default: 18)'
    )
    
    # Other options
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file (optional)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = get_config_for_scenario(args.config)
    logger.info(f"Using configuration preset: {args.config}")
    
    # Override config with command-line arguments
    if args.tile_size:
        config['tile_size'] = args.tile_size
    if args.overlap:
        config['overlap'] = args.overlap
    if args.conf_threshold:
        config['conf_threshold'] = args.conf_threshold
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.eps:
        config['eps'] = args.eps
    if args.min_samples:
        config['min_samples'] = args.min_samples
    
    # Determine input source
    if args.image:
        image_source = args.image
        is_url = False
        bbox = None
        logger.info(f"Processing local file: {args.image}")
    elif args.bbox:
        image_source = None
        is_url = False
        bbox = tuple(args.bbox)
        logger.info(f"Downloading from bounding box: {bbox}")
    else:  # args.url
        image_source = args.url
        is_url = True
        bbox = None
        logger.info(f"Downloading from URL: {args.url}")
    
    # Verify model exists
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    # Run pipeline
    logger.info("=" * 80)
    logger.info("Starting Qanat Detection Pipeline")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Tile size: {config['tile_size']}, Overlap: {config['overlap']}")
    logger.info(f"Confidence threshold: {config.get('conf_threshold', 0.25)}")
    logger.info("=" * 80)
    
    try:
        gdf = run_qanat_detection_pipeline(
            image_source=image_source,
            model_path=args.model,
            output_dir=args.output,
            is_url=is_url,
            bbox=bbox,
            tile_size=config['tile_size'],
            overlap=config['overlap'],
            conf_threshold=config.get('conf_threshold', 0.25)
        )
        
        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Detected {len(gdf)} qanat shafts")
        logger.info(f"Results saved to: {args.output}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"Pipeline failed with error: {e}")
        logger.error("=" * 80)
        
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
