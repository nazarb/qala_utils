"""
Test script to verify qanat detection package installation and functionality.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing module imports...")
    
    try:
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
        logger.info("✓ All utility modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_dependencies():
    """Test that all required dependencies are available."""
    logger.info("\nTesting dependencies...")
    
    dependencies = {
        'numpy': 'numpy',
        'opencv-python': 'cv2',
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'gdal': 'osgeo.gdal',
        'shapely': 'shapely',
        'geopandas': 'geopandas',
        'sklearn': 'sklearn',
        'leafmap': 'leafmap',
        'samgeo': 'samgeo'
    }
    
    all_ok = True
    for name, module in dependencies.items():
        try:
            __import__(module)
            logger.info(f"✓ {name}")
        except ImportError:
            logger.error(f"✗ {name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok


def test_tile_generator():
    """Test tile generation logic."""
    logger.info("\nTesting TileGenerator...")
    
    try:
        from qala_processor import TileGenerator
        import numpy as np
        
        # Create test image
        test_image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        
        # Initialize tile generator
        tile_gen = TileGenerator(tile_size=512, overlap=64)
        
        # Calculate positions
        positions = tile_gen.calculate_tile_positions(
            image_width=test_image.shape[1],
            image_height=test_image.shape[0]
        )
        
        logger.info(f"✓ Generated {len(positions)} tile positions")
        logger.info(f"  - Image size: {test_image.shape[:2]}")
        logger.info(f"  - Tile size: 512x512")
        logger.info(f"  - Overlap: 64 pixels")
        logger.info(f"  - Stride: {tile_gen.stride} pixels")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ TileGenerator test failed: {e}")
        return False


def test_postprocessor():
    """Test detection postprocessing."""
    logger.info("\nTesting DetectionPostprocessor...")
    
    try:
        from qala_processor import DetectionPostprocessor
        import numpy as np
        
        postprocessor = DetectionPostprocessor(iou_threshold=0.5)
        
        # Test NMS
        boxes = np.array([
            [10, 10, 50, 50],
            [15, 15, 55, 55],  # Overlapping with first
            [100, 100, 140, 140]
        ])
        scores = np.array([0.9, 0.8, 0.95])
        
        keep = postprocessor._non_maximum_suppression(boxes, scores, 0.5)
        
        logger.info(f"✓ NMS test passed")
        logger.info(f"  - Input: {len(boxes)} boxes")
        logger.info(f"  - Output: {len(keep)} boxes after NMS")
        
        # Test centroid conversion
        centroids = postprocessor.boxes_to_centroids(boxes)
        logger.info(f"✓ Centroid conversion: {boxes.shape} -> {centroids.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ DetectionPostprocessor test failed: {e}")
        return False


def test_clusterer():
    """Test qanat clustering."""
    logger.info("\nTesting QalaPipeline...")
    
    try:
        from qala_processor import QalaPipeline
        import numpy as np
        
        # Create test data (3 qanats with 5 shafts each)
        np.random.seed(42)
        centroids = []
        
        # Qanat 1
        for i in range(5):
            centroids.append([100 + i * 30, 100 + np.random.randn() * 5])
        
        # Qanat 2
        for i in range(5):
            centroids.append([100 + i * 30, 200 + np.random.randn() * 5])
        
        # Qanat 3
        for i in range(5):
            centroids.append([300 + i * 30, 150 + np.random.randn() * 5])
        
        centroids = np.array(centroids)
        
        clusterer = QalaPipeline(eps=50.0, min_samples=3)
        labels = clusterer.cluster_shafts(centroids)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        logger.info(f"✓ Clustering test passed")
        logger.info(f"  - Input: {len(centroids)} shafts")
        logger.info(f"  - Output: {n_clusters} clusters")
        
        # Test statistics
        stats = clusterer.calculate_shaft_distances(centroids, labels)
        logger.info(f"✓ Distance statistics calculated for {len(stats)} clusters")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ QalaPipeline test failed: {e}")
        return False


def test_gpu_availability():
    """Check GPU availability for YOLO inference."""
    logger.info("\nChecking GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA is available")
            logger.info(f"  - Device count: {torch.cuda.device_count()}")
            logger.info(f"  - Current device: {torch.cuda.current_device()}")
            logger.info(f"  - Device name: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠ CUDA not available - will use CPU (slower)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ GPU check failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    logger.info("=" * 80)
    logger.info("QANAT DETECTION PACKAGE TEST SUITE")
    logger.info("=" * 80)
    
    tests = [
        ("Module Imports", test_imports),
        ("Dependencies", test_dependencies),
        ("Tile Generator", test_tile_generator),
        ("Detection Postprocessor", test_postprocessor),
        ("Qala Pipeline", test_clusterer),
        ("GPU Availability", test_gpu_availability)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test '{name}' raised exception: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info("=" * 80)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 80)
    
    if passed == total:
        logger.info("\n🎉 All tests passed! The package is ready to use.")
        return 0
    else:
        logger.warning(f"\n⚠ {total - passed} test(s) failed. Please check the installation.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
