"""
Image I/O utilities for loading and downloading satellite imagery.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from osgeo import gdal
from samgeo import download_file
try:
    from samgeo import tms_to_geotiff
except ImportError:
    from samgeo.common import tms_to_geotiff

logger = logging.getLogger(__name__)


class ImageDownloader:
    """Handle downloading satellite imagery from various sources."""
    
    def __init__(self, output_dir: str = "temp"):
        """
        Initialize ImageDownloader.
        
        Args:
            output_dir: Directory to save downloaded images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_from_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        filename: str,
        zoom: int = 18,
        source: str = "Satellite",
        overwrite: bool = False
    ) -> str:
        """
        Download satellite imagery for a bounding box.
        
        Args:
            bbox: Bounding box as (minx, miny, maxx, maxy) in EPSG:4326
            filename: Output filename (without extension)
            zoom: Zoom level for tile download
            source: Tile source (e.g., 'Satellite', 'OpenStreetMap')
            overwrite: Whether to overwrite existing files
            
        Returns:
            Path to downloaded image
        """
        output_path = self.output_dir / f"{filename}.tif"
        
        if output_path.exists() and not overwrite:
            logger.info(f"File {output_path} already exists, skipping download")
            return str(output_path)
        
        logger.info(f"Downloading imagery for bbox: {bbox}")
        
        try:
            tms_to_geotiff(
                output=str(output_path),
                bbox=bbox,
                zoom=zoom,
                source=source,
                overwrite=overwrite
            )
            logger.info(f"Successfully downloaded to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to download imagery: {e}")
            raise
    
    def download_from_url(
        self,
        url: str,
        filename: Optional[str] = None,
        overwrite: bool = False
    ) -> str:
        """
        Download image from URL.
        
        Args:
            url: URL of the image
            filename: Optional custom filename
            overwrite: Whether to overwrite existing files
            
        Returns:
            Path to downloaded image
        """
        if filename is None:
            filename = Path(url).name
        
        output_path = self.output_dir / filename
        
        if output_path.exists() and not overwrite:
            logger.info(f"File {output_path} already exists, skipping download")
            return str(output_path)
        
        logger.info(f"Downloading from URL: {url}")
        
        try:
            download_file(url, str(output_path))
            logger.info(f"Successfully downloaded to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to download from URL: {e}")
            raise


class ImageLoader:
    """Load and validate GeoTIFF images."""
    
    @staticmethod
    def load_image(image_path: str) -> Tuple[np.ndarray, dict]:
        """
        Load a GeoTIFF image and extract metadata.
        
        Args:
            image_path: Path to the GeoTIFF file
            
        Returns:
            Tuple of (image array, metadata dict)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        ds = gdal.Open(image_path)
        if ds is None:
            raise ValueError(f"Unable to open image: {image_path}")
        
        # Read image data
        image = ds.ReadAsArray()
        
        # Handle different channel configurations
        if len(image.shape) == 3:
            # Multi-band image (bands, height, width) -> (height, width, bands)
            image = np.transpose(image, (1, 2, 0))
        
        # Extract metadata
        metadata = {
            'width': ds.RasterXSize,
            'height': ds.RasterYSize,
            'bands': ds.RasterCount,
            'geotransform': ds.GetGeoTransform(),
            'projection': ds.GetProjection(),
            'driver': ds.GetDriver().ShortName
        }
        
        ds = None  # Close dataset
        
        logger.info(f"Loaded image: {image.shape}, {metadata['bands']} bands")
        
        return image, metadata
    
    @staticmethod
    def validate_image(image_path: str) -> bool:
        """
        Validate that an image file is a valid GeoTIFF.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            ds = gdal.Open(image_path)
            if ds is None:
                return False
            
            # Check basic requirements
            if ds.RasterXSize <= 0 or ds.RasterYSize <= 0:
                return False
            
            if ds.GetProjection() == "":
                logger.warning(f"Image {image_path} has no projection information")
            
            ds = None
            return True
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
    
    @staticmethod
    def get_image_info(image_path: str) -> dict:
        """
        Get detailed information about an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image information
        """
        ds = gdal.Open(image_path)
        if ds is None:
            raise ValueError(f"Unable to open image: {image_path}")
        
        info = {
            'path': image_path,
            'size': os.path.getsize(image_path),
            'width': ds.RasterXSize,
            'height': ds.RasterYSize,
            'bands': ds.RasterCount,
            'projection': ds.GetProjection(),
            'geotransform': ds.GetGeoTransform(),
            'driver': ds.GetDriver().ShortName,
            'datatype': gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
        }
        
        ds = None
        return info
