"""
Image preprocessing utilities including resizing, tiling with overlap, and coordinate mapping.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
from osgeo import gdal, osr

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocess images for YOLO inference."""
    
    def __init__(
        self,
        target_width: int = 8192,
        target_height: int = 8192,
        interpolation: int = cv2.INTER_LINEAR,
        convert_to_grayscale: bool = True
    ):
        """
        Initialize ImagePreprocessor.
        
        Args:
            target_width: Target width for clipping (must be multiple of tile_size)
            target_height: Target height for clipping (must be multiple of tile_size)
            interpolation: Interpolation method for resizing
            convert_to_grayscale: Whether to convert RGB to grayscale 8-bit
        """
        self.target_width = target_width
        self.target_height = target_height
        self.interpolation = interpolation
        self.convert_to_grayscale = convert_to_grayscale
    
    def convert_to_grayscale_8bit(
        self,
        input_path: str,
        output_path: str
    ) -> str:
        """
        Convert RGB image to grayscale 8-bit, preserving georeference.
        If already grayscale, just ensures it's 8-bit.
        
        Args:
            input_path: Path to input GeoTIFF (RGB or grayscale)
            output_path: Path for output grayscale GeoTIFF
            
        Returns:
            Path to output file
        """
        logger.info(f"Converting to grayscale 8-bit: {input_path}")
        
        # Create output directory if needed
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ds = gdal.Open(input_path)
        if not ds:
            raise ValueError(f"Cannot open {input_path}")
        
        # Read image data
        input_array = ds.ReadAsArray()
        num_bands = ds.RasterCount
        
        logger.info(f"Input has {num_bands} band(s)")
        
        # Handle different input types
        if num_bands == 1:
            # Already grayscale, just ensure 8-bit
            logger.info("Input is already grayscale, converting to 8-bit")
            grayscale_array = input_array
        elif num_bands >= 3:
            # RGB or more - convert to grayscale
            logger.info("Converting RGB to grayscale")
            if len(input_array.shape) == 3:
                # Shape is (bands, height, width)
                r, g, b = input_array[0], input_array[1], input_array[2]
            else:
                # Shape is (height, width) - shouldn't happen with num_bands >= 3
                raise ValueError(f"Unexpected array shape: {input_array.shape} with {num_bands} bands")
            
            grayscale_array = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            raise ValueError(f"Input image must have 1 band (grayscale) or 3+ bands (RGB), got {num_bands}")
        
        # Ensure 2D array
        if len(grayscale_array.shape) != 2:
            raise ValueError(f"Grayscale array must be 2D, got shape {grayscale_array.shape}")
        
        # Scale to 8-bit
        min_val = np.min(grayscale_array)
        max_val = np.max(grayscale_array)
        
        if max_val > min_val:
            scale_factor = 255.0 / (max_val - min_val)
            output_array = ((grayscale_array - min_val) * scale_factor).astype(np.uint8)
        else:
            # All values are the same
            output_array = np.zeros_like(grayscale_array, dtype=np.uint8)
        
        logger.info(f"Output array shape: {output_array.shape}, dtype: {output_array.dtype}")
        
        # Create output file
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            output_path,
            ds.RasterXSize,
            ds.RasterYSize,
            1,
            gdal.GDT_Byte
        )
        
        if not out_ds:
            raise RuntimeError(f"Failed to create output file: {output_path}")
        
        # Copy geotransform and projection
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        
        # Write data
        out_ds.GetRasterBand(1).WriteArray(output_array)
        out_ds.FlushCache()
        
        # Close datasets
        ds = None
        out_ds = None
        
        logger.info(f"Grayscale conversion completed: {output_path}")
        
        return output_path
    
    def reproject_to_4326(
        self,
        input_path: str,
        output_path: str
    ) -> str:
        """
        Reproject image to EPSG:4326.
        
        Args:
            input_path: Path to input GeoTIFF
            output_path: Path for output GeoTIFF in EPSG:4326
            
        Returns:
            Path to output file
        """
        logger.info(f"Reprojecting to EPSG:4326: {input_path}")
        
        # Use gdalwarp for reprojection
        warp_options = gdal.WarpOptions(
            dstSRS='EPSG:4326',
            resampleAlg=gdal.GRA_Bilinear
        )
        
        ds = gdal.Warp(output_path, input_path, options=warp_options)
        
        if ds is None:
            raise RuntimeError(f"Failed to reproject to {output_path}")
        
        ds = None
        
        logger.info(f"Reprojection completed: {output_path}")
        
        return output_path
    
    def clip_or_pad_to_regular_size(
        self,
        input_path: str,
        output_path: str,
        tile_size: int = 1024,
        use_padding: bool = False
    ) -> str:
        """
        Clip or pad image to regular size (multiple of tile_size).
        
        AUTOMATIC SIZING: 
        - If use_padding=False: floors to nearest multiple (clips excess)
        - If use_padding=True: ceils to nearest multiple (pads to fill)
        
        Example with tile_size=1024:
        - Input 8500x9200
        - Clip mode: 8192x9216 (8*1024 x 9*1024)
        - Pad mode:  9216x10240 (9*1024 x 10*1024)
        
        This preserves the geotransform correctly for the processed area.
        
        Args:
            input_path: Path to input GeoTIFF
            output_path: Path for output GeoTIFF
            tile_size: Tile size for calculating regular dimensions
            use_padding: If True, pad to next multiple; if False, clip to previous multiple
            
        Returns:
            Path to output file
        """
        import rasterio
        from rasterio.windows import Window
        import math
        
        logger.info(f"Processing to regular size with tile_size={tile_size}, padding={use_padding}")
        
        # Create output directory if needed
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with rasterio.open(input_path) as src:
            width = src.width
            height = src.height
            
            if use_padding:
                # Pad to next multiple of tile_size
                width_set = math.ceil(width / tile_size) * tile_size
                height_set = math.ceil(height / tile_size) * tile_size
                logger.info(f"Original size: {width}x{height}")
                logger.info(f"Padded size: {width_set}x{height_set}")
                
                # Read all data
                data = src.read()
                
                # Calculate padding needed
                pad_width = width_set - width
                pad_height = height_set - height
                
                # Pad the data (pad on right and bottom)
                if len(data.shape) == 3:
                    # Multi-band: (bands, height, width)
                    padded_data = np.pad(
                        data,
                        ((0, 0), (0, pad_height), (0, pad_width)),
                        mode='constant',
                        constant_values=0
                    )
                else:
                    # Single band: (height, width)
                    padded_data = np.pad(
                        data,
                        ((0, pad_height), (0, pad_width)),
                        mode='constant',
                        constant_values=0
                    )
                
                # Update metadata for padded image
                meta = src.meta.copy()
                meta.update({
                    'width': width_set,
                    'height': height_set
                    # Transform stays the same - we're padding, not moving origin
                })
                
                # Save padded image
                with rasterio.open(output_path, 'w', **meta) as dest:
                    dest.write(padded_data)
                
            else:
                # Clip to previous multiple of tile_size (original behavior)
                width_set = math.floor(width / tile_size) * tile_size
                height_set = math.floor(height / tile_size) * tile_size
                
                logger.info(f"Original size: {width}x{height}")
                logger.info(f"Clipped size: {width_set}x{height_set}")
                
                # Define clipping window (from top-left)
                window = Window(0, 0, min(width, width_set), min(height, height_set))
                
                # Read data from window
                data = src.read(window=window)
                
                # Get metadata and update for clipped image
                meta = src.meta.copy()
                meta.update({
                    'width': window.width,
                    'height': window.height,
                    'transform': src.window_transform(window)  # CRITICAL: updates geotransform
                })
                
                # Save clipped image with correct geotransform
                with rasterio.open(output_path, 'w', **meta) as dest:
                    dest.write(data)
        
        logger.info(f"Processed image saved to {output_path}")
        
        return output_path
    
    def clip_to_regular_size(
        self,
        input_path: str,
        output_path: str,
        regular_width: Optional[int] = None,
        regular_height: Optional[int] = None
    ) -> str:
        """
        DEPRECATED: Use clip_or_pad_to_regular_size() instead for automatic sizing.
        
        Clip image to exact regular size dimensions.
        This preserves the geotransform correctly for the clipped area.
        
        Args:
            input_path: Path to input GeoTIFF
            output_path: Path for output clipped GeoTIFF
            regular_width: Exact target width
            regular_height: Exact target height
            
        Returns:
            Path to clipped file
        """
        import rasterio
        from rasterio.windows import Window
        
        regular_width = regular_width or self.target_width
        regular_height = regular_height or self.target_height
        
        logger.info(f"Clipping to exact size: {regular_width}x{regular_height}")
        
        # Create output directory if needed
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with rasterio.open(input_path) as src:
            width = src.width
            height = src.height
            
            logger.info(f"Original size: {width}x{height}")
            
            # Define clipping window (from top-left)
            window = Window(0, 0, min(width, regular_width), min(height, regular_height))
            
            # Read data from window
            data = src.read(window=window)
            
            # Get metadata and update for clipped image
            meta = src.meta.copy()
            meta.update({
                'width': window.width,
                'height': window.height,
                'transform': src.window_transform(window)  # CRITICAL: updates geotransform
            })
            
            # Save clipped image with correct geotransform
            with rasterio.open(output_path, 'w', **meta) as dest:
                dest.write(data)
        
        logger.info(f"Clipped image saved to {output_path}")
        
        return output_path
    
    def prepare_for_yolo(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image for YOLO inference (ensure correct format).
        For grayscale images, convert to 3-channel by replicating.
        
        Args:
            image: Input image
            
        Returns:
            Prepared image (always 3-channel)
        """
        # Ensure image is in the correct format
        if len(image.shape) == 2:
            # Grayscale to 3-channel (replicate, not convert to RGB)
            # This is important for models trained on grayscale
            image = np.stack([image, image, image], axis=2)
        elif image.shape[2] == 4:
            # RGBA to RGB
            image = image[:, :, :3]
        
        return image
    
    def save_image(
        self,
        image: np.ndarray,
        output_path: str,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Save image with optional geospatial metadata.
        
        Args:
            image: Image array to save
            output_path: Output file path
            metadata: Optional geospatial metadata
            
        Returns:
            Path to saved image
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if metadata and 'geotransform' in metadata and 'projection' in metadata:
            # Save as GeoTIFF with georeferencing
            self._save_geotiff(image, str(output_path), metadata)
        else:
            # Save as regular image
            cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        logger.info(f"Saved image to {output_path}")
        return str(output_path)
    
    def _save_geotiff(self, image: np.ndarray, output_path: str, metadata: dict):
        """Save image as GeoTIFF with georeferencing."""
        driver = gdal.GetDriverByName('GTiff')
        
        if len(image.shape) == 3:
            height, width, bands = image.shape
            # Transpose back to (bands, height, width) for GDAL
            image = np.transpose(image, (2, 0, 1))
        else:
            height, width = image.shape
            bands = 1
        
        ds = driver.Create(output_path, width, height, bands, gdal.GDT_Byte)
        
        if bands == 1:
            ds.GetRasterBand(1).WriteArray(image)
        else:
            for i in range(bands):
                ds.GetRasterBand(i + 1).WriteArray(image[i])
        
        ds.SetGeoTransform(metadata['geotransform'])
        ds.SetProjection(metadata['projection'])
        
        ds.FlushCache()
        ds = None


class TileGenerator:
    """Generate tiles from large images with overlap support."""
    
    def __init__(
        self,
        tile_size: int = 1024,
        overlap: int = 128,
        min_tile_coverage: float = 0.5
    ):
        """
        Initialize TileGenerator.
        
        Args:
            tile_size: Size of each tile (width and height)
            overlap: Overlap between adjacent tiles in pixels
            min_tile_coverage: Minimum fraction of tile that must contain data (0-1)
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap  # Step size between tiles
        self.min_tile_coverage = min_tile_coverage
        
        logger.info(f"Initialized TileGenerator: tile_size={tile_size}, overlap={overlap}, stride={self.stride}")
    
    def calculate_tile_positions(
        self,
        image_width: int,
        image_height: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Calculate tile positions with overlap.
        
        Args:
            image_width: Width of the source image
            image_height: Height of the source image
            
        Returns:
            List of (x_start, y_start, x_end, y_end) tuples
        """
        tiles = []
        
        # Calculate number of tiles needed
        n_cols = int(np.ceil((image_width - self.overlap) / self.stride))
        n_rows = int(np.ceil((image_height - self.overlap) / self.stride))
        
        logger.info(f"Generating {n_rows}x{n_cols} = {n_rows * n_cols} tiles")
        
        for row in range(n_rows):
            y_start = row * self.stride
            y_end = min(y_start + self.tile_size, image_height)
            
            # Adjust if the last tile is too small
            if y_end - y_start < self.tile_size * self.min_tile_coverage:
                y_start = max(0, image_height - self.tile_size)
                y_end = image_height
            
            for col in range(n_cols):
                x_start = col * self.stride
                x_end = min(x_start + self.tile_size, image_width)
                
                # Adjust if the last tile is too small
                if x_end - x_start < self.tile_size * self.min_tile_coverage:
                    x_start = max(0, image_width - self.tile_size)
                    x_end = image_width
                
                tiles.append((x_start, y_start, x_end, y_end))
        
        # Remove duplicate tiles (can occur with boundary adjustments)
        tiles = list(set(tiles))
        
        logger.info(f"Final tile count after deduplication: {len(tiles)}")
        
        return tiles
    
    def generate_tiles_from_geotiff(
        self,
        input_path: str,
        output_dir: str,
        basename: str = "tile"
    ) -> List[dict]:
        """
        Generate tiles from a GeoTIFF, preserving georeference for each tile.
        This is the PREFERRED method for georeferenced data.
        
        Args:
            input_path: Path to input GeoTIFF
            output_dir: Directory to save tiles
            basename: Base name for tile files
            
        Returns:
            List of dictionaries with tile information including geotransforms
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating georeferenced tiles from {input_path}")
        
        # Open source raster
        src_ds = gdal.Open(input_path)
        if not src_ds:
            raise ValueError(f"Cannot open {input_path}")
        
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        geotransform = src_ds.GetGeoTransform()
        projection = src_ds.GetProjection()
        
        tile_positions = self.calculate_tile_positions(width, height)
        tile_info = []
        
        driver = gdal.GetDriverByName('GTiff')
        
        for idx, (x_start, y_start, x_end, y_end) in enumerate(tile_positions):
            x_size = x_end - x_start
            y_size = y_end - y_start
            
            # Read data for this tile
            data = src_ds.ReadAsArray(x_start, y_start, x_size, y_size)
            
            # Calculate geotransform for this tile
            # Origin of tile in geo coordinates
            x_origin = geotransform[0] + x_start * geotransform[1]
            y_origin = geotransform[3] + y_start * geotransform[5]
            
            tile_geotransform = (
                x_origin,           # x origin
                geotransform[1],    # pixel width
                0,                  # rotation (usually 0)
                y_origin,           # y origin  
                0,                  # rotation (usually 0)
                geotransform[5]     # pixel height (usually negative)
            )
            
            # Create output tile
            tile_filename = f"{idx}.tif"  # Keep as TIF to preserve georeference
            tile_path = output_dir / tile_filename
            
            output_ds = driver.Create(
                str(tile_path),
                x_size,
                y_size,
                1,  # Single band (grayscale)
                gdal.GDT_Byte
            )
            
            # Set geotransform and projection for tile
            output_ds.SetGeoTransform(tile_geotransform)
            output_ds.SetProjection(projection)
            
            # Write data
            output_ds.GetRasterBand(1).WriteArray(data)
            
            # Close tile
            output_ds = None
            
            # Store tile information
            info = {
                'tile_id': idx,
                'filename': tile_filename,
                'path': str(tile_path),
                'x_start': x_start,
                'y_start': y_start,
                'x_end': x_end,
                'y_end': y_end,
                'width': x_size,
                'height': y_size,
                'geotransform': tile_geotransform,
                'projection': projection
            }
            
            tile_info.append(info)
        
        # Close source
        src_ds = None
        
        logger.info(f"Generated {len(tile_info)} georeferenced tiles")
        
        # Save metadata
        import json
        metadata_path = output_dir / f"{basename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'source_file': input_path,
                'tile_size': self.tile_size,
                'overlap': self.overlap,
                'stride': self.stride,
                'source_width': width,
                'source_height': height,
                'source_geotransform': geotransform,
                'source_projection': projection,
                'tiles': tile_info
            }, f, indent=2)
        logger.info(f"Saved tile metadata to {metadata_path}")
        
        return tile_info
    
    def generate_tiles(
        self,
        image: np.ndarray,
        output_dir: str,
        basename: str = "tile",
        save_metadata: bool = True
    ) -> List[dict]:
        """
        Generate and save tiles from an image array (no georeference).
        For georeferenced data, use generate_tiles_from_geotiff() instead.
        
        Args:
            image: Source image array
            output_dir: Directory to save tiles
            basename: Base name for tile files
            save_metadata: Whether to save tile metadata
            
        Returns:
            List of dictionaries with tile information
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        height, width = image.shape[:2]
        tile_positions = self.calculate_tile_positions(width, height)
        
        tile_info = []
        
        for idx, (x_start, y_start, x_end, y_end) in enumerate(tile_positions):
            # Extract tile
            tile = image[y_start:y_end, x_start:x_end]
            
            # Pad if necessary to reach tile_size
            if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                pad_height = max(0, self.tile_size - tile.shape[0])
                pad_width = max(0, self.tile_size - tile.shape[1])
                
                if len(tile.shape) == 3:
                    tile = np.pad(
                        tile,
                        ((0, pad_height), (0, pad_width), (0, 0)),
                        mode='constant',
                        constant_values=0
                    )
                else:
                    tile = np.pad(
                        tile,
                        ((0, pad_height), (0, pad_width)),
                        mode='constant',
                        constant_values=0
                    )
            
            # Save tile
            tile_filename = f"{basename}_{idx:04d}.jpg"
            tile_path = output_dir / tile_filename
            
            if len(tile.shape) == 3:
                cv2.imwrite(str(tile_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(str(tile_path), tile)
            
            # Store tile information
            info = {
                'tile_id': idx,
                'filename': tile_filename,
                'path': str(tile_path),
                'x_start': x_start,
                'y_start': y_start,
                'x_end': x_end,
                'y_end': y_end,
                'width': x_end - x_start,
                'height': y_end - y_start,
                'actual_width': tile.shape[1],
                'actual_height': tile.shape[0]
            }
            
            tile_info.append(info)
        
        logger.info(f"Generated {len(tile_info)} tiles in {output_dir}")
        
        # Save metadata if requested
        if save_metadata:
            import json
            metadata_path = output_dir / f"{basename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'tile_size': self.tile_size,
                    'overlap': self.overlap,
                    'stride': self.stride,
                    'source_width': width,
                    'source_height': height,
                    'tiles': tile_info
                }, f, indent=2)
            logger.info(f"Saved tile metadata to {metadata_path}")
        
        return tile_info
    
    def map_detection_to_original(
        self,
        detection_coords: np.ndarray,
        tile_info: dict,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Map detection coordinates from tile space to original image space.
        
        Args:
            detection_coords: Detection coordinates in tile space (N, 2) or (N, 4)
            tile_info: Tile information dictionary
            image_shape: Shape of the original image (height, width)
            
        Returns:
            Coordinates in original image space
        """
        # Add tile offset
        original_coords = detection_coords.copy()
        original_coords[:, 0::2] += tile_info['x_start']  # x coordinates
        original_coords[:, 1::2] += tile_info['y_start']  # y coordinates
        
        # Clip to image bounds
        height, width = image_shape
        original_coords[:, 0::2] = np.clip(original_coords[:, 0::2], 0, width)
        original_coords[:, 1::2] = np.clip(original_coords[:, 1::2], 0, height)
        
        return original_coords