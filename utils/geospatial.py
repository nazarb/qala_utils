"""
Geospatial utilities for coordinate transformations and image clipping.
"""

import os
import logging
from typing import Tuple, Optional
from pathlib import Path
import numpy as np
from osgeo import gdal, osr
from shapely.geometry import Polygon, box
import geopandas as gpd

logger = logging.getLogger(__name__)


class GeospatialUtils:
    """Utilities for geospatial operations on raster data."""
    
    @staticmethod
    def get_image_bbox(image_path: str, target_crs: str = "EPSG:4326") -> Polygon:
        """
        Get bounding box of an image as a polygon.
        
        Args:
            image_path: Path to the image
            target_crs: Target CRS for the bounding box
            
        Returns:
            Shapely Polygon representing the bounding box
        """
        ds = gdal.Open(image_path)
        if not ds:
            raise ValueError(f"Unable to open image: {image_path}")
        
        # Get geotransform and projection
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        width = ds.RasterXSize
        height = ds.RasterYSize
        
        # Create source spatial reference
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(projection)
        
        # Create target spatial reference
        tgt_srs = osr.SpatialReference()
        tgt_srs.ImportFromProj4(target_crs.replace("EPSG:", "+init=epsg:"))
        
        # Create coordinate transformation
        transform = osr.CoordinateTransformation(src_srs, tgt_srs)
        
        # Get corner coordinates
        def get_corner_coords(x, y):
            """Transform pixel coordinates to geographic coordinates."""
            geo_x = geotransform[0] + x * geotransform[1] + y * geotransform[2]
            geo_y = geotransform[3] + x * geotransform[4] + y * geotransform[5]
            lon, lat, _ = transform.TransformPoint(geo_x, geo_y)
            return lon, lat
        
        # Get all four corners
        top_left = get_corner_coords(0, 0)
        top_right = get_corner_coords(width, 0)
        bottom_right = get_corner_coords(width, height)
        bottom_left = get_corner_coords(0, height)
        
        # Create polygon
        bbox_polygon = Polygon([top_left, top_right, bottom_right, bottom_left, top_left])
        
        ds = None
        
        logger.info(f"Extracted bounding box: {bbox_polygon.bounds}")
        
        return bbox_polygon
    
    @staticmethod
    def clip_raster_to_bbox(
        input_path: str,
        output_path: str,
        bbox: Tuple[float, float, float, float],
        target_crs: Optional[str] = None
    ) -> str:
        """
        Clip a raster to a bounding box.
        
        Args:
            input_path: Path to input raster
            output_path: Path to output raster
            bbox: Bounding box as (minx, miny, maxx, maxy)
            target_crs: Optional target CRS (e.g., "EPSG:4326")
            
        Returns:
            Path to clipped raster
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build gdal_translate command options
        translate_options = gdal.TranslateOptions(
            projWin=[bbox[0], bbox[3], bbox[2], bbox[1]],  # ulx, uly, lrx, lry
            projWinSRS=target_crs if target_crs else None
        )
        
        # Clip the raster
        ds = gdal.Translate(str(output_path), input_path, options=translate_options)
        
        if ds is None:
            raise RuntimeError(f"Failed to clip raster to {output_path}")
        
        ds = None
        
        logger.info(f"Clipped raster to {output_path}")
        
        return str(output_path)
    
    @staticmethod
    def reproject_raster(
        input_path: str,
        output_path: str,
        target_crs: str = "EPSG:4326",
        resampling: int = gdal.GRA_Bilinear
    ) -> str:
        """
        Reproject a raster to a different CRS.
        
        Args:
            input_path: Path to input raster
            output_path: Path to output raster
            target_crs: Target CRS
            resampling: Resampling method
            
        Returns:
            Path to reprojected raster
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build gdalwarp options
        warp_options = gdal.WarpOptions(
            dstSRS=target_crs,
            resampleAlg=resampling
        )
        
        # Reproject
        ds = gdal.Warp(str(output_path), input_path, options=warp_options)
        
        if ds is None:
            raise RuntimeError(f"Failed to reproject raster to {output_path}")
        
        ds = None
        
        logger.info(f"Reprojected raster to {output_path}")
        
        return str(output_path)
    
    @staticmethod
    def pixel_to_geo_coords(
        pixel_coords: np.ndarray,
        geotransform: Tuple[float, ...]
    ) -> np.ndarray:
        """
        Convert pixel coordinates to geographic coordinates.
        
        Args:
            pixel_coords: Pixel coordinates (N, 2) as [x, y]
            geotransform: GDAL geotransform tuple
            
        Returns:
            Geographic coordinates (N, 2)
        """
        x_pixel = pixel_coords[:, 0]
        y_pixel = pixel_coords[:, 1]
        
        x_geo = geotransform[0] + x_pixel * geotransform[1] + y_pixel * geotransform[2]
        y_geo = geotransform[3] + x_pixel * geotransform[4] + y_pixel * geotransform[5]
        
        geo_coords = np.column_stack([x_geo, y_geo])
        
        return geo_coords
    
    @staticmethod
    def geo_to_pixel_coords(
        geo_coords: np.ndarray,
        geotransform: Tuple[float, ...]
    ) -> np.ndarray:
        """
        Convert geographic coordinates to pixel coordinates.
        
        Args:
            geo_coords: Geographic coordinates (N, 2) as [x, y]
            geotransform: GDAL geotransform tuple
            
        Returns:
            Pixel coordinates (N, 2)
        """
        x_geo = geo_coords[:, 0]
        y_geo = geo_coords[:, 1]
        
        # Inverse geotransform
        det = geotransform[1] * geotransform[5] - geotransform[2] * geotransform[4]
        
        x_pixel = (
            geotransform[5] * (x_geo - geotransform[0]) -
            geotransform[2] * (y_geo - geotransform[3])
        ) / det
        
        y_pixel = (
            -geotransform[4] * (x_geo - geotransform[0]) +
            geotransform[1] * (y_geo - geotransform[3])
        ) / det
        
        pixel_coords = np.column_stack([x_pixel, y_pixel])
        
        return pixel_coords
    
    @staticmethod
    def calculate_ground_sample_distance(geotransform: Tuple[float, ...]) -> Tuple[float, float]:
        """
        Calculate ground sample distance (GSD) from geotransform.
        
        Args:
            geotransform: GDAL geotransform tuple
            
        Returns:
            Tuple of (gsd_x, gsd_y) in the units of the CRS
        """
        gsd_x = abs(geotransform[1])
        gsd_y = abs(geotransform[5])
        
        return gsd_x, gsd_y
    
    @staticmethod
    def get_raster_extent(image_path: str) -> dict:
        """
        Get the extent and metadata of a raster.
        
        Args:
            image_path: Path to the raster
            
        Returns:
            Dictionary with extent information
        """
        ds = gdal.Open(image_path)
        if not ds:
            raise ValueError(f"Unable to open image: {image_path}")
        
        geotransform = ds.GetGeoTransform()
        width = ds.RasterXSize
        height = ds.RasterYSize
        
        minx = geotransform[0]
        maxy = geotransform[3]
        maxx = minx + width * geotransform[1]
        miny = maxy + height * geotransform[5]
        
        extent = {
            'minx': minx,
            'miny': miny,
            'maxx': maxx,
            'maxy': maxy,
            'width': width,
            'height': height,
            'geotransform': geotransform,
            'projection': ds.GetProjection()
        }
        
        ds = None
        
        return extent
    
    @staticmethod
    def create_spatial_index(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Create a spatial index for a GeoDataFrame.
        
        Args:
            gdf: Input GeoDataFrame
            
        Returns:
            GeoDataFrame with spatial index
        """
        gdf.sindex  # This creates the spatial index
        logger.info(f"Created spatial index for GeoDataFrame with {len(gdf)} features")
        return gdf
    
    @staticmethod
    def buffer_points(
        gdf: gpd.GeoDataFrame,
        distance: float,
        crs: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Buffer point geometries.
        
        Args:
            gdf: GeoDataFrame with point geometries
            distance: Buffer distance in CRS units
            crs: Optional target CRS for buffering
            
        Returns:
            GeoDataFrame with buffered geometries
        """
        original_crs = gdf.crs
        
        # Reproject if needed
        if crs and crs != original_crs:
            gdf = gdf.to_crs(crs)
        
        # Apply buffer
        gdf_buffered = gdf.copy()
        gdf_buffered.geometry = gdf.geometry.buffer(distance)
        
        # Reproject back if needed
        if crs and crs != original_crs:
            gdf_buffered = gdf_buffered.to_crs(original_crs)
        
        logger.info(f"Buffered {len(gdf)} points by {distance} units")
        
        return gdf_buffered
