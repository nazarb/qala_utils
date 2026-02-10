"""
Postprocessing utilities for detection merging, NMS, and qanat-specific analysis.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix

try:
    from scipy import ndimage
    from skimage import measure
except ImportError:
    ndimage = None
    measure = None
    logger.warning("scipy or skimage not available. Binary mask conversion may not work.")

logger = logging.getLogger(__name__)


class DetectionPostprocessor:
    """Postprocess detections from tiled inference (detection or instance segmentation)."""
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        class_name: str = "qala"
    ):
        """
        Initialize DetectionPostprocessor.
        
        Args:
            iou_threshold: IOU threshold for merging overlapping detections
            class_name: Class name for single-class instance segmentation (default: "qala").
                Used when class_names are missing from detection output.
        """
        self.iou_threshold = iou_threshold
        self.class_name = class_name
    
    def _boxes_from_masks(
        self,
        masks,
        confidences: np.ndarray,
        class_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract bounding boxes from instance segmentation masks.
        
        Args:
            masks: List of polygon arrays (N, 2) or binary mask arrays (H, W)
            confidences: Confidence scores (N,)
            class_ids: Class IDs (N,)
            
        Returns:
            Tuple of (boxes, confidences, class_ids) with invalid masks filtered out
        """
        boxes_list = []
        conf_list = []
        ids_list = []
        for i, mask in enumerate(masks):
            mask = np.asarray(mask)
            if mask.ndim == 2:
                # Binary mask: get bounding box from True/nonzero pixels
                rows, cols = np.where(mask > 0)
                if len(rows) == 0 or len(cols) == 0:
                    continue
                x1, x2 = float(cols.min()), float(cols.max())
                y1, y2 = float(rows.min()), float(rows.max())
            else:
                # Polygon format (N, 2)
                x1, y1 = float(mask[:, 0].min()), float(mask[:, 1].min())
                x2, y2 = float(mask[:, 0].max()), float(mask[:, 1].max())
            boxes_list.append([x1, y1, x2, y2])
            conf_list.append(confidences[i] if i < len(confidences) else 0.0)
            ids_list.append(class_ids[i] if i < len(class_ids) else 0)
        boxes = np.array(boxes_list) if boxes_list else np.array([]).reshape(0, 4)
        return boxes, np.array(conf_list), np.array(ids_list)

    def merge_tile_detections(
        self,
        tile_detections: List[Dict],
        image_shape: Tuple[int, int]
    ) -> Dict:
        """
        Merge detections from multiple tiles, handling overlaps.
        
        Supports both object detection and instance segmentation formats.
        For single-class instance segmentation with class "qala", use class_name="qala".
        
        Args:
            tile_detections: List of detection dictionaries with tile_info.
                Each detection may have:
                - boxes: (N, 4) in xyxy format
                - masks: optional, used to extract boxes when boxes are missing
                - class_names: optional, inferred from class_name when missing
            image_shape: Shape of the original image (height, width)
            
        Returns:
            Merged detection dictionary
        """
        all_boxes = []
        all_masks = []  # Store masks for instance segmentation
        all_confidences = []
        all_class_ids = []
        all_class_names = []
        all_tile_offsets = []  # Store tile offsets for mask transformation
        
        # Map tile detections to original image coordinates
        for detection in tile_detections:
            if detection['num_detections'] == 0:
                continue
            
            tile_info = detection['tile_info']
            
            # Get boxes: use boxes if present, else extract from masks (instance segmentation)
            class_ids = np.asarray(detection['class_ids'])
            confidences = np.asarray(detection['confidences'])
            has_masks = 'masks' in detection and len(detection['masks']) > 0
            
            if 'boxes' in detection and len(detection['boxes']) > 0:
                boxes = np.asarray(detection['boxes']).copy()
                if boxes.ndim == 1:
                    boxes = boxes.reshape(1, -1)
                # Store masks if available (for instance segmentation)
                if has_masks:
                    all_masks.extend(detection['masks'])
                    all_tile_offsets.extend([(tile_info['x_start'], tile_info['y_start'])] * len(detection['masks']))
            elif has_masks:
                boxes, confidences, class_ids = self._boxes_from_masks(
                    detection['masks'], confidences, class_ids
                )
                if len(boxes) == 0:
                    continue
                # Store masks with tile offsets
                all_masks.extend(detection['masks'])
                all_tile_offsets.extend([(tile_info['x_start'], tile_info['y_start'])] * len(detection['masks']))
            else:
                continue
            
            n = len(class_ids)
            # Get class names: use provided or infer for single-class instance segmentation
            if 'class_names' in detection and len(detection['class_names']) == n:
                names = list(detection['class_names'])
            else:
                names = [self.class_name] * n
            
            # Transform coordinates to original image space
            boxes[:, [0, 2]] += tile_info['x_start']  # x coordinates
            boxes[:, [1, 3]] += tile_info['y_start']  # y coordinates
            
            # Clip to image bounds
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_shape[1])
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_shape[0])
            
            all_boxes.append(boxes)
            all_confidences.append(confidences)
            all_class_ids.append(class_ids)
            all_class_names.extend(names)
        
        if len(all_boxes) == 0:
            return {
                'num_detections': 0,
                'boxes': np.array([]),
                'confidences': np.array([]),
                'class_ids': np.array([]),
                'class_names': []
            }
        
        # Concatenate all detections
        boxes = np.vstack(all_boxes)
        confidences = np.concatenate(all_confidences)
        class_ids = np.concatenate(all_class_ids)
        
        # Apply NMS to remove duplicates from overlapping tiles
        keep_indices = self._non_maximum_suppression(
            boxes,
            confidences,
            self.iou_threshold
        )
        
        merged = {
            'num_detections': len(keep_indices),
            'boxes': boxes[keep_indices],
            'confidences': confidences[keep_indices],
            'class_ids': class_ids[keep_indices],
            'class_names': [all_class_names[i] for i in keep_indices]
        }
        
        # Include masks if available (for instance segmentation)
        if len(all_masks) > 0 and len(all_masks) == len(boxes):
            merged['masks'] = [all_masks[i] for i in keep_indices]
            merged['tile_offsets'] = [all_tile_offsets[i] for i in keep_indices]
        
        logger.info(f"Merged {len(boxes)} detections to {len(keep_indices)} after NMS")
        
        return merged
    
    def _non_maximum_suppression(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float
    ) -> np.ndarray:
        """
        Apply Non-Maximum Suppression.
        
        Args:
            boxes: Bounding boxes (N, 4) in xyxy format
            scores: Confidence scores (N,)
            iou_threshold: IOU threshold
            
        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return np.array([], dtype=int)
        
        # Get coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Compute areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by confidence
        order = scores.argsort()[::-1]
        
        keep = []
        
        while len(order) > 0:
            # Pick box with highest confidence
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Compute IOU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Keep boxes with IOU below threshold
            mask = iou <= iou_threshold
            order = order[1:][mask]
        
        return np.array(keep)
    
    def boxes_to_centroids(self, boxes: np.ndarray) -> np.ndarray:
        """
        Convert bounding boxes to centroids.
        
        Args:
            boxes: Bounding boxes (N, 4) in xyxy format
            
        Returns:
            Centroids (N, 2)
        """
        if len(boxes) == 0:
            return np.array([])
        
        centroids = np.column_stack([
            (boxes[:, 0] + boxes[:, 2]) / 2,  # x
            (boxes[:, 1] + boxes[:, 3]) / 2   # y
        ])
        
        return centroids


class QalaPipeline:
    """Process detected qanats using spatial join approach (without clustering)."""
    
    def __init__(
        self,
        eps: float = 0.0008,  # DBSCAN epsilon in geographic degrees (deprecated, kept for compatibility)
        min_samples: int = 4,  # Minimum points per cluster (deprecated, kept for compatibility)
        min_confidence_qala: float = 0.1,  # Minimum confidence for qala class
        min_confidence_qala_pair: float = None,  # Minimum confidence for qala_pair class (only for two-class pipeline)
        overlap_threshold: float = 0.9  # Minimum overlap ratio to merge geometries (90%)
    ):
        """
        Initialize QalaPipeline with processing parameters.
        
        Args:
            eps: DBSCAN epsilon parameter (deprecated, kept for backward compatibility)
            min_samples: DBSCAN minimum samples (deprecated, kept for backward compatibility)
            min_confidence_qala: Minimum confidence for qala detections (default: 0.1)
            min_confidence_qala_pair: Minimum confidence for qala_pair detections (default: None, only used in two-class pipeline)
            overlap_threshold: Minimum overlap ratio (0.0-1.0) to merge geometries (default: 0.9 = 90%)
        """
        self.eps = eps
        self.min_samples = min_samples
        self.min_confidence_qala = min_confidence_qala
        self.min_confidence_qala_pair = min_confidence_qala_pair if min_confidence_qala_pair is not None else min_confidence_qala
        self.overlap_threshold = overlap_threshold
        
        logger.info(f"QalaPipeline initialized: min_confidence_qala={min_confidence_qala}, overlap_threshold={overlap_threshold*100}%")
    
    def masks_to_geodataframe(
        self,
        masks: List,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        class_names: List[str],
        reference_geotransform: Tuple[float, ...],
        tile_offsets: List[Tuple[int, int]] = None,
        crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """
        Convert instance segmentation masks to GeoDataFrame with polygon geometries.
        
        Args:
            masks: List of mask arrays (polygon format (N, 2) or binary masks (H, W))
            confidences: Detection confidences (N,)
            class_ids: Class IDs (N,)
            class_names: List of class names
            reference_geotransform: Geotransform from clipped reference image
            tile_offsets: Optional list of (x_offset, y_offset) for each mask (if masks are in tile coordinates)
            crs: Target CRS
            
        Returns:
            GeoDataFrame with polygon geometries from masks
        """
        from shapely.geometry import Polygon
        
        if measure is None:
            raise ImportError("skimage.measure is required for binary mask conversion. Install with: pip install scikit-image")
        
        if len(masks) == 0:
            return gpd.GeoDataFrame(columns=['geometry', 'class', 'confidence'], crs=crs)
        
        geometries = []
        valid_confidences = []
        valid_class_ids = []
        valid_class_names = []
        
        for i, mask in enumerate(masks):
            mask = np.asarray(mask)
            
            # Handle polygon format (N, 2) - direct polygon coordinates
            if mask.ndim == 2 and mask.shape[1] == 2:
                # Polygon format: (N, 2) array of (x, y) coordinates
                if len(mask) < 3:  # Need at least 3 points for a polygon
                    continue
                
                # Apply tile offset if provided
                if tile_offsets and i < len(tile_offsets):
                    x_offset, y_offset = tile_offsets[i]
                    mask = mask.copy()
                    mask[:, 0] += x_offset
                    mask[:, 1] += y_offset
                
                # Transform polygon coordinates to geographic
                coords_geo = []
                for x, y in mask:
                    x_geo = reference_geotransform[0] + x * reference_geotransform[1] + y * reference_geotransform[2]
                    y_geo = reference_geotransform[3] + x * reference_geotransform[4] + y * reference_geotransform[5]
                    coords_geo.append((x_geo, y_geo))
                
                # Close polygon if not closed
                if coords_geo[0] != coords_geo[-1]:
                    coords_geo.append(coords_geo[0])
                
                try:
                    geom = Polygon(coords_geo)
                    if geom.is_valid and geom.area > 0:
                        geometries.append(geom)
                        valid_confidences.append(confidences[i] if i < len(confidences) else 0.0)
                        valid_class_ids.append(class_ids[i] if i < len(class_ids) else 0)
                        valid_class_names.append(class_names[i] if i < len(class_names) else 'qala')
                except:
                    continue
            
            # Handle binary mask format (H, W)
            elif mask.ndim == 2:
                # Binary mask: convert to polygon using contour detection
                try:
                    # Find contours
                    contours = measure.find_contours(mask, 0.5)
                    if len(contours) == 0:
                        continue
                    
                    # Use largest contour
                    largest_contour = max(contours, key=len)
                    if len(largest_contour) < 3:
                        continue
                    
                    # Apply tile offset if provided
                    if tile_offsets and i < len(tile_offsets):
                        x_offset, y_offset = tile_offsets[i]
                        largest_contour = largest_contour.copy()
                        largest_contour[:, 1] += x_offset  # x coordinate (column)
                        largest_contour[:, 0] += y_offset  # y coordinate (row)
                    
                    # Transform to geographic coordinates
                    coords_geo = []
                    for y, x in largest_contour:  # Note: contours are (row, col) = (y, x)
                        x_geo = reference_geotransform[0] + x * reference_geotransform[1] + y * reference_geotransform[2]
                        y_geo = reference_geotransform[3] + x * reference_geotransform[4] + y * reference_geotransform[5]
                        coords_geo.append((x_geo, y_geo))
                    
                    # Close polygon
                    if coords_geo[0] != coords_geo[-1]:
                        coords_geo.append(coords_geo[0])
                    
                    geom = Polygon(coords_geo)
                    if geom.is_valid and geom.area > 0:
                        geometries.append(geom)
                        valid_confidences.append(confidences[i] if i < len(confidences) else 0.0)
                        valid_class_ids.append(class_ids[i] if i < len(class_ids) else 0)
                        valid_class_names.append(class_names[i] if i < len(class_names) else 'qala')
                except Exception as e:
                    logger.debug(f"Failed to convert binary mask {i} to polygon: {e}")
                    continue
        
        if len(geometries) == 0:
            return gpd.GeoDataFrame(columns=['geometry', 'class', 'confidence'], crs=crs)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'geometry': geometries,
            'class': valid_class_names,
            'confidence': valid_confidences,
            'class_id': valid_class_ids
        }, crs=crs)
        
        logger.info(f"Created GeoDataFrame with {len(gdf)} mask geometries")
        
        return gdf
    
    def boxes_to_geodataframe(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        class_names: List[str],
        reference_geotransform: Tuple[float, ...],
        crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """
        Convert YOLO detection boxes to GeoDataFrame with bounding box geometries.
        
        Args:
            boxes: Detection boxes (N, 4) in xyxy format (pixel coordinates)
            confidences: Detection confidences (N,)
            class_ids: Class IDs (N,)
            class_names: List of class names
            reference_geotransform: Geotransform from clipped reference image
            crs: Target CRS
            
        Returns:
            GeoDataFrame with bbox geometries
        """
        from shapely.geometry import box
        
        if len(boxes) == 0:
            return gpd.GeoDataFrame(columns=['geometry', 'class', 'confidence'], crs=crs)
        
        # Convert pixel coords to geographic coords
        geometries = []
        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            
            # Transform corners to geographic coordinates
            x1_geo = reference_geotransform[0] + x1 * reference_geotransform[1] + y1 * reference_geotransform[2]
            y1_geo = reference_geotransform[3] + x1 * reference_geotransform[4] + y1 * reference_geotransform[5]
            x2_geo = reference_geotransform[0] + x2 * reference_geotransform[1] + y2 * reference_geotransform[2]
            y2_geo = reference_geotransform[3] + x2 * reference_geotransform[4] + y2 * reference_geotransform[5]
            
            # Create box geometry (minx, miny, maxx, maxy)
            geometries.append(box(
                min(x1_geo, x2_geo),
                min(y1_geo, y2_geo),
                max(x1_geo, x2_geo),
                max(y1_geo, y2_geo)
            ))
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'geometry': geometries,
            'class': class_names,
            'confidence': confidences,
            'class_id': class_ids
        }, crs=crs)
        
        logger.info(f"Created GeoDataFrame with {len(gdf)} detections")
        
        return gdf
    
    def dissolve_overlapping_masks(
        self,
        gdf: gpd.GeoDataFrame,
        class_filter: str,
        overlap_threshold: float = 0.9
    ) -> gpd.GeoDataFrame:
        """
        Dissolve overlapping mask geometries, keeping max confidence.
        Merges overlapping geometries by union, keeping the geometry with highest confidence.
        Only merges geometries that overlap by at least overlap_threshold (default 90%).
        
        Args:
            gdf: GeoDataFrame with polygon geometries (from masks)
            class_filter: Class name to filter (e.g., 'qala')
            overlap_threshold: Minimum overlap ratio to merge (0.0-1.0, default: 0.9)
            
        Returns:
            GeoDataFrame with dissolved overlapping masks
        """
        # Filter by class
        gdf_class = gdf[gdf['class'] == class_filter].copy()
        
        if len(gdf_class) == 0:
            logger.warning(f"No detections of class '{class_filter}'")
            return gdf_class
        
        # Reset index
        gdf_class = gdf_class.reset_index(drop=True)
        
        logger.info(f"Dissolving {len(gdf_class)} {class_filter} masks (overlap threshold: {overlap_threshold*100}%)")
        
        # Create spatial index
        gdf_class.sindex
        
        # Find overlapping geometries
        dissolved = []
        processed = set()
        
        for idx in range(len(gdf_class)):
            if idx in processed:
                continue
            
            row = gdf_class.iloc[idx]
            
            # Find all geometries that intersect with this one
            candidates = list(gdf_class.sindex.intersection(row.geometry.bounds))
            
            # Filter candidates by overlap ratio
            overlapping_indices = []
            for i in candidates:
                if i == idx:
                    continue
                    
                candidate_geom = gdf_class.iloc[i].geometry
                
                # Calculate overlap ratio (intersection area / smaller area)
                intersection = row.geometry.intersection(candidate_geom)
                intersection_area = intersection.area
                
                area1 = row.geometry.area
                area2 = candidate_geom.area
                smaller_area = min(area1, area2)
                
                if smaller_area > 0:
                    overlap_ratio = intersection_area / smaller_area
                    
                    # Only consider as overlapping if ratio exceeds threshold
                    if overlap_ratio >= overlap_threshold:
                        overlapping_indices.append(i)
            
            if len(overlapping_indices) == 0:
                # No overlaps above threshold, keep as is
                dissolved.append({
                    'geometry': row.geometry,
                    'class': row['class'],
                    'confidence': row['confidence'],
                    'class_id': row['class_id']
                })
                processed.add(idx)
            else:
                # Merge overlapping geometries
                overlapping_indices.append(idx)
                overlapping = gdf_class.iloc[overlapping_indices]
                
                # Union geometries (merge masks)
                merged_geom = overlapping.geometry.unary_union
                
                # Keep max confidence
                max_conf = overlapping['confidence'].max()
                max_conf_idx = overlapping['confidence'].idxmax()
                max_class_id = overlapping.loc[max_conf_idx, 'class_id']
                
                dissolved.append({
                    'geometry': merged_geom,  # Actual merged geometry, not envelope
                    'class': class_filter,
                    'confidence': max_conf,
                    'class_id': max_class_id
                })
                
                # Mark all as processed
                processed.update(overlapping_indices)
        
        result = gpd.GeoDataFrame(dissolved, crs=gdf.crs)
        logger.info(f"Dissolved to {len(result)} masks (from {len(gdf_class)})")
        logger.info(f"Merged {len(gdf_class) - len(result)} masks with >{overlap_threshold*100}% overlap")
        
        return result
    
    def dissolve_overlapping_bboxes(
        self,
        gdf: gpd.GeoDataFrame,
        class_filter: str,
        overlap_threshold: float = 0.9
    ) -> gpd.GeoDataFrame:
        """
        Dissolve overlapping bounding boxes, keeping max confidence.
        Only merges boxes that overlap by at least overlap_threshold (default 90%).
        
        Args:
            gdf: GeoDataFrame with bbox geometries
            class_filter: Class name to filter (e.g., 'qanat')
            overlap_threshold: Minimum overlap ratio to merge (0.0-1.0, default: 0.9)
            
        Returns:
            GeoDataFrame with dissolved overlapping boxes
        """
        # Filter by class
        gdf_class = gdf[gdf['class'] == class_filter].copy()
        
        if len(gdf_class) == 0:
            logger.warning(f"No detections of class '{class_filter}'")
            return gdf_class
        
        # CRITICAL: Reset index to ensure it's 0-based integer
        gdf_class = gdf_class.reset_index(drop=True)
        
        logger.info(f"Dissolving {len(gdf_class)} {class_filter} bboxes (overlap threshold: {overlap_threshold*100}%)")
        
        # Create spatial index
        gdf_class.sindex
        
        # Find overlapping boxes
        dissolved = []
        processed = set()
        
        for idx in range(len(gdf_class)):
            if idx in processed:
                continue
            
            row = gdf_class.iloc[idx]
            
            # Find all boxes that intersect with this one
            candidates = list(gdf_class.sindex.intersection(row.geometry.bounds))
            
            # Filter candidates by overlap ratio
            overlapping_indices = []
            for i in candidates:
                if i == idx:
                    continue
                    
                candidate_geom = gdf_class.iloc[i].geometry
                
                # Calculate overlap ratio
                intersection = row.geometry.intersection(candidate_geom)
                intersection_area = intersection.area
                
                # Calculate overlap as fraction of smaller box
                area1 = row.geometry.area
                area2 = candidate_geom.area
                smaller_area = min(area1, area2)
                
                if smaller_area > 0:
                    overlap_ratio = intersection_area / smaller_area
                    
                    # Only consider as overlapping if ratio exceeds threshold
                    if overlap_ratio >= overlap_threshold:
                        overlapping_indices.append(i)
            
            if len(overlapping_indices) == 0:
                # No overlaps above threshold, keep as is
                dissolved.append({
                    'geometry': row.geometry,
                    'class': row['class'],
                    'confidence': row['confidence'],
                    'class_id': row['class_id']
                })
                processed.add(idx)
            else:
                # Merge overlapping boxes
                overlapping_indices.append(idx)
                overlapping = gdf_class.iloc[overlapping_indices]
                
                # Union geometries
                merged_geom = overlapping.geometry.unary_union
                
                # Keep max confidence
                max_conf = overlapping['confidence'].max()
                max_conf_idx = overlapping['confidence'].idxmax()
                max_class_id = overlapping.loc[max_conf_idx, 'class_id']
                
                dissolved.append({
                    'geometry': merged_geom.envelope,  # Envelope for bbox
                    'class': class_filter,
                    'confidence': max_conf,
                    'class_id': max_class_id
                })
                
                # Mark all as processed
                processed.update(overlapping_indices)
        
        result = gpd.GeoDataFrame(dissolved, crs=gdf.crs)
        logger.info(f"Dissolved to {len(result)} bboxes (from {len(gdf_class)})")
        logger.info(f"Merged {len(gdf_class) - len(result)} boxes with >{overlap_threshold*100}% overlap")
        
        return result
    
    def spatial_join_qanat_pairs(
        self,
        qanat_gdf: gpd.GeoDataFrame,
        qanat_pair_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Perform spatial join between qanat and qanat_pair.
        Keep only qanats that intersect with qanat_pairs.
        
        Args:
            qanat_gdf: GeoDataFrame with qanat detections
            qanat_pair_gdf: GeoDataFrame with qanat_pair detections
            
        Returns:
            Filtered qanat GeoDataFrame
        """
        logger.info(f"Spatial join: {len(qanat_gdf)} qanats with {len(qanat_pair_gdf)} qanat_pairs")
        
        if len(qanat_pair_gdf) == 0:
            logger.warning("No qanat_pairs found, returning empty result")
            return qanat_gdf.iloc[0:0].copy()  # Empty with same structure
        
        # Spatial join: keep qanats that intersect with any qanat_pair
        joined = gpd.sjoin(
            qanat_gdf,
            qanat_pair_gdf[['geometry']],
            how='inner',
            predicate='intersects'
        )
        
        # Remove duplicate qanats (can match multiple qanat_pairs)
        joined = joined.drop_duplicates(subset=['geometry'])
        
        # Keep only original qanat columns
        result = joined[qanat_gdf.columns].copy()
        
        logger.info(f"After spatial join: {len(result)} qanats within qanat_pairs")
        
        return result
    
    def filter_by_confidence(
        self,
        gdf: gpd.GeoDataFrame,
        min_confidence: float
    ) -> gpd.GeoDataFrame:
        """
        Filter detections by minimum confidence.
        
        Args:
            gdf: GeoDataFrame with detections
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered GeoDataFrame
        """
        filtered = gdf[gdf['confidence'] >= min_confidence].copy()
        
        logger.info(f"Confidence filter: {len(gdf)} → {len(filtered)} (min={min_confidence})")
        
        return filtered
    
    def cluster_with_dbscan(
        self,
        gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Cluster geometries using DBSCAN on centroids.
        
        Args:
            gdf: GeoDataFrame with geometries
            
        Returns:
            GeoDataFrame with 'cluster' column
        """
        if len(gdf) < self.min_samples:
            logger.warning(f"Too few points ({len(gdf)}) for clustering (min={self.min_samples})")
            gdf['cluster'] = -1
            return gdf
        
        # Extract centroids
        centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry])
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(centroids)
        
        gdf['cluster'] = labels
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"DBSCAN clustering: {n_clusters} clusters, {n_noise} noise points")
        
        return gdf
    
    def remove_outliers(
        self,
        gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Remove outlier points (cluster == -1).
        
        Args:
            gdf: GeoDataFrame with 'cluster' column
            
        Returns:
            Filtered GeoDataFrame without outliers
        """
        filtered = gdf[gdf['cluster'] != -1].copy()
        
        n_removed = len(gdf) - len(filtered)
        logger.info(f"Removed {n_removed} outliers")
        
        return filtered
    
    def process_instance_segmentation(
        self,
        merged_detections: Dict,
        reference_geotransform: Tuple[float, ...],
        class_name: str = "qala",
        crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """
        Process instance segmentation detections with single class.
        
        Pipeline:
        1. Convert masks to GeoDataFrame with polygon geometries
        2. Filter by confidence threshold
        3. Dissolve overlapping masks (keep max confidence)
        
        Args:
            merged_detections: Dictionary with detection results (must include 'masks')
            reference_geotransform: Geotransform from clipped reference image
            class_name: Class name (default: "qala")
            crs: Target CRS
            
        Returns:
            Final GeoDataFrame with merged mask geometries
        """
        logger.info("=" * 60)
        logger.info("INSTANCE SEGMENTATION PROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # Check if masks are available
        if 'masks' not in merged_detections or len(merged_detections['masks']) == 0:
            logger.warning("No masks found in detections! Falling back to bounding boxes.")
            return self.process_detections(merged_detections, reference_geotransform, crs)
        
        # Step 1: Convert masks to GeoDataFrame
        logger.info("Step 1: Converting masks to GeoDataFrame")
        gdf = self.masks_to_geodataframe(
            merged_detections['masks'],
            merged_detections['confidences'],
            merged_detections['class_ids'],
            merged_detections['class_names'],
            reference_geotransform,
            tile_offsets=merged_detections.get('tile_offsets'),
            crs=crs
        )
        
        logger.info(f"Total mask geometries: {len(gdf)}")
        
        if len(gdf) == 0:
            logger.warning("No valid mask geometries found!")
            return gpd.GeoDataFrame(columns=['geometry', 'class', 'confidence'], crs=crs)
        
        # Step 2: Filter by confidence
        logger.info("\nStep 2: Filtering by confidence")
        gdf_filtered = self.filter_by_confidence(
            gdf,
            self.min_confidence_qala
        )
        
        if len(gdf_filtered) == 0:
            logger.warning("No masks after confidence filtering!")
            return gpd.GeoDataFrame(columns=['geometry', 'class', 'confidence'], crs=crs)
        
        # Step 3: Dissolve overlapping masks
        logger.info("\nStep 3: Dissolving overlapping masks")
        gdf_merged = self.dissolve_overlapping_masks(
            gdf_filtered,
            class_name,
            overlap_threshold=self.overlap_threshold
        )
        
        logger.info("=" * 60)
        logger.info(f"FINAL RESULT: {len(gdf_merged)} merged mask geometries")
        logger.info("=" * 60)
        
        return gdf_merged
    
    def process_detections(
        self,
        merged_detections: Dict,
        reference_geotransform: Tuple[float, ...],
        crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """
        Complete processing pipeline matching original notebook.
        
        Pipeline:
        1. Convert boxes to GeoDataFrame with bbox geometries
        2. Separate qanat and qanat_pair classes
        3. Dissolve overlapping qanat bboxes (keep max confidence)
        4. Keep qanat_pair bboxes as-is
        5. Filter by confidence thresholds
        6. Spatial join: keep qanats within qanat_pairs
        
        Args:
            merged_detections: Dictionary with detection results
            reference_geotransform: Geotransform from clipped reference image
            crs: Target CRS
            
        Returns:
            Final GeoDataFrame with filtered qanats
        """
        # Check if this is instance segmentation (has masks)
        if 'masks' in merged_detections and len(merged_detections['masks']) > 0:
            # Use instance segmentation pipeline
            return self.process_instance_segmentation(
                merged_detections,
                reference_geotransform,
                class_name=merged_detections.get('class_names', ['qala'])[0] if merged_detections.get('class_names') else 'qala',
                crs=crs
            )
        
        logger.info("=" * 60)
        logger.info("QANAT DETECTION PROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Convert to GeoDataFrame
        logger.info("Step 1: Converting detections to GeoDataFrame")
        gdf_all = self.boxes_to_geodataframe(
            merged_detections['boxes'],
            merged_detections['confidences'],
            merged_detections['class_ids'],
            merged_detections['class_names'],
            reference_geotransform,
            crs
        )
        
        logger.info(f"Total detections: {len(gdf_all)}")
        
        # Normalize class column to strings (model may return 0, "0", etc.)
        gdf_all['class'] = gdf_all['class'].astype(str)
        logger.info(f"Classes: {gdf_all['class'].value_counts().to_dict()}")
        
        # Detect single-class vs two-class pipeline
        unique_classes = gdf_all['class'].unique().tolist()
        has_qanat_pair = 'qanat_pair' in unique_classes or 'qala_pair' in unique_classes
        has_qanat = 'qanat' in unique_classes
        has_qala = 'qala' in unique_classes
        
        # Determine primary class name
        if has_qala:
            primary_class = 'qala'
        elif has_qanat:
            primary_class = 'qanat'
        else:
            # Use first class found (model may return "0", "1", etc. for single-class)
            primary_class = str(unique_classes[0]) if len(unique_classes) > 0 else 'qala'
        
        # Step 2: Handle single-class or two-class pipeline
        if not has_qanat_pair:
            # Single-class pipeline (qala only or model's single class)
            logger.info(f"\nStep 2: Single-class pipeline - processing '{primary_class}' detections")
            gdf_primary = gdf_all[gdf_all['class'] == primary_class].copy()
            
            # If no match (e.g. model returns "0" but we looked for "qala"), use ALL detections
            if len(gdf_primary) == 0 and len(gdf_all) > 0:
                logger.info(f"No '{primary_class}' class found. Using all {len(gdf_all)} detections as single class.")
                gdf_primary = gdf_all.copy()
                gdf_primary['class'] = 'qala'  # Normalize to qala for downstream
                primary_class = 'qala'
            
            if len(gdf_primary) == 0:
                logger.warning(f"No detections found!")
                return gpd.GeoDataFrame(columns=['geometry', 'class', 'confidence'], crs=crs)
            
            # Step 3: Dissolve overlapping geometries
            logger.info(f"\nStep 3: Dissolving overlapping {primary_class} geometries")
            gdf_dissolved = self.dissolve_overlapping_bboxes(
                gdf_primary,
                primary_class,
                overlap_threshold=self.overlap_threshold
            )
            
            # Step 4: Filter by confidence
            logger.info("\nStep 4: Filtering by confidence")
            gdf_filtered = self.filter_by_confidence(
                gdf_dissolved,
                self.min_confidence_qala
            )
            
            if len(gdf_filtered) == 0:
                logger.warning(f"No {primary_class} detections after confidence filtering!")
                return gpd.GeoDataFrame(columns=['geometry', 'class', 'confidence'], crs=crs)
            
            logger.info("=" * 60)
            logger.info(f"FINAL RESULT: {len(gdf_filtered)} {primary_class} detections")
            logger.info("=" * 60)
            
            return gdf_filtered
        
        else:
            # Two-class pipeline (qanat/qala + qanat_pair/qala_pair)
            logger.info("\nStep 2: Two-class pipeline - separating primary and pair classes")
            gdf_primary = gdf_all[gdf_all['class'] == primary_class].copy()
            pair_class = 'qanat_pair' if 'qanat_pair' in unique_classes else 'qala_pair'
            gdf_pair = gdf_all[gdf_all['class'] == pair_class].copy()
            
            logger.info(f"{primary_class}: {len(gdf_primary)}, {pair_class}: {len(gdf_pair)}")
            
            if len(gdf_primary) == 0:
                logger.warning(f"No {primary_class} detections found!")
                return gpd.GeoDataFrame(columns=['geometry', 'class', 'confidence'], crs=crs)
            
            if len(gdf_pair) == 0:
                logger.warning(f"No {pair_class} detections found!")
                return gpd.GeoDataFrame(columns=['geometry', 'class', 'confidence'], crs=crs)
            
            # Step 3: Dissolve overlapping primary class bboxes
            logger.info(f"\nStep 3: Dissolving overlapping {primary_class} bboxes")
            gdf_primary_dissolved = self.dissolve_overlapping_bboxes(
                gdf_primary,
                primary_class,
                overlap_threshold=self.overlap_threshold
            )
            
            # Step 4: Keep pair class as-is (no dissolving)
            logger.info(f"\nStep 4: Keeping {pair_class} bboxes as-is")
            
            # Step 5: Filter by confidence
            logger.info("\nStep 5: Filtering by confidence")
            gdf_primary_filtered = self.filter_by_confidence(
                gdf_primary_dissolved,
                self.min_confidence_qala
            )
            gdf_pair_filtered = self.filter_by_confidence(
                gdf_pair,
                self.min_confidence_qala_pair
            )
            
            if len(gdf_primary_filtered) == 0:
                logger.warning(f"No {primary_class} detections after confidence filtering!")
                return gpd.GeoDataFrame(columns=['geometry', 'class', 'confidence'], crs=crs)
            
            # Step 6: Spatial join
            logger.info(f"\nStep 6: Spatial join - keeping {primary_class} within {pair_class}")
            gdf_within = self.spatial_join_qanat_pairs(
                gdf_primary_filtered,
                gdf_pair_filtered
            )
            
            if len(gdf_within) == 0:
                logger.warning(f"No {primary_class} within {pair_class} after spatial join!")
                return gpd.GeoDataFrame(columns=['geometry', 'class', 'confidence'], crs=crs)
            
            logger.info("=" * 60)
            logger.info(f"FINAL RESULT: {len(gdf_within)} {primary_class} detections")
            logger.info("=" * 60)
            
            return gdf_within
    
    def export_results(
        self,
        gdf: gpd.GeoDataFrame,
        output_path: str,
        driver: str = "GPKG",
        include_centroids: bool = True
    ):
        """
        Export results to file with both bounding boxes and centroids.
        
        Args:
            gdf: GeoDataFrame to export (with bbox geometries)
            output_path: Output file path
            driver: GDAL driver name ('GPKG', 'GeoJSON', 'ESRI Shapefile')
            include_centroids: If True, adds centroid columns and creates separate centroids file
        """
        # Make a copy to avoid modifying original
        gdf_export = gdf.copy()
        
        if include_centroids:
            # Calculate centroids
            logger.info("Calculating centroids from bounding boxes")
            centroids = gdf_export.geometry.centroid
            
            # Add centroid coordinates as columns
            gdf_export['centroid_x'] = centroids.x
            gdf_export['centroid_y'] = centroids.y
            
            # Add centroid as WKT text (not geometry - to avoid multiple geometry columns issue)
            gdf_export['centroid_wkt'] = centroids.to_wkt()
            
            logger.info(f"Added centroid coordinates (centroid_x, centroid_y, centroid_wkt)")
        
        # Export with bbox as main geometry
        gdf_export.to_file(output_path, driver=driver)
        logger.info(f"Exported {len(gdf_export)} features to {output_path}")
        logger.info(f"  - Main geometry: bounding boxes")
        if include_centroids:
            logger.info(f"  - Centroid coords: centroid_x, centroid_y")
            logger.info(f"  - Centroid WKT: centroid_wkt")
        
        # Also export a separate centroids-only file
        if include_centroids:
            centroids_path = output_path.replace('.gpkg', '_centroids.gpkg')
            centroids_path = centroids_path.replace('.geojson', '_centroids.geojson')
            centroids_path = centroids_path.replace('.shp', '_centroids.shp')
            
            # Create centroids GeoDataFrame
            gdf_centroids = gdf.copy()
            gdf_centroids['geometry'] = gdf.geometry.centroid
            
            # Add bbox coordinates as attributes
            bbox_bounds = gdf.geometry.bounds
            gdf_centroids['bbox_minx'] = bbox_bounds['minx'].values
            gdf_centroids['bbox_miny'] = bbox_bounds['miny'].values
            gdf_centroids['bbox_maxx'] = bbox_bounds['maxx'].values
            gdf_centroids['bbox_maxy'] = bbox_bounds['maxy'].values
            
            # Add bbox as WKT
            gdf_centroids['bbox_wkt'] = gdf.geometry.to_wkt()
            
            gdf_centroids.to_file(centroids_path, driver=driver)
            logger.info(f"Exported centroids separately to {centroids_path}")
        
        return output_path