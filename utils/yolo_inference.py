"""
YOLO model inference utilities for qanat detection.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLODetector:
    """Handle YOLO model loading and inference."""
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 1024,
        device: Optional[str] = None
    ):
        """
        Initialize YOLODetector.
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            image_size: Image size for inference (default: 1024)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        self.model = self._load_model()
        
        logger.info(f"Initialized YOLODetector on {self.device}")
        logger.info(f"Model: {model_path}")
        logger.info(f"Conf threshold: {conf_threshold}, IOU threshold: {iou_threshold}")
        logger.info(f"Image size: {image_size}")
    
    def _load_model(self) -> YOLO:
        """Load YOLO model from path."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            logger.info(f"Successfully loaded YOLO model from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_single(
        self,
        image_path: str,
        save_results: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Run detection on a single image.
        
        Args:
            image_path: Path to input image
            save_results: Whether to save visualization
            output_dir: Directory to save results
            
        Returns:
            Dictionary with detection results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info(f"Running detection on {image_path}")
        
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False
        )[0]
        
        # Parse results
        detections = self._parse_results(results)
        
        # Save visualization if requested
        if save_results and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"{Path(image_path).stem}_detected.jpg"
            results.save(filename=str(output_path))
            logger.info(f"Saved detection visualization to {output_path}")
        
        return detections
    
    def detect_batch(
        self,
        image_paths: List[str],
        batch_size: int = 16,
        save_results: bool = False,
        output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Run detection on a batch of images.
        
        Args:
            image_paths: List of paths to input images
            batch_size: Batch size for inference
            save_results: Whether to save visualizations
            output_dir: Directory to save results
            
        Returns:
            List of detection dictionaries
        """
        logger.info(f"Running batch detection on {len(image_paths)} images")
        
        all_detections = []
        
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            
            # Run inference
            results = self.model.predict(
                source=batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                device=self.device,
                verbose=False
            )
            
            # Parse results for each image
            for result, image_path in zip(results, batch):
                detections = self._parse_results(result)
                detections['image_path'] = image_path
                all_detections.append(detections)
                
                # Save visualization if requested
                if save_results and output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_path = output_dir / f"{Path(image_path).stem}_detected.jpg"
                    result.save(filename=str(output_path))
        
        logger.info(f"Completed batch detection: {len(all_detections)} results")
        
        return all_detections
    
    def detect_tiles(
        self,
        tile_info: List[Dict],
        batch_size: int = 16,
        save_results: bool = False,
        output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Run detection on tiles with metadata preservation.
        
        Args:
            tile_info: List of tile information dictionaries
            batch_size: Batch size for inference
            save_results: Whether to save visualizations
            output_dir: Directory to save results
            
        Returns:
            List of detection dictionaries with tile metadata
        """
        tile_paths = [t['path'] for t in tile_info]
        
        # Run batch detection
        detections = self.detect_batch(
            tile_paths,
            batch_size=batch_size,
            save_results=save_results,
            output_dir=output_dir
        )
        
        # Add tile metadata to detections
        for detection, tile in zip(detections, tile_info):
            detection['tile_info'] = tile
        
        return detections
    
    def _parse_results(self, results) -> Dict:
        """
        Parse YOLO results into a structured dictionary.
        
        Args:
            results: YOLO results object
            
        Returns:
            Dictionary with parsed detections
        """
        boxes = results.boxes
        
        if boxes is None or len(boxes) == 0:
            return {
                'num_detections': 0,
                'boxes': np.array([]),
                'confidences': np.array([]),
                'class_ids': np.array([]),
                'class_names': []
            }
        
        # Extract box coordinates (xyxy format)
        box_coords = boxes.xyxy.cpu().numpy()
        
        # Extract confidences
        confidences = boxes.conf.cpu().numpy()
        
        # Extract class IDs
        class_ids = boxes.cls.cpu().numpy().astype(int)
        
        # Get class names
        class_names = [results.names[int(cls_id)] for cls_id in class_ids]
        
        detection = {
            'num_detections': len(boxes),
            'boxes': box_coords,
            'confidences': confidences,
            'class_ids': class_ids,
            'class_names': class_names
        }
        
        # Extract masks if available (instance segmentation)
        if results.masks is not None:
            # .xy gives list of polygon arrays (N, 2) in pixel coordinates
            detection['masks'] = [seg.cpu().numpy() if hasattr(seg, 'cpu') else np.asarray(seg)
                                  for seg in results.masks.xy]
        
        return detection
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_path': self.model_path,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'image_size': self.image_size,
            'class_names': self.model.names
        }
    
    def filter_by_confidence(
        self,
        detections: Dict,
        min_confidence: float
    ) -> Dict:
        """
        Filter detections by confidence threshold.
        
        Args:
            detections: Detection dictionary
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered detection dictionary
        """
        if detections['num_detections'] == 0:
            return detections
        
        mask = detections['confidences'] >= min_confidence
        
        filtered = {
            'num_detections': int(mask.sum()),
            'boxes': detections['boxes'][mask],
            'confidences': detections['confidences'][mask],
            'class_ids': detections['class_ids'][mask],
            'class_names': [name for name, m in zip(detections['class_names'], mask) if m]
        }
        
        # Preserve masks if present
        if 'masks' in detections and len(detections['masks']) > 0:
            filtered['masks'] = [seg for seg, m in zip(detections['masks'], mask) if m]
        
        return filtered
    
    def filter_by_class(
        self,
        detections: Dict,
        class_names: List[str]
    ) -> Dict:
        """
        Filter detections by class name.
        
        Args:
            detections: Detection dictionary
            class_names: List of class names to keep
            
        Returns:
            Filtered detection dictionary
        """
        if detections['num_detections'] == 0:
            return detections
        
        mask = np.array([name in class_names for name in detections['class_names']])
        
        filtered = {
            'num_detections': int(mask.sum()),
            'boxes': detections['boxes'][mask],
            'confidences': detections['confidences'][mask],
            'class_ids': detections['class_ids'][mask],
            'class_names': [name for name, m in zip(detections['class_names'], mask) if m]
        }
        
        # Preserve masks if present
        if 'masks' in detections and len(detections['masks']) > 0:
            filtered['masks'] = [seg for seg, m in zip(detections['masks'], mask) if m]
        
        return filtered
