"""
Evaluation Module for Qanat Detection Results
Compares detected features with reference data and calculates performance metrics.
Updated to work with bbox and centroid files from current pipeline.
"""

import os
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class ResultsEvaluator:
    """Evaluate detection results by comparing with reference data."""
    
    def __init__(self, output_folder: str = "evaluation"):
        """Initialize evaluator.
        
        Args:
            output_folder: Base folder for evaluation outputs
        """
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Initialized evaluator: {output_folder}")
    
    def load_detected_features(
        self,
        detected_file: str,
        use_centroids: bool = True
    ) -> gpd.GeoDataFrame:
        """Load detected features, optionally using centroids file.
        
        Args:
            detected_file: Path to detection results GeoPackage
            use_centroids: If True, try to use centroids file for point-based evaluation
            
        Returns:
            GeoDataFrame with detected features
        """
        # Try to use centroids file if available
        if use_centroids:
            centroids_file = detected_file.replace('.gpkg', '_centroids.gpkg')
            if os.path.exists(centroids_file):
                logger.info(f"Using centroids file: {centroids_file}")
                return gpd.read_file(centroids_file)
            else:
                logger.info("Centroids file not found, using main file")
        
        gdf = gpd.read_file(detected_file)
        
        # If main file has bbox geometries, convert to centroids for point evaluation
        if gdf.geometry.type.iloc[0] == 'Polygon':
            logger.info("Converting bbox geometries to centroids for evaluation")
            gdf['geometry'] = gdf.geometry.centroid
        
        return gdf
    
    def create_buffers(
        self,
        input_file: str,
        output_file: str,
        distance: float,
        use_centroids: bool = True
    ) -> str:
        """Create buffer zones around features.
        
        Args:
            input_file: Path to input GeoPackage/Shapefile
            output_file: Path to output buffer file
            distance: Buffer distance (in degrees for EPSG:4326)
            use_centroids: If True, use centroids for detected features
            
        Returns:
            Path to output buffer file
            
        Example:
            >>> evaluator = ResultsEvaluator()
            >>> buff_path = evaluator.create_buffers("detected.gpkg", "detected_buff.gpkg", 0.00004)
        """
        try:
            logger.info(f"Creating buffers with distance {distance}...")
            
            # Load features (use centroids for detected features)
            if 'detected' in input_file.lower() and use_centroids:
                gdf = self.load_detected_features(input_file, use_centroids=True)
            else:
                gdf = gpd.read_file(input_file)
            
            logger.info(f"Loaded {len(gdf)} features")
            gdf['geometry'] = gdf.geometry.buffer(distance)
            
            gdf.to_file(output_file, driver='GPKG', engine='fiona')
            logger.info(f"✓ Buffers created: {output_file}")
            
            return output_file
        
        except Exception as e:
            logger.error(f"Failed to create buffers: {e}", exc_info=True)
            return None
    
    def spatial_join_detected_with_reference(
        self,
        detected_file: str,
        reference_buffer_file: str,
        output_file: str,
        use_centroids: bool = True
    ) -> str:
        """Spatial join: detected features with reference buffers.
        
        Identifies True Positives and False Positives.
        
        Args:
            detected_file: Path to detected features
            reference_buffer_file: Path to buffered reference data
            output_file: Path to output
            use_centroids: If True, use centroids for detected features
            
        Returns:
            Path to output file
        """
        try:
            logger.info("Spatial join: detected features with reference buffers...")
            
            detected_gdf = self.load_detected_features(detected_file, use_centroids=use_centroids)
            reference_buff_gdf = gpd.read_file(reference_buffer_file)
            
            # Spatial join
            joined = gpd.sjoin(
                detected_gdf,
                reference_buff_gdf,
                how='left',
                predicate='intersects',
                lsuffix='detected_',
                rsuffix='reference_'
            )
            
            # Mark True Positives (TP) and False Positives (FP)
            joined['TP'] = joined['index_reference_'].apply(
                lambda x: 1 if pd.notna(x) else 0
            ).astype('int64')
            
            joined['FP'] = joined['TP'].apply(lambda x: 0 if x == 1 else 1).astype('int64')
            
            # FN is 0 for detected features (not missing)
            joined['FN'] = 0
            joined['FN'] = joined['FN'].astype('int64')
            
            tp_count = int(joined['TP'].sum())
            fp_count = int(joined['FP'].sum())
            
            logger.info(f"TP: {tp_count}, FP: {fp_count}")
            logger.info(f"Data types - TP: {joined['TP'].dtype}, FP: {joined['FP'].dtype}, FN: {joined['FN'].dtype}")
            
            joined.to_file(output_file, driver='GPKG', engine='fiona')
            logger.info(f"✓ Spatial join completed: {output_file}")
            
            return output_file
        
        except Exception as e:
            logger.error(f"Spatial join failed: {e}", exc_info=True)
            return None
    
    def spatial_join_reference_with_detected(
        self,
        reference_file: str,
        detected_buffer_file: str,
        output_file: str
    ) -> str:
        """Spatial join: reference features with detected buffers.
        
        Identifies False Negatives.
        
        Args:
            reference_file: Path to reference features
            detected_buffer_file: Path to buffered detected data
            output_file: Path to output
            
        Returns:
            Path to output file
        """
        try:
            logger.info("Spatial join: reference features with detected buffers...")
            
            reference_gdf = gpd.read_file(reference_file)
            detected_buff_gdf = gpd.read_file(detected_buffer_file)
            
            # Spatial join
            joined = gpd.sjoin(
                reference_gdf,
                detected_buff_gdf,
                how='left',
                predicate='intersects',
                lsuffix='reference_',
                rsuffix='detected_'
            )
            
            # Get features with no intersection (False Negatives)
            fn_features = joined[pd.isna(joined['index_detected_'])]
            
            # Ensure consistent data types (int64)
            fn_features = fn_features.copy()
            fn_features['FN'] = 1
            fn_features['FN'] = fn_features['FN'].astype('int64')
            
            fn_features['TP'] = 0
            fn_features['TP'] = fn_features['TP'].astype('int64')
            
            fn_features['FP'] = 0
            fn_features['FP'] = fn_features['FP'].astype('int64')
            
            fn_count = len(fn_features)
            logger.info(f"FN: {fn_count}")
            logger.info(f"Data types - TP: {fn_features['TP'].dtype}, FP: {fn_features['FP'].dtype}, FN: {fn_features['FN'].dtype}")
            
            fn_features.to_file(output_file, driver='GPKG', engine='fiona')
            logger.info(f"✓ Reference-detected join completed: {output_file}")
            
            return output_file
        
        except Exception as e:
            logger.error(f"Reference-detected join failed: {e}", exc_info=True)
            return None
    
    def merge_evaluation_results(
        self,
        tp_fp_file: str,
        fn_file: str,
        output_file: str
    ) -> str:
        """Merge TP/FP and FN results into single file.
        
        Args:
            tp_fp_file: Path to TP/FP file
            fn_file: Path to FN file
            output_file: Path to merged output
            
        Returns:
            Path to merged file
        """
        try:
            logger.info("Merging evaluation results...")
            
            tp_fp_gdf = gpd.read_file(tp_fp_file)
            fn_gdf = gpd.read_file(fn_file)
            
            # Select common columns
            common_cols = [col for col in tp_fp_gdf.columns if col in fn_gdf.columns]
            
            tp_fp_gdf = tp_fp_gdf[common_cols]
            fn_gdf = fn_gdf[common_cols]
            
            # Concatenate
            merged_gdf = pd.concat([tp_fp_gdf, fn_gdf], ignore_index=True)
            
            logger.info(f"Merged {len(tp_fp_gdf)} + {len(fn_gdf)} = {len(merged_gdf)} features")
            
            merged_gdf.to_file(output_file, driver='GPKG', engine='fiona')
            logger.info(f"✓ Merged file saved: {output_file}")
            
            return output_file
        
        except Exception as e:
            logger.error(f"Merge failed: {e}", exc_info=True)
            return None
    
    def calculate_metrics(self, merged_evaluation_file: str) -> Dict:
        """Calculate precision, recall, and F1-score.
        
        Args:
            merged_evaluation_file: Path to merged evaluation results
            
        Returns:
            Dictionary with metrics
        """
        try:
            logger.info("Calculating performance metrics...")
            gdf = gpd.read_file(merged_evaluation_file)
            
            # Ensure int types
            gdf['TP'] = gdf['TP'].astype('int64')
            gdf['FP'] = gdf['FP'].astype('int64')
            gdf['FN'] = gdf['FN'].astype('int64')
            
            tp = int(gdf['TP'].sum())
            fp = int(gdf['FP'].sum())
            fn = int(gdf['FN'].sum())
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'total_detected': tp + fp,
                'total_reference': tp + fn
            }
            
            logger.info(f"✓ Metrics calculated")
            logger.info(f"   TP={tp}, FP={fp}, FN={fn}")
            logger.info(f"   Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}", exc_info=True)
            return {}
    
    def save_metrics_report(self, metrics: Dict, report_file: str) -> str:
        """Save metrics to text file.
        
        Args:
            metrics: Dictionary with metrics
            report_file: Path to output report file
            
        Returns:
            Path to report file
        """
        try:
            logger.info(f"Saving metrics report to {report_file}...")
            
            with open(report_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("QANAT DETECTION EVALUATION REPORT\n")
                f.write("="*80 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("CONFUSION MATRIX:\n")
                f.write("-"*40 + "\n")
                f.write(f"True Positives (TP):   {metrics['TP']}\n")
                f.write(f"False Positives (FP):  {metrics['FP']}\n")
                f.write(f"False Negatives (FN):  {metrics['FN']}\n\n")
                
                f.write("PERFORMANCE METRICS:\n")
                f.write("-"*40 + "\n")
                f.write(f"Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
                f.write(f"Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
                f.write(f"F1-Score:    {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)\n\n")
                
                f.write("SUMMARY:\n")
                f.write("-"*40 + "\n")
                f.write(f"Total Detected:   {metrics['total_detected']}\n")
                f.write(f"Total Reference:  {metrics['total_reference']}\n\n")
                
                f.write("="*80 + "\n")
            
            logger.info(f"✓ Report saved: {report_file}")
            return report_file
        
        except Exception as e:
            logger.error(f"Report save failed: {e}", exc_info=True)
            return None
    
    def print_metrics(self, metrics: Dict) -> None:
        """Print metrics to console.
        
        Args:
            metrics: Dictionary with metrics
        """
        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)
        print(f"\n📊 Confusion Matrix:")
        print(f"   • True Positives (TP):   {metrics['TP']}")
        print(f"   • False Positives (FP):  {metrics['FP']}")
        print(f"   • False Negatives (FN):  {metrics['FN']}")
        print(f"\n🎯 Performance:")
        print(f"   • Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"   • Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"   • F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"\n📈 Summary:")
        print(f"   • Total Detected:   {metrics['total_detected']}")
        print(f"   • Total Reference:  {metrics['total_reference']}")
        print("="*80 + "\n")
    
    def run_evaluation_pipeline(
        self,
        detected_file: str,
        reference_file: str,
        buffer_distance: float = 0.00004,
        save_report: bool = True,
        use_centroids: bool = True,
        save_comparison_files: bool = True,
        export_formats: list = ['gpkg']
    ) -> Dict:
        """Run complete evaluation pipeline.
        
        Args:
            detected_file: Path to detected features GeoPackage
            reference_file: Path to reference features
            buffer_distance: Buffer distance in degrees (default: 0.00004)
            save_report: Whether to save text report
            use_centroids: Whether to use centroids file for detected features
            save_comparison_files: Whether to save separate TP/FP/FN files
            export_formats: List of formats for comparison files ['gpkg', 'geojson', 'csv', 'shapefile']
            
        Returns:
            Dictionary with results, metrics, and file paths
            
        Example:
            >>> evaluator = ResultsEvaluator("evaluation")
            >>> results = evaluator.run_evaluation_pipeline(
            ...     detected_file="detected.gpkg",
            ...     reference_file="reference.gpkg",
            ...     buffer_distance=0.00004,
            ...     save_comparison_files=True,
            ...     export_formats=['gpkg', 'geojson', 'csv']
            ... )
            >>> print(f"TP file: {results['comparison_files']['gpkg']['true_positives']}")
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING EVALUATION PIPELINE")
        logger.info("="*80)
        
        try:
            # Step 1: Create buffers
            logger.info("\n[Step 1] Creating buffers...")
            detected_buff = os.path.join(self.output_folder, "detected_buffer.gpkg")
            reference_buff = os.path.join(self.output_folder, "reference_buffer.gpkg")
            
            self.create_buffers(detected_file, detected_buff, buffer_distance, use_centroids=use_centroids)
            self.create_buffers(reference_file, reference_buff, buffer_distance, use_centroids=False)
            
            # Step 2: Spatial join - detected with reference
            logger.info("\n[Step 2] Spatial join (detected vs reference)...")
            tp_fp_file = os.path.join(self.output_folder, "detected_TP_FP.gpkg")
            self.spatial_join_detected_with_reference(
                detected_file, reference_buff, tp_fp_file, use_centroids=use_centroids
            )
            
            # Step 3: Spatial join - reference with detected
            logger.info("\n[Step 3] Spatial join (reference vs detected)...")
            fn_file = os.path.join(self.output_folder, "reference_FN.gpkg")
            self.spatial_join_reference_with_detected(reference_file, detected_buff, fn_file)
            
            # Step 4: Merge results
            logger.info("\n[Step 4] Merging evaluation results...")
            merged_file = os.path.join(self.output_folder, "evaluation_results.gpkg")
            self.merge_evaluation_results(tp_fp_file, fn_file, merged_file)
            
            # Step 5: Calculate metrics
            logger.info("\n[Step 5] Calculating metrics...")
            metrics = self.calculate_metrics(merged_file)
            
            # Step 6: Save report
            report_file = None
            if save_report:
                logger.info("\n[Step 6] Saving report...")
                report_file = os.path.join(self.output_folder, "evaluation_report.txt")
                self.save_metrics_report(metrics, report_file)
            
            # Step 7: Save comparison files
            comparison_files = {}
            if save_comparison_files:
                logger.info("\n[Step 7] Saving comparison files...")
                comparison_files = self.export_comparison_to_formats(
                    merged_file,
                    output_folder=self.output_folder,
                    formats=export_formats
                )
            
            logger.info("\n" + "="*80)
            logger.info("EVALUATION COMPLETE!")
            logger.info("="*80 + "\n")
            
            return {
                'merged_file': merged_file,
                'metrics': metrics,
                'report_file': report_file,
                'buffer_distance': buffer_distance,
                'comparison_files': comparison_files,
                'tp_fp_file': tp_fp_file,
                'fn_file': fn_file
            }
        
        except Exception as e:
            logger.error(f"Evaluation pipeline failed: {e}", exc_info=True)
            return {}
    
    def save_comparison_results(
        self,
        merged_evaluation_file: str,
        output_folder: str = None
    ) -> Dict[str, str]:
        """Save comparison results as separate files for TP, FP, and FN.
        
        Args:
            merged_evaluation_file: Path to merged evaluation results GeoPackage
            output_folder: Output folder (uses self.output_folder if None)
            
        Returns:
            Dictionary with paths to saved files
            
        Example:
            >>> evaluator = ResultsEvaluator("evaluation")
            >>> files = evaluator.save_comparison_results("evaluation_results.gpkg")
            >>> print(f"TP file: {files['true_positives']}")
        """
        if output_folder is None:
            output_folder = self.output_folder
        
        try:
            logger.info("Saving comparison results to separate files...")
            
            # Load merged results
            gdf = gpd.read_file(merged_evaluation_file)
            
            # Separate by classification
            gdf_tp = gdf[gdf['TP'] == 1].copy()
            gdf_fp = gdf[gdf['FP'] == 1].copy()
            gdf_fn = gdf[gdf['FN'] == 1].copy()
            
            logger.info(f"TP: {len(gdf_tp)}, FP: {len(gdf_fp)}, FN: {len(gdf_fn)}")
            
            # Save files
            output_files = {}
            
            # True Positives
            if len(gdf_tp) > 0:
                tp_file = os.path.join(output_folder, "true_positives.gpkg")
                gdf_tp.to_file(tp_file, driver='GPKG')
                output_files['true_positives'] = tp_file
                logger.info(f"✓ True Positives saved: {tp_file}")
            else:
                logger.warning("No True Positives to save")
                output_files['true_positives'] = None
            
            # False Positives
            if len(gdf_fp) > 0:
                fp_file = os.path.join(output_folder, "false_positives.gpkg")
                gdf_fp.to_file(fp_file, driver='GPKG')
                output_files['false_positives'] = fp_file
                logger.info(f"✓ False Positives saved: {fp_file}")
            else:
                logger.warning("No False Positives to save")
                output_files['false_positives'] = None
            
            # False Negatives
            if len(gdf_fn) > 0:
                fn_file = os.path.join(output_folder, "false_negatives.gpkg")
                gdf_fn.to_file(fn_file, driver='GPKG')
                output_files['false_negatives'] = fn_file
                logger.info(f"✓ False Negatives saved: {fn_file}")
            else:
                logger.warning("No False Negatives to save")
                output_files['false_negatives'] = None
            
            # Also save combined file for reference
            combined_file = os.path.join(output_folder, "comparison_all.gpkg")
            gdf.to_file(combined_file, driver='GPKG')
            output_files['all_results'] = combined_file
            logger.info(f"✓ All results saved: {combined_file}")
            
            logger.info(f"✓ Comparison results saved to {output_folder}")
            
            return output_files
        
        except Exception as e:
            logger.error(f"Failed to save comparison results: {e}", exc_info=True)
            return {}
    
    def export_comparison_to_formats(
        self,
        merged_evaluation_file: str,
        output_folder: str = None,
        formats: list = ['gpkg', 'geojson', 'csv']
    ) -> Dict[str, Dict[str, str]]:
        """Export comparison results to multiple formats.
        
        Args:
            merged_evaluation_file: Path to merged evaluation results
            output_folder: Output folder (uses self.output_folder if None)
            formats: List of formats to export ['gpkg', 'geojson', 'csv', 'shapefile']
            
        Returns:
            Nested dictionary with format -> category -> filepath
            
        Example:
            >>> evaluator = ResultsEvaluator("evaluation")
            >>> exports = evaluator.export_comparison_to_formats(
            ...     "evaluation_results.gpkg",
            ...     formats=['gpkg', 'geojson', 'csv']
            ... )
            >>> print(f"TP GeoJSON: {exports['geojson']['true_positives']}")
        """
        if output_folder is None:
            output_folder = self.output_folder
        
        try:
            logger.info("\n" + "="*60)
            logger.info("EXPORTING COMPARISON RESULTS TO MULTIPLE FORMATS")
            logger.info("="*60)
            
            # Load merged results
            gdf = gpd.read_file(merged_evaluation_file)
            
            # Separate by classification
            gdf_tp = gdf[gdf['TP'] == 1].copy()
            gdf_fp = gdf[gdf['FP'] == 1].copy()
            gdf_fn = gdf[gdf['FN'] == 1].copy()
            
            categories = {
                'true_positives': gdf_tp,
                'false_positives': gdf_fp,
                'false_negatives': gdf_fn,
                'all_results': gdf
            }
            
            exports = {fmt: {} for fmt in formats}
            
            for category, gdf_cat in categories.items():
                if len(gdf_cat) == 0 and category != 'all_results':
                    logger.warning(f"No {category} to export")
                    for fmt in formats:
                        exports[fmt][category] = None
                    continue
                
                logger.info(f"\nExporting {category} ({len(gdf_cat)} features)...")
                
                # GeoPackage
                if 'gpkg' in formats:
                    gpkg_file = os.path.join(output_folder, f"{category}.gpkg")
                    gdf_cat.to_file(gpkg_file, driver='GPKG')
                    exports['gpkg'][category] = gpkg_file
                    logger.info(f"  ✓ GPKG: {gpkg_file}")
                
                # GeoJSON
                if 'geojson' in formats:
                    geojson_file = os.path.join(output_folder, f"{category}.geojson")
                    # Drop WKT columns for GeoJSON
                    gdf_export = gdf_cat.drop(columns=[col for col in gdf_cat.columns if col.endswith('_wkt')], errors='ignore')
                    gdf_export.to_file(geojson_file, driver='GeoJSON')
                    exports['geojson'][category] = geojson_file
                    logger.info(f"  ✓ GeoJSON: {geojson_file}")
                
                # Shapefile
                if 'shapefile' in formats:
                    shp_file = os.path.join(output_folder, f"{category}.shp")
                    # Drop WKT columns and truncate column names for shapefile
                    gdf_export = gdf_cat.drop(columns=[col for col in gdf_cat.columns if col.endswith('_wkt')], errors='ignore')
                    gdf_export.to_file(shp_file, driver='ESRI Shapefile')
                    exports['shapefile'][category] = shp_file
                    logger.info(f"  ✓ Shapefile: {shp_file}")
                
                # CSV
                if 'csv' in formats:
                    csv_file = os.path.join(output_folder, f"{category}.csv")
                    df = gdf_cat.copy()
                    
                    # Add coordinates
                    centroids = df.geometry.centroid
                    df['longitude'] = centroids.x
                    df['latitude'] = centroids.y
                    
                    # Select columns (exclude geometry and WKT)
                    cols = ['longitude', 'latitude']
                    for col in df.columns:
                        if col not in ['geometry', 'longitude', 'latitude'] and not col.endswith('_wkt'):
                            cols.append(col)
                    
                    df[cols].to_csv(csv_file, index=False)
                    exports['csv'][category] = csv_file
                    logger.info(f"  ✓ CSV: {csv_file}")
            
            logger.info("\n" + "="*60)
            logger.info("EXPORT COMPLETE!")
            logger.info("="*60)
            
            # Print summary
            logger.info("\nExported files:")
            for fmt in formats:
                logger.info(f"\n{fmt.upper()}:")
                for category, filepath in exports[fmt].items():
                    if filepath:
                        logger.info(f"  • {category}: {filepath}")
            
            return exports
        
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            return {}
            
            """detected_file: Path to detected features GeoPackage
            reference_file: Path to reference features
            buffer_distance: Buffer distance in degrees (default: 0.00004)
            save_report: Whether to save text report
            use_centroids: Whether to use centroids file for detected features
            
        Returns:
            Dictionary with results and metrics
            
        Example:
            >>> evaluator = ResultsEvaluator("evaluation")
            >>> results = evaluator.run_evaluation_pipeline(
            ...     detected_file="detected.gpkg",
            ...     reference_file="reference.gpkg",
            ...     buffer_distance=0.00004
            ... )
            >>> evaluator.print_metrics(results['metrics'])"""

        logger.info("\n" + "="*80)
        logger.info("STARTING EVALUATION PIPELINE")
        logger.info("="*80)
        
        try:
            # Step 1: Create buffers
            logger.info("\n[Step 1] Creating buffers...")
            detected_buff = os.path.join(self.output_folder, "detected_buffer.gpkg")
            reference_buff = os.path.join(self.output_folder, "reference_buffer.gpkg")
            
            self.create_buffers(detected_file, detected_buff, buffer_distance, use_centroids=use_centroids)
            self.create_buffers(reference_file, reference_buff, buffer_distance, use_centroids=False)
            
            # Step 2: Spatial join - detected with reference
            logger.info("\n[Step 2] Spatial join (detected vs reference)...")
            tp_fp_file = os.path.join(self.output_folder, "detected_TP_FP.gpkg")
            self.spatial_join_detected_with_reference(
                detected_file, reference_buff, tp_fp_file, use_centroids=use_centroids
            )
            
            # Step 3: Spatial join - reference with detected
            logger.info("\n[Step 3] Spatial join (reference vs detected)...")
            fn_file = os.path.join(self.output_folder, "reference_FN.gpkg")
            self.spatial_join_reference_with_detected(reference_file, detected_buff, fn_file)
            
            # Step 4: Merge results
            logger.info("\n[Step 4] Merging evaluation results...")
            merged_file = os.path.join(self.output_folder, "evaluation_results.gpkg")
            self.merge_evaluation_results(tp_fp_file, fn_file, merged_file)
            
            # Step 5: Calculate metrics
            logger.info("\n[Step 5] Calculating metrics...")
            metrics = self.calculate_metrics(merged_file)
            
            # Step 6: Save report
            if save_report:
                logger.info("\n[Step 6] Saving report...")
                report_file = os.path.join(self.output_folder, "evaluation_report.txt")
                self.save_metrics_report(metrics, report_file)
            else:
                report_file = None
            
            logger.info("\n" + "="*80)
            logger.info("EVALUATION COMPLETE!")
            logger.info("="*80 + "\n")
            
            return {
                'merged_file': merged_file,
                'metrics': metrics,
                'report_file': report_file,
                'buffer_distance': buffer_distance,
            }
        
        except Exception as e:
            logger.error(f"Evaluation pipeline failed: {e}", exc_info=True)
            return {}


class ComparisonVisualizer:
    """Visualize comparison of detected vs reference features."""
    
    def __init__(self):
        """Initialize visualizer."""
        logger.info("Initialized comparison visualizer")
    
    def create_comparison_map(
        self,
        merged_evaluation_file: str,
        center: Tuple[float, float] = None,
        zoom: int = 12,
        raster_file: str = None,
        basemap: str = "Esri.WorldImagery"
    ):
        """Create interactive map showing TP, FP, and FN.
        
        Args:
            merged_evaluation_file: Path to merged evaluation results GeoPackage
            center: Map center as (latitude, longitude), auto-detected if None
            zoom: Initial zoom level
            raster_file: Optional raster file to overlay
            basemap: Basemap name
            
        Returns:
            leafmap.Map object
            
        Example:
            >>> visualizer = ComparisonVisualizer()
            >>> m = visualizer.create_comparison_map(
            ...     "evaluation_results.gpkg",
            ...     center=(31.528, 65.247),
            ...     zoom=14,
            ...     raster_file="image.tif"
            ... )
            >>> display(m)
        """
        try:
            try:
                import leafmap
            except ImportError:
                logger.error("leafmap not installed. Install with: pip install leafmap")
                return None
            
            logger.info("\n[VISUALIZATION] Creating comparison map...")
            
            # Load evaluation results
            logger.info(f"Loading evaluation results from {merged_evaluation_file}...")
            gdf = gpd.read_file(merged_evaluation_file)
            logger.info(f"Loaded {len(gdf)} features")
            
            # Ensure WGS84
            if gdf.crs != "EPSG:4326":
                logger.info("Reprojecting to EPSG:4326...")
                gdf = gdf.to_crs("EPSG:4326")
            
            # Auto-detect center
            if center is None:
                bounds = gdf.total_bounds
                center = (
                    (bounds[1] + bounds[3]) / 2,  # latitude
                    (bounds[0] + bounds[2]) / 2   # longitude
                )
                logger.info(f"Auto-detected center: {center}")
            
            # Create map
            logger.info(f"Creating map at {center}, zoom={zoom}...")
            m = leafmap.Map(center=center, zoom=zoom, height="800px", basemap=basemap)
            
            # Add raster if provided
            if raster_file and os.path.exists(raster_file):
                try:
                    logger.info(f"Adding raster: {raster_file}")
                    m.add_raster(raster_file, layer_name="Original Image", crs="EPSG:4326")
                except Exception as e:
                    logger.warning(f"Could not add raster: {e}")
            
            # Separate by TP, FP, FN
            logger.info("Separating by classification...")
            gdf_tp = gdf[gdf['TP'] == 1].copy()
            gdf_fp = gdf[gdf['FP'] == 1].copy()
            gdf_fn = gdf[gdf['FN'] == 1].copy()
            
            logger.info(f"TP: {len(gdf_tp)}, FP: {len(gdf_fp)}, FN: {len(gdf_fn)}")
            
            # Add layers
            logger.info("Adding layers to map...")
            
            if len(gdf_tp) > 0:
                centroids_tp = gdf_tp.geometry.centroid
                gdf_tp["longitude"] = centroids_tp.x
                gdf_tp["latitude"] = centroids_tp.y
                m.add_circle_markers_from_xy(
                    gdf_tp,
                    x="longitude",
                    y="latitude",
                    radius=4,
                    color="#0dff0d",
                    fill_color="#0dff0d",
                    layer_name=f"True Positives ({len(gdf_tp)})",
                    stroke=False,
                    fill_opacity=0.7
                )
            
            if len(gdf_fp) > 0:
                centroids_fp = gdf_fp.geometry.centroid
                gdf_fp["longitude"] = centroids_fp.x
                gdf_fp["latitude"] = centroids_fp.y
                m.add_circle_markers_from_xy(
                    gdf_fp,
                    x="longitude",
                    y="latitude",
                    radius=4,
                    color="#e5350e",
                    fill_color="#e5350e",
                    layer_name=f"False Positives ({len(gdf_fp)})",
                    stroke=False,
                    fill_opacity=0.7
                )
            
            if len(gdf_fn) > 0:
                centroids_fn = gdf_fn.geometry.centroid
                gdf_fn["longitude"] = centroids_fn.x
                gdf_fn["latitude"] = centroids_fn.y
                m.add_circle_markers_from_xy(
                    gdf_fn,
                    x="longitude",
                    y="latitude",
                    radius=4,
                    color="#729bff",
                    fill_color="#729bff",
                    layer_name=f"False Negatives ({len(gdf_fn)})",
                    stroke=False,
                    fill_opacity=0.7
                )
            
            # Add legend
            legend_dict = {
                "True Positives (Correct)": "#0dff0d",
                "False Positives (Over-detected)": "#e5350e",
                "False Negatives (Missed)": "#729bff"
            }
            m.add_legend(title="Evaluation Results", legend_dict=legend_dict)
            
            # Add layer control
            m.add_layer_control()
            
            logger.info("✓ Map created successfully!")
            return m
        
        except Exception as e:
            logger.error(f"Visualization failed: {e}", exc_info=True)
            return None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)