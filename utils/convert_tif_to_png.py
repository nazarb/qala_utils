"""
Convert TIF images to PNG format for YOLO training dataset.
Handles both single-channel and multi-channel TIF files.
Preserves directory structure and updates YOLO annotation files.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def convert_tif_to_png(tif_path, png_path, preserve_range=True):
    """
    Convert a single TIF file to PNG.
    
    Args:
        tif_path: Path to input TIF file
        png_path: Path to output PNG file
        preserve_range: If True, normalize to 8-bit range
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read TIF image (supports multi-channel and 16-bit)
        img = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print(f"❌ Failed to read: {tif_path}")
            return False
        
        # Handle different bit depths
        if img.dtype == np.uint16:
            # Normalize 16-bit to 8-bit
            img = (img / 256).astype(np.uint8)
        elif img.dtype == np.float32 or img.dtype == np.float64:
            # Normalize float to 8-bit
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        
        # Save as PNG
        cv2.imwrite(str(png_path), img)
        return True
        
    except Exception as e:
        print(f"❌ Error converting {tif_path}: {e}")
        return False


def update_annotation_file(txt_path, old_image_name, new_image_name):
    """
    Update image reference in annotation file if needed.
    (Usually YOLO annotations don't include image name, but just in case)
    
    Args:
        txt_path: Path to annotation .txt file
        old_image_name: Original image filename
        new_image_name: New image filename
    """
    # Most YOLO annotations don't need updating, but this is here for completeness
    pass


def convert_dataset_tif_to_png(input_dir, output_dir=None, recursive=True):
    """
    Convert all TIF images in a dataset directory to PNG.
    
    Args:
        input_dir: Input directory containing TIF images
        output_dir: Output directory (if None, uses input_dir)
        recursive: If True, process subdirectories
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_path = input_path
        in_place = True
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        in_place = False
    
    # Find all TIF files
    if recursive:
        tif_files = list(input_path.rglob("*.tif")) + list(input_path.rglob("*.tiff"))
    else:
        tif_files = list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff"))
    
    if not tif_files:
        print(f"⚠️  No TIF files found in {input_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Found {len(tif_files)} TIF files to convert")
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Mode: {'In-place' if in_place else 'Copy to new location'}")
    print(f"{'='*60}\n")
    
    success_count = 0
    fail_count = 0
    
    for tif_file in tqdm(tif_files, desc="Converting TIF to PNG"):
        # Determine output path
        if in_place:
            png_file = tif_file.with_suffix('.png')
        else:
            # Preserve relative directory structure
            rel_path = tif_file.relative_to(input_path)
            png_file = output_path / rel_path.with_suffix('.png')
            png_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert
        if convert_tif_to_png(tif_file, png_file):
            success_count += 1
            
            # If in-place, remove original TIF
            if in_place:
                try:
                    tif_file.unlink()
                except Exception as e:
                    print(f"⚠️  Couldn't remove {tif_file}: {e}")
            
            # Check for corresponding annotation file and copy if needed
            txt_file = tif_file.with_suffix('.txt')
            if txt_file.exists() and not in_place:
                new_txt_file = png_file.with_suffix('.txt')
                new_txt_file.write_text(txt_file.read_text())
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"✓ Success: {success_count}")
    print(f"✗ Failed: {fail_count}")
    print(f"{'='*60}\n")


def convert_yolo_dataset(dataset_yaml_path, output_dir=None):
    """
    Convert all TIF images in a YOLO dataset to PNG.
    Reads the dataset YAML and processes train/val/test splits.
    
    Args:
        dataset_yaml_path: Path to YOLO dataset YAML file
        output_dir: Optional output directory (None for in-place)
    """
    import yaml
    
    yaml_path = Path(dataset_yaml_path)
    
    with open(yaml_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Get dataset root path
    dataset_root = Path(dataset_config.get('path', yaml_path.parent)).resolve()
    
    print(f"\n{'='*60}")
    print(f"Processing YOLO dataset: {yaml_path.name}")
    print(f"Dataset root: {dataset_root}")
    print(f"{'='*60}\n")
    
    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        if split in dataset_config:
            split_path = dataset_root / dataset_config[split]
            
            if split_path.exists():
                print(f"\n📁 Processing {split} split...")
                
                if output_dir:
                    split_output = Path(output_dir) / dataset_config[split]
                else:
                    split_output = None
                
                convert_dataset_tif_to_png(
                    input_dir=split_path,
                    output_dir=split_output,
                    recursive=False
                )
            else:
                print(f"⚠️  {split} split not found: {split_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TIF images to PNG for YOLO datasets"
    )
    parser.add_argument(
        'input',
        help='Input directory or YAML file path'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output directory (default: in-place conversion)',
        default=None
    )
    parser.add_argument(
        '-r', '--recursive',
        help='Process subdirectories recursively',
        action='store_true',
        default=True
    )
    parser.add_argument(
        '--yaml',
        help='Treat input as YOLO dataset YAML file',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    if args.yaml or args.input.endswith('.yaml'):
        # Process as YOLO dataset
        convert_yolo_dataset(args.input, args.output)
    else:
        # Process as directory
        convert_dataset_tif_to_png(
            args.input,
            args.output,
            args.recursive
        )


if __name__ == "__main__":
    # Example usage (uncomment and modify as needed):
    
    # Option 1: Convert a single directory
    # convert_dataset_tif_to_png(
    #     input_dir="./datasets/qanats/train/images",
    #     output_dir=None,  # None for in-place conversion
    #     recursive=False
    # )
    
    # Option 2: Convert entire YOLO dataset using YAML
    # convert_yolo_dataset(
    #     dataset_yaml_path="./datasets/qanats_256_synt_G1_AFG1_pairs_single_4.yaml",
    #     output_dir=None  # None for in-place conversion
    # )
    
    # Option 3: Run from command line
    main()
