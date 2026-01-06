"""
Convert LabelMe JSON format to simple training format.

LabelMe format:
{
  "shapes": [
    {"label": "face", "points": [[x1, y1], [x2, y2]]},
    {"label": "mouth_open", "points": [[x1, y1], [x2, y2]]}  # Optional
  ],
  "imageWidth": 1920,
  "imageHeight": 1080
}

Our format:
{
  "class": 1,
  "bbox": [x1, y1, x2, y2],  # Normalized 0-1
  "mouth_open": 0 or 1
}
"""

import json
import os
from pathlib import Path


def convert_labelme_to_simple(labelme_path: str) -> dict:
    """
    Convert a single LabelMe JSON file to our simple format.
    
    Args:
        labelme_path: Path to LabelMe JSON file
        
    Returns:
        Dict with 'class', 'bbox', and 'mouth_open' fields
    """
    with open(labelme_path, 'r') as f:
        labelme_data = json.load(f)
    
    # Get image dimensions
    img_width = labelme_data.get('imageWidth', 1920)
    img_height = labelme_data.get('imageHeight', 1080)
    
    # Find face box
    face_box = None
    mouth_detected = False
    
    for shape in labelme_data.get('shapes', []):
        label = shape.get('label', '').lower()
        points = shape.get('points', [])
        
        if len(points) != 2:
            continue
            
        # Extract box coordinates
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # Ensure x1 < x2 and y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        if label == 'face':
            # Normalize coordinates to 0-1 range
            face_box = [
                x1 / img_width,
                y1 / img_height,
                x2 / img_width,
                y2 / img_height
            ]
        elif 'mouth' in label:
            # If there's any box with "mouth" in the label, mouth is open
            mouth_detected = True
    
    if face_box is None:
        raise ValueError(f"No 'face' box found in {labelme_path}")
    
    # Round to 3 decimal places for cleaner JSON
    face_box = [round(coord, 3) for coord in face_box]
    
    return {
        "class": 1,
        "bbox": face_box,
        "mouth_open": 1 if mouth_detected else 0
    }


def convert_directory(input_dir: str, output_dir: str = None, overwrite: bool = False):
    """
    Convert all LabelMe JSON files in a directory.
    
    Args:
        input_dir: Directory containing LabelMe JSON files
        output_dir: Output directory (default: same as input_dir)
        overwrite: Whether to overwrite existing files
    """
    if output_dir is None:
        output_dir = input_dir
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_path.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    print(f"Output directory: {output_dir}")
    print()
    
    converted = 0
    skipped = 0
    errors = 0
    
    for json_file in json_files:
        output_file = output_path / json_file.name
        
        # Check if already in simple format
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # If it has 'class' and 'bbox', it's already converted
            if 'class' in data and 'bbox' in data:
                print(f"✓ Already converted: {json_file.name}")
                skipped += 1
                continue
                
        except Exception:
            pass
        
        # Skip if file exists and overwrite is False
        if output_file.exists() and not overwrite:
            print(f"⊘ Skipping (exists): {json_file.name}")
            skipped += 1
            continue
        
        try:
            # Convert
            simple_data = convert_labelme_to_simple(str(json_file))
            
            # Save
            with open(output_file, 'w') as f:
                json.dump(simple_data, f, indent=2)
            
            mouth_status = "OPEN" if simple_data['mouth_open'] == 1 else "closed"
            print(f"✓ Converted: {json_file.name} (mouth: {mouth_status})")
            converted += 1
            
        except Exception as e:
            print(f"✗ Error: {json_file.name} - {str(e)}")
            errors += 1
    
    print()
    print(f"Summary:")
    print(f"  Converted: {converted}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Total: {len(json_files)}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python convert_labelme.py <directory>")
        print("  python convert_labelme.py <directory> --overwrite")
        print()
        print("Example:")
        print("  python convert_labelme.py src/data/labels")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    overwrite = '--overwrite' in sys.argv
    
    convert_directory(input_dir, overwrite=overwrite)
