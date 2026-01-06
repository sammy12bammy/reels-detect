#!/usr/bin/env python3
"""
Reset data structure - move all images and labels back to source directories.
Run this before trainer.py to start fresh.
"""
import os
import shutil
from pathlib import Path

def reset_data():
    """Move all images back to data/images and labels back to data/labels"""
    
    # Move images back to data/images
    for split in ['train', 'val', 'test']:
        img_dir = f'data/{split}/images'
        if os.path.exists(img_dir):
            for img_file in os.listdir(img_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(img_dir, img_file)
                    dst = os.path.join('data/images', img_file)
                    shutil.move(src, dst)
                    print(f"Moved {img_file} from {split} back to images/")
    
    # Move labels back to data/labels
    for split in ['train', 'val', 'test']:
        label_dir = f'data/{split}/labels'
        if os.path.exists(label_dir):
            for label_file in os.listdir(label_dir):
                if label_file.endswith('.json'):
                    src = os.path.join(label_dir, label_file)
                    dst = os.path.join('data/labels', label_file)
                    shutil.move(src, dst)
                    print(f"Moved {label_file} from {split} back to labels/")
    
    # Count final totals
    total_images = len([f for f in os.listdir('data/images') 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total_labels = len([f for f in os.listdir('data/labels') 
                       if f.endswith('.json')])
    
    print(f"\n✓ Reset complete!")
    print(f"  Total images in data/images/: {total_images}")
    print(f"  Total labels in data/labels/: {total_labels}")
    print(f"\nYou can now run trainer.py")

if __name__ == '__main__':
    reset_data()
