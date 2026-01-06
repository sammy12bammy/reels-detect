import os
import json
import numpy as np
import cv2
import albumentations as alb
from tqdm import tqdm


class AugmentationPipeline:
    """Augmentation pipeline for face detection training data."""
    
    def __init__(self, 
                 data_dir: str = 'data',
                 output_dir: str = 'aug_data',
                 target_width: int = 711,
                 target_height: int = 400,
                 augmentations_per_image: int = 60):
        """
        Initialize the augmentation pipeline.
        
        Args:
            data_dir: Source directory containing train/test/val folders
            output_dir: Output directory for augmented data
            target_width: Target image width after resize/crop
            target_height: Target image height after resize/crop
            augmentations_per_image: Number of augmented versions per image
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.target_width = target_width
        self.target_height = target_height
        self.augmentations_per_image = augmentations_per_image
        
        # Define augmentation transform
        self.transform = alb.Compose([
            alb.HorizontalFlip(p=0.5),
            alb.RandomBrightnessContrast(p=0.2),
            alb.RandomGamma(p=0.2),
            alb.RGBShift(p=0.2),
            alb.VerticalFlip(p=0.5)
        ], bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))
    
    def _create_output_dirs(self):
        """Create output directory structure."""
        for split in ['train', 'test', 'val']:
            os.makedirs(os.path.join(self.output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, split, 'labels'), exist_ok=True)
    
    def _resize_and_crop(self, img):
        """Resize and center crop image to target dimensions."""
        orig_h, orig_w = img.shape[:2]
        
        # Scale so height becomes target_height
        scale = self.target_height / orig_h
        new_w = int(orig_w * scale)
        new_h = self.target_height
        img = cv2.resize(img, (new_w, new_h))
        
        # Center crop width to target_width
        start_x = (new_w - self.target_width) // 2
        if start_x < 0:
            start_x = 0
        img = img[:, start_x:start_x + self.target_width]
        
        # Pad if image is smaller than target width
        if img.shape[1] < self.target_width:
            pad_left = (self.target_width - img.shape[1]) // 2
            pad_right = self.target_width - img.shape[1] - pad_left
            img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, 
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])
            start_x = -pad_left
        
        return img, scale, start_x, orig_w, orig_h
    
    def _transform_coords(self, coords, scale, start_x):
        """Transform bounding box coordinates for resized/cropped image."""
        coords[0] = (coords[0] * scale - start_x) / self.target_width   # x1
        coords[1] = (coords[1] * scale) / self.target_height            # y1
        coords[2] = (coords[2] * scale - start_x) / self.target_width   # x2
        coords[3] = (coords[3] * scale) / self.target_height            # y2
        # Clip to valid range [0, 1]
        return [max(0, min(1, c)) for c in coords]
    
    def run(self):
        """Run the augmentation pipeline."""
        self._create_output_dirs()
        
        # Count total images
        total_images = sum(
            len(os.listdir(os.path.join(self.data_dir, t, 'images'))) 
            for t in ['train', 'test', 'val']
        )
        print(f"Processing {total_images} images x {self.augmentations_per_image} augmentations = {total_images * self.augmentations_per_image} total outputs")
        
        image_count = 0
        for split in ['train', 'test', 'val']:
            images_dir = os.path.join(self.data_dir, split, 'images')
            images_list = os.listdir(images_dir)
            
            for image_name in tqdm(images_list, desc=f"Augmenting {split}", unit="img"):
                image_count += 1
                self._process_image(split, image_name)
        
        print(f"\n✓ Generated {image_count * self.augmentations_per_image} augmented images from {image_count} originals.")
        return image_count * self.augmentations_per_image
    
    def _process_image(self, split: str, image_name: str):
        """
        Process a single image through the augmentation pipeline.
        
        Preserves all labels including mouth_open classification.
        """
        # Load image
        img_path = os.path.join(self.data_dir, split, 'images', image_name)
        img = cv2.imread(img_path)
        
        # Resize and crop (get original dimensions for denormalization)
        img, scale, start_x, orig_width, orig_height = self._resize_and_crop(img)
        
        # Load label if exists
        base_name = image_name.split(".")[0]
        label_path = os.path.join(self.data_dir, split, 'labels', f'{base_name}.json')
        
        coords = [0, 0, 0.00001, 0.00001]  # default coords if no label
        mouth_open = 0  # default: mouth closed
        has_label = os.path.exists(label_path)
        
        if has_label:
            with open(label_path, 'r') as f:
                label = json.load(f)
            
            # Check if label has 'class' field (simple format) or 'shapes' (LabelMe format)
            if 'class' in label:
                # Simple format: {"class": 1, "bbox": [x1, y1, x2, y2], "mouth_open": 0/1}
                # Skip if no face detected (class 0 or empty bbox)
                if label.get('class', 0) == 0 or not label.get('bbox'):
                    print(f"⊘ Skipping {os.path.basename(img_path)} - no face detected")
                    return  # Skip this image
                
                # Denormalize bbox coordinates (convert from 0-1 to pixel values)
                bbox = label['bbox']
                coords[0] = bbox[0] * orig_width   # x1
                coords[1] = bbox[1] * orig_height  # y1
                coords[2] = bbox[2] * orig_width   # x2
                coords[3] = bbox[3] * orig_height  # y2
                
                mouth_open = label.get('mouth_open', 0)
            else:
                # LabelMe format (legacy support)
                coords[0] = label['shapes'][0]['points'][0][0]  # x1
                coords[1] = label['shapes'][0]['points'][0][1]  # y1
                coords[2] = label['shapes'][0]['points'][1][0]  # x2
                coords[3] = label['shapes'][0]['points'][1][1]  # y2
                
                mouth_open = label.get('mouth_open', 0)
            
            coords = self._transform_coords(coords, scale, start_x)
        
        # Generate augmentations
        try:
            for i in range(self.augmentations_per_image):
                augmented = self.transform(image=img, bboxes=[coords], class_labels=['face'])
                
                # Save augmented image
                out_img_path = os.path.join(self.output_dir, split, 'images', f'{base_name}.{i}.jpg')
                cv2.imwrite(out_img_path, augmented['image'])
                
                # Save augmented label (with mouth_open preserved)
                aug_label_data = {'image': image_name}
                
                if has_label and len(augmented['bboxes']) > 0:
                    aug_label_data['bbox'] = augmented['bboxes'][0]
                    aug_label_data['class'] = 1
                else:
                    aug_label_data['bbox'] = [0, 0, 0, 0]
                    aug_label_data['class'] = 0
                
                # Preserve mouth_open label (NEW!)
                aug_label_data['mouth_open'] = mouth_open
                
                out_label_path = os.path.join(self.output_dir, split, 'labels', f'{base_name}.{i}.json')
                with open(out_label_path, 'w') as f:
                    json.dump(aug_label_data, f)
                    
        except Exception as e:
            print(f"Error processing {image_name}: {e}")


# Allow running directly for standalone use
if __name__ == "__main__":
    pipeline = AugmentationPipeline()
    pipeline.run()