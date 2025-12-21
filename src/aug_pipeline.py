import os
import json
import numpy as np
import cv2
import albumentations as alb
from tqdm import tqdm

# Define augmentation pipeline for 711x400 images
transform = alb.Compose([
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),
    alb.VerticalFlip(p=0.5)
], bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))

# Count total images for progress bar
total_images = sum(
    len(os.listdir(os.path.join('data', t, 'images'))) 
    for t in ['train', 'test', 'val']
)
print(f"Processing {total_images} images x 60 augmentations = {total_images * 60} total outputs")

image_count = 0
for type in ['train','test','val']:
    images_list = os.listdir(os.path.join('data', type, 'images'))
    for image in tqdm(images_list, desc=f"Augmenting {type}", unit="img"):
        image_count += 1
        
        img = cv2.imread(os.path.join('data', type, 'images', image))
        orig_h = img.shape[0]
        orig_w = img.shape[1]
        
        # Resize to 711x400 while keeping aspect ratio
        # Scale so height becomes 400
        scale = 400 / orig_h
        new_w = int(orig_w * scale)
        new_h = 400
        img = cv2.resize(img, (new_w, new_h))
        
        # Center crop width to 711
        start_x = (new_w - 711) // 2
        if start_x < 0:
            start_x = 0
        img = img[:, start_x:start_x+711]
        
        # Pad if image is smaller than 711 wide
        if img.shape[1] < 711:
            pad_left = (711 - img.shape[1]) // 2
            pad_right = 711 - img.shape[1] - pad_left
            img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0,0,0])
            start_x = -pad_left  # Adjust for coordinate calculation
        
        coords = [0,0,0.00001,0.00001] # default coords (near 0) if label does not exist
        label_path = os.path.join('data', type, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as label_file:
                label = json.load(label_file)

            coords[0] = label['shapes'][0]['points'][0][0] # x1
            coords[1] = label['shapes'][0]['points'][0][1] # y1
            coords[2] = label['shapes'][0]['points'][1][0] # x2
            coords[3] = label['shapes'][0]['points'][1][1] # y2
            
            # Rescale and adjust for center crop
            coords[0] = (coords[0] * scale - start_x) / 711  # x1
            coords[1] = (coords[1] * scale) / 400            # y1
            coords[2] = (coords[2] * scale - start_x) / 711  # x2
            coords[3] = (coords[3] * scale) / 400            # y2
            # Clip to valid range [0, 1]
            coords = [max(0, min(1, c)) for c in coords]

        try: 
            for i in range(60):
                augmented = transform(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join('aug_data', type, 'images', f'{image.split(".")[0]}.{i}.jpg'), augmented['image'])

                aug_label_data = {}
                aug_label_data['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0: # no bounding box in the augmented image
                        aug_label_data['bbox'] = [0, 0, 0, 0]
                        aug_label_data['class'] = 0
                    else:
                        aug_label_data['bbox'] = augmented['bboxes'][0]
                        aug_label_data['class'] = 1
                else:
                    aug_label_data['bbox'] = [0, 0, 0, 0]
                    aug_label_data['class'] = 0

                with open(os.path.join('aug_data', type, 'labels', f'{image.split(".")[0]}.{i}.json'), 'w') as aug_label_file:
                    json.dump(aug_label_data, aug_label_file)
                    
        except Exception as e:
            print(f"Error processing {image}: {e}")

print(f"\nGenerated {image_count * 60} augmented images from {image_count} originals.")