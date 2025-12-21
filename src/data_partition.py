# process data
import warnings
warnings.filterwarnings('ignore', category=Warning)

import tensorflow as tf
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt
# moving images
import os
import random
import shutil
# augmentation
import albumentations as alb

# move from images to train
def move_all_data(total_images: int):
    # 70% into train - 84
    # 15% into val - 18
    # 15% into test - 18
    src_dir_ims = 'data/images'

    def move_images(src, dest, num):

        images = [
            f for f in os.listdir(src)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        selected = random.sample(images, num)

        for img in selected:
            src_path = os.path.join(src, img)
            dst_path = os.path.join(dest, img)
            shutil.move(src_path, dst_path)

        print(f"Moved {num} images to {dest}")

    move_images(src_dir_ims, 'data/train/images', total_images * 70 // 100)
    move_images(src_dir_ims, 'data/val/images', total_images * 15 // 100)
    move_images(src_dir_ims, 'data/test/images', total_images * 15 // 100)

    # move matching labels
    for folder in ['train', 'val', 'test']:
        for file in os.listdir(os.path.join('data', folder, 'images')):
            filename = file.split('.')[0]+'.json'
            exisiting_fp = os.path.join('data','labels', filename)
            if os.path.exists(exisiting_fp):
                new_fp = os.path.join('data', folder, 'labels', filename)
                os.replace(exisiting_fp, new_fp)
        print(f"Moved labels to {folder} folder")


def plot_images():
    # load images from data/images with wildcard and shuffle off
    images = tf.data.Dataset.list_files("data/images/*.jpg", shuffle=False)

    def load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        return img

    images = images.map(load_image)

    # get one batch of 4
    batch = next(iter(images.batch(4)))

    # plot on matplotlib visualize
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        ax[i].imshow(batch[i])
        # ax[i].axis("off")

    plt.tight_layout()
    plt.show()

# limit GPU mem growth
# avoid OOM errors by setting memory growth - good practice for tensor flow
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# I dont have a gpu - so this wont show anything
tf.config.list_physical_devices('GPU')
# display what the model is going to be trained on - cpu for my local
print(tf.config.list_physical_devices())







# test augmentation 

augmentor = alb.Compose([alb.RandomCrop(400, 600),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                    bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels'])
)
# load test image and annotation with OpenCV and JSON
img = cv2.imread(os.path.join('data','train','images', 'a961b80c-de00-11f0-b8da-129bd4b76ee8.jpg'))
# get original dimensions BEFORE resizing
orig_h = img.shape[0]
orig_w = img.shape[1]

# resize + center crop to 600x400 while keeping aspect ratio
# Resize so the SHORTER side fits (height to 400)
scale = 400 / orig_h
new_w = int(orig_w * scale)
new_h = int(orig_h * scale)
img = cv2.resize(img, (new_w, new_h))

# Center crop to 600x400
start_x = (new_w - 600) // 2
start_y = 0  # No vertical crop needed since height is exactly 400
img = img[start_y:start_y+400, start_x:start_x+600]

h = img.shape[0]
w = img.shape[1]
with open(os.path.join('data','train','labels', 'a961b80c-de00-11f0-b8da-129bd4b76ee8.json'), 'r') as f:
    label = json.load(f)
# print(label)
# extract cords
coords = [0,0,0,0]
coords[0] = label['shapes'][0]['points'][0][0] #x1
coords[1] = label['shapes'][0]['points'][0][1] #y1
coords[2] = label['shapes'][0]['points'][1][0] #x2
coords[3] = label['shapes'][0]['points'][1][1] #y2
print(f'Without rescale: {coords}')
# rescale and adjust for center crop
# First normalize to resized image, then adjust for crop offset
coords[0] = (coords[0] * scale - start_x) / 600  # x1
coords[1] = (coords[1] * scale - start_y) / 400  # y1
coords[2] = (coords[2] * scale - start_x) / 600  # x2
coords[3] = (coords[3] * scale - start_y) / 400  # y2
# Clip to valid range [0, 1] in case bbox is partially outside crop
coords = [max(0, min(1, c)) for c in coords]
print(f'With rescale: {coords}')
# run the augmented thing
# Augment image
augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
cv2.imwrite(os.path.join('aug_data', 'test', 'images', 'a961b80c-de00-11f0-b8da-129bd4b76ee8.jpg'), augmented['image'])
print(augmented.keys())

# Show image
cv2.rectangle(augmented['image'],
              tuple(np.multiply(augmented['bboxes'][0][:2], [600, 400]).astype(int)),
              tuple(np.multiply(augmented['bboxes'][0][2:], [600, 400]).astype(int)),
              (255, 0, 0), 2)

plt.imshow(augmented['image'])
plt.show()
