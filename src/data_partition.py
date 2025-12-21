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

# limit GPU mem growth
# avoid OOM errors by setting memory growth - good practice for tensor flow
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# I dont have a gpu - so this wont show anything
tf.config.list_physical_devices('GPU')
# display what the model is going to be trained on - cpu for my local
print(tf.config.list_physical_devices())

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

# plt.tight_layout()
# plt.show()

# move from images to train
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

move_images(src_dir_ims, 'data/train/images', 84)
move_images(src_dir_ims, 'data/val/images', 18)
move_images(src_dir_ims, 'data/test/images', 18)

# move matching labels
for folder in ['train', 'val', 'test']:
    for file in os.listdir(os.path.join('data', folder, 'images')):
        filename = file.split('.')[0]+'.json'
        exisiting_fp = os.path.join('data','labels', filename)
        if os.path.exists(exisiting_fp):
            new_fp = os.path.join('data', folder, 'labels', filename)
            os.replace(exisiting_fp, new_fp)
    print(f"Moved labels to {folder} folder")