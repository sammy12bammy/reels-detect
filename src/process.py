# process data
import warnings
warnings.filterwarnings('ignore', category=Warning)

import tensorflow as tf
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt

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
plt.show()