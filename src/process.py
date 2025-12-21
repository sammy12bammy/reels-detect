# process data
import warnings
warnings.filterwarnings('ignore', category=Warning)

import tensorflow as tf
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt

# param x - full file path
def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

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
images = tf.data.Dataset.list_files('data/images/*.jpg', shuffle=False)
images.as_numpy_iterator().next()
# apply the load_image function to each image
images = images.map(load_image)
images.as_numpy_iterator().next()

# visualize images - view raw img with matplotlib
img_gen = images.batch(4).as_numpy_iterator()
plot_imgs = img_gen.next()  # Get one batch of 4 images
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4):
    ax[idx].imshow(plot_imgs[idx])
plt.show()