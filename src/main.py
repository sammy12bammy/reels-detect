import warnings
warnings.filterwarnings('ignore', category=Warning)

import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt
import os
import random
import shutil
from tensorflow.keras.models import Model # base model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16 # image classification neural network


from get_data import ImageCapture
from aug_pipeline import AugmentationPipeline


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


#Step 1 - Capture data from webcam
# print("Starting data capture from webcam")
# capture = ImageCapture(images_path='data/images', camera_id=0)
# capture.capture(num_images=30, delay=0.5, show_preview=True)

# Step 1.5 - extra check
# plot_images()

# Step 2 - partion the data
# print("Partitioning data into train, val, test")
# move_all_data(total_images=120)

#Step 3 - Augmentation pipeline - see aug_pipeline.py for details
# print("Starting augmentation pipeline")
# pipeline = AugmentationPipeline(data_dir='data',output_dir='aug_data',target_width=711,target_height=400,augmentations_per_image=60)
# pipeline.run()

# Step 4 - put it into tensorflow dataset
def load_image(file):
    encoded = tf.io.read_file(file)
    img = tf.io.decode_jpeg(encoded)
    return img

aug_images = {}
for type in ['train', 'val', 'test']:
    type_images = tf.data.Dataset.list_files(f'aug_data/{type}/images/*.jpg', shuffle=False)
    type_images = type_images.map(load_image)
    type_images = type_images.map(lambda x: tf.image.resize(x, (120,120))) # compress for efficiency
    type_images = type_images.map(lambda x: x/255) # scale by 255
    aug_images[type] = type_images
print(aug_images.keys())
print(aug_images['test'].as_numpy_iterator().next())

# Step 5 - Labels
def load_labels(label_path: str):
    with open(label_path.numpy(), 'r', encoding='utf-8') as f:
        label = json.load(f)
    return [label['class']],label['bbox']

aug_labels = {}
for type in ['train', 'val', 'test']:
    type_labels = tf.data.Dataset.list_files(f'aug_data/{type}/labels/*.json', shuffle=False)
    type_labels = type_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16])) 
    aug_labels[type] = type_labels
print(aug_labels.keys())
aug_labels['train'].as_numpy_iterator().next()

print(f"lengths of aug images train: {len(aug_images['train'])}")
print(f"lengths of aug labels train: {len(aug_labels['train'])}")

# Step 6 - combine images and labels

aug_data = {}
for type in ['train', 'val', 'test']:
    data = tf.data.Dataset.zip((aug_images[type], aug_labels[type]))
    if (type == 'train'):
        data = data.shuffle(5000)
    elif (type == 'test'):
        data = data.shuffle(1000)
    else: # val
        data = data.shuffle(1000)
    data = data.batch(8)    
    data = data.prefetch(4)  # helps with bottle necks
    aug_data[type] = data
print("printing images and labels together")
print(f"Images : {aug_data['test'].as_numpy_iterator().next()[0]}")
print(f"Labels : {aug_data['test'].as_numpy_iterator().next()[1]}")

# Step 7 - Building Neural Network model
vgg = VGG16(include_top=False) # don't need the top classification layers
vgg.summary()
# sigmoid should be between 0 - 1 for both classification and regression
def build_model():
    input_layer = Input(shape=(120,120,3)) # shape of input images
    vgg = VGG16(include_top=False)(input_layer) # base model
    # 2 different predicition heads 
    # Classification model
    f1 = GlobalMaxPooling2D()(vgg) # reduce dimensions - condense
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)
    # Bounding Box model
    f2 = GlobalMaxPooling2D()(vgg) 
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    facetracker = Model(inputs=input_layer, outputs=[class2, regress2]) # classifcation and regression output
    return facetracker

# test
facetracker = build_model()
facetracker.summary()
tf.keras.utils.plot_model(facetracker, "initial model", show_shapes=True, show_layer_names=True, show_dtype=True)

# quick test run
# X, y = train.as_numpy_iterator().next() # x - images, y -labels
# print(X.shape)
# classes, coords = facetracker.predict(X)
# print(f"Classes: {classes}, Coords: {coords}")

