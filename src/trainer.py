import warnings
warnings.filterwarnings('ignore', category=Warning)

import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt
import os
import random
import shutil
import cv2
from tensorflow.keras.models import Model # base model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16 # image classification neural network

# custom classes 
from get_data import ImageCapture
from aug_pipeline import AugmentationPipeline
from train_pipeline import FaceTracker


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
# This should be run n times because it captures only 30 images at a time
# print("Starting data capture from webcam")
inp = int(input("Enter the amount of session you would like to capture (each session captures 30 images): "))
for i in range(inp):
    print(f"Starting session {i+1} of {inp}")
    capture = ImageCapture(images_path='data/images', camera_id=0)
    capture.capture(num_images=30, delay=0.5, show_preview=True)

# Step 1.5 - extra check
# plot_images()

# Wait for user
inp = input("Press 'y' to continue to data partitioning: ")
if inp.lower() == 'y':
    inp = input("Press 'y' to continue to data partitioning: ")
    # Step 2 - partion the data

    print("Partitioning data into train, val, test")
    move_all_data(total_images=90)

#Step 3 - Augmentation pipeline - see aug_pipeline.py for details
inp = input("Press 'y' to continue to data augmentation: ")
while inp.lower() != 'y':
    inp = input("Press 'y' to continue to data augmentation: ")

print("Starting augmentation pipeline")
pipeline = AugmentationPipeline(data_dir='data', output_dir='aug_data', target_width=711, target_height=400, augmentations_per_image=60)
pipeline.run()
print("Augmentation complete!")
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
    """
    Load labels from JSON file.
    
    Returns:
        face_class: [1] if face present, [0] otherwise
        bbox: [x1, y1, x2, y2] normalized face bounding box coordinates
        mouth_open: [1] if mouth is open, [0] if closed
    """
    with open(label_path.numpy(), 'r', encoding='utf-8') as f:
        label = json.load(f)
    
    face_class = label['class']
    bbox = label['bbox']
    mouth_open = label.get('mouth_open', 0)  # Default to 0 (closed) if not labeled yet
    
    return [face_class], bbox, [mouth_open]

def set_label_shapes(face_class, bbox, mouth_open):
    """Set explicit shapes after tf.py_function (which loses shape info)."""
    face_class.set_shape([1])
    bbox.set_shape([4])
    mouth_open.set_shape([1])  # New: mouth open classification
    return face_class, bbox, mouth_open

aug_labels = {}
for type in ['train', 'val', 'test']:
    type_labels = tf.data.Dataset.list_files(f'aug_data/{type}/labels/*.json', shuffle=False)
    type_labels = type_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16, tf.uint8]))  # 3 outputs now
    type_labels = type_labels.map(set_label_shapes)  # Fix: set explicit shapes
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

print("Images have been augmented and combined with labels into tensorflow datasets")
inp = input("Press 'y' to continue to building the model")
while inp.lower() != 'y':
    inp = input("Press 'y' to continue to building the model")


# Building Neural Network model
vgg = VGG16(include_top=False) # don't need the top classification layers
vgg.summary()

def build_model():
    """
    Build 3-head face tracking model with VGG16 base.
    
    Architecture:
        - Shared VGG16 feature extractor (pretrained on ImageNet)
        - Head 1: Face classification (binary) - is there a face?
        - Head 2: Face bounding box regression - where is the face?
        - Head 3: Mouth open classification (binary) - is mouth open?
    
    Returns:
        Model with 3 outputs: [face_class, face_bbox, mouth_open]
    """
    input_layer = Input(shape=(120,120,3)) # shape of input images
    vgg = VGG16(include_top=False)(input_layer) # base model - shared feature extractor
    
    # Head 1: Face Classification
    f1 = GlobalMaxPooling2D()(vgg) # reduce dimensions - condense
    class1 = Dense(2048, activation='relu')(f1)
    face_class = Dense(1, activation='sigmoid', name='face_class')(class1)  # Binary: face present?
    
    # Head 2: Face Bounding Box Regression
    f2 = GlobalMaxPooling2D()(vgg) 
    regress1 = Dense(2048, activation='relu')(f2)
    face_bbox = Dense(4, activation='sigmoid', name='face_bbox')(regress1)  # 4 coords: [x1, y1, x2, y2]
    
    # Head 3: Mouth Open Classification (NEW!)
    f3 = GlobalMaxPooling2D()(vgg)
    mouth1 = Dense(2048, activation='relu')(f3)
    mouth_open = Dense(1, activation='sigmoid', name='mouth_open')(mouth1)  # Binary: mouth open?

    facetracker = Model(inputs=input_layer, outputs=[face_class, face_bbox, mouth_open])
    return facetracker

# test
facetracker = build_model()
facetracker.summary()
tf.keras.utils.plot_model(facetracker, "initial model", show_shapes=True, show_layer_names=True, show_dtype=True)

# quick test run
# Predicting without training
train = aug_data['train']

x, y = train.as_numpy_iterator().next()
print(x.shape)
print(f"Labels shape: face_class={y[0].shape}, bbox={y[1].shape}, mouth_open={y[2].shape}")

face_class_pred, face_bbox_pred, mouth_open_pred = facetracker.predict(x)
print(f"Face class predictions: {face_class_pred}")
print(f"Face bbox predictions: {face_bbox_pred}")
print(f"Mouth open predictions: {mouth_open_pred}")


# Step 8 Define Losses and Optimizers

# Define Optimizer and LR
batches_per_epoch = len(train)
lr_decay = (1./0.75 -1)/batches_per_epoch # how much learning rate is going to drop
opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)
# Create localization loss and classification loss
def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))

    h_true = y_true[:,3] - y_true[:,1] # height of box
    w_true = y_true[:,2] - y_true[:,0] # width of box
    h_pred = yhat[:,3] - yhat[:,1] # predicted height
    w_pred = yhat[:,2] - yhat[:,0] # predicted width

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))
    return delta_coord + delta_size

# Define losses for all 3 heads
classloss = tf.keras.losses.BinaryCrossentropy()  # For face classification
regressloss = localization_loss  # For bounding box regression
mouth_classloss = tf.keras.losses.BinaryCrossentropy()  # For mouth open classification

# Test losses on a sample
print(f'Face classification loss: {classloss(y[0], face_class_pred)}')
print(f'Bounding box loss: {regressloss(y[1], face_bbox_pred)}')
print(f'Mouth open loss: {mouth_classloss(y[2], mouth_open_pred)}')

# Step 9 - Train Model
model = FaceTracker(facetracker)
model.compile(opt, classloss, regressloss, mouth_classloss)

# logs
logdir='logs'
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir) # review model performance

# train for 10 epochs - one epoch is one complete pass of the entire training dataset
hist = model.fit(aug_data['train'].take(100), epochs=10, 
                 validation_data=aug_data['val'], callbacks=[tensorboard_cb])

# Step 10 - Test and visualize results

# show training data + making predictions
test_data = train.as_numpy_iterator()
test_sample = test_data.next()
prediction = facetracker.predict(test_sample[0])
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4): 
    sample_image = test_sample[0][idx].copy()  # Make a writable copy
    sample_coords = prediction[1][idx]
    
    # draw rectangle if confidence is greater than 0.9
    if prediction[0][idx] > 0.9:
        cv2.rectangle(sample_image, 
                      tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)), 
                            (255,0,0), 2)
    
    ax[idx].imshow(sample_image)
plt.show()

# Step 11 - Save Model
facetracker.save('facedetection.keras')