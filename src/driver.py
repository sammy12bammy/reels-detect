# Live Test 

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
facetracker = load_model('facedetection.keras')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    retval, frame = cap.read() # returns a boolean, and captured frame which is a numpy array
    frame = frame[50:500, 50:500,:] 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(frame_rgb, (120,120))

    pred = facetracker.predict(np.expand_dims(resized/255, 0))
    sample_coords = pred[1][0]

    if pred[0] > 0.5:
        # controls main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                            (255,0,0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        # Controls text render
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow('EyeTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()