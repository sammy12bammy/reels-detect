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
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Optional: center crop to square (comment out to use full frame)
    # size = min(h, w)
    # start_x = (w - size) // 2
    # start_y = (h - size) // 2
    # frame = frame[start_y:start_y+size, start_x:start_x+size, :]
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(frame_rgb, (120,120))

    pred = facetracker.predict(np.expand_dims(resized/255, 0))
    sample_coords = pred[1][0]

    if pred[0] > 0.5:
        # controls main rectangle - scale to actual frame size
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [w, h]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [w, h]).astype(int)), 
                            (255,0,0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [w, h]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [w, h]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        # Controls text render
        cv2.putText(frame, 'WANGHAF Detect', tuple(np.add(np.multiply(sample_coords[:2], [w, h]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow('EyeTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()