# Live Face and Mouth Detection
# 
# Real-time inference using 3-head model:
#   - Face detection (binary classification)
#   - Face localization (bounding box regression)
#   - Mouth open detection (binary classification)

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained 3-head model
facetracker = load_model('facedetection.keras')

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while cap.isOpened():
    retval, frame = cap.read()
    if not retval:
        break
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Prepare frame for model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(frame_rgb, (120,120))

    # Get predictions from all 3 heads
    face_class_pred, face_bbox_pred, mouth_open_pred = facetracker.predict(
        np.expand_dims(resized/255, 0), verbose=0
    )
    
    face_detected = face_class_pred[0][0] > 0.5
    mouth_is_open = mouth_open_pred[0][0] > 0.5
    bbox_coords = face_bbox_pred[0]

    if face_detected:
        # Draw face bounding box
        cv2.rectangle(frame, 
                      tuple(np.multiply(bbox_coords[:2], [w, h]).astype(int)),
                      tuple(np.multiply(bbox_coords[2:], [w, h]).astype(int)), 
                      (0, 255, 0), 2)  # Green box for face
        
        # Determine label text and color based on mouth state
        label_text = 'MOUTH OPEN' if mouth_is_open else 'Mouth Closed'
        label_color = (0, 0, 255) if mouth_is_open else (255, 0, 0)  # Red if open, Blue if closed
        
        # Draw label background rectangle
        text_pos = tuple(np.multiply(bbox_coords[:2], [w, h]).astype(int))
        cv2.rectangle(frame, 
                      tuple(np.add(text_pos, [0, -35])),
                      tuple(np.add(text_pos, [200, 0])), 
                      label_color, -1)
        
        # Draw label text
        cv2.putText(frame, label_text, 
                    tuple(np.add(text_pos, [5, -10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display confidence scores (optional - for debugging)
        conf_text = f"Face: {face_class_pred[0][0]:.2f} | Mouth: {mouth_open_pred[0][0]:.2f}"
        cv2.putText(frame, conf_text,
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imshow('Face & Mouth Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Tracker closed")