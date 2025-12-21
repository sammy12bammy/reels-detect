# getting data
import os
import time
import uuid
import cv2


# getting images
IMAGES_PATH = os.path.join('data', 'images')
number_images = 30

# take 30 images
# test camera number right now 1 prob 0
# right now using 0 for built in webcam
cap = cv2.VideoCapture(0)
for img_num in range(number_images):
    print('Taking image {}'.format(img_num))
    ret, frame = cap.read()
    # uuid1 is unqiue file name
    img_name = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(img_name, frame)
    # show to screen
    cv2.imshow('frame', frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print('Finished taking images')

import tensorflow as tf
