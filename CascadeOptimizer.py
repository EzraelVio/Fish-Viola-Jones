import numpy as np
import os
import cv2
from Prototype import *
from Cascade import *
from Utilities import *

images, labels = prototype.start_load()
image_width = 350
image_height = 200
window_cascade = [None, None, None]
window_cascade[0] = Utilities.read_from_pickle('window_0_cascade') #for left side/mouth detection
window_cascade[1] = Utilities.read_from_pickle('window_1_cascade') #for mid side/fin detection
window_cascade[2] = Utilities.read_from_pickle('window_2_cascade') #for right side/tail detection

positive_tracker = np.zeros((image_height, image_width))




for x in range(0, int(image_width / 3) - 50 +1):
    for y in range(0, image_height - 50 + 1):
        for i in range(len(images)):
            prediction = window_cascade[0].final_cascade_classification(images[i], x, y)
            if prediction == labels[i] and prediction != 0:
                positive_tracker[y][x] += 1
            
nonzero_indices = np.transpose(np.nonzero(positive_tracker))
# Sort indices based on values in descending order
sorted_indices = sorted(nonzero_indices, key=lambda idx: positive_tracker[idx[0], idx[1]], reverse=True)
orderlist = np.array(sorted_indices)

while True:
    alpha_list = np.zeros(image_width * image_height)
    prediction = np.zeros(len(images))
    for i in range(len(output)):
        prediction = np.zeros(len(images))


