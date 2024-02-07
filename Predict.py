import numpy as np
import os
import cv2
from Cascade import *
from Utilities import *

# load cascades for each window
window_cascade = [None, None, None]
window_prediction = np.zeros(3)
window_cascade[0] = Utilities.read_from_pickle('window_0_cascade') #for left side/mouth detection
window_cascade[1] = Utilities.read_from_pickle('window_1_cascade') #for mid side/fin detection
window_cascade[2] = Utilities.read_from_pickle('window_2_cascade') #for right side/tail detection

directory = "classification_target"

for filename in os.listdir(directory):
    if filename.endswith(".png"):
        window_prediction = np.zeros(3)
        image_path = os.path.join(directory, filename)
        image_name = filename
        window_index = [None] * 3

        #load target image for classification
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_unedited = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (350, 200))
        image_width = 350
        image_height = 200
 
        # scan for the whole image using sliding windows
        for i in range(len(window_cascade)):
            # offset for different part all 3 window 
            match i:
                case 0: left_window_width = 0
                case 1: left_window_width = int(image_width / 3)
                case 2: left_window_width = int(image_width / 3 * 2)
            for x in range(0, int(image_width / 3) - 50 +1):
                for y in range(0, image_height - 50 + 1):
                    prediction = window_cascade[i].final_cascade_classification(image, x + left_window_width, y)
                    print(f' window: {i}, x: {x}, y: {y} complete! class: {prediction}')
                    if prediction != 0: break
                if prediction != 0: break
            # print(f'classification result for window {i}: {prediction}')
            print(f'classified in window: {i} x: {x}, y: {y}, class: {prediction}')
            window_index[i] = f'[{x+left_window_width},{y}]'
            window_prediction[i] = prediction

        # count majority vote and predict class
        print(f'result of {image_name} classification: {window_prediction}')
        unique_elements, counts = np.unique(window_prediction, return_counts=True)
        max_count_index = np.argmax(counts)

        if counts[max_count_index] > len(window_prediction) // 2:
            image_class = unique_elements[max_count_index]
        else:
            image_class = 0

        match image_class:
            case 0: image_class = 'None'
            case 1: image_class = 'Ikan Emas'
            case 2: image_class = 'Ikan Lele'
            case 3: image_class = 'Ikan Nila'

        position = (10, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        font_color = (0, 0, 255)

        output_image_path = os.path.join(f'classification_results\\ {window_prediction}{window_index}' + os.path.splitext(filename)[0] + '.jpg')
        cv2.putText(image_unedited, image_class, position, font, font_scale, font_color, font_thickness)
        cv2.imwrite(output_image_path, image_unedited)
        print('anotation image completed!')
