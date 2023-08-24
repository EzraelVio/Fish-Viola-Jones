import numpy as np

from HaarFeatures import *
from IntegralImage import *

class Dataset:
    class_Window_offset_1 = [
        # order according to label's order in LoadImages
        # for searching mouth feature
        (0, 0), 
        (0, 0),
        (0, 0),
        (0, 0)
    ]

    class_Window_offset_2 = [
        # order according to label's order in LoadImages
        # for searching fin feature
        (0, 0), 
        (0, 0),
        (0, 0),
        (0, 0)
    ]

    class_Window_offset_3 = [
        # order according to label's order in LoadImages
        # for searching tail feature
        (0, 0), 
        (0, 0),
        (0, 0),
        (0, 0)
    ]

    def __init__(self, image, label, feature_list):
        self.image = image
        self.label = label
        self.blue_Channel, self.green_Channel, self.red_Channel = combine_integral_grb(image)
        self.window_1_features = self.Find_Feature_Value(image, feature_list, self.class_Window_offset_1[label, 0], self.class_Window_offset_1[label, 1])

    def Find_Feature_Value(image, feature_list, x_offset, y_offset):
        features = []
        for i in range(0, feature_list):
            feature_type, x, y, width, height = feature_list[i]
            x += x_offset
            y += y_offset
            updated_feature = (feature_type, x, y, width, height)
            blue_channel_feature_value = compute_feature_with_matrix(image, 0, updated_feature)
            green_channel_feature_value = compute_feature_with_matrix(image, 1, updated_feature)
            red_channel_feature_value = compute_feature_with_matrix(image, 3, updated_feature)
            data_features = (blue_channel_feature_value, green_channel_feature_value, red_channel_feature_value)
            features.append(data_features)
        