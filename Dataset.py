import numpy as np

from HaarFeatures import *
from IntegralImage import *

class Dataset:
    # Note: ini offset buat daun sekarang, jangan lupa ganti buat ikan

    class_Window_offset_1 = [
        # order according to label's order in LoadImages
        # for searching mouth feature
        (103, 0),
        (106, 18), 
        (106, 34),
        (0, 0)
    ]

    class_Window_offset_2 = [
        # order according to label's order in LoadImages
        # for searching fin feature
        (106, 163),
        (106, 178), 
        (106, 171),
        (0, 0)
    ]

    class_Window_offset_3 = [
        # order according to label's order in LoadImages
        # for searching tail feature
        (110, 300),
        (106, 274), 
        (106, 257),
        (0, 0)
    ]

    def __init__(self, image, label, feature_list):
        self.image = image
        self.label = label
        # self.blue_Channel, self.green_Channel, self.red_Channel = combine_integral_grb(image)
        self.window_1_features = self.Find_Feature_Value(image, feature_list, self.class_Window_offset_1[label][0], self.class_Window_offset_1[label][1])
        self.window_2_features = self.Find_Feature_Value(image, feature_list, self.class_Window_offset_2[label][0], self.class_Window_offset_2[label][1])
        self.window_3_features = self.Find_Feature_Value(image, feature_list, self.class_Window_offset_3[label][0], self.class_Window_offset_3[label][1])

    def Find_Feature_Value(self, image, feature_list, x_offset, y_offset):
        features = np.zeros(len(feature_list), dtype=object)
        for i in range(len(feature_list)):
            feature_type, x, y, width, height = feature_list[i]
            x += x_offset
            y += y_offset
            updated_feature = (feature_type, x, y, width, height)
            # blue_channel_feature_value = compute_feature_with_matrix(image, 0, updated_feature)
            # green_channel_feature_value = compute_feature_with_matrix(image, 1, updated_feature)
            # red_channel_feature_value = compute_feature_with_matrix(image, 2, updated_feature)
            data_features = compute_feature_with_matrix(image, 0, updated_feature)
            # data_features = (blue_channel_feature_value, green_channel_feature_value, red_channel_feature_value)
            features[i] = data_features
        return features
        