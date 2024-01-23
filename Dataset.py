import numpy as np

from HaarFeatures import *
from IntegralImage import *

class Dataset:

    class_Window_offset_1 = [
        # order according to label's order in LoadImages
        # for searching mouth features

        (0, 0),
        (0, 70),
        (17, 77),
        (1, 87)
    ]

    class_Window_offset_2 = [
        # order according to label's order in LoadImages
        # for searching fin features

        (0, 0),
        (116, 15),
        (117, 50),
        (116, 15)
    ]

    class_Window_offset_3 = [
        # order according to label's order in LoadImages
        # for searching tail feature

        (0, 0),
        (279, 65),
        (300, 90),
        (300, 69)
    ]

    def __init__(self, image, label, feature_list):
        self.image = image
        self.label = label
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
            data_features = compute_feature_with_matrix(image, updated_feature)
            features[i] = data_features
        return features
        