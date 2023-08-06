import numpy as np

# feature_height and feature_width is the minimal size of the feature that are going to be used
# 2 rectangle feature and 4 rectangle feature minimum size is: 2x2
# while a 3 rectangle feature minimum size is either: 1x3 or 3x1

def generate_features(image_height, image_width, feature_height, feature_width):
    features = []
    for w in range (feature_width, image_width+1, feature_width):
        for h in range (feature_height, image_height+1, feature_height):
            for x in range (0, image_width - w):
                for y in range (0, image_height - h):
                    feature = (x, y, w, h)
                    features.append(feature)
    return features

