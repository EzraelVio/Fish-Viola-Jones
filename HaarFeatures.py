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

def compute_square_value(integral_image, feature):
    x, y, width, height = feature
    return integral_image[width, height] + integral_image[x, y] - integral_image[width, y] - integral_image[x, height]

    # if width > 2:
    #     integral_a = integral_image[x - 1, y - 1]
    #     integral_b = integral_image[x + width/2 - 1, y - 1]
    #     integral_c = integral_image[x + width - 1, y - 1]
    #     integral_d = integral_image[x - 1, y + height - 1]
    #     integral_e = integral_image[x + width/2 - 1, y + height - 1]
    #     integral_f = integral_image[x + width - 1, y + height - 1]
    #     white_sum = integral_e - integral_b - integral_d + integral_a
    #     black_sum = integral_f - integral_c - integral_e + integral_b

    # return white_sum - black_sum

def compute_feature_value(integral_image, feature_type, feature):
    x, y, width, height = feature
    if feature_type == "Two Horizontal":
        white = compute_square_value(x, y, width/2, height)
        black = compute_square_value(x + width/2, y, width/2, height)
    elif feature_type == "Two Vertical":
        white = compute_square_value(x, y, width, height/2)
        black = compute_square_value(x, y + height/2, width, height/2)
    elif feature_type == "Three Horizontal":    
        white = compute_square_value(x, y, width/3, height) + compute_square_value(x + width*2/3, y, width/3, height)
        black = compute_square_value(x + width/3, y, width/3, height)
    elif feature_type == "Three Vertical":    
        white = compute_square_value(x, y, width, height/3) + compute_square_value(x, y + height*2/3, width, height/3)
        black = compute_square_value(x, y + height/3, width, height/3)
    elif feature_type == "Four Diagonal":
        white = compute_square_value(x, y, width/2, height/2) + compute_square_value(x + width/2, y + height/2, width/2, height/2)
        black = compute_square_value(x + width/2, y, width/2, height/2) + compute_square_value(x, y + height/2, width/2, height/2)
    return white - black
