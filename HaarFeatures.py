import numpy as np

# feature_height and feature_width is the minimal size of the feature that are going to be used
# 2 rectangle feature and 4 rectangle feature minimum size is: 2x2
# while a 3 rectangle feature minimum size is either: 1x3 or 3x1

def generate_features(image_height, image_width, feature_type):

    # minimum feature heigth and width go here
    match feature_type:
        case "Two Horizontal" | "Two Vertical" | "Four Diagonal" | "Right Triangular" | "Left Triangular":
            feature_height = 50
            feature_width = 50
        case "Three Horizontal":
            feature_height = 50
            feature_width = 150
        case "Three Vertical":
            feature_height = 150
            feature_width = 50
        
    features = []
    for w in range (feature_width, image_width+1, feature_width):
        for h in range (feature_height, image_height+1, feature_height):
            for x in range (0, image_width - w):
                for y in range (0, image_height - h):
                    feature = (x, y, w, h)
                    features.append(feature)
    return features

# using integral image
def compute_square_value(integral_image, x, y, width, height):
    return integral_image[width, height] + integral_image[x, y] - integral_image[width, y] - integral_image[x, height]

def compute_feature_value(integral_image, feature_type, feature):
    x, y, width, height = feature
    match feature_type:
        case "Two Horizontal":
            white = compute_square_value(integral_image, x, y, int(width/2), height)
            black = compute_square_value(integral_image, x + int(width/2), y, int(width/2), height)
        case "Two Vertical":
            white = compute_square_value(integral_image, x, y, width, int(height/2))
            black = compute_square_value(integral_image, x, y + int(height/2), width, int(height/2))
        case "Three Horizontal":    
            white = compute_square_value(integral_image, x, y, int(width/3), height) + compute_square_value(integral_image, x + int(width*2/3), y, int(width/3), height)
            black = compute_square_value(integral_image, x + int(width/3), y, int(width/3), height)
        case "Three Vertical":    
            white = compute_square_value(integral_image, x, y, width, int(height/3)) + compute_square_value(integral_image, x, y + int(height*2/3), width, int(height/3))
            black = compute_square_value(integral_image, x, y + int(height/3), width, int(height/3))
        case "Four Diagonal":
            white = compute_square_value(integral_image, x, y, int(width/2), int(height/2)) + compute_square_value(integral_image, x + int(width/2), y + int(height/2), int(width/2), int(height/2))
            black = compute_square_value(integral_image, x + int(width/2), y, int(width/2), int(height/2)) + compute_square_value(integral_image, x, y + int(height/2), int(width/2), int(height/2))
    return white - black

# using Matrices
# color channel BRG = 0, 1, 2
def compute_feature_with_matrix(image, color_channel, feature_type ,feature):
    x, y, width, height = feature
    # image [y vertical:y vertical +1, x horizontal: x horizontal +1, color_channel]
    # +1 due to slicing paramter = start at:stop before
    match feature_type:
        case "Two Horizontal":
            white = np.sum(image[y:y + height + 1, x:x + int(width/2) + 1, color_channel])
            black = np.sum(image[y:y + height + 1, x + int(width/2):x + width + 1, color_channel])
        case "Two Vertical":
            white = np.sum(image[y:y + int(height/2) + 1, x:x + width+1, color_channel])
            black = np.sum(image[y + int(height/2):y + height + 1, x:x + width+1, color_channel])
        case "Three Horizontal":
            white = np.sum(image[y: y + height + 1, x:x + int(width/3) + 1, color_channel]) + np.sum(image[y: y + height + 1, x + int(width*2/3):x + width + 1, color_channel])
            black = np.sum(image[y: y + height + 1, x + width/3:x + int(width*2/3) + 1, color_channel])
        case "Three Vertical":
            white = np.sum(image[y:y + int(height/3) + 1, x:x + width + 1, color_channel]) + np.sum(image[y + int(height*2/3):y + height + 1, x: x + width + 1, color_channel])
            black = np.sum(image[y + int(height/3):y + int(height*2/3) + 1, x:x + width + 1, color_channel])
        case "Four Diagonal":
            white = np.sum(image[y:y + int(height/2) + 1, x + int(width/2): x + width + 1, color_channel]) + np.sum(image[y + int(height/2):y + height + 1, x: x + int(width/2) + 1, color_channel])
            black = np.sum(image[y:y + int(height/2) + 1, x:x + int(width/2) + 1, color_channel]) + np.sum(image[y + int(height/2): y + height + 1, x + int(width/2):x + width + 1, color_channel])
        case "Right Triangular":
            matrix = image[y:y + height + 1, x:x + width + 1, color_channel]
            white = np.sum(np.tril(matrix))
            black = np.sum(np.triu(matrix))
        case "Left Triangular":
            matrix = np.rot90(image[y:y + height + 1, x:x + width + 1, color_channel], k=3)
            white = np.sum(np.tril(matrix))
            black = np.sum(np.triu(matrix))
    return int(white) - int(black)
    
        




