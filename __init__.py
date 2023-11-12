import numpy as np
from LoadImages import *
from HaarFeatures import *
from IntegralImage import *
from Dataset import *
import pickle

images, labels = combine_dataset()

# for testing only
features = generate_features(50, 50)
print(len(features))
print(labels[0])

# testing matrix calculation
# features = (235, 576, 50, 50)
# feature_value_matrice = compute_feature_with_matrix(images[0], 0, "Two Horizontal", features[94702])

# testing integral image calculation
# b, g, r = combine_integral_grb(images[0])
# feature_value_integral = compute_feature_value(b, "Two Horizontal", features[94702])

# print("starting...")
# images_data = np.zeros(len(images), dtype=object)
# for i in range (len(images)):
#     if labels[i] == 1:
#         new_Data = Dataset(images[i], labels[i], features)
#         images_data[i] = new_Data

# with open('data.fish', 'wb') as file:
#     pickle.dump(images_data, file)

# with open('data.fish', 'rb') as file:
#     images_data = pickle.load(file)
# for i in range (len(images_data)):
#     if labels[i] == 1:
#         test_image = images_data[i]
#         print(i)
#          print(test_image.window_1_features)

# toprint = images[0]
# np.savetxt('Feature_visualisation.txt', toprint[185:185+50, 61:61+50, 0], fmt='%d')
# # print(feature_value_matrice)
# # print(feature_value_integral)
# cv2.imwrite('ImageTest.png', images[0])