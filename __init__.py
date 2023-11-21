import numpy as np
import pickle
import pandas as pd
from LoadImages import *
from HaarFeatures import *
from IntegralImage import *
from Dataset import *

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

print("starting...")
# windows_1_value = [0] * len(images)

# for i in range (len(images)):
#     if labels[i] == 1:
#         new_Data = Dataset(images[i], labels[i], features)
#         # images_data[i] = new_Data
#         windows_1_value[i] = new_Data.window_1_features

# initialize main Dataframe for keys, images, and labels
# image_ids = np.arange(len(images))

# final_dataset = {
#     'image_ids' : image_ids,
#     'images' : images,
#     'labels' : labels,
# }
# final_table = pd.DataFrame(final_dataset)

# initialize separate Dataframe for features list (Data gembrot)
temp_window_value = np.zeros(len(images), dtype=object)
temp_value = np.zeros(len(features))
image_ids = np.arange(len(images))

for i in range(len(images)):
    new_Data = Dataset(images[i], labels[i], features)
    temp_window_value[i] = new_Data.window_1_features

window_1_feature = {
    'image_ids' : image_ids,
}
for i in range(len(features)):
    for j in range(len(images)):
        temp_value[j] = temp_window_value[j][i]
    column_name = f'win_1_feature_{i}'
    window_1_feature[column_name] = temp_value

print(window_1_feature.shape)
print(len(image_ids))
print(len(window_1_feature['win_1_feature_1']))
print(len(window_1_feature['win_1_feature_2']))

window_1_feature = pd.DataFrame(window_1_feature)
window_1_feature.to_csv('Data/image_features.csv', index=False)

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