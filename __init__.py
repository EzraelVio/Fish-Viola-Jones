import numpy as np
import pickle
import pandas as pd
from LoadImages import *
from HaarFeatures import *
from IntegralImage import *
from Dataset import *
from Utilities import *
from DecisionTree import *
from Boosting import *

images, labels = combine_dataset()

# for testing only
features = generate_features(50, 50)
print(len(features))

print("starting...")

# csv_name = "leaves"
# Utilities.write_csv(images, labels, features, csv_name)

csv_name = "leaves_window_1"
splits = DecisionTree.split_data(features, csv_name, labels)
# trees, accuracies = DecisionTree.build_all_tree(splits, features)

# window_1_decision_trees = PickleTree(features, trees, accuracies) 
# Utilities.dump_to_pickle('window_1_decision_trees', window_1_decision_trees)

window_1_decision_trees = Utilities.read_from_pickle('window_1_decision_trees')
trees = window_1_decision_trees.trees
accuracies = window_1_decision_trees.accuracies

Boosting.training_strong_classifier(trees, splits, accuracies)

# testing matrix calculation
# features = (235, 576, 50, 50)
# feature_value_matrice = compute_feature_with_matrix(images[0], 0, "Two Horizontal", features[94702])

# testing integral image calculation
# b, g, r = combine_integral_grb(images[0])
# feature_value_integral = compute_feature_value(b, "Two Horizontal", features[94702])

# i = 1
# temp_window_value1 = np.zeros(len(images), dtype=object)
# new_Data = Dataset(images[i], labels[i], features)
# temp_window_value1[i] = new_Data.window_1_features

# feat1 = str(temp_window_value1[i][0])
# feat2 = str(temp_window_value1[i][520703])
# print (feat1)
# print (feat2)

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