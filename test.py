import cv2
import numpy as np
from Utilities import *

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
from Cascade import *

from Dataset import *
from sklearn.metrics import accuracy_score

# # === GROUP A ===
# target_image_name = 'target.png'
# image_unedited = cv2.imread(target_image_name, cv2.IMREAD_UNCHANGED)

# print(np.shape(image_unedited))

# position = (10, 30)
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 1
# font_thickness = 2
# font_color = (0, 0, 255)

# cv2.putText(image_unedited, 'contoh', position, font, font_scale, font_color, font_thickness)
# cv2.imwrite('hasil.jpg', image_unedited)

# === GROUP B ===
images, labels = combine_dataset()
initial_features = generate_features(50, 50)
# csv_name_loop = f'fish_window_1'
# splits = DecisionTree.split_data(initial_features, csv_name_loop, labels)
# X_train, Y_train, X_test, Y_test, X_valid, Y_valid = splits
# # print(Y_train)
# testing =[]

# pickle_name = f'window_1_decision_trees'
# window_1_decision_trees = Utilities.read_from_pickle(pickle_name)
# trees = window_1_decision_trees.trees
# print(window_1_decision_trees.accuracies[:10])
# for i in range(len(window_1_decision_trees.accuracies)):
#     # if window_1_decision_trees.accuracies[i] > 0.17857142857142858 and window_1_decision_trees.accuracies[i] < 0.25:
#     if window_1_decision_trees.accuracies[i] > 0.4:
#         testing.append(window_1_decision_trees.accuracies[i])
# print(np.shape(testing))
# print(testing[:10])

for i in range(3):
    pickle_name = f'window_{i}_strong_classsifier'
    strong_classifier = Utilities.read_from_pickle(pickle_name)
    # print(strong_classifier.trees[:10])
    print(np.shape(strong_classifier.trees))
    print(np.shape(strong_classifier.features))
    print(strong_classifier.features[:5])
    print(strong_classifier.alpha_list[:5])

    # csv_name_loop = f'fish_window_{i}'
    # splits = DecisionTree.split_data(initial_features, csv_name_loop, labels)
    # X_train, Y_train, X_test, Y_test, X_valid, Y_valid = splits

    # pickle_name = f'window_{i}_cascade'
    # orderlist = np.arange(len(strong_classifier.features))
    # validation_prediction = Boosting.strong_prediction(strong_classifier.trees, orderlist, X_valid, strong_classifier.alpha_list)
    # current_accuracy = accuracy_score(Y_valid, validation_prediction)
    # print(current_accuracy)

    # cascade = Cascade()
    # cascade.fill_cascade(strong_classifier.features, strong_classifier.trees, strong_classifier.alpha_list, splits)
    # cascade.save_to_pickle(pickle_name)

# orderlist = np.arange(len(strong_classifier.trees))
# prediction = Boosting.strong_prediction(strong_classifier.trees, orderlist, X_test, strong_classifier.alpha_list)
# # prediction = window_1_decision_trees.trees[100].predict(X_test)
# classifiers_accuracy = accuracy_score(Y_test, prediction)
# print(Y_test.flatten())
# print(prediction)
# print(classifiers_accuracy)

# # === GROUP C ===
# # read image and assign appropriate labels
# images, labels = combine_dataset()

# # initialize haar_like features. Change 50 x 50 according to neccesity
# initial_features = generate_features(50, 50)

# print(f'Numbers of features generated: {len(initial_features)}')

# print("starting...")

# # generate CSV of image feature values. Run if CSV have not been made
# # write_csv splits data into 3 dataframes. Image feature value depends on preassigned windows and class in Dataset.py
# csv_name = "fish" # change name accordingly
# # Utilities.write_csv(images, labels, initial_features, csv_name)
# # temp_window_values = np.zeros((len(images), len(initial_features)), dtype=object)
# # image_ids = np.arange(len(images))
# # for i in range(len(images)):
# #     new_data = Dataset(images[i], labels[i], initial_features)
# #     temp_window_values[i] = new_data.window_3_features

# # window_feature = {'image_ids': image_ids}
# # for i in range(len(initial_features)):
# #     column_name = f'win_{2 + 1}_feature_{i}'
# #     window_feature[column_name] = temp_window_values[:, i]

# # directory = f"Data/{csv_name}_window_2.csv"

# # window_feature = pd.DataFrame(window_feature)
# # window_feature.to_csv(directory, index=False)

# # create weak classifiers (Decision Trees) for each window

# splits = []
# trees = []
# accuracies = []
# features = initial_features
# csv_name_loop = f'{csv_name}_window_0'
# # split data into 3 part for training and saving it inside splits:
# # X_train and Y_train for creating trees
# # X_test and Y_test for boosting
# # X_valid and Y_valid for training final strong classifier and cascade
# splits = DecisionTree.split_data(features, csv_name_loop, labels)

# # create decision tree and saving it in pickle for later. Skip if Pickel has already been made
# # Long ahh progress est. 2+ hours for all 3 window
# # trees, accuracies = DecisionTree.build_all_tree(splits, features)
# # decision_trees = PickleTree(features, trees, accuracies)
# # pickle_name = f'window_2_decision_trees'
# # Utilities.dump_to_pickle(pickle_name, decision_trees)

# pickle_name = f'window_0_decision_trees'
# window_1_decision_trees = Utilities.read_from_pickle(pickle_name)
# trees = window_1_decision_trees.trees
# accuracies = window_1_decision_trees.accuracies
    

# # train strong classifier which also double as feature elimination. Saving it into another pickle
# pickle_name = f'window_0_strong_classsifier'
# Boosting.training_strong_classifier(features, trees, splits, accuracies, pickle_name)