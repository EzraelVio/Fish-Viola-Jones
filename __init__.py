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

# read image and assign appropriate labels
images, labels = combine_dataset()

# initialize haar_like features. Change 50 x 50 according to neccesity
initial_features = generate_features(50, 50)

print(f'Numbers of features generated: {len(initial_features)}')

print("starting...")

# generate CSV of image feature values. Run if CSV have not been made
# write_csv splits data into 3 dataframes. Image feature value depends on preassigned windows and class in Dataset.py
csv_name = "fish" # change name accordingly
# Utilities.write_csv(images, labels, features, csv_name)

# create weak classifiers (Decision Trees) for each window
for i in range(3):
    features = initial_features
    csv_name_loop = f'{csv_name}_window_{i}'
    # split data into 3 part for training and saving it inside splits:
    # X_train and Y_train for creating trees
    # X_test and Y_test for boosting
    # X_valid and Y_valid for training final strong classifier and cascade
    splits = DecisionTree.split_data(features, csv_name_loop, labels)

    # create decision tree and saving it in pickle for later. Skip if Pickel has already been made
    # Long ahh progress est. 2+ hours for all 3 window
    trees, accuracies = DecisionTree.build_all_tree(splits, features)
    decision_trees = PickleTree(features, trees, accuracies)
    pickle_name = f'window_{i}_decision_trees'
    Utilities.dump_to_pickle(pickle_name, decision_trees)

    # read pickle for further use in creating strong classifier.
    # Run only if pickle for decision trees exist
    '''
    window_1_decision_trees = Utilities.read_from_pickle(pickle_name)
    trees = window_1_decision_trees.trees
    accuracies = window_1_decision_trees.accuracies
    '''

    # train strong classifier which also double as feature elimination. Saving it into another pickle
    pickle_name = f'window_{i}_strong_classsifier'
    Boosting.training_strong_classifier(features, trees, splits, accuracies, pickle_name)


# for j in range (3):
#     strong_classifier = Utilities.read_from_pickle(pickle_name)
#     pickle_name = f'window_{j}_cascade'
#     cascade = Cascade()
#     cascade.fill_cascade(strong_classifier.features, strong_classifier.trees, strong_classifier.alpha_list, splits)
#     cascade.save_to_pickle(pickle_name)


# i = 1
# temp_window_value1 = np.zeros(len(images), dtype=object)
# new_Data = Dataset(images[i], labels[i], features)
# temp_window_value1[i] = new_Data.window_1_features

# feat1 = str(temp_window_value1[i][0])
# feat2 = str(temp_window_value1[i][520703])
# print (feat1)
# print (feat2)

# toprint = images[0]
# np.savetxt('Feature_visualisation.txt', toprint[185:185+50, 61:61+50, 0], fmt='%d')
# # print(feature_value_matrice)
# # print(feature_value_integral)
# cv2.imwrite('ImageTest.png', images[0])